from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
import tarfile
from .pipeline import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 repeat = 1,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False,):
        super().__init__()
        self.use_tar_format = True if ".tar" in data_prefix else False
        data_prefix = data_prefix.replace(".tar", "")
        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline) # todo   self.pipeline is callable
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        # here  results is a dict of 5 keys:  filename, ps_scores, tar, modality, start_index
        aug1 = self.pipeline(results)  #  a dict of 'imgs': tensor (n_frames, 3, 224, 224)   'label'
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {"imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                    "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)

class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, train_ps_scores = None,  soft_k = None,   **kwargs):
        self.train_ps_scores = train_ps_scores
        self.soft_k = soft_k
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file


    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        include_ps_scores = not (self.train_ps_scores == '' or self.train_ps_scores is None)
        if include_ps_scores:
            vid_scores = np.load(self.train_ps_scores)
            # todo  keep only the top k scores
            n_vids = vid_scores.shape[0]
            idx = np.argpartition(vid_scores, -self.soft_k, axis=1)[:, -self.soft_k:] # Indices are not sorted
            vid_scores_new = np.zeros_like(vid_scores)
            vid_scores_new[np.arange(n_vids)[:, None], idx] = vid_scores[np.arange(n_vids)[:, None], idx]  # todo  keep only the top k scores

            vid_scores = vid_scores_new
            vid_scores = vid_scores / np.sum(vid_scores, axis=1, keepdims=True) # normalize scores

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line_id, line in enumerate(fin):
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                if include_ps_scores:
                    video_infos.append(dict(filename=filename, ps_scores= vid_scores[line_id,:], tar=self.use_tar_format))
                else:
                    video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        return video_infos


class VideoDataset_w_idx(VideoDataset):
    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx), idx

        return self.prepare_train_frames(idx), idx


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

class SubsetSequentialSampler(SubsetRandomSampler):
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    include_ps_scores = not (config.DATA.TRAIN_PS_SCORES == '' or config.DATA.TRAIN_PS_SCORES is None)
    if include_ps_scores:
        label_key = 'ps_scores'
    else:
        label_key = 'label'

    train_pipeline = [    #  todo pipeline is the list of transformations
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES), #  sample consecutive frames
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),  #
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', label_key], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', label_key]),
    ]

    
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline, train_ps_scores= config.DATA.TRAIN_PS_SCORES, soft_k= config.DATA.SOFT_K )
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )

    train_data_w_idx = VideoDataset_w_idx(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline, train_ps_scores= config.DATA.TRAIN_PS_SCORES, soft_k= config.DATA.SOFT_K )
    sampler_train_w_idx = torch.utils.data.DistributedSampler(
        train_data_w_idx, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader_w_idx = DataLoader(
        train_data_w_idx, sampler=sampler_train_w_idx,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    #######################################################################################################################################################
    #############################################  validation dataloader     ##########################################################################################################
    #######################################################################################################################################################
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)

    multi_class = True if config.DATA.DATASET == 'Charades' else False
    num_classes = config.DATA.NUM_CLASSES if config.DATA.DATASET == 'Charades' else None
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline, multi_class=multi_class, num_classes=num_classes)

    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())  # indices of validation videos,  0,1,2,...
    sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=config.VAL_BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,  # why should we drop last?
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.VAL_BATCH_SIZE),
    )

    val_data_w_idx = VideoDataset_w_idx(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline, multi_class=multi_class, num_classes=num_classes)
    sampler_val_sequential = SubsetSequentialSampler(indices)
    val_loader_sequential = DataLoader(
        val_data_w_idx, sampler=sampler_val_sequential,
        batch_size=config.VAL_BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.VAL_BATCH_SIZE),
    )

    #######################################################################################################################################################
    #############################################  validation zero-shot dataloader     ##########################################################################################################
    #######################################################################################################################################################
    if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap', 'train cls', 'train cls+synonym', 'train cls+cap max',
                                                                            'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max' ]:

        val_data_zs = VideoDataset(ann_file=config.DATA.VAL_ZS_FILE, data_prefix=config.DATA.ROOT_VAL_ZS, labels_file=config.DATA.VAL_ZS_LABEL_LIST, pipeline=val_pipeline)
        indices_zs = np.arange(dist.get_rank(), len(val_data_zs),  dist.get_world_size())  # indices of validation videos,  0,1,2,...
        sampler_val_zs = SubsetRandomSampler(indices_zs)
        val_loader_zs = DataLoader(
            val_data_zs, sampler=sampler_val_zs,
            batch_size=config.VAL_BATCH_SIZE,
            num_workers=config.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,  # why should we drop last?
            collate_fn=partial(mmcv_collate, samples_per_gpu=config.VAL_BATCH_SIZE),
        )
        return train_data, val_data, train_loader, train_loader_w_idx, val_loader, val_loader_sequential, val_loader_zs
    else:
        return train_data, val_data, train_loader, train_loader_w_idx, val_loader, val_loader_sequential