
from datasets.build import *



def build_dataloader_glob_loc(config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
    include_ps_scores = not (config.DATA.TRAIN_PS_SCORES == '' or config.DATA.TRAIN_PS_SCORES is None)
    if include_ps_scores:
        label_key = 'ps_scores'
    else:
        label_key = 'label'

    global_transform1 = None
    global_transform2 = None
    local_transform = None

    decode_and_resize = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        # sample consecutive frames
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),  # the shorter side is resized to 256
    ]


    collect_and_totensor = [
        dict(type='Collect', keys=['imgs', label_key], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', label_key]),
    ]

    # todo  RandomResizedCrop,  Flip, ColorJitter, GrayScale, Normalize, FormatShape

    # todo ##########################################################################################
    train_pipeline = [  # todo pipeline is the list of transformations
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        # sample consecutive frames
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
    # todo ##########################################################################################


class VideoDataset_glob_loc(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index = 0, train_ps_scores = None,
                 pipeline_glo1 = None, pipeline_glo2 = None, pipeline_loc =None,
                 **kwargs):
        self.train_ps_scores = train_ps_scores
        super().__init__(ann_file, pipeline, start_index= start_index, **kwargs)
        # BaseDataset
        #  load_annotations,  load_json_annotations, parse_by_class, label2array, dump_results, prepare_train_frames, prepare_test_frames
        #  __len__,  __getitem__
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
            vid_scores = vid_scores / np.sum(vid_scores, axis=1, keepdims=True)  # normalize scores

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
                    video_infos.append(
                        dict(filename=filename, ps_scores=vid_scores[line_id, :], tar=self.use_tar_format))
                else:
                    video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        return video_infos

    def prepare_train_frames(self, idx):
        """ prepare the frames for training given the index. """
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot



    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)