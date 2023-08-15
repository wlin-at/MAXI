import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.vificlip import TextEncoder, load_clip_to_cpu

"""
only for evaluation, no training

there are 3 cases 

1) eval_nn_for_each_class
the text features are normalized within descriptions of one class
for each video, find the nearest description for each class

2) eval_all_descriptions
text features are normalized within all descriptions of all classes


3) eval_text_ensemble
ensemble text features,  average among   self.n_description_per_class


"""


class CLIP_NearestNeighbor(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        """
        find the nearest neighbors between visual embedding  and  text embedding of a bunch of text descriptions
        :param cfg:
        :param classnames:
        :param clip_model:
        :param logger:
        """
        super().__init__()
        self.only_test = cfg.TEST.ONLY_TEST
        assert self.only_test == True
        self.dtype = clip_model.dtype
        self.description_dir = cfg.DATA.DESCRIPTION_DIR
        self.n_description_per_class = cfg.DATA.N_DESCRIPTION
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # self.clip_model = clip_model  #  this is only for compute_text_features
        self.logit_scale = clip_model.logit_scale
        # classnames = classnames[:5]
        self.classnames = classnames
        self.n_class = cfg.DATA.NUM_CLASSES
        self.use_description = cfg.DATA.USE_DESCRIPTION
        self.use_description_type = cfg.DATA.USE_DESCRIPTION_TYPE
        assert self.use_description_type in ['eval_nn_for_each_class', 'eval_all_descriptions', 'eval_text_ensemble' ]
        # with torch.no_grad():
        #     for classname in classnames:
        #         description_file = osp.join(self.description_dir, f'{classname}.txt')
        #         tokenized_prompts += [clip.tokenize(line.strip('\n')) for line in open(description_file)]#
        #     tokenized_prompts = torch.cat( tokenized_prompts)
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # prompts
        #     self.text_features = self.text_encoder(embedding,tokenized_prompts)
        #     self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.vid_descript_similarity = dict()
    def compute_text_features(self, loaded_clip_model):
        # todo during test time, the text encoder will not be updated anymore, there is no need to compute text features in every iteration
        if self.only_test:
            with torch.no_grad():
                if self.use_description:
                    if self.use_description_type == 'eval_nn_for_each_class':
                        # todo 1) the text features are normalized within descriptions of one class
                        #    for each video, find the nearest description for each class
                        self.text_feature_dict = dict()
                        for class_id, classname in enumerate(self.classnames):
                            description_file = osp.join(self.description_dir, f'{classname}.txt')
                            # tokenized_prompts += [clip.tokenize(line.strip('\n')) for line in open(description_file)]  #

                            tokenized_prompts = []
                            for line_id, line in enumerate(open(description_file)):
                                if line_id < self.n_description_per_class:
                                    tokenized_prompts.append(clip.tokenize(line.strip('\n')))
                                else:
                                    break
                            tokenized_prompts = torch.cat(tokenized_prompts).cuda()
                            # tokenized_prompts = torch.cat([clip.tokenize(line.strip('\n')) for line in open(description_file)]).cuda()  #
                            embedding = loaded_clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # prompts
                            text_features = self.text_encoder(embedding, tokenized_prompts)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            self.text_feature_dict.update({class_id: text_features})
                    elif self.use_description_type == 'eval_all_descriptions':
                        # todo 2) text features are normalized within all descriptions of all classes
                        tokenized_prompts = []
                        for class_id, classname in enumerate(self.classnames):
                            description_file = osp.join(self.description_dir, f'{classname}.txt')
                            for line_id, line in enumerate(open(description_file)):
                                if line_id < self.n_description_per_class:
                                    tokenized_prompts.append(clip.tokenize(line.strip('\n')))
                                else:
                                    break

                            # tokenized_prompts += [clip.tokenize(line.strip('\n')) for line in open(description_file)]
                        tokenized_prompts = torch.cat(tokenized_prompts).cuda()
                        embedding = loaded_clip_model.token_embedding(tokenized_prompts).type(self.dtype)  # prompts
                        self.text_features = self.text_encoder(embedding, tokenized_prompts)
                        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                    elif self.use_description_type == 'eval_text_ensemble':
                        # todo 3) ensemble text features,  average among   self.n_description_per_class
                        # text_features_avg = torch.zeros((self.n_class, ))
                        text_features_avg = []
                        for class_id, classname in enumerate(self.classnames):
                            action_name_as_filename = classname.replace('/', '') if '/' in classname else classname
                            description_file = osp.join(self.description_dir, f'{action_name_as_filename}.txt')
                            # for line_id, line in enumerate(open(description_file)):
                            tokenized_prompts = torch.cat([clip.tokenize(line.strip('\n'), truncate=True) for line in open(description_file)]).cuda()  # (n_descriptions, 77)
                            embedding = loaded_clip_model.token_embedding(tokenized_prompts).type(self.dtype)
                            text_features = self.text_encoder(embedding, tokenized_prompts)  # (n_descriptions, 512)
                            text_features_avg.append(torch.mean( text_features, dim=0, keepdim=True )) # take the average feature among all descriptions
                        self.text_features = torch.cat(text_features_avg)  # (n_cls, 512)
                        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, batch_idx):
        logit_scale = self.logit_scale.exp()
        # Lets encode the video into required format
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)  # (bz*t, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image.type(self.dtype))  # todo  (bz*t, 512)
        # Now again attach the batch dimensions
        image_features = image_features.view(b, t, -1)  # todo  [B, t, 512]
        # todo   Now take the mean along the temporal direction,   video feature the temporal averaging of frame features
        image_features = image_features.mean(dim=1, keepdim=False)  # todo  [B, t, 512]  -> ( B, 512 )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # todo  ( B, 512 )
        # for class_id, classname in enumerate
        if self.use_description:
            if self.use_description_type == 'eval_nn_for_each_class':
                # todo  1) the text features are normalized within descriptions of one class
                #    for each video, find the nearest descriptions for each class
                logits_dict = dict()
                for class_id, classname in enumerate(self.classnames):
                    text_features = self.text_feature_dict[class_id]
                    logits = logit_scale * image_features @ text_features.t()
                    logits_dict.update({class_id: logits})  #   dict of  (B, n_descriptions_per_cls)
                return logits_dict

            elif self.use_description_type in ['eval_all_descriptions', 'eval_text_ensemble' ]:
                # todo  2) text features are normalized within all descriptions of all classes
                #
                # or  todo  3) ensemble text features,  average among   self.n_description_per_class
                text_features = self.text_features
                logits =  logit_scale * image_features @ text_features.t()  #  (B, 8000 )  similarity betwen video features and text features of all descriptions
                return logits


def returnCLIP(config, logger=None, class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building ViFi-CLIP CLIP")
    model = CLIP_NearestNeighbor(config, class_names, clip_model, logger)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    if not config.TEST.ONLY_TEST:
        logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model,  clip_model



