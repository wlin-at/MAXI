import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.vificlip import TextEncoder, load_clip_to_cpu, freeze_model
import numpy as np


"""
text augmentation with action class descriptions from GPT-3 
each action class has a few descriptions, saved in   osp.join(self.description_dir, f'{classname}.txt')

"""


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, train_data = None):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.ViFi_CLIP.PROMPT_MODEL
        # ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT  #   a photo of a
        self.use_description_type = cfg.DATA.USE_DESCRIPTION_TYPE
        assert self.use_description_type == 'train_text_aug'

        self.description_dir = cfg.DATA.DESCRIPTION_DIR
        self.n_description_per_class = cfg.DATA.N_DESCRIPTION
        self.aug_n_neighbors_to_choose = self.n_description_per_class if self.use_description_type == 'train_text_aug' else int(self.use_description_type[len('train_text_aug'):])
        self.classnames = classnames
        self.n_class = cfg.DATA.NUM_CLASSES
        # ZS_evaluation = cfg.TRAINER.ViFi_CLIP.ZS_EVAL
        # No prompting

        # ctx_init = ctx_init.replace("_", " ")
        # prompt_prefix = ctx_init
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # todo comnpute the tokenization and embedding of all descriptions
        #  during training, randomly use  a tokenization and embedding  for each class,  forming   (n_cls, 77)  and  (n_cls, 77, 512)
        self.input_tokenized_dict = dict()
        input_embed_dict = dict()
        for class_id, classname in enumerate(self.classnames):
            description_file = osp.join(self.description_dir, f'{classname}.txt')
            tokenized_prompts = []
            for line_id, line in enumerate(open(description_file)):
                if line_id < self.n_description_per_class:
                    tokenized_prompts.append(clip.tokenize(line.strip('\n')))
                else:
                    break
            tokenized_prompts = torch.cat(tokenized_prompts) #  (n_description, 77)
            with torch.no_grad():
                embedding  = clip_model.token_embedding(tokenized_prompts).type(dtype) #  (n_description, 77)
            self.input_tokenized_dict.update({class_id: tokenized_prompts})
            input_embed_dict.update({class_id: embedding})

        # self.register_buffer( 'input_embed_dict', input_embed_dict)
        self.input_embed_dict = input_embed_dict

        if self.aug_n_neighbors_to_choose != self.n_description_per_class:
            raise Exception('This is deprecated! ')
            assert self.aug_n_neighbors_to_choose < self.n_description_per_class
            #  select the neighbor text descriptions
            self.description_result_dir = cfg.DATA.NN_DESCRIPTION_RESULT
            self.vid_text_aug_description_id = dict()
            for vid_id in range(len(train_data)):
                lines = open(osp.join(self.description_result_dir, f'vid{vid_id}.txt')).readlines()
                label = train_data.video_infos[vid_id]['label']
                similarities = np.array([ float(value)  for value  in   lines[label].strip('\n').split(' ')])
                indices = np.argsort( - similarities)[:self.aug_n_neighbors_to_choose]  # sort in descending order, choose the nearest neighbor
                self.vid_text_aug_description_id.update({vid_id:  indices})


    def forward(self):
        return self.input_embed_dict




class ViFiCLIP_text_aug(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, train_data = None):
        super().__init__()
        #   todo   the prompt template and tokenization of prompted texts, token embedding (input to transformer) are done in VLPromptLearner
        self.only_test = cfg.TEST.ONLY_TEST
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger, train_data=train_data)
        self.lambda_classification = cfg.TRAIN.LAMBDA_CLASSIFICATION
        self.lambda_contrast = cfg.TRAIN.LAMBDA_CONTRAST

        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts  #  (n_cls,  77)

        self.input_tokenized_dict = self.prompt_learner.input_tokenized_dict
        self.input_embed_dict = self.prompt_learner.input_embed_dict
        self.n_description_per_class = cfg.DATA.N_DESCRIPTION
        self.aug_n_neighbors_to_choose = self.prompt_learner.aug_n_neighbors_to_choose
        self.n_class = cfg.DATA.NUM_CLASSES

        self.image_encoder = clip_model.visual    #  change 'visual' into 'image_encoder
        self.text_encoder = TextEncoder(clip_model)  #  add 'text_encoder'
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # if self.only_test:
        #     with torch.no_grad():
        #         # todo during test time, the text encoder will not be updated anymore, there is no need to compute text features in every iteration
        #         prompts = self.prompt_learner()  # todo the precomputed  token embeddings (input to the transformer)  in shape   (n_cls, 77, 512), computed from  tokenized prompts  (n_cls,  77)
        #         # self.text_features = self.text_encoder(prompts, self.tokenized_prompts)
        #         self.text_features = self.text_encoder(prompts[:50, :, :], self.tokenized_prompts[:50, :])
        #         self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)  # todo  (n_class, 512)


    def forward(self, image):

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


        # Finally, make the text features

        # tokenized_prompts = self.tokenized_prompts  # todo  tokenized prompts  (n_cls,  77)
        # prompts = self.prompt_learner()  # todo the precomputed  token embeddings (input to the transformer)  in shape   (n_cls, 77, 512), computed from  tokenized prompts  (n_cls,  77)

        # todo randomly sample a tokenized input and input embedding for each class
        input_tokenized = []
        input_embed = []
        # if self.aug_n_neighbors_to_choose == self.n_description_per_class:

        text_id = np.random.randint(self.n_description_per_class, size= self.n_class)
        # input_tokenized = torch.stack( [  for class_id in range(self.n_class) ] )
        for class_id in range(self.n_class):
            input_tokenized.append( self.input_tokenized_dict[class_id][text_id[class_id]] )
            input_embed.append( self.input_embed_dict[class_id][text_id[class_id]] )
        # else:

        input_tokenized = torch.stack(input_tokenized, dim=0).cuda()
        input_embed = torch.stack(input_embed, dim=0).cuda()

        text_features = self.text_encoder(input_embed,  input_tokenized)   #  todo  prompts:  the precomputed  token embeddings  in shape   (n_cls, 77, 512),      tokenized prompts  (n_cls,  77)                         text_features (n_class, 512)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)     #   todo  (n_class, 512)

        logits = logit_scale * image_features @ text_features.t()

        return logits



def returnCLIP(config, logger=None,
               class_names=None, train_data = None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP_text_aug(config, class_names, clip_model, logger, train_data=train_data)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'

    freeze_model(model, config, logger)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    if not config.TEST.ONLY_TEST:
        logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model