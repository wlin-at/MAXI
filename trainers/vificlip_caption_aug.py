

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.vificlip import TextEncoder, load_clip_to_cpu, freeze_model
import numpy as np
import os.path as osp
import glob





def load_vid_captions(video_infos, description_dir, ):
    vid_caption_list = []
    for vid_id, vid_info in enumerate(video_infos):
        class_id = vid_info['filename'].split('/')[-2]
        vidname = vid_info['filename'].split('/')[-1].split('.')[0]
        caption_file = osp.join(description_dir, class_id, f'{vidname}.txt')
        vid_caption_list.append([line.strip('\n') for line in open(caption_file)])
    return vid_caption_list

def load_descriptions(description_dir, classnames):
    description_list = []  #  a list of list,  the order is the same as order in classnames
    for classname in classnames:
        descript_file = osp.join( description_dir, f'{classname}.txt' )
        description_list.append(  [  line.strip('\n') for line in open(descript_file)] )
    return description_list


def compute_tokenized_and_embed_for_cap(vid_caption_list ,  token_embedding, dtype):
    input_tokenized_vid_list = []
    input_embed_vid_list = []
    for vid_captions in vid_caption_list:   # each video has 8 captions
        with torch.no_grad():
            input_tokenized = torch.cat([clip.tokenize(vid_caption +'.' )  for vid_caption in vid_captions  ]  )  #  (8, 77)
            input_embed = token_embedding(input_tokenized).type(dtype)   #  (8, 77, 512)
            input_tokenized_vid_list.append(input_tokenized.cuda()  )
            input_embed_vid_list.append( input_embed.cuda() )

    # for vid_id, vid_info in enumerate(video_infos):
    #     class_id = vid_info['filename'].split('/')[-2]
    #     vidname = vid_info['filename'].split('/')[-1].split('.')[0]
    #     caption_file = osp.join(description_dir, class_id, f'{vidname}.txt')
    #     with torch.no_grad():
    #         input_tokenized = torch.cat([clip.tokenize(line.strip('\n') +'.' )  for line in open(caption_file)]  )
    #         input_embed = token_embedding(input_tokenized).type(dtype)
    #         input_tokenized_vid_list.append(input_tokenized.cuda()  )
    #         input_embed_vid_list.append( input_embed.cuda() )
    return input_tokenized_vid_list, input_embed_vid_list

def load_vid_blip_textbag(video_infos, description_dir, ):
    vid_caption_list = []
    for vid_id, vid_info in enumerate(video_infos):
        class_id = vid_info['filename'].split('/')[-2]
        vidname = vid_info['filename'].split('/')[-1].split('.')[0]
        caption_file = osp.join(description_dir, class_id, f'{vidname}.txt')
        vid_caption_list.append([line.strip('\n') for line in open(caption_file)])
    return vid_caption_list

def load_gpt3_verbs(classnames, bag_dir):

    class_bag_list = []
    for class_id, classname in enumerate(classnames):
        bag_file = osp.join( bag_dir, f'{classname}.txt' )

        class_bag_list.append( [ line.strip('\n')  for line in open( bag_file) ]  )

    for class_id, classname in enumerate(classnames):
        class_bag_list[class_id].remove(classname)

    return class_bag_list


class ViFiCLIP_caption_aug(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, train_data = None, val_zs_classnames = None):
        super().__init__()
        self.only_test = cfg.TEST.ONLY_TEST
        # self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger, train_data=train_data)
        # todo  load all video instance captions, in the order given in the training file list, every video has a list of 8 captions
        self.use_description_type = cfg.DATA.USE_DESCRIPTION_TYPE
        assert self.use_description_type in ['train cls+cap', 'train cap', 'train synonym+cap', 'train cls',  'train cls+synonym', 'train cls+cap max']  # class name prompt template + caption
        self.bag_type = cfg.DATA.BAG_TYPE

        if 'blipwords' in self.bag_type:
            self.vid_blip_textbag_list = load_vid_blip_textbag(train_data.video_infos, description_dir=cfg.DATA.CAPTION_BAG_DIR)
        if 'gpt3verb' in self.bag_type:
            # todo  gpt3_verb_list does not include the original class name
            self.gpt3_verb_list = load_gpt3_verbs(classnames, bag_dir= cfg.DATA.GPT3_BAG_DIR)
        # if self.use_description_type in ['train cls+cap', 'train cap', 'train synonym+cap', 'train cls', 'train cls+cap max']:
        #     self.vid_caption_list = load_vid_captions(train_data.video_infos, description_dir=cfg.DATA.DESCRIPTION_DIR)
        #     self.n_captions_per_vid = len(self.vid_caption_list[0])
        elif self.use_description_type == 'train cls+synonym':
            self.description_list = load_descriptions( description_dir=cfg.DATA.DESCRIPTION_DIR, classnames=classnames )
            self.n_descripts_per_class = len(self.description_list[0])

        ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT


        self.n_class = cfg.DATA.NUM_CLASSES


        if self.use_description_type in ['train cls+cap','train cls', 'train cls+synonym', 'train cls+cap max']:
            self.cls_prompts = [ctx_init.replace("_", " ") + " " + name + "." for name in classnames]
        # elif self.use_description_type == 'train cap':
        #     self.input_tokenized_vid_list, self.input_embed_vid_list = compute_tokenized_and_embed_for_cap(self.vid_caption_list, token_embedding=clip_model.token_embedding, dtype= clip_model.dtype)
        # todo  if  'train cap',  token embeddings can be computed offline
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # todo ###################################################################################
        # todo ###################################################################################
        #  for validation, we perform zero-shot classification task on the UCF dataset
        # todo ###################################################################################
        # todo ###################################################################################
        zs_cls_prompts = [ctx_init.replace("_", " ") + " " + name + "." for name in val_zs_classnames]
        self.zs_tokenized_cls_prompts = torch.cat([clip.tokenize(p) for p in zs_cls_prompts])
        with torch.no_grad():
            self.zs_input_embed_cls = clip_model.token_embedding(self.zs_tokenized_cls_prompts).type(self.dtype)
        self.zs_tokenized_cls_prompts = self.zs_tokenized_cls_prompts.cuda()
        self.zs_input_embed_cls = self.zs_input_embed_cls.cuda()


    def compute_visual_features(self, image):
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
        return image_features


    def forward(self, image, batch_idx, labels, token_embedding = None):
        logit_scale = self.logit_scale.exp()

        # Lets encode the video into required format
        b, t, c, h, w = image.size()

        image_features = self.compute_visual_features(image)

        # todo randomĺy sample a caption for each video in a batch

        # if self.use_description_type in ['train cap', 'train cls+cap','train cls']:
        #     text_ids = np.random.randint(self.n_captions_per_vid, size=b)
        # elif self.use_description_type in ['train cls+synonym']:
        #     text_ids = np.random.randint(self.n_descripts_per_class, size=b)

        if self.use_description_type in ['train cap', 'train cls+cap', 'train cls', 'train cls+synonym']:
            tokenized_prompts = []
            with torch.no_grad():
                for sample_id in range(b):
                    video_prompt = self.get_video_prompt(batch_idx[sample_id], labels[sample_id])  # , text_ids[sample_id]
                    tokenized_prompts.append(clip.tokenize(video_prompt))
                # compute  tokenized prompts and  embedding for an entire batch
                tokenized_prompts = torch.cat(tokenized_prompts).cuda()  # (bz, 77)
                input_embed = token_embedding(tokenized_prompts).type(self.dtype)  # (bz, 512, 77)


            text_features = self.text_encoder(input_embed, tokenized_prompts)  # (bz,  512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # todo  (bz, 512)

            logits = logit_scale * image_features @ text_features.t()

        elif self.use_description_type in ['train cls+cap max']:
            tokenized_prompts = []
            # bag_size_list = []
            start_position_list = []
            end_position_list = []
            start_position = 0
            with torch.no_grad():
                for sample_id in range(b):
                    #  todo it can happen that different instance bags have different sizes
                    video_prompt_bag = self.get_video_prompt( batch_idx[sample_id],  labels[sample_id] )
                    start_position_list.append(start_position)
                    start_position += len(video_prompt_bag)
                    end_position_list.append( start_position  )
                    tokenized_prompts += [clip.tokenize( video_prompt )  for video_prompt in video_prompt_bag]
                tokenized_prompts = torch.cat( tokenized_prompts ).cuda()  #  (total number of prompts in all videos,  77)
                input_embed = token_embedding(tokenized_prompts).type(self.dtype) #  (total number of prompts in all videos, 512,  77)

            text_features = self.text_encoder(input_embed, tokenized_prompts)  # (total number of prompts in all videos,  512 )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (total number of prompts in all videos,  512 )
            logits_raw = logit_scale * image_features @ text_features.t()  # (bz,   total number of prompts in all videos )

            # logits = torch.zeros((b, b)).to(logits_raw.device)
            #  todo check the digonal blocks  for the idx for columns !!!!!!!!!!!!

            max_idx_global_list = []

            for i in range(b):
                max_idx_local =  torch.argmax(logits_raw[i, start_position_list[i]: end_position_list[i] ])
                max_idx_global_list.append(  start_position_list[i] + max_idx_local  )

            logits = logits_raw[:, max_idx_global_list]
            # for i in range(b):
            #     for j in range(b):
            #         #  todo take the maximum similarity score between a video embedding and all text embedding in the bag
            #         logits[i, j] = torch.max(logits_raw[i, start_position_list[j]: end_position_list[j]  ] )

        return logits

    def val_zs_classification(self, image):
        logit_scale = self.logit_scale.exp()
        # b, t, c, h, w = image.size()
        image_features = self.compute_visual_features(image)
        text_features = self.text_encoder( self.zs_input_embed_cls, self.zs_tokenized_cls_prompts )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # todo  (bz, 512)

        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_video_prompt(self, vid_id, label):
        if self.use_description_type == 'train cls+cap':
            # todo   randomly choose one caption out of a bag
            # todo  it is possible that there is nothing in the caption file, e.g. for the verb VBG case
            cls_prompt = self.cls_prompts[label]
            # n_captions = len( self.vid_caption_list[vid_id])

            if self.bag_type == 'gpt3verb':
                combined_bag = self.gpt3_verb_list[label]
            elif self.bag_type == 'blipwords':
                combined_bag = self.vid_blip_textbag_list[vid_id]
            elif self.bag_type == 'gpt3verb+blipwords':
                combined_bag = self.gpt3_verb_list[label] + self.vid_blip_textbag_list[vid_id]
                combined_bag = list(set(combined_bag))
            n_captions = len(combined_bag)

            if n_captions == 0:
                return cls_prompt
            else:
                text_id = np.random.randint( n_captions )
                # frame_caption = self.vid_caption_list[vid_id][text_id]
                selected_text =combined_bag[text_id]

                return cls_prompt + ' ' + selected_text + '.'
        elif self.use_description_type == 'train cls+cap max':
            # todo return the bag of video prompts
            # todo  it is possible that there is nothing in the caption file, e.g. for the verb VBG case
            cls_prompt = self.cls_prompts[label]
            frame_caption_list = self.vid_caption_list[vid_id]
            video_prompt_bag = []
            n_captions = len(frame_caption_list)
            if n_captions == 0:
                video_prompt_bag.append(cls_prompt)
            else:
                for frame_caption in frame_caption_list:
                    video_prompt_bag.append( cls_prompt + ' ' + frame_caption + '.' )
            return video_prompt_bag
        elif self.use_description_type == 'train cls+synonym':
            cls_prompt = self.cls_prompts[label]
            text_id = np.random.randint( len( self.description_list[label] ) )
            descript = self.description_list[label][text_id]
            return cls_prompt + ' ' + descript + '.'
        elif self.use_description_type == 'train cap':
            # todo  the computing embedding offline - out of memory error.
            text_id = np.random.randint(len(self.vid_caption_list[vid_id]))
            frame_caption = self.vid_caption_list[vid_id][text_id]
            return frame_caption + '.'
        elif self.use_description_type == 'train synonym+cap':
            return None
        elif self.use_description_type == 'train cls':
            return self.cls_prompts[label]







def returnCLIP(config, logger=None,
               class_names=None, train_data = None, val_zs_classnames = None ):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)


    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP_caption_aug(config, class_names, clip_model, logger, train_data=train_data, val_zs_classnames=val_zs_classnames)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'

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
    return model, clip_model





