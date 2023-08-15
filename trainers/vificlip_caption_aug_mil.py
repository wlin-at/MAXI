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
import random

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


def load_class_textbag(video_infos, description_dir, top_k = None):
    vid_bag_list = []
    for vid_id, vid_info in enumerate(video_infos):
        class_id = vid_info['filename'].split('/')[-2]
        vidname = vid_info['filename'].split('/')[-1].split('.')[0]
        caption_file = osp.join(description_dir, class_id, f'{vidname}.txt')
        vid_bag = []
        for line_id, line in enumerate(open(caption_file)):
            if line_id +1 <= top_k:
                vid_bag.append( line.strip('\n') )
        vid_bag_list.append(vid_bag)
    return vid_bag_list



class ViFiCLIP_caption_aug_MIL(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, train_data = None, val_zs_classnames = None):
        super().__init__()
        self.only_test = cfg.TEST.ONLY_TEST
        self.classnames = classnames
        # self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger, train_data=train_data)
        # todo  load all video instance captions, in the order given in the training file list, every video has a list of 8 captions
        self.bag_type = cfg.DATA.BAG_TYPE
        self.n_samples_in_bag = cfg.DATA.N_SAMPLES_IN_BAG  # todo randomly sample this amount of elements in each bag
        # self.vid_caption_list = load_vid_captions(train_data.video_infos, description_dir=cfg.DATA.DESCRIPTION_DIR)
        if 'blipwords' in self.bag_type:
            self.vid_blip_textbag_list = load_vid_blip_textbag(train_data.video_infos, description_dir=cfg.DATA.CAPTION_BAG_DIR)

        if 'gpt3verb' in self.bag_type:
            # todo  gpt3_verb_list does not include the original class name
            self.gpt3_verb_list = load_gpt3_verbs(classnames, bag_dir= cfg.DATA.GPT3_BAG_DIR)
        if self.bag_type == 'classbag':
            #  todo   words in class bag are already ordered w.r.t prediction score,  no random sampling
            self.class_bag = load_class_textbag(train_data.video_infos, description_dir=cfg.DATA.TOPK_CLASS_BAG_DIR, top_k= self.n_samples_in_bag +1)

        # self.n_captions_per_vid = len(self.vid_caption_list[0])

        ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT
        self.ctx_init = ctx_init.replace("_", " ")

        self.n_class = cfg.DATA.NUM_CLASSES
        self.use_description_type = cfg.DATA.USE_DESCRIPTION_TYPE
        assert self.use_description_type in ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train mil extract max']  # class name prompt template + caption


        if self.use_description_type in ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train mil extract max']:
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
        # cap_ids = np.random.randint( self.n_captions_per_vid, size=b )

        # todo
        #  MIL-Max
        #  1) video-to-textbag score,  1 postive,  (B-1)M negative
        #          for each video, there are (B-1) negative bags, each bag contains M texts
        #  2) textbag-to-video score, 1 positive, (B-1)M negative
        #          for each bag, there are (B-1) negative videos,  the current bag has M texts
        #  MIL-NCE loss,
        #       video-to-textbag similarity score:
        #                   M positive,     the positive bag has M texts
        #                   (B-1)M negative   there are (B-1) negative bags, each bag contains M texts
        #       textbag-to-video similarity score:  None
        # if self.use_description_type in ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce']:
        if self.use_description_type in ['train cls+cap mil nce', 'train mil extract max']:
            # todo  MIL-NCE loss, only video-to-textbag similarity score,  no textbag-to-video similarity score
            tokenized_prompts = []
            with torch.no_grad():
                for sample_id in range(b):
                    video_prompt_bag = self.get_video_prompt_bag(batch_idx[sample_id], labels[sample_id])  # a list of 8 video prompts
                    tokenized_prompts +=  [ clip.tokenize( video_prompt)  for video_prompt in video_prompt_bag]
            tokenized_prompts = torch.cat(tokenized_prompts).cuda() # (bz *n_captions_per_vid , 77 )
            input_embed = token_embedding(tokenized_prompts).type(self.dtype) # (bz *n_captions_per_vid, 512, 77 )

            # with torch.no_grad():
            #     for sample_id in range(b):
            #         video_prompt = self.get_video_prompt(batch_idx[sample_id], labels[sample_id], cap_ids[sample_id])
            #         tokenized_prompts.append(clip.tokenize(video_prompt))
            #     # compute  tokenized prompts and  embedding for an entire batch
            #     tokenized_prompts = torch.cat(tokenized_prompts).cuda()  # (bz, 77)
            #     input_embed = token_embedding(tokenized_prompts).type(self.dtype)  # (bz, 512, 77)


        text_features = self.text_encoder(input_embed, tokenized_prompts)  # (bz*n_captions_per_vid,  512)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # todo  (bz*n_captions_per_vid, 512)

        logits = logit_scale * image_features @ text_features.t()  # (bz, bz*n_captions_per_vid)

        return logits

    def val_zs_classification(self, image):
        logit_scale = self.logit_scale.exp()
        # b, t, c, h, w = image.size()
        image_features = self.compute_visual_features(image)
        text_features = self.text_encoder( self.zs_input_embed_cls, self.zs_tokenized_cls_prompts )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # todo  (bz, 512)

        logits = logit_scale * image_features @ text_features.t()

        return logits


    # def control_bag_size(self, bag, target_bag_size):
    #     bag_size = len(bag)
    #
    #     if bag_size == target_bag_size:
    #         target_bag = bag
    def random_sample_from_bag(self, bag, sample_size):
        multiples = sample_size // len(bag)
        residual_ = sample_size % len(bag)

        sampled_list = bag * multiples  + random.sample(bag, residual_)

        return sampled_list



    def get_video_prompt_bag(self, vid_id, label):
        # get a bag of video prompt
        # frame_caption_list = self.vid_caption_list[vid_id]  #  a list of 8 frame captions for a video.
        # verb_bag = self.vid_caption_verb_list[vid_id] + self.class_verb_list[label]
        # class_verb_list = self.class_verb_list
        # verb_bag = [ self.classnames[label] ]  +  self.random_sample_from_bag(  self.class_verb_list[label],  self.n_samples_in_bag -1 )
        if self.bag_type == 'gpt3verb':
            # todo only GPT3 verbs
            combined_bag = self.gpt3_verb_list[label]
        elif self.bag_type == 'blipwords':
            # todo only BLIP caption words
            combined_bag = self.vid_blip_textbag_list[vid_id]
        elif self.bag_type == 'gpt3verb+blipwords':
            # todo GPT3 verbs + BLIP caption verbs
            combined_bag = self.gpt3_verb_list[label] + self.vid_blip_textbag_list[vid_id]
            combined_bag = list(set(combined_bag))


        if self.bag_type != 'classbag':
            if len(combined_bag) > 0:
                combined_bag = self.random_sample_from_bag(combined_bag, self.n_samples_in_bag)
        video_prompt_bag = []
        if self.use_description_type in ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train mil extract max']:
            cls_prompt = self.cls_prompts[label]


            classname = self.classnames[label]
            if classname in combined_bag:
                combined_bag.remove( classname )  # todo remove the classname
            else:
                combined_bag = combined_bag[: self.n_samples_in_bag] # todo remove the last word (lowest confidence)

            if self.bag_type == 'classbag':

                # for text_ in combined_bag[1:]:
                for text_ in combined_bag:
                    video_prompt_bag.append( cls_prompt + ' ' + text_ + '.' )
            else:
                if len(combined_bag) == 0:
                    video_prompt_bag = [cls_prompt] * self.n_samples_in_bag
                else:
                    for text_ in combined_bag:
                        # video_prompt_bag.append(self.ctx_init + ' ' + verb_ + '.' )
                        video_prompt_bag.append(cls_prompt + ' ' + text_ + '.')

            return video_prompt_bag



def returnCLIP(config, logger=None,
               class_names=None, train_data = None, val_zs_classnames = None ):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP_caption_aug_MIL(config, class_names, clip_model, logger, train_data=train_data, val_zs_classnames=val_zs_classnames)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'

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