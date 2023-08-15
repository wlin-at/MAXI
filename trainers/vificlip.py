import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()






def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = cfg.MODEL.MODEL_PATH
    init_temperature = cfg.MODEL.INIT_TEMPERATURE
    min_temperature = cfg.MODEL.MIN_TEMPERATURE

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViFi_CLIP',
                      "vision_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_VISION,  # 0
                      "language_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT,  # 1
                      "vision_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_VISION,  # 0   CTX context tokens
                      "language_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT}  # 0
    model = clip.build_model(state_dict or model.state_dict(), design_details, init_temperature=init_temperature,
                             n_frames= cfg.DATA.NUM_FRAMES, use_temp_attentino=cfg.MODEL.USE_TEMP_ATTENTION  )  # model is set to eval

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final   #  final layernorm layer of CLIP
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):   #  todo prompts are computed text embeddings (n_cls, 77, 512), computed from tokenized prompts,   tokenized_prompts (n_cls, 77)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # todo  EOT tokens
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.ViFi_CLIP.PROMPT_MODEL
        ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT  #   a photo of a
        ZS_evaluation = cfg.TRAINER.ViFi_CLIP.ZS_EVAL
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)  # todo register a buffer for data that are not model parameters, but saved alongside parameters. the data are saved into self._buffers
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            logger.info(f"V-L design")
            logger.info(f'Initial text context: "{prompt_prefix}"')
            logger.info(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            logger.info(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.ViFi_CLIP.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)    (n_cls, 77)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)   # (n_class, 77, 512)  todo  there is no gradinent in the token embedding layer, token embedding layer is not trained
            #  todo complete_text_embeddings  will be re-computed during test time.
            self.register_buffer("complete_text_embeddings", embedding)  # todo register a buffer for data that are not model parameters, but saved alongside parameters. the data are saved into self._buffers
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings   #  todo  computed as      clip_model.token_embedding(tokenized_prompts).type(dtype)

        return prompts


class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        #   todo   the prompt template and tokenization of prompted texts, token embedding (input to transformer) are done in VLPromptLearner
        self.only_test = cfg.TEST.ONLY_TEST
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  #  (n_cls,  77)
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

    def compute_text_features(self):
        # todo during test time, the text encoder will not be updated anymore, there is no need to compute text features in every iteration
        if self.only_test:
            with torch.no_grad():
                prompts = self.prompt_learner()  # todo the precomputed  token embeddings (input to the transformer)  in shape   (n_cls, 77, 512), computed from  tokenized prompts  (n_cls,  77)
                self.text_features = self.text_encoder(prompts, self.tokenized_prompts)
                # self.text_features = self.text_encoder(prompts[:50, :, :], self.tokenized_prompts[:50, :])
                self.text_features = self.text_features / self.text_features.norm(dim=-1,  keepdim=True)  # todo  (n_class, 512)
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
        if self.only_test:
            text_features = self.text_features
        else:
            # tokenized_prompts = self.tokenized_prompts  # todo  tokenized prompts  (n_cls,  77)
            prompts = self.prompt_learner()  # todo the precomputed  token embeddings (input to the transformer)  in shape   (n_cls, 77, 512), computed from  tokenized prompts  (n_cls,  77)
            text_features = self.text_encoder(prompts, self.tokenized_prompts)   #  todo  prompts:  the precomputed  token embeddings  in shape   (n_cls, 77, 512),      tokenized prompts  (n_cls,  77)                         text_features (n_class, 512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)     #   todo  (n_class, 512)

        logits = logit_scale * image_features @ text_features.t()

        return logits




def freeze_model(model, config, logger):
    if config.TRAINER.ViFi_CLIP.PROMPT_MODEL:
        logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.ViFi_CLIP.USE
        if train_complete_clip == "both":
            logger.info("Turning on gradients for COMPLETE ViFi-CLIP model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        else:
            if train_complete_clip == "image":
                logger.info("Turning on gradients for image side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            elif train_complete_clip == 'text':
                logger.info("Turning on gradients for TEXT side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            elif 'image_vit_last' in train_complete_clip:
                n_blocks_unfreeze = int(train_complete_clip[-1])
                # todo  train only the last few blocks in image encoder ViT
                logger.info(f"Unfreeze image encoder ViT - only the last {n_blocks_unfreeze} blocks")
                n_blocks_total = len(model.image_encoder.transformer.resblocks)
                trainable_names = []
                trainable_names.append('logit_scale')
                for idx in range( n_blocks_total - n_blocks_unfreeze,  n_blocks_total ):
                    trainable_names.append(f'image_encoder.transformer.resblocks.{idx}')
                trainable_names.append('image_encoder.ln_post')
                trainable_names.append('image_encoder.proj')
                logger.info(trainable_names)

                for name, param in model.named_parameters():
                    require_grad = False
                    for trainable_name in trainable_names:
                        if trainable_name in name:
                            require_grad = True
                    if require_grad == True:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

            elif 'both_last' in train_complete_clip:
                # todo  unfreeze the last few blocks in both visual and test encoders
                n_blocks_unfreeze = int(train_complete_clip[-1])
                logger.info(f"Unfreeze image and text encoders - Unfreeze the last {n_blocks_unfreeze} blocks")
                n_blocks_total_vis = len(model.image_encoder.transformer.resblocks)
                n_blocks_total_text = len(model.text_encoder.transformer.resblocks)
                trainable_names = []
                trainable_names.append('logit_scale')
                # trainable modules for visual encoder
                for idx in range(n_blocks_total_vis - n_blocks_unfreeze, n_blocks_total_vis):
                    trainable_names.append(f'image_encoder.transformer.resblocks.{idx}')
                trainable_names.append('image_encoder.ln_post')
                trainable_names.append('image_encoder.proj')
                # trainable modules for text encoder
                for idx in range(n_blocks_total_text - n_blocks_unfreeze, n_blocks_total_text):
                    trainable_names.append(f'text_encoder.transformer.resblocks.{idx}')
                trainable_names.append('text_encoder.ln_final')
                trainable_names.append('text_encoder.text_projection')
                logger.info(trainable_names)

                for name, param in model.named_parameters():
                    require_grad = False
                    for trainable_name in trainable_names:
                        if trainable_name in name:
                            require_grad = True
                    if require_grad == True:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)



def returnCLIP(config, logger=None,
               class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)



    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP(config, class_names, clip_model, logger)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'


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
