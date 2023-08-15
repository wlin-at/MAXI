

# from trainers.vificlip import load_clip_to_cpu

import torch
import torch.nn as nn
from clip import clip_lora
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = cfg.MODEL.MODEL_PATH
    lora = cfg.MODEL.R_LORA
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
    model = clip_lora.build_model(state_dict or model.state_dict(), design_details, lora=lora)  # model is set to eval, all modules excepf for lora are frozen

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final   #  final layernorm layer of CLIP
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):   #  todo prompts are embeddings (n_cls, 77, 512),   tokenized_prompts (n_cls, 77)
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
            tokenized_prompts = torch.cat([clip_lora.tokenize(text_aug.format(c), context_length=77) for c in classnames])
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
                prompt = clip_lora.tokenize(ctx_init)
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

            tokenized_prompts = torch.cat([clip_lora.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
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
            tokenized_prompts = torch.cat([clip_lora.tokenize(p) for p in prompts])  # (n_cls, n_tkn)    (n_cls, 77)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)   # (n_class, 77, 512)
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
            prompts = self.complete_text_embeddings

        return prompts



class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  #  (n_cls,  77)
        self.image_encoder = clip_model.visual    #  change 'visual' into 'image_encoder
        self.text_encoder = TextEncoder(clip_model)  #  add 'text_encoder'
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts  # todo  tokenized prompts  (n_cls,  77)
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()  #  todo the precomputed  token embeddings  in shape   (n_cls, 77, 512), computed from  tokenized prompts  (n_cls,  77)

        # b = image.shape[0]
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

        # Finally, make the text features
        text_features = self.text_encoder(prompts, tokenized_prompts)   #  todo  (n_class, 512)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  #   todo  ( B, 512 )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)     #   todo  (n_class, 512)
        logits = logit_scale * image_features @ text_features.t()

        return logits



def returnCLIP(config, logger=None,
               class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building ViFi-CLIP CLIP")
    model = ViFiCLIP(config, class_names, clip_model, logger)  # initialize a class,   #  change 'visual' into 'image_encoder     #  add 'text_encoder'


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