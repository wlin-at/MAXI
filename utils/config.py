import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.ROOT_VAL_ZS =''
_C.DATA.TRAIN_FILE = ''
_C.DATA.TRAIN_PS_SCORES = ''
_C.DATA.SOFT_K=5
_C.DATA.VAL_FILE = ''
_C.DATA.VAL_ZS_FILE=''
_C.DATA.DATASET = 'kinetics400'
_C.DATA.DATASET_VAL_ZS = ''
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 400
_C.DATA.NUM_CLASSES_ZS = 101
_C.DATA.LABEL_LIST = 'labels/kinetics_400_labels.csv'
_C.DATA.VAL_ZS_LABEL_LIST=None   #  todo  training with contrastive loss, validation loss of ZS classification on a different dataset
_C.DATA.DESCRIPTION_DIR = ''
_C.DATA.CAPTION_BAG_DIR=''
_C.DATA.GPT3_BAG_DIR=''
_C.DATA.TOPK_CLASS_BAG_DIR=''
_C.DATA.N_SAMPLES_IN_BAG=None
_C.DATA.USE_DESCRIPTION = False
_C.DATA.USE_DESCRIPTION_TYPE = ''
_C.DATA.BAG_TYPE=''
_C.DATA.N_DESCRIPTION = 20
_C.DATA.NN_DESCRIPTION_RESULT = ''
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.USE_TEMP_ATTENTION = None
_C.MODEL.MODEL_PATH = None
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = True
_C.MODEL.R_LORA = -1
_C.MODEL.INIT_TEMPERATURE=0.05
_C.MODEL.MIN_TEMPERATURE=0.001
_C.MODEL.FUSE_WEIGHT_FOR_ORIGIN=0.5
# -----------------------------------------------------------------------------
# Custom trainer settings
# -----------------------------------------------------------------------------
_C.TRAINER = CN()
# Config for ViFi-CLIP
_C.TRAINER.ViFi_CLIP = CN()
_C.TRAINER.ViFi_CLIP.PROMPT_MODEL = False # second stage prompting?  todo only finetune the prompt module?? freeze the other modules of both visual and text encoder?
_C.TRAINER.ViFi_CLIP.N_CTX_VISION = 0  # number of context vectors at the vision branch
_C.TRAINER.ViFi_CLIP.N_CTX_TEXT = 0  # number of context vectors at the language branch
_C.TRAINER.ViFi_CLIP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
_C.TRAINER.ViFi_CLIP.PROMPT_DEPTH_VISION = 0  # max 12, min 0, for 0 it will act as shallow vision prompting (first layer)
_C.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT = 1  # max 12, min 0, for 0 it will act as shallow language prompting (first layer)
_C.TRAINER.ViFi_CLIP.USE = "both"  # fine-tuning complete CLIP model by default
_C.TRAINER.ViFi_CLIP.ZS_EVAL = False  # make True only during test mode to evaluate zero-shot vanilla CLIP performance
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1' # todo check here https://nvidia.github.io/apex/amp.html     "O0" - FP32 training  O1 - Mixed precision   O2 - "Almost FP16" mixed precision    O3 - FP16 training
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.NUM_WORKERS = 16
_C.TRAIN.LAMBDA_CLASSIFICATION = 1
_C.TRAIN.LAMBDA_CONTRAST = 0
_C.TRAIN.LOAD_OPTIMIZER_STATE = True
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False
_C.TEST.MULTI_VIEW_INFERENCE = False
_C.TEST.COMPUTE_PS = False
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.VAL_BATCH_SIZE = 2
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 50
_C.SEED = 1024



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.output:
        config.OUTPUT = args.output
    if args.only_test:
        config.TEST.ONLY_TEST = True
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config