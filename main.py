import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import datetime
import shutil
from pathlib import Path

from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import  epoch_saving, load_checkpoint,  auto_resume_helper, model_analysis
from datasets.build import build_dataloader
from utils.logger import create_logger
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending

from trainers import vificlip, vificlip_lora, clip_nn, vificlip_text_aug, vificlip_caption_aug, vificlip_caption_aug_mil
# import os.path as osp
from clip.train import train_ce_loss, train_contrast
from clip.val import validate, validate_compute_description_nn, validate_description_ensemble, compute_description_nn_for_each_class
from utils.utils_ import parse_option
from clip.loss import NCELoss, MIL_NCEloss,  MIL_Max_Loss, MIL_extract_max_Loss

def main(config):
    if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap', 'train cls', 'train cls+synonym', 'train cls+cap max',
                                                                            'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max']:
        train_data, val_data, train_loader, train_loader_w_idx, val_loader, val_loader_sequential, \
                    val_loader_zs = build_dataloader(logger, config)
        val_zs_classnames = [class_name for i, class_name in val_loader_zs.dataset.classes]
    else:
        train_data, val_data, train_loader, train_loader_w_idx,  val_loader, val_loader_sequential = build_dataloader(logger, config)
    class_names = [class_name for i, class_name in train_data.classes]

    # Custom trainer for different variants of ViFi-CLIP
    # todo ####################################################################################################################
    # todo ######################################## model #################################################################
    # todo ####################################################################################################################
    if config.DATA.USE_DESCRIPTION:
        if config.DATA.USE_DESCRIPTION_TYPE in ['eval_nn_for_each_class', 'eval_all_descriptions', 'eval_text_ensemble' ]  :
            # todo test-only models
            model, loaded_clip_model = clip_nn.returnCLIP(config, logger=logger, class_names=class_names)
        elif 'train_text_aug' in config.DATA.USE_DESCRIPTION_TYPE:
            # todo model to train with text augmentation
            model = vificlip_text_aug.returnCLIP( config, logger=logger, class_names=class_names, train_data=train_data)
        elif config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap', 'train cls', 'train cls+synonym', 'train cls+cap max']:
            model,loaded_clip_model = vificlip_caption_aug.returnCLIP(config, logger=logger, class_names=class_names, train_data=train_data, val_zs_classnames=val_zs_classnames)
            train_loader = train_loader_w_idx
        elif config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train mil extract max']:
            model, loaded_clip_model = vificlip_caption_aug_mil.returnCLIP(config, logger=logger, class_names=class_names, train_data=train_data, val_zs_classnames=val_zs_classnames  )
            train_loader = train_loader_w_idx
        elif config.DATA.USE_DESCRIPTION_TYPE in ['train topk class bag mil']:
            model, loaded_clip_model = vificlip_caption_aug_mil_topk_class_bag.returnCLIP(config, logger=logger, class_names=class_names, train_data=train_data, val_zs_classnames=val_zs_classnames  )
            train_loader = train_loader_w_idx

    else:
        if config.MODEL.R_LORA <= 0:
            # todo model to train w/o text augmentation (dummy prompt template)
            model = vificlip.returnCLIP(config, logger=logger, class_names=class_names,)
        else:
            model = vificlip_lora.returnCLIP(config, logger=logger, class_names= class_names)

    model = model.cuda()  # changing to cuda here
    if 'loaded_clip_model' in locals():
        loaded_clip_model = loaded_clip_model.cuda()

    # todo ####################################################################################################################
    # todo ######################################## criterion #################################################################
    # todo ####################################################################################################################
    mixup_fn = None
    if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max']:
        criterion = NCELoss()
    elif config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in  ['train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce']:
        criterion = MIL_NCEloss()
    elif config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in [ 'train mil extract max']:
        criterion = MIL_extract_max_Loss()
    elif config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train topk class bag mil']:
        # criterion = MIL_NCEloss_topk_class_bag()
        criterion = MIL_Max_Loss()
    else:
        if config.AUG.MIXUP > 0:
            criterion = SoftTargetCrossEntropy()  #  the loss used together with MixUp
            mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                           smoothing=config.AUG.LABEL_SMOOTH,
                                           mixup_alpha=config.AUG.MIXUP,
                                           cutmix_alpha=config.AUG.CUTMIX,
                                           switch_prob=config.AUG.MIXUP_SWITCH_PROB)
        elif config.AUG.LABEL_SMOOTH > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
        else:
            criterion = nn.CrossEntropyLoss()
    # todo ####################################################################################################################
    # todo ####################################################################################################################
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    if config.TRAIN.OPT_LEVEL != 'O0': # todo  amp = automatic mixed precision   "O0" - FP32 training  O1 - Mixed precision   O2 - "Almost FP16" mixed precision    O3 - FP16 training
        if config.MODEL.R_LORA <= 0: # does not use LoRA
            model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
        # else: #  LoRA
        #     model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    model_analysis(model, logger )


    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        if start_epoch > 1:
            logger.info("resetting epochs no and max. accuracy to 0 after loading pre-trained weights")
            start_epoch = 0
            max_accuracy = 0
    if config.TEST.ONLY_TEST:
        # Test only
        if config.DATA.USE_DESCRIPTION:
            if config.DATA.USE_DESCRIPTION_TYPE == 'eval_nn_for_each_class':
                # todo  1) the text features are normalized within descriptions of one class
                #    for each video, find the nearest descriptions for each class
                compute_description_nn_for_each_class(val_loader_sequential, model, config, loaded_clip_model=loaded_clip_model, logger=logger)
            elif config.DATA.USE_DESCRIPTION_TYPE == 'eval_all_descriptions' :
                # todo  2) text features are normalized within all descriptions of all classes
                acc1, description_top20_indices_dict = validate_compute_description_nn(val_loader_sequential, model, config, loaded_clip_model=loaded_clip_model, logger=logger)
                # assert len(description_top5_indices_dict) == len(val_loader_sequential.dataset)
                f_write = open(config.DATA.NN_DESCRIPTION_RESULT + f'_rank{dist.get_rank()}.txt', 'w+')
                # for idx in range(len(description_top5_indices_dict)):
                for idx in description_top20_indices_dict.keys():
                    f_write.write(str(idx) + ' '  + ' '.join([str(value_) for value_ in description_top20_indices_dict[idx]]) + '\n')
                f_write.close()
                logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
            elif config.DATA.USE_DESCRIPTION_TYPE == 'eval_text_ensemble':
                # todo  3) ensemble text features,  average among   self.n_description_per_class
                acc1 = validate_description_ensemble(val_loader_sequential, model, config, loaded_clip_model=loaded_clip_model, logger=logger)
                logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        else:
            if config.DATA.DATASET == 'Charades':
                mAP = validate(val_loader_sequential, model, config, logger=logger)
                logger.info(f"mAP of the network on the {len(val_data)} test videos: {mAP * 100.0:.1f}%")
            else:
                if config.TEST.COMPUTE_PS:
                    acc1 = validate(val_loader_sequential, model, config, logger=logger)
                else:
                    acc1 = validate(val_loader, model, config, logger=logger)
                logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")

    else:
        #  Training
        max_logit_scale = nn.Parameter(torch.tensor([- np.log(config.MODEL.MIN_TEMPERATURE)]), requires_grad=False).float().cuda()
        for epoch in range(start_epoch, config.TRAIN.EPOCHS):
            train_loader.sampler.set_epoch(epoch)
            if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max',
                                                                                    'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max']:
                train_contrast(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn, clip_model=loaded_clip_model, scaler=scaler, logger=logger, max_logit_scale=max_logit_scale)
            else:
                train_ce_loss(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn, scaler=scaler, logger=logger)

            if (epoch +1) % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):

                if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max',
                                                                                        'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max' ]:
                    # todo  zero-shot classification for validation
                    acc1 = validate(val_loader_zs, model, config, logger=logger)
                else:
                    acc1 = validate(val_loader, model, config, logger=logger)
                logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
                is_best = acc1 > max_accuracy
                max_accuracy = max(max_accuracy, acc1)
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')
                # todo save only the last epoch
                # if dist.get_rank() == 0 and ( (epoch +1) % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or is_best):
                if dist.get_rank() == 0 and (  epoch == (config.TRAIN.EPOCHS - 1) or is_best) :
                    epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)
        # Now doing the multi-view inference crop for videos
        # 4 CLIPs are obtained from each video, and for each CLIP, we get 3 crops (augmentations)
        multi_view_inference = config.TEST.MULTI_VIEW_INFERENCE
        if multi_view_inference:
            config.defrost()
            config.TEST.NUM_CLIP = 4
            config.TEST.NUM_CROP = 3
            config.freeze()
            if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap','train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max']:
                train_data, val_data, train_loader, train_loader_w_idx, val_loader, val_loader_sequential, val_loader_zs =  build_dataloader(logger, config)
                # todo  zero-shot classification for validation
                acc1 = validate(val_loader_zs, model, config, logger=logger)
            else:
                train_data, val_data, train_loader, train_loader_w_idx, val_loader, val_loader_sequential = build_dataloader(logger, config)
                # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
                acc1 = validate(val_loader, model, config, logger=logger)
            logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")







if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
