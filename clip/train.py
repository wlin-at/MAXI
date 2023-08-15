



import torch
import datetime
from utils.tools import AverageMeter
import time
from apex import amp

from clip.loss import gen_mask


def train_contrast(epoch, model, criterion, optimizer, lr_scheduler, train_loader_w_idx, config, mixup_fn = None, clip_model = None,  scaler = None, logger = None, max_logit_scale = None):
    #  todo train with contrastive loss
    #       #  video embedding (bz, 512)
    #     #   text embedding ( bz, 512)

    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader_w_idx)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    include_ps_scores = not (config.DATA.TRAIN_PS_SCORES == '' or config.DATA.TRAIN_PS_SCORES is None)

    for idx, (batch_data, batch_idx) in enumerate(train_loader_w_idx):
        images = batch_data["imgs"].cuda(non_blocking=True)  # (bz, n_frames, 3, 224, 224)
        if config.DATA.USE_DESCRIPTION_TYPE != 'train topk class bag mil':
            label_target = batch_data["label"].cuda(non_blocking=True)  # (bz, 1)
            label_target = label_target.reshape(-1)
            label_mask = gen_mask(label_target).cuda(non_blocking=True)

        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])


        if config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max']:
            output = model(images, batch_idx, label_target,  token_embedding=clip_model.token_embedding)  # (bz, n_class )
            # todo for NCE loss, scores   (bz, bz )
            total_loss = criterion(output, label_mask )
        elif config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap mil max', 'train cls+cap mil softmax', 'train cls+cap mil nce', 'train mil extract max']:
            output = model(images, batch_idx, label_target, token_embedding=clip_model.token_embedding)  # (bz, n_class )
            #  todo     for MIL NCE loss,  scores (bz, bz*n_captions_per_vid)
            bz = images.size(0)
            # n_captions_per_vid = int(output.size(-1) / bz)
            total_loss = criterion(output, label_mask, bz=bz, n_captions_per_vid= model.module.n_samples_in_bag )
        elif config.DATA.USE_DESCRIPTION_TYPE in ['train topk class bag mil']:
            # todo label_target is not needed,  the label_target is only within 400-class space
            # output, label_mask = model( images, batch_idx,  token_embedding=clip_model.token_embedding )
            # total_loss = criterion( output, label_mask )
            total_loss = model( images, batch_idx,  token_embedding=clip_model.token_embedding )


        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else: # todo  no mixed precision
            total_loss.backward()
        # todo ############################# update model weights ####################################
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            # # todo  gradient accumulation
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()   # todo   if accumulation_steps > 1,  update model weights evey  [accumulation_steps] steps
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # todo  no gradient accumulation
            if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step() # todo   if accumulation_steps = 1,  update model weights at every step
                lr_scheduler.step_update(epoch * num_steps + idx)

        if model.module.logit_scale > max_logit_scale:
            model.module.logit_scale.copy_(max_logit_scale)


        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(),  images.size(0)) # len(label_target)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'logit_scale {model.module.logit_scale:.4f}')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_ce_loss(epoch, model, criterion, optimizer, lr_scheduler, train_loader, config, mixup_fn, scaler = None, logger = None):

    #  todo  train with cross entropy loss  for classification
    #   video embedding (bz, 512)
    #   text embedding (n_cls, 512)
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    include_ps_scores = not (config.DATA.TRAIN_PS_SCORES == '' or config.DATA.TRAIN_PS_SCORES is None)


    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True) # (bz, n_frames, 3, 224, 224)
        if include_ps_scores:
            label_target = batch_data['ps_scores'].cuda(non_blocking=True)  # todo ps scores as soft pseudo labels

        else:
            label_target = batch_data["label"].cuda(non_blocking=True)  # (bz, 1)
            label_target = label_target.reshape(-1)

        if mixup_fn is not None:
            images, label_target = mixup_fn(images, label_target, if_soft_labels=include_ps_scores)
            # CutmixMixupBlending:  for 0.5, do Cutmix,    for 0.5 do Mixup

        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
            with torch.autocast(device_type='cuda', dtype=torch.float16):  # todo torch.autocast  or torch.cuda.amp.autocast
                output = model(images)
                total_loss = criterion(output, label_target)
                total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        else:
            output = model(images)  # (bz, n_class )
            total_loss = criterion(output, label_target)  # Here todo  criterion is SoftTargetCrossEntropy, this is the loss used with mixup
            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        # todo ############################# compute gradients ####################################
        #   gradient accumulation:  https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
        #      use a small batch but save the gradients and update network weights once every couple of batches
        #      instead of updating  the network weights on every batch, we can save gradient values, proceed to the next batch and add up the new gradients.
        #      the weight update is done only after several batches have been processed by the model.
        #      Gradient accumulation helps to imitate a larger batch size. Imagine you want to use 32 images in one batch,
        #      but your hardware crashes once you go beyond 8. In that case, you can use batches of 8 images and update weights once every 4 batches.
        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            if config.MODEL.R_LORA <= 0:  # does not use LoRA
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: # LoRA is used
                scaler.scale(total_loss).backward()
        else: # todo  no mixed precision
            total_loss.backward()
        # todo ############################# update model weights ####################################
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            # # todo  gradient accumulation
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()   # todo   if accumulation_steps > 1,  update model weights evey  [accumulation_steps] steps
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # todo  no gradient accumulation
            if config.TRAIN.OPT_LEVEL != 'O0' and config.MODEL.R_LORA > 0:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step() # todo   if accumulation_steps = 1,  update model weights at every step
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_target))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
