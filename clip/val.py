



import torch
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, model_analysis, meanAP_Tool, computePseudoLabelTool
from utils.utils_ import make_dir

import os.path as osp




@torch.no_grad()
def compute_description_nn_for_each_class(val_loader, model, config, loaded_clip_model = None, logger = None):
    make_dir(config.DATA.NN_DESCRIPTION_RESULT)
    model.eval()
    n_description_per_class = config.DATA.N_DESCRIPTION
    n_cls = config.DATA.NUM_CLASSES
    with torch.no_grad():
        model.module.compute_text_features(loaded_clip_model)  # self.text_feature_dict contain description text features of each class
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            batch_data, batch_idx = batch_data
            _image = batch_data["imgs"]  # (bz, n_frames * n_clip * n_crop, 3, 224, 224)
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)  # (bz, )

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t  # number of views    n_clip * n_crop
            _image = _image.view(b, n, t, c, h, w)  # (bz, n_clip * n_crop, n_frames, 3, 224, 224)

            tot_similarity_dict = dict()
            for class_id in range(n_cls):
                tot_similarity_dict.update({class_id: torch.zeros((b,  n_description_per_class)).cuda()})  # dict of  (bz, n_description_per_class)
            for i in range(n): # loop over all views
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                output_dict = model(image_input, batch_idx)
                for class_id in range(n_cls):
                    similarity = output_dict[class_id].view(b, -1).softmax(dim=-1) # (bz, n_descriptions)
                    tot_similarity_dict[class_id] += similarity  # todo the similartiy scores of all viwes are summed up
            f_write_list = []
            for idx_ in range(b):
                vid_id = batch_idx[idx_].item()
                f_write_list.append(  open( osp.join(config.DATA.NN_DESCRIPTION_RESULT , f'vid{vid_id}.txt'), 'w+') )
                # f_write = open( osp.join(config.DATA.NN_DESCRIPTION_RESULT , f'vid{vid_id}.txt'), 'w+')
            for class_id in range(n_cls):
                similarity = tot_similarity_dict[class_id].cpu().numpy()  # (bz, n_description_per_class)

                for idx_ in range(b):

                    f_write_list[idx_].write(' '.join([f'{value:.3f}' for value in  list( similarity[idx_, :]  )]) + '\n' )
                    # vid_id = batch_idx[idx].item()
                    # f_write_list[idx].write( tot_similarity_dict[class_id] )

            for idx_ in range(b):
                f_write_list[idx_].close()

            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]'
                )

@torch.no_grad()
def validate_compute_description_nn(val_loader, model, config, loaded_clip_model = None, logger = None):
    model.eval()
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    n_description_per_class = config.DATA.N_DESCRIPTION
    description_top20_indices_dict = dict()

    with torch.no_grad():
        model.module.compute_text_features(loaded_clip_model)
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            batch_data, batch_idx = batch_data
            _image = batch_data["imgs"]  # (bz, n_frames * n_clip * n_crop, 3, 224, 224)
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)  # (bz, )

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t  # number of views    n_clip * n_crop
            _image = _image.view(b, n, t, c, h, w)  # (bz, n_clip * n_crop, n_frames, 3, 224, 224)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES * n_description_per_class)).cuda()  # todo sum over the similartiy scores over all views
            for i in range(n):  # loop over all views
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                output = model(image_input, batch_idx)  # (bz,  n_cls*n_descriptions)
                similarity = output.view(b, -1).softmax(dim=-1)  # # (bz,  n_cls*n_descriptions)
                tot_similarity += similarity  # todo the similarty scores of all views are summed up
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)  # indices_1 (bz, 1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1) # indices_5 (bz, 5)
            values_20, indices_20 = tot_similarity.topk(20, dim=-1)  # indices_5 (bz, 10)
            # ps_label_top1 = indices_1 //n_description_per_class
            # ps_label_top5 = indices_5 //n_description_per_class
            ps_label_top1 = torch.div(indices_1, n_description_per_class, rounding_mode='floor')
            ps_label_top5 = torch.div(indices_5, n_description_per_class, rounding_mode='floor')

            acc1, acc5 = 0, 0
            for i in range(b):
                description_top20_indices_dict.update({ batch_idx[i].item() :  list(indices_20[i].cpu().numpy()) })
                if ps_label_top1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in ps_label_top5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, description_top20_indices_dict

@torch.no_grad()
def validate_description_ensemble(val_loader, model, config, loaded_clip_model = None, logger = None):
    model.eval()
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    # n_description_per_class = config.DATA.N_DESCRIPTION
    # description_top20_indices_dict = dict()

    with torch.no_grad():
        model.module.compute_text_features(loaded_clip_model)
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            batch_data, batch_idx = batch_data
            _image = batch_data["imgs"]  # (bz, n_frames * n_clip * n_crop, 3, 224, 224)
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)  # (bz, )

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t  # number of views    n_clip * n_crop
            _image = _image.view(b, n, t, c, h, w)  # (bz, n_clip * n_crop, n_frames, 3, 224, 224)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES )).cuda()  # todo sum over the similartiy scores over all views
            for i in range(n):  # loop over all views
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                output = model(image_input, batch_idx)  # (bz,  n_cls*n_descriptions)
                similarity = output.view(b, -1).softmax(dim=-1)  # # (bz,  n_cls*n_descriptions)
                tot_similarity += similarity  # todo the similarty scores of all views are summed up
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)  # indices_1 (bz, 1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1) # indices_5 (bz, 5)

            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg

@torch.no_grad()
def validate(val_loader, model, config, logger = None):
    model.eval()
    compute_ps = config.TEST.COMPUTE_PS
    if config.DATA.DATASET == 'Charades':
        n_test_samples = len(val_loader.dataset.video_infos)
        mAP_tool = meanAP_Tool(n_test_data=n_test_samples, n_class=config.DATA.NUM_CLASSES)
    else:
        acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    if compute_ps:
        n_test_samples = len(val_loader.dataset.video_infos)
        compute_ps_tool = computePseudoLabelTool(n_test_data=n_test_samples, n_class=config.DATA.NUM_CLASSES)
    with torch.no_grad():
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")

        if config.TEST.ONLY_TEST:
            model.module.compute_text_features()

        for idx, batch_data in enumerate(val_loader):
            if config.DATA.DATASET == 'Charades' or  config.TEST.COMPUTE_PS:
                batch_data, batch_idx = batch_data
            _image = batch_data["imgs"]   # (bz, n_frames * n_clip * n_crop, 3, 224, 224)
            label_id = batch_data["label"]
            if label_id.size(-1) == 1:
                label_id = label_id.reshape(-1) # (bz, )

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t   #  number of views    n_clip * n_crop
            _image = _image.view(b, n, t, c, h, w) # (bz, n_clip * n_crop, n_frames, 3, 224, 224)

            if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max',
                                                                                    'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max']:
                # todo  validation task is the zero-shot classification on another dataset
                tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES_ZS)).cuda()
            else:
                tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):   # loop over all views
                image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                if config.DATA.USE_DESCRIPTION and config.DATA.USE_DESCRIPTION_TYPE in ['train cls+cap', 'train cap', 'train synonym+cap','train cls', 'train cls+synonym', 'train cls+cap max',
                                                                                        'train cls+cap mil max','train cls+cap mil softmax', 'train cls+cap mil nce', 'train topk class bag mil', 'train mil extract max']:
                    # todo  validation task is the zero-shot classification on another dataset
                    output = model.module.val_zs_classification(image_input)
                else:
                    output = model(image_input)


                similarity = output.view(b, -1).softmax(dim=-1)  # (bz, n_cls)
                tot_similarity += similarity   # todo the similarty scores of all views are summed up  as the prediction score
            if config.DATA.DATASET == 'Charades':
                # todo collecting the gt_onehot_label and prediction score
                mAP_tool.update( batch_idx, tot_similarity, label_id )
                if idx % config.PRINT_FREQ == 0:
                    logger.info(
                        f'Test mAP: [{idx}/{len(val_loader)}]\t'
                    )
            else:
                # compute top1 and top5 accuracy
                values_1, indices_1 = tot_similarity.topk(1, dim=-1)  # indices_1 (bz, 1)
                values_5, indices_5 = tot_similarity.topk(5, dim=-1)  # indices_5 (bz, 5)

                acc1, acc5 = 0, 0
                for i in range(b):
                    if indices_1[i] == label_id[i]:
                        acc1 += 1
                    if label_id[i] in indices_5[i]:
                        acc5 += 1

                acc1_meter.update(float(acc1) / b * 100, b)
                acc5_meter.update(float(acc5) / b * 100, b)
                if idx % config.PRINT_FREQ == 0:
                    logger.info(
                        f'Test: [{idx}/{len(val_loader)}]\t'
                        f'Acc@1: {acc1_meter.avg:.3f}\t'
                    )
            if compute_ps:
                compute_ps_tool.update(batch_idx, tot_similarity)
    if compute_ps:
        compute_ps_tool.write_to_file(file_path= osp.join(config.OUTPUT, 'ps_label.txt'),  video_infos= val_loader.dataset.video_infos,
                                      pred_score_file= osp.join(config.OUTPUT, 'ps_pred_score.npy'  ))
    if config.DATA.DATASET == 'Charades':
        mAP_tool.compute_mAP()
        logger.info(f' * mAP {mAP_tool.mAP * 100.0:.3f}')
        return mAP_tool.mAP
    else:
        acc1_meter.sync()
        acc5_meter.sync()
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
        return acc1_meter.avg