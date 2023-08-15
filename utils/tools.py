import numpy
import torch.distributed as dist
import torch
import clip
import os
import numpy as np
from sklearn import metrics
import copy as cp
import os.path as osp
def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    #  todo all_reduce :  reduces the tensor data across all machines in such a way that all get the final result
    #    after the call tensor is going to be bitwise identical in all processes
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # in default mode, async:_op is set to False
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # here  n  is set to batch_size
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()  # take the average among all GPUs
        self.sum = reduce_tensor(sum_v, 1).item()  # todo  sum up the values in all GPUs
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


class meanAP_Tool:
    def __init__(self, n_test_data, n_class):
        self.y_true = torch.zeros((n_test_data, n_class)).cuda()
        self.y_pred = torch.zeros((n_test_data, n_class)).cuda()
    def update(self, batch_idx, prediction_score, gt_onehot_label):
        self.y_true[batch_idx, :] = gt_onehot_label
        self.y_pred[batch_idx, :] = prediction_score
    def sync(self):
        self.y_true_all = reduce_tensor( self.y_true, 1 ) # todo  sum up the values in all GPUs
        self.y_pred_all = reduce_tensor(self.y_pred, 1)
        print(self.y_true_all)
        print(self.y_pred_all)
    def compute_mAP(self):
        self.sync()
        # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        self.mAP = metrics.average_precision_score( self.y_true_all.cpu().numpy(), self.y_pred_all.cpu().numpy(), average='macro')

class computePseudoLabelTool:
    def __init__(self, n_test_data, n_class):
        self.y_pred = torch.zeros((n_test_data, n_class)).cuda()  #  prediction scores
    def update(self, batch_idx, prediction_score):
        self.y_pred[batch_idx, :] = prediction_score
    def sync(self):
        self.y_pred_all = reduce_tensor(self.y_pred, 1)
    def write_to_file(self, file_path, video_infos, pred_score_file):
        self.sync()
        self.y_pred_all = self.y_pred_all.cpu().numpy()
        np.save( pred_score_file, self.y_pred_all)

        pred_labels = np.argmax(self.y_pred_all, axis= -1 )
        n_correct = 0
        f_write = open(file_path, 'w+')
        for vid_id, video_info in enumerate(video_infos):
            vid_path = '/'.join( video_info['filename'].split('/')[-3:])
            gt_label = video_info['label']
            pred_label = pred_labels[vid_id]
            f_write.write( f'{vid_path} {pred_label}\n' )
            if pred_label == gt_label:
                n_correct += 1
        f_write.close()
        acc = float(n_correct) / len(video_infos)
        print(f'Acc Top1 {acc*100.0:.3f}')
def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")



def wise_state_dict(ori_model, loaded_state_dict, weight_for_origin = None):
    finetuned_model = cp.deepcopy( ori_model)
    msg = finetuned_model.load_state_dict(loaded_state_dict, strict = False)
    # todo   the  logit_scale does not affect the inference procedure
    print(f'load finetuned model {msg}')

    state_dict_ori = dict(ori_model.named_parameters())
    state_dict_finetuned = dict(finetuned_model.named_parameters())

    assert set(state_dict_ori) == set(state_dict_finetuned)

    fused_dict = {   k:  weight_for_origin * state_dict_ori[k]  + (1-weight_for_origin) * state_dict_finetuned[k] for k in state_dict_ori }
    return fused_dict
    # finetuned_model.load_state_dict( fused_dict )
    # return finetuned_model


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        if config.TEST.ONLY_TEST and config.MODEL.FUSE_WEIGHT_FOR_ORIGIN != 0:
            # todo Wise FT, fuse the loaded model weights with original CLIP weights
            fused_state_dict = wise_state_dict(ori_model=model, loaded_state_dict=load_state_dict, weight_for_origin=config.MODEL.FUSE_WEIGHT_FOR_ORIGIN)
            msg = model.load_state_dict(fused_state_dict, strict = False)
            logger.info(f"Wise FT weight for origin {config.MODEL.FUSE_WEIGHT_FOR_ORIGIN}, fused model {msg}")
        else:
            msg = model.load_state_dict(load_state_dict, strict=False)
            logger.info(f"resume model: {msg}")

        if config.TRAIN.LOAD_OPTIMIZER_STATE:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

                start_epoch = checkpoint['epoch'] + 1
                max_accuracy = checkpoint['max_accuracy']

                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

                del checkpoint
                torch.cuda.empty_cache()

                return start_epoch, max_accuracy
            except:
                del checkpoint
                torch.cuda.empty_cache()
                return 0, 0.
        else:
            logger.info(f'=> loaded model weights, but not optimizer state, from {config.MODEL.RESUME}')
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes

def model_analysis(model, logger):
    # print("Model Structure")
    # print(model)

    counter = 0
    for name, params in model.named_parameters():
        counter += 1
        print(name, params.requires_grad, np.prod(params.size()))
    print(f'length of list of named parameters {counter}')

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # param_list = [np.prod(p.size()) for p in model_parameters]
    # print(param_list)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_list = [np.prod(p.size()) for p in model_parameters]
    print(param_list)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.debug('#################################################')
    logger.debug(f'Number of trainable parameters: {params}')
    logger.debug('#################################################')