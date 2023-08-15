




from torch.nn import functional as F
import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss
import numpy as np


def gen_mask(labels):
    bz = labels.shape[0]
    mask = torch.zeros((bz, bz), dtype=torch.bool)
    for i in range(bz):
        for j in range(bz):
            if labels[i] == labels[j]:
                mask[i,j] = True
    return mask

def _rows_to_columns_nce_loss(scores: torch.Tensor, reduction = "mean") -> torch.Tensor:
    loss = - F.log_softmax(scores, dim=-1).diag()   #  scores (bz, bz)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def _rows_to_columns_nce_loss_w_labelmask(scores: torch.Tensor, reduction ="mean", label_mask = None) -> torch.Tensor:
    #  todo  NCE version 2
    #  todo  instead of only taking the diagonal similarities,
    #        take all visual-text similarities whose visual and text are from the same category (according to the pseudo label).
    # loss = - F.log_softmax(scores, dim=-1).diag()   #  scores (bz, bz)
    loss = - torch.masked_select( F.log_softmax(scores, dim=-1) ,  label_mask  )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

# todo video-to-text and text-to-video
def nce_loss(scores: torch.Tensor, reduction = "mean", label_mask = None) -> torch.Tensor:
    if False:
        return (_rows_to_columns_nce_loss(scores, reduction=reduction)
                + _rows_to_columns_nce_loss(scores.T, reduction=reduction))
    else:
        return (_rows_to_columns_nce_loss_w_labelmask(scores, reduction=reduction, label_mask=label_mask)
                + _rows_to_columns_nce_loss_w_labelmask(scores.T, reduction=reduction, label_mask=label_mask))

class NCELoss(_Loss):
    # @overrides(check_signature=False)
    def forward(self, scores: torch.Tensor, label_mask = None) -> torch.Tensor:
        return nce_loss(scores, reduction=self.reduction, label_mask=label_mask)  # noqa       self.reduction, default is 'mean'

def mil_nce(scores: torch.Tensor, reduction = "mean", bz=None,  n_captions_per_vid= None,) -> torch.Tensor:
    # todo only take the diagonal similarities
    # bz = scores.size(1) / n_captions_per_vid
    softmax_scores = F.softmax(scores, dim=-1)
    textbag_scores = torch.zeros((bz, bz)).to(softmax_scores.device)
    for idx in range(bz):  # sum up the softmax scores within each textbag
        textbag_scores[:, idx] = torch.sum( softmax_scores[:, idx * n_captions_per_vid :(idx+1) * n_captions_per_vid], dim=1)
    loss = - torch.log(textbag_scores).diag() # todo only take the diagonal similarities
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def mil_nce_w_labelmask(scores: torch.Tensor, reduction ="mean", label_mask = None, bz= None, n_captions_per_vid= None, ) -> torch.Tensor:
    # MIL NCE loss v2
    #  todo   MIL-NCE loss, only video-to-textbag similarities
    #  todo  instead of only taking the diagonal similarities,
    #        take all visual-text similarities whose visual and text are from the same category (according to the pseudo label).
    #     scores  # (bz,   bz*n_captions_per_vid)
    # bz = scores.size(1) / n_captions_per_vid
    softmax_scores = F.softmax(scores, dim=-1)
    textbag_scores = torch.zeros((bz, bz)).to(softmax_scores.device)
    for idx in range(bz):  # sum up the softmax scores within each textbag
        textbag_scores[:, idx] = torch.sum( softmax_scores[:, idx * n_captions_per_vid :(idx+1) * n_captions_per_vid], dim=1)
    loss = - torch.masked_select(torch.log(textbag_scores), label_mask)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def mil_nce_w_labelmask_vid2txt_txt2vid(scores: torch.Tensor, reduction ="mean", label_mask = None, bz= None, n_captions_per_vid= None, ) -> torch.Tensor:
    # todo MIL-NCE loss,  video-to-textbag similarities,  and max-text-to-video similarities
    #  todo  instead of only taking the diagonal similarities,
    #        take all visual-text similarities whose visual and text are from the same category (according to the pseudo label).
    #     scores  # (bz,   bz*n_captions_per_vid)
    softmax_scores = F.softmax(scores, dim=-1)  # (bz,   bz*n_captions_per_vid)  todo normalization is done among the words
    textbag_scores = torch.zeros((bz, bz)).to(softmax_scores.device)
    for idx in range(bz):  # sum up the softmax scores within each textbag
        textbag_scores[:, idx] = torch.sum( softmax_scores[:, idx * n_captions_per_vid :(idx+1) * n_captions_per_vid], dim=1)
    loss1 = - torch.masked_select(torch.log(textbag_scores), label_mask)
    loss1 = loss1.mean()

    # todo loss term 2 : text to video score
    indices_to_select = []
    for idx in range(bz):
        max_idx_in_bag = torch.argmax( scores[idx, idx * n_captions_per_vid :(idx+1) * n_captions_per_vid] ) # index of the maximum word within the bag
        indices_to_select.append( max_idx_in_bag + idx * n_captions_per_vid )
    text_to_video_score = scores[:, torch.tensor( indices_to_select )].t()   #  (bz = n_selected_words,   bz = n_vids )
    log_softmax = - F.log_softmax( text_to_video_score,  dim=-1 )  #  todo normalization is done among the videos
    loss2 = torch.masked_select( log_softmax, label_mask ) #  label_mask is a synmmetric matrix
    loss2 = loss2.mean()

    return loss1 + loss2


class MIL_NCEloss(_Loss):
    def forward(self, scores: torch.Tensor, label_mask = None, bz =None, n_captions_per_vid= None,) -> torch.Tensor:

        # return mil_nce(scores, reduction=self.reduction, bz=bz, n_captions_per_vid=n_captions_per_vid)

        return mil_nce_w_labelmask(scores, reduction=self.reduction, label_mask=label_mask, bz=bz, n_captions_per_vid=n_captions_per_vid)  # noqa       self.reduction, default is 'mean'

        # return mil_nce_w_labelmask_vid2txt_txt2vid(scores, reduction=self.reduction, label_mask=label_mask, bz=bz, n_captions_per_vid=n_captions_per_vid)  # noqa       self.reduction, default is 'mean'


class MIL_extract_max_Loss(_Loss):
    def forward(self, scores: torch.Tensor, label_mask = None, bz =None, n_captions_per_vid= None, ) -> torch.Tensor:
        # todo     scores  # (bz,   bz*n_captions_per_vid)
        max_scores = torch.zeros((bz, bz)).to(scores.device)
        for i in range(bz):
            for j in range(bz):
                #  todo extract the maximum word out of each bag
                max_scores[i,j] = torch.max( scores[i, j * n_captions_per_vid  : (j+1) * n_captions_per_vid ] )

        return (_rows_to_columns_nce_loss_w_labelmask(max_scores, label_mask=label_mask)
                + _rows_to_columns_nce_loss_w_labelmask(max_scores.T, label_mask=label_mask))


class MIL_NCEloss_topk_class_bag(_Loss ):
    def forward(self, scores: torch.Tensor, label_mask: torch.Tensor , reduction = 'mean'):
        #  scores   (bz, bz*n_samples_in_bag),   label_mask  (bz, bz*n_samples_in_bag)
        softmax_scores = F.softmax( scores, dim=-1 )
        loss = -  torch.masked_select(   torch.log( softmax_scores )   , label_mask)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


class MIL_Max_Loss(_Loss):
    def forward(self, scores: torch.Tensor,  vid_bag_list = None, b = None,  n_samples_in_bag = None , reduction = 'mean'):



        loss1 = torch.tensor(0).float().to(scores.device)
        best_words_list = []
        best_words_global_colum_id_list = []
        best_word_to_pos_vid_id_dict = dict()

        # todo term 1 :  video to text score
        #     for all the videos in the batch
        #     positive: the maximum word in the pose bag,
        #     negatives:  all the unique words in the all negative bags,  each word is summed only once
        #     dont'care:  the other non-maximum words in the bag are dont-cares
        for i in range(b):
            pos_bag = vid_bag_list[i]
            indices_to_select = []
            # find the maximum word in bag of current video --  this is the positive
            max_idx_in_bag = torch.argmax(scores[i, i * n_samples_in_bag: ( i + 1) * n_samples_in_bag])  # index of the maximum within the bag
            pos_idx = max_idx_in_bag + i * n_samples_in_bag
            best_word = pos_bag[max_idx_in_bag]
            if best_word not in best_words_list:
                best_words_list.append(  best_word )
                best_words_global_colum_id_list.append( pos_idx )
            if best_word not in best_word_to_pos_vid_id_dict:
                best_word_to_pos_vid_id_dict.update({best_word : []})

            best_word_to_pos_vid_id_dict[best_word].append( i )
            # collect the id of the best word as the positive

            indices_to_select.append(pos_idx)
            neg_bag_collection = []

            for j in range(b):
                if j != i:
                    for word_idx, word_ in enumerate(vid_bag_list[j]):
                        if (word_ not in pos_bag) and (word_ not in neg_bag_collection):
                            neg_bag_collection.append(word_)  # add the negative word to collection
                            neg_idx = word_idx + j * n_samples_in_bag
                            indices_to_select.append(neg_idx)
            # softmax_scores = - torch.log(  F.softmax(  logits[i, indices_to_select]  ))
            indices_to_select = torch.tensor(indices_to_select)
            loss1 += - F.log_softmax(scores[i, indices_to_select])[0]

        loss1 /= b
        # todo term 2:  text to video score
        #  for the words that have a best match to a video  (it is possible that one word has multiples best-match videos )
        #  positive:  the best matched videos
        #   negative:  the videos whose bag does not contain this word
        #   dont-care:  the videos whose bag contains this word, but this word is not best match
        loss2 = torch.tensor(0).float().to(scores.device)
        best_word_to_neg_vid_id_dict = dict()
        for best_word in best_words_list:
            # if best_word not in best_word_to_neg_vid_id_dict:
            best_word_to_neg_vid_id_dict.update({best_word: []})
            for vid_id, vid_bag in enumerate(vid_bag_list):
                if best_word not in vid_bag:
                    best_word_to_neg_vid_id_dict[best_word].append(vid_id)

        text_to_video_score = scores[:, torch.tensor( best_words_global_colum_id_list ) ].t()  # (n_best_words,  b )
        for i, best_word in enumerate(best_words_list):
            positive_id = best_word_to_pos_vid_id_dict[best_word]
            negative_id = best_word_to_neg_vid_id_dict[best_word]
            indices_to_select = torch.tensor(  positive_id + negative_id  )
            log_softmax = -F.log_softmax( text_to_video_score[i,  indices_to_select ])
            loss2 +=  log_softmax[:len(positive_id)].sum()
        loss2 /= len(best_words_list)


        return loss1 + loss2


        # if reduction == "mean":
        #     loss /= b
        #     return loss
        # elif reduction == "sum":
        #     return loss
        # else:
        #     return loss
