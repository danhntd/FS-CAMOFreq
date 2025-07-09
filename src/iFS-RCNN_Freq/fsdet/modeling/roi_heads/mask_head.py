# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
import math
from .triplet_custom import TripletMarginWithDistanceLoss

__all__ = [
    "BaseMaskRCNNHead",
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "ROI_MASK_HEAD_REGISTRY",
]


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""

# our triplet loss and memory loss

class PairwiseCosine(torch.nn.Module):
    def __init__(self, dim=-1):
        self.cos = torch.nn.CosineSimilarity(dim=dim)
    def forward(self, input1, input2):
        return 1 - self.cos(input1, input2)


class memory_bank_func(torch.nn.Module):
    def __init__(self, num_classes=2, capacity=32, input_dim=1024, device='cpu', tau=1):
        super(memory_bank_func, self).__init__()
        self.memory = torch.nn.Parameter(torch.randn(num_classes, capacity, input_dim), requires_grad=False).to(device)
        self.cap = capacity
        self.num_classes = num_classes
        self.device = device
        self.tau = tau

    @torch.no_grad()
    def update(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        for cls in __unique:
            if cls == self.num_classes: continue
            candidates = self.memory[cls]

            index     = torch.where(classes==cls)
            #print('candidates.shape', candidates.shape)
            #print('instances[index[0]].shape', instances[index[0]].shape)

            new_ins = torch.cat([instances[index[0]], candidates])
            self.memory[cls] = new_ins[:self.cap]
            del new_ins, candidates

    @torch.no_grad()
    def get_mem(self, cls):
        return self.memory[cls]

    def forward(self, x, classes):
        self.update(x, classes)
        return self.call_loss(x, classes, self.tau)

    def call_loss(self, feat, labels, tau): #tau=5e-3
        # tau càng tiến đến 1 thì sự phân biệt giữa negative và anchor càng lớn.
        # và sự giống nhau giữa postive và anchor càng lớn.

        memory = self.memory.detach().clone()
        ###memory = torch.nn.functional.normalize(memory, p=2, dim=-1) ###

        centroids = memory.mean(dim=1)

        #print("feat.shape", feat.shape)

        # apply cosine by divide to norm, create unit vector for memory and centroids
        # memory = memory/(torch.norm(memory, p=2, dim=-1, keepdim=True) + 1e-5)
        # centroids = centroids/(torch.norm(centroids, p=2, dim=-1, keepdim=True) + 1e-5)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1) ###

        b = feat.shape[0]
        index = torch.arange(b)
        positive_logits = (feat@centroids.T/tau)[index, labels]
        # torch.einsum('bij,kij-->', a)
        mask = torch.ones(b, self.cap, self.num_classes, dtype=torch.bool)
        mask[index, :, labels] = 0

        index_negative = torch.where(mask.view(b, -1)==1)
        negative_logits = (feat@memory.view(-1, memory.shape[-1]).T)/tau
        negative_logits = negative_logits[index_negative[0], index_negative[1]].view(b, -1)

        # print(positive_logits.shape)
        # print(negative_logits.shape)

        final_logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        final_labels = torch.zeros(final_logits.shape[0]).to(torch.long).to(self.device)

        return torch.nn.functional.cross_entropy(final_logits, final_labels)

    def call_loss2(self, feat, labels, tau):
        memory = self.memory.detach().clone()
        normalize = torch.nn.functional.normalize

        norm_memory = normalize(memory, p=2, dim=-1)
        norm_feat = normalize(feat, p=2, dim=-1)
        norm_centroids = normalize(memory.mean(dim=1), p=2, dim=-1)

        b = feat.shape[0]
        index = torch.arange(b)
        positive_logits = (norm_feat@norm_centroids.T/tau)[index, labels]

        mask = torch.ones(b, self.cap, self.num_classes, dtype=torch.bool)
        mask[index, :, labels] = 0

        index_negative = torch.where(mask.view(b, -1)==1)
        negative_logits = (norm_feat@norm_memory.view(-1, memory.shape[-1]).T)/tau
        negative_logits = negative_logits[index_negative[0], index_negative[1]].view(b, -1)

        final_logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        final_labels = torch.zeros(final_logits.shape[0]).to(torch.long).to(self.device)

        return torch.nn.functional.cross_entropy(final_logits, final_labels)#, final_logits


def nansum(x):
    x[torch.isnan(x)] = 0
    return x.sum()

@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], feat_triplet, vis_period: int = 0, memory_loss_func=None, loss_type_camo='origin', triplet_margin=0.5, alpha_triplet=10e-2, beta_memory=10e-3):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")

    output_return = {}

    if loss_type_camo == 'origin':
        output_return['loss_mask'] = mask_loss
    elif loss_type_camo == 'memory':
        # memory on each instance
        all_ins_memory = []
        for idx_ins in range(feat_triplet.shape[0]):
            fg_idx = torch.where(gt_masks[idx_ins]==1)
            bg_idx = torch.where(gt_masks[idx_ins]==0)
            ins_cls = gt_classes[idx_ins]
            # print('fg_idx[0].shape', fg_idx[0].shape)
            # print('bg_idx[0].shape', bg_idx[0].shape)

            # print("feat_triplet.shape", feat_triplet.shape)
            # print("gt_classes.shape", gt_classes.shape)
            # print("gt_classes", gt_classes)

            fg = feat_triplet[idx_ins,:, fg_idx[0], fg_idx[1]].T
            bg = feat_triplet[idx_ins,:, bg_idx[0], bg_idx[1]].T
            # print('0 fg.shape', fg.shape)
            # print('0 bg.shape', bg.shape)

            ###fg = fg.mean(dim=(0,),keepdim=True)
            ###bg = bg.mean(dim=(0,),keepdim=True)

            #print('0 fg.shape', fg.shape)
            #print('0 bg.shape', bg.shape)

            len_to_keep = len(fg_idx[1]) if len(fg_idx[1]) < len(bg_idx[1]) else len(bg_idx[1])
            # print('len_to_keep', len_to_keep)
            fg = fg[:len_to_keep,:]
            bg = bg[:len_to_keep,:]
            #print('1 fg.shape', fg.shape)
            #print('1 bg.shape', bg.shape)

            __memory_feature = torch.cat([fg, bg], dim=0).cuda()
            if __memory_feature.shape[0] == 0:
                continue
            __memory_labels  = torch.zeros(__memory_feature.shape[0]).cuda().to(torch.long)
            __memory_labels[:fg.shape[0]] += 1

            loss_memory = memory_loss_func[ins_cls](__memory_feature,__memory_labels)
            all_ins_memory.append(loss_memory)

        all_output_memory = torch.mean(torch.stack(all_ins_memory))

        output_return['loss_mask'] = mask_loss
        output_return['loss_memory'] = all_output_memory*beta_memory#*10e-3 was used when ablation

    elif loss_type_camo == 'triplet':
        # triplet on each instance
        # triplet_loss = nn.TripletMarginLoss(margin=2.0, p=2, reduction='none')
        triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=triplet_margin, reduction='none')
        all_ins_triplet = []
        for idx_ins in range(feat_triplet.shape[0]):
            fg_idx = torch.where(gt_masks[idx_ins]==1)
            bg_idx = torch.where(gt_masks[idx_ins]==0)
            ins_cls = gt_classes[idx_ins]
            # print('fg_idx[0].shape', fg_idx[0].shape)
            # print('bg_idx[0].shape', bg_idx[0].shape)

            # print("feat_triplet.shape", feat_triplet.shape)
            # print("gt_classes.shape", gt_classes.shape)
            # print("gt_classes", gt_classes)

            fg = feat_triplet[idx_ins,:, fg_idx[0], fg_idx[1]].T
            bg = feat_triplet[idx_ins,:, bg_idx[0], bg_idx[1]].T
            # print('0 fg.shape', fg.shape)
            # print('0 bg.shape', bg.shape)

            #print("fg_idx[1].shape", fg_idx[1].shape)
            #print("fg_idx[1]", fg_idx[1])
            #print("bg_idx[1].shape", bg_idx[1].shape)
            #print("bg_idx[1]", bg_idx[1])

            #print("fg_idx[0].shape", fg_idx[0].shape)
            #print("fg_idx[0]", fg_idx[0])
            #print("bg_idx[0].shape", bg_idx[0].shape)
            #print("bg_idx[0]", bg_idx[0])

            #print("fg.shape", fg.shape)
            #print("bg.shape", bg.shape)
            len_to_keep = len(fg_idx[1]) if len(fg_idx[1]) < len(bg_idx[1]) else len(bg_idx[1])
            # print('len_to_keep', len_to_keep)
            fg = fg[:len_to_keep,:]
            bg = bg[:len_to_keep,:]

            ###fg = fg.mean(dim=(0,),keepdim=True)
            ###bg = bg.mean(dim=(0,),keepdim=True)

            #print('1 fg', fg)
            #print('1 bg', bg)

            ###
            #print("torch.sum(fg, 1)", torch.sum(fg, 1))
            #print(torch.sum(bg, 1))


            #flag = 0
            #print("torch.where(torch.sum(fg, 1)==0))", torch.where(torch.sum(fg, 1)==0)[0])
            #print("len(torch.where(torch.sum(fg, 1))==0)", len(torch.where(torch.sum(fg, 1)==0)[0]))
            #print("len(torch.where(torch.sum(bg, 1))==0)", len(torch.where(torch.sum(bg, 1)==0)[0]))

            #if len(torch.where(torch.sum(fg, 1)==0)[0])>0 or len(torch.where(torch.sum(bg, 1)==0)[0])>0: #eleminate cases that has 0 fg or fg leading to nan
            #    flag = 1

            #for i in range(fg.shape[0]):
            #    if len(torch.unique(fg[i].clone().detach())) == 1 or len(torch.unique(bg[i].clone().detach())) == 1:
            #        flag = 1

            #if flag == 1:
            #    print("flag", flag)
            #    continue
            ###

            #print('1 fg.shape', fg.shape)
            #print('1 bg.shape', bg.shape)


            anchor = torch.mean(fg, dim=0, keepdim=True)
            #print('anchor.shape', anchor.shape)
            anchor = anchor.repeat(fg.shape[0], 1)
            #print('anchor.shape', anchor.shape)
            #print('anchor', anchor)

            #print("torch.any(torch.isnan(fg))", torch.any(torch.isnan(fg)))
            #print("torch.any(torch.isnan(bg))", torch.any(torch.isnan(bg)))
            #print("torch.any(torch.isnan(anchor))", torch.any(torch.isnan(anchor)))


            output_triplet_loss = triplet_loss(anchor, fg, bg)
            #print('output_triplet_loss', output_triplet_loss)

            output_triplet_loss = nansum(output_triplet_loss)/output_triplet_loss.shape[0] # avoid nan in loss
            if torch.isnan(output_triplet_loss):
                print("loss is nan here")
                continue

            #print('output_triplet_loss', output_triplet_loss)
            all_ins_triplet.append(output_triplet_loss)

        all_output_triplet_loss = torch.mean(torch.stack(all_ins_triplet))

        output_return['loss_mask'] = mask_loss
        output_return['loss_triplet'] = all_output_triplet_loss*alpha_triplet#*10e-2

    elif loss_type_camo == 'both':
        # triplet and memory on each instance
        #triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=triplet_margin, reduction='none')
        all_ins_triplet = []
        all_ins_memory = []
        for idx_ins in range(feat_triplet.shape[0]):
            fg_idx = torch.where(gt_masks[idx_ins]==1)
            bg_idx = torch.where(gt_masks[idx_ins]==0)
            ins_cls = gt_classes[idx_ins]
            # print('fg_idx[0].shape', fg_idx[0].shape)
            # print('bg_idx[0].shape', bg_idx[0].shape)

            # print("feat_triplet.shape", feat_triplet.shape)
            # print("gt_classes.shape", gt_classes.shape)
            # print("gt_classes", gt_classes)

            fg = feat_triplet[idx_ins,:, fg_idx[0], fg_idx[1]].T
            bg = feat_triplet[idx_ins,:, bg_idx[0], bg_idx[1]].T

            # print('0 fg.shape', fg.shape)
            # print('0 bg.shape', bg.shape)

            ###fg = fg.mean(dim=(0,),keepdim=True)
            ###bg = bg.mean(dim=(0,),keepdim=True)

            # print('0 fg.shape', fg.shape)
            # print('0 bg.shape', bg.shape)

            len_to_keep = len(fg_idx[1]) if len(fg_idx[1]) < len(bg_idx[1]) else len(bg_idx[1])
            # print('len_to_keep', len_to_keep)
            fg = fg[:len_to_keep,:]
            bg = bg[:len_to_keep,:]
            # print('1 fg.shape', fg.shape)
            # print('1 bg.shape', bg.shape)

            anchor = torch.mean(fg, dim=0, keepdim=True).repeat(fg.shape[0], 1)

            output_triplet_loss = triplet_loss(anchor, fg, bg)

            #print('output_triplet_loss', output_triplet_loss)

            output_triplet_loss = nansum(output_triplet_loss)/output_triplet_loss.shape[0] # avoid nan in loss

            if torch.isnan(output_triplet_loss):
                print("loss is nan here")
                continue

            #print('output_triplet_loss', output_triplet_loss)
            all_ins_triplet.append(output_triplet_loss)

            __memory_feature = torch.cat([fg, bg], dim=0).cuda()
            if __memory_feature.shape[0] == 0:
                continue
            __memory_labels  = torch.zeros(__memory_feature.shape[0]).cuda().to(torch.long)
            __memory_labels[:fg.shape[0]] += 1
            loss_memory = memory_loss_func[ins_cls](__memory_feature,__memory_labels)
            all_ins_memory.append(loss_memory)

        all_output_triplet_loss = torch.mean(torch.stack(all_ins_triplet))
        all_output_memory = torch.mean(torch.stack(all_ins_memory))

        output_return['loss_mask'] = mask_loss
        output_return['loss_memory'] = all_output_memory*beta_memory#10e-3
        output_return['loss_triplet'] = all_output_triplet_loss*alpha_triplet#*10e-2


    # all ins together
    # print('gt_masks.shape', gt_masks.shape)
    #print('gt_masks[0]', gt_masks[0])
    # print('gt_masks[0].shape', gt_masks[0].shape)
    ###fg_idx = torch.where(gt_masks==1)
    # print('torch.where(gt_masks==1)', fg_idx)
    # print('len fg_idx', len(fg_idx[0]), len(fg_idx[1]), len(fg_idx[2]))
    ###bg_idx = torch.where(gt_masks==0)
    # print('torch.where(gt_masks==0)', bg_idx)
    # print('len bg_idx', len(bg_idx[0]), len(bg_idx[1]), len(bg_idx[2]))

    # print('fg_idx[0]', len(fg_idx[0]))
    # print('bg_idx[0]', len(bg_idx[0]))
    ###len_to_keep = len(fg_idx[0]) if len(fg_idx[0]) < len(bg_idx[0]) else len(bg_idx[0])
    # print('len_to_keep', len_to_keep)

    ###fg = feat_triplet[fg_idx[0],:, fg_idx[1], fg_idx[2]]
    ###bg = feat_triplet[bg_idx[0],:, bg_idx[1], bg_idx[2]]
    # print('fg.shape', fg.shape)
    # print('bg.shape', bg.shape)

    #len_to_keep = fg.shape[0] if fg.shape[0] < bg.shape[0] else bg.shape[0]
    #print('len_to_keep', len_to_keep)
    #print('bg.shape[1]', bg.shape[1])

    ###fg = fg[:len_to_keep,:]
    ###bg = bg[:len_to_keep,:]

    #for id_ins in range(fg_idx.shape[0]):
    #    fg = feat_triplet[id_ins, :, fg_idx[1], fg_idx[2]

    # print('fg.shape', fg.shape)
    # print('bg.shape', bg.shape)

    # print('pred_mask_logits.shape', pred_mask_logits.shape)

    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # anchor = torch.mean(fg)
    # anchor = torch.full(fg.shape, anchor).cuda()
    # print('anchor.shape', anchor.shape)
    # output_triplet_loss = triplet_loss(anchor, fg, bg)
    # print('output_triplet_loss', output_triplet_loss)

    # output_triplet_loss.requires_grad = True
    # output_triplet_loss.backward()
    # print('mask_loss', mask_loss)


    ###__memory_feature = torch.cat([fg, bg], dim=0).cuda()
    ###__memory_labels  = torch.zeros(__memory_feature.shape[0]).cuda().to(torch.long)
    ###__memory_labels[:fg.shape[0]] += 1

    ###loss_memory = memory_loss_func(__memory_feature,__memory_labels)

    #total_loss = mask_loss + output_triplet_loss
    #output_return =  {"loss_mask": mask_loss, 'loss_memory': all_output_memory}# 'loss_triplet': all_output_triplet_loss}#, 'loss_memory': loss_memory}
    #print('output_return', output_return)

    return output_return


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, vis_period=0):
        """
        NOTE: this interface is experimental.
        Args:
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.bayesian = False

    @classmethod
    def from_config(self, cfg, input_shape):
        # ours
        self.vis_period = cfg.VIS_PERIOD
        self.loss_type_camo = cfg.MODEL.LOSS_TYPE_CAMO
        self.triplet_margin = cfg.MODEL.TRIPLET_MARGIN
        self.tau = cfg.MODEL.TAU
        self.capacity = cfg.MODEL.CAPACITY
        self.alpha_triplet = cfg.MODEL.ALPHA_TRIPLET
        self.beta_memory = cfg.MODEL.BETA_MEMORY
        print('self.loss_type_camo', self.loss_type_camo, 'self.triplet_margin', self.triplet_margin, 'self.tau', self.tau, 'self.capacity', self.capacity, 'self.alpha_triplet', self.alpha_triplet, 'self.beta_memory', self.beta_memory)
        self.memory_bank = torch.nn.ModuleList([
                            memory_bank_func(num_classes=2, capacity=self.capacity, input_dim=256, device='cuda', tau=self.tau)
                            for _ in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)])

        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x, feat_triplet = self.layers(x)
        #print('x forward .shape', x.shape)
        if self.training:
            return_losses = mask_rcnn_loss(x, instances, feat_triplet, self.vis_period,  memory_loss_func=self.memory_bank, loss_type_camo=self.loss_type_camo, triplet_margin=self.triplet_margin, alpha_triplet=self.alpha_triplet, beta_memory=self.beta_memory)

            if self.bayesian:
                return_losses['loss_mask_reg'] = self.var.mean()
                # print(self.var.min().item(), self.var.max().item())
            return return_losses
        else:
            mask_rcnn_inference(x, instances)
            return instances


    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels[0]
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def set_freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    #def layers(self, x):
    #    for i, layer in enumerate(self):
    #        x = layer(x)
    #    return x

    def layers(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        # print("x.shape", x.shape)
        #print("x[0]", x[0])
        #print("x", np.argmax(x.cpu()))

        # feed forward for cosine in memory bank
        ###weight_data  = self.predictor.weight.data
        ###weight_data = weight_data/(torch.norm(weight_data, p=2, dim=1, keepdim=True) + 1e-5)
        ###self.predictor.weight.data = weight_data * 20
        ###y_norm = layer(x)

        # return self.predictor(x/(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-5)), x
        return self.predictor(x), x

@ROI_MASK_HEAD_REGISTRY.register()
class UncertaintyMaskRCNNConvUpsampleHead(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels[0]
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes*2, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def set_freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def loss(self, pred_mask_logits, instances):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(3)
        assert pred_mask_logits.size(4) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        mean, var = pred_mask_logits[:, 0], pred_mask_logits[:, 1]
        mean = mean.sigmoid()
        var = F.softplus(var) + 1e-6 # consider sigmoid here

        mask_loss = 1/2 * (mean - gt_masks).pow(2) / var + 1/2 * var

        return {"loss_mask": mask_loss.mean()}

    def inference(self, pred_mask_logits, pred_instances):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits[:, 0:1].sigmoid()
            mask_probs_var = F.softplus(pred_mask_logits[:, 1:2])
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            mask_pred = pred_mask_logits[indices, class_pred]
            mask_probs_pred = mask_pred[:, 0:].sigmoid()
            mask_probs_var = F.softplus(mask_pred[:, 1:2])

        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
        mask_probs_var = mask_probs_var.split(num_boxes_per_image, dim=0)

        for prob, var, instances in zip(mask_probs_pred, mask_probs_var, pred_instances):
            instances.pred_masks = prob  # (1, Hmask, Wmask)
            instances.pred_uncertainty = var[:, 0]

    def forward(self, x, instances):
        x = self.layers(x)

        if self.training:
            results = self.loss(x, instances)
        else:
            self.inference(x, instances)
            results = instances

        return results

    def layers(self, x):
        for i, layer in enumerate(self):
            x = layer(x)

        n, _, h, w = x.shape

        x = x.reshape(n, -1, 2, h, w)

        return x


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_MASK_HEAD_REGISTRY.register()
class BayesianMaskRCNNConvUpsampleHead(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, scale, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []
        self.bayesian = True

        cur_channels = input_shape.channels[0]
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_sigma = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.predictor_sigma.weight, math.log(math.e-1))
        nn.init.constant_(self.predictor_sigma.bias, 6)

        self.scale = scale

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def set_freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.predictor.parameters():
            param.requires_grad = True

        for param in self.predictor_sigma.parameters():
            param.requires_grad = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )

        ret["scale"] = cfg.MODEL.ROI_HEADS.COSINE_SCALE

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def loss(self, pred_mask_logits, instances):
        # print(self.var.min().item(), self.var.max().item())
        loss_dict = mask_rcnn_loss(x, instances, feat_triplet, self.vis_period,  memory_loss_func=self.memory_bank, loss_type_camo=self.loss_type_camo, triplet_margin=self.triplet_margin, alpha_triplet=self.alpha_triplet, beta_memory=self.beta_memory)
        loss_dict.update({'loss_mask_reg': self.var.mean()})
        return loss_dict
        #return {"loss_mask": mask_rcnn_loss(pred_mask_logits, instances, self.vis_period), 'loss_mask_reg': self.var.mean()}

    def inference(self, pred_mask_logits, pred_instances):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits[:, 0:1].sigmoid()
            mask_probs_var = pred_mask_logits[:, 1:2]
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            mask_pred = pred_mask_logits[indices, class_pred]
            mask_probs_pred = mask_pred[:, 0:].sigmoid()
            mask_probs_var = mask_pred[:, 1:2]

        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
        mask_probs_var = mask_probs_var.split(num_boxes_per_image, dim=0)

        for prob, var, instances in zip(mask_probs_pred, mask_probs_var, pred_instances):
            instances.pred_masks = prob  # (1, Hmask, Wmask)
            instances.pred_uncertainty = var[:, 0]

    def forward(self, x, instances):
        x = self.layers(x)

        if self.training:
            results = self.loss(x, instances)
        else:
            self.inference(x, instances)
            results = instances

        return results

    def layers(self, x):

        for layer in self.conv_norm_relus:
            x = layer(x)

        x = self.deconv(x)
        x = self.deconv_relu(x)

        self.var = F.softplus(self.predictor_sigma.weight).unsqueeze(0)
        weight = self.predictor.weight.unsqueeze(0)
        x = x.unsqueeze(1)

        # some constraints on x, weight to be non-negative to form parts

        m = (x * weight).sum(2) # E[a]: predicted map
        v = (x * x * self.var).sum(2) # V[a]: uncertainty map
        k = (1 + math.pi * v / 8).pow(-1/2)

        out = k * m * self.scale

        if not self.training:
            out = torch.stack([out, k], 2)

        return out


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)