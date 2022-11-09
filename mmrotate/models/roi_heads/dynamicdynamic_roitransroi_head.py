from abc import ABCMeta

import torch
from mmcv.runner import BaseModule, ModuleList
from mmdet.core import bbox2roi

from mmrotate.core import (build_assigner, build_sampler, obb2xyxy,
                           rbbox2result, rbbox2roi)
from ..builder import ROTATED_HEADS, build_head, build_roi_extractor
import numpy as np

EPS = 1e-15

@ROTATED_HEADS.register_module()
class DynamicRoiTransRoiDynamic(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 version='oc',
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        super(DynamicRoiTransRoiDynamic, self).__init__(init_cfg)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history0 = []
        self.iou_history1 = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history0 = []
        self.beta_history1 = []


        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.version = version

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if self.bbox_head is not None else False

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_assigner_sampler(self):
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                if i > 0:
                    rois = rbbox2roi([proposals])
                bbox_results = self._bbox_forward(i, x, rois)
                proposals = torch.randn(1000, 6).to(proposals.device)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg,img_metas):
        num_imgs = len(img_metas)
        if stage == 0:
            rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        if True:
            pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
            num_pos = len(pos_inds)
            cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
            beta_topk = min(rcnn_train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                            num_pos)
            cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
            if stage==0:
                self.beta_history0.append(cur_target)    
            else: 
                self.beta_history1.append(cur_target)    
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                cur_iou = []
                for j in range(num_imgs):
                    if i == 0:
                        gt_tmp_bboxes = obb2xyxy(gt_bboxes[j], self.version)
                    else:
                        gt_tmp_bboxes = gt_bboxes[j]
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_tmp_bboxes, gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_tmp_bboxes,
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    if True:
                        # record the `iou_topk`-th largest IoU in an image
                        iou_topk = min(rcnn_train_cfg.dynamic_rcnn.iou_topk,
                                    len(assign_result.max_overlaps))
                        ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                        cur_iou.append(ious[-1].item())

                    if gt_bboxes[j].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_bboxes[j].new(
                            (0, gt_bboxes[0].size(-1))).zero_()
                    else:
                        sampling_result.pos_gt_bboxes = \
                            gt_bboxes[j][
                                sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)
                if i==0:
                    cur_iou = np.mean(cur_iou)
                    self.iou_history0.append(cur_iou)
                else:
                    cur_iou = np.mean(cur_iou)
                    self.iou_history1.append(cur_iou)
            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg,img_metas)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)
            if i==1:
                # update IoU threshold and SmoothL1 beta
                update_iter_interval = rcnn_train_cfg.dynamic_rcnn.update_iter_interval
                if len(self.iou_history1) % update_iter_interval == 0:
                    new_iou_thr, new_beta = self.update_hyperparameters1()
                    print("Stage 1: ",new_iou_thr, new_beta)
            else:
                # update IoU threshold and SmoothL1 beta
                update_iter_interval = rcnn_train_cfg.dynamic_rcnn.update_iter_interval
                if len(self.iou_history0) % update_iter_interval == 0:
                    new_iou_thr, new_beta = self.update_hyperparameters0()
                    print("Stage 0: ",new_iou_thr, new_beta)
        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
                                                       bbox_pred[j],
                                                       img_metas[j])
                    for j in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results
        results = ms_bbox_result['ensemble']

        return results

    def update_hyperparameters1(self):
        new_iou_thr = max(self.train_cfg[1].dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history1))
        self.iou_history1 = []
        self.bbox_assigner[1].pos_iou_thr = new_iou_thr
        self.bbox_assigner[1].neg_iou_thr = new_iou_thr
        self.bbox_assigner[1].min_pos_iou = new_iou_thr
        if (np.median(self.beta_history1) < EPS):
            # avoid 0 or too small value for new_beta
            new_beta = self.bbox_head[1].loss_bbox.beta
        else:
            new_beta = min(self.train_cfg[1].dynamic_rcnn.initial_beta,
                           np.median(self.beta_history1))
        self.beta_history1 = []
        self.bbox_head[1].loss_bbox.beta = new_beta
        return new_iou_thr, new_beta

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError

    def update_hyperparameters0(self):
        new_iou_thr = max(self.train_cfg[0].dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history0))
        self.iou_history0 = []
        self.bbox_assigner[0].pos_iou_thr = new_iou_thr
        self.bbox_assigner[0].neg_iou_thr = new_iou_thr
        self.bbox_assigner[0].min_pos_iou = new_iou_thr
        if (np.median(self.beta_history0) < EPS):
            # avoid 0 or too small value for new_beta
            new_beta = self.bbox_head[0].loss_bbox.beta
        else:
            new_beta = min(self.train_cfg[0].dynamic_rcnn.initial_beta,
                           np.median(self.beta_history0))
        self.beta_history0 = []
        self.bbox_head[0].loss_bbox.beta = new_beta
        return new_iou_thr, new_beta
