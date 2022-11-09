import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)

# from ..builder import HEADS, build_head, build_roi_extractor
from mmrotate.core import rbbox2roi
from mmrotate.core import build_assigner, build_sampler, obb2xyxy, rbbox2result
from ..builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)
from .rotate_standard_roi_head import RotatedStandardRoIHead


@ROTATED_HEADS.register_module()
class CasacadeRoiTransRoi(RotatedStandardRoIHead):

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                #  mask_roi_extractor=None,
                #  mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CasacadeRoiTransRoi, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            # mask_roi_extractor=mask_roi_extractor,
            # mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

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
                if(i>0):
                    rois=rbbox2roi([proposals])
                bbox_results = self._bbox_forward(i, x, rois)
                proposals = torch.randn(1000, 6).to(proposals.device)                
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # # mask heads
        # if self.with_mask:
        #     mask_rois = rois[:100]
        #     for i in range(self.num_stages):
        #         mask_results = self._mask_forward(i, x, mask_rois)
        #         outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        if stage==0:
            rois = bbox2roi([res.bboxes for res in sampling_results])
        else:
            rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
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
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    if i == 0:
                        gt_tmp_bboxes = obb2xyxy(gt_bboxes[j], 'le90')
                    else:
                        gt_tmp_bboxes = gt_bboxes[j]
                    #gt_hbboxes = obb2xyxy(gt_bboxes[j], self.version)
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_tmp_bboxes, gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_tmp_bboxes,
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    if gt_bboxes[j].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_bboxes[j].new((0, gt_bboxes[0].size(-1))).zero_()
                    else:
                        sampling_result.pos_gt_bboxes = gt_bboxes[j][sampling_result.pos_assigned_gt_inds, :]
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # # mask head forward and loss
            # if self.with_mask:
            #     mask_results = self._mask_forward_train(
            #         i, x, sampling_results, gt_masks, rcnn_train_cfg,
            #         bbox_results['bbox_feats'])
            #     for name, value in mask_results['loss_mask'].items():
            #         losses[f's{i}.{name}'] = (
            #             value * lw if 'loss' in name else value)

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

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    if i==0:
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                    else:
                        proposal_list = self.bbox_head[i].refine_bboxes1(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)                        

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        # if rois.shape[0] == 0:
        #     # There is no proposal in the whole batch
        #     bbox_results = [[
        #         np.zeros((0, 5), dtype=np.float32)
        #         for _ in range(self.bbox_head[-1].num_classes)
        #     ]] * num_imgs

        #     if self.with_mask:
        #         mask_classes = self.mask_head[-1].num_classes
        #         segm_results = [[[] for _ in range(mask_classes)]
        #                         for _ in range(num_imgs)]
        #         results = list(zip(bbox_results, segm_results))
        #     else:
        #         results = bbox_results

        #     return results

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
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        if i==0:
                            refined_rois = self.bbox_head[i].regress_by_class(
                                rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        else:
                            refined_rois = self.bbox_head[i].regress_by_class1(
                                rois[j], bbox_label, bbox_pred[j], img_metas[j])                            
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

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

        # if self.with_mask:
        #     if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
        #         mask_classes = self.mask_head[-1].num_classes
        #         segm_results = [[[] for _ in range(mask_classes)]
        #                         for _ in range(num_imgs)]
        #     else:
        #         if rescale and not isinstance(scale_factors[0], float):
        #             scale_factors = [
        #                 torch.from_numpy(scale_factor).to(det_bboxes[0].device)
        #                 for scale_factor in scale_factors
        #             ]
        #         _bboxes = [
        #             det_bboxes[i][:, :4] *
        #             scale_factors[i] if rescale else det_bboxes[i][:, :4]
        #             for i in range(len(det_bboxes))
        #         ]
        #         mask_rois = bbox2roi(_bboxes)
        #         num_mask_rois_per_img = tuple(
        #             _bbox.size(0) for _bbox in _bboxes)
        #         aug_masks = []
        #         for i in range(self.num_stages):
        #             mask_results = self._mask_forward(i, x, mask_rois)
        #             mask_pred = mask_results['mask_pred']
        #             # split batch mask prediction back to each image
        #             mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
        #             aug_masks.append([
        #                 m.sigmoid().cpu().detach().numpy() for m in mask_pred
        #             ])

        #         # apply mask post-processing to each image individually
        #         segm_results = []
        #         for i in range(num_imgs):
        #             if det_bboxes[i].shape[0] == 0:
        #                 segm_results.append(
        #                     [[]
        #                      for _ in range(self.mask_head[-1].num_classes)])
        #             else:
        #                 aug_mask = [mask[i] for mask in aug_masks]
        #                 merged_masks = merge_aug_masks(
        #                     aug_mask, [[img_metas[i]]] * self.num_stages,
        #                     rcnn_test_cfg)
        #                 segm_result = self.mask_head[-1].get_seg_masks(
        #                     merged_masks, _bboxes[i], det_labels[i],
        #                     rcnn_test_cfg, ori_shapes[i], scale_factors[i],
        #                     rescale)
        #                 segm_results.append(segm_result)
        #     ms_segm_result['ensemble'] = segm_results

        # if self.with_mask:
        #     results = list(
        #         zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        # else:
        results = ms_bbox_result['ensemble']

        return results