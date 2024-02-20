# -----------------------------------------------------------------------------
# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
from typing import Tuple, NamedTuple

import torch
import torchvision.ops
from torch import Tensor


class NMSResults(NamedTuple):
    boxes: Tensor
    scores: Tensor
    labels: Tensor
    valid_detections: Tensor


class MultiClassNMS(torch.nn.Module):
    """
        Multi-class non-maximum suppression.
        Detections are returned in a decreasing scores order.
        Output tensors always contain a fixed number of detections (defined by max_detections).
        If less detections are selected, output tensors are zero-padded to max_detections.
    """

    def __init__(self, score_threshold: float, iou_threshold: float, max_detections: int):
        """
        Args:
            score_threshold: score threshold for boxes selection
            iou_threshold: intersection over union threshold
            max_detections: the number of detections to return
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, boxes: Tensor, scores: Tensor):
        """
       Args:
            boxes: input boxes of shape [batch, n_boxes, 4] in corner coordinates (y_min, x_min, y_max, x_max)
            scores: input scores of shape [batch, n_boxes, n_classes]

        Returns:
            NMSResults containing:
                boxes: boxes of shape [batch, max_detections, 4]
                scores: scores in decreasing order of shape [batch, max_detections]
                labels: labels of shape [batch, max_detections]
                valid_detections: number of valid detections out of max_detections
        """
        return torch.ops.sony.multiclass_nms(boxes, scores, self.score_threshold, self.iou_threshold,
                                             self.max_detections)


torch.library.define(
    'sony::multiclass_nms',
    "(Tensor boxes, Tensor scores, float score_threshold, float iou_threshold, int max_detections) -> "
    "(Tensor, Tensor, Tensor, Tensor)")


@torch.library.impl('sony::multiclass_nms', 'default')
def multiclass_nms_op(boxes: torch.Tensor, scores: torch.Tensor, score_threshold: float, iou_threshold: float,
                      max_detections: int) -> NMSResults:
    return multiclass_nms_impl(boxes,
                               scores,
                               score_threshold=score_threshold,
                               iou_threshold=iou_threshold,
                               max_detections=max_detections)


def multiclass_nms_impl(boxes: Tensor, scores: Tensor, score_threshold: float, iou_threshold: float,
                        max_detections: int) -> NMSResults:
    """
    Multi-class non-maximum suppression.
    Detections are returned in a decreasing scores order.
    Output tensors always contain a fixed number of detections (defined by max_detections).
    If less detections are selected, output tensors are zero-padded to max_detections.

    Args:
        boxes: input boxes of shape [batch, n_boxes, 4] in corner coordinates (y_min, x_min, y_max, x_max)
        scores: input scores of shape [batch, n_boxes, n_classes]
        score_threshold: score threshold for boxes selection
        iou_threshold: intersection over union threshold
        max_detections: the number of detections to return

    Returns:
        NMSResults containing:
        boxes: boxes of shape [batch, max_detections, 4]
        scores: scores in decreasing order of shape [batch, max_detections]
        labels: labels of shape [batch, max_detections]
        valid_detections: number of valid detections out of max_detections
    """

    _validate_inputs(boxes, scores, batch=True)
    batch = boxes.shape[0]
    res = [
        _image_multiclass_nms(boxes[i],
                              scores[i],
                              score_threshold=score_threshold,
                              iou_threshold=iou_threshold,
                              max_detections=max_detections) for i in range(batch)
    ]
    out_boxes = torch.stack([r.boxes for r in res], dim=0)
    out_scores = torch.stack([r.scores for r in res], dim=0)
    out_labels = torch.stack([r.labels for r in res], dim=0)
    out_valid_dets = torch.stack([r.valid_detections for r in res], dim=0)
    return NMSResults(boxes=out_boxes, scores=out_scores, labels=out_labels, valid_detections=out_valid_dets)


def _image_multiclass_nms(boxes: Tensor, scores: Tensor, score_threshold: float, iou_threshold: float,
                          max_detections: int) -> NMSResults:
    """
    Performs multi-class non-maximum suppression on a single image
    Same interface as multiclass_nms, only without the batch dimension.
    """
    flat_boxes, flat_scores, labels = _flatten_image_inputs(boxes, scores)
    score_mask = flat_scores >= score_threshold
    flat_boxes = flat_boxes[score_mask]
    flat_scores = flat_scores[score_mask]
    labels = labels[score_mask]

    idxs = torchvision.ops.batched_nms(flat_boxes, flat_scores, labels, iou_threshold=iou_threshold)

    idxs = idxs[:max_detections]
    valid_dets = idxs.numel()

    out_boxes = torch.zeros(max_detections, 4)
    out_boxes[:valid_dets] = flat_boxes[idxs]

    out_scores = torch.zeros(max_detections)
    out_scores[:valid_dets] = flat_scores[idxs]

    out_labels = torch.zeros(max_detections)
    out_labels[:valid_dets] = labels[idxs]

    return NMSResults(boxes=out_boxes,
                      scores=out_scores,
                      labels=out_labels,
                      valid_detections=Tensor([valid_dets]).to(torch.int64))


def _flatten_image_inputs(boxes: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converts MultiClassNMS inputs for a single image into inputs expected by torchvision.ops.batched_nms.
    Args:
        boxes: tensor of shape [n_boxes, 4]
        scores: tensor of shape [n_boxes, n_classes]

    Returns:
        boxes: flattened boxes vector of size n_boxes * n_classes
        scores: flattened scores vector of size n_boxes * n_classes
        labels: labels indices vector of size n_boxes * n_classes. Label i corresponds to scores[:, i]
    """
    _validate_inputs(boxes, scores, batch=False)
    n_boxes, n_classes = scores.shape
    labels = torch.arange(end=n_classes).expand(n_boxes, n_classes).flatten()
    flat_boxes = boxes.view(n_boxes, 1, 4).expand(n_boxes, n_classes, 4).flatten(end_dim=1)
    flat_scores = scores.flatten()
    return flat_boxes, flat_scores, labels


def _validate_inputs(boxes: Tensor, scores: Tensor, batch: bool):
    """
    Validates input boxes and scores
    Args:
        boxes: expected shape [batch, n_boxes, 4] if batch is True or [n_boxes, 4] otherwise
        scores: expected shape [batch, n_boxes, n_classes] if batch is True or [n_boxes, n_classes] otherwise
        batch: whether the inputs are expected to contain the batch dims

    Raises:
        ValueError
    """
    exp_ndims = 2 + int(batch)
    if boxes.ndim != exp_ndims or boxes.shape[-1] != 4:
        raise ValueError(f'Invalid input boxes shape {boxes.shape}. '
                         f'Expected shape ({"batch, " if batch else ""}n_boxes, 4).')
    if scores.ndim != exp_ndims:
        raise ValueError(f'Invalid input scores shape {scores.shape}. '
                         f'Expected shape ({"batch, " if batch else ""}n_boxes, n_classes).')
    if boxes.shape[-2] != scores.shape[-2]:
        raise ValueError(f'Mismatch in the number of boxes between input boxes ({boxes.shape[-2]}) '
                         f'and scores ({scores.shape[-2]})')
