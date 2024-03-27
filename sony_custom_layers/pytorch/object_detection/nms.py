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
from typing import Tuple, NamedTuple, Union

import numpy as np
import torch
from torch import Tensor
import torchvision    # noqa: F401 # needed for torch.ops.torchvision

MULTICLASS_NMS_TORCH_OP = 'sony::multiclass_nms'

__all__ = ['multiclass_nms', 'NMSResults']


class NMSResults(NamedTuple):
    boxes: Tensor
    scores: Tensor
    labels: Tensor
    n_valid: Tensor


def multiclass_nms(boxes, scores, score_threshold: float, iou_threshold: float, max_detections: int) -> NMSResults:
    """
    Multi-class non-maximum suppression.
    Detections are returned in descending order of their scores.
    The output tensors always contain a fixed number of detections, as defined by 'max_detections'.
    If fewer detections are selected, the output tensors are zero-padded up to 'max_detections'.

    Args:
        boxes (Tensor): Input boxes with shape [batch, n_boxes, 4], specified in corner coordinates
                        (x_min, y_min, x_max, y_max). Agnostic to the x-y axes order.
        scores (Tensor): Input scores with shape [batch, n_boxes, n_classes].
        score_threshold (float): The score threshold. Candidates with scores below the threshold are discarded.
        iou_threshold (float): The Intersection Over Union (IOU) threshold for boxes overlap.
        max_detections (int): The number of detections to return.

    Returns:
        'NMSResults' named tuple:
            boxes (Tensor): The selected boxes with shape [batch, max_detections, 4].
            scores (Tensor): The corresponding scores in descending order with shape [batch, max_detections].
            labels (Tensor): The labels for each box with shape [batch, max_detections].
            n_valid (Tensor): The number of valid detections out of 'max_detections' with shape [batch, 1]
    """
    return NMSResults(*torch.ops.sony.multiclass_nms(boxes, scores, score_threshold, iou_threshold, max_detections))


torch.library.define(
    MULTICLASS_NMS_TORCH_OP,
    "(Tensor boxes, Tensor scores, float score_threshold, float iou_threshold, SymInt max_detections) -> "
    "(Tensor, Tensor, Tensor, Tensor)")


@torch.library.impl(MULTICLASS_NMS_TORCH_OP, 'default')
def _multiclass_nms_op(boxes: torch.Tensor, scores: torch.Tensor, score_threshold: float, iou_threshold: float,
                       max_detections: int) -> NMSResults:
    """ Registers the torch op as torch.ops.sony.multiclass_nms """
    return _multiclass_nms_impl(boxes,
                                scores,
                                score_threshold=score_threshold,
                                iou_threshold=iou_threshold,
                                max_detections=max_detections)


@torch.library.impl_abstract(MULTICLASS_NMS_TORCH_OP)
def _multiclass_nms_meta(boxes: torch.Tensor, scores: torch.Tensor, score_threshold: float, iou_threshold: float,
                         max_detections: int) -> NMSResults:
    """ Registers torch op's abstract implementation. It specifies the properties of the output tensors.
        Needed for torch.export """
    ctx = torch.library.get_ctx()
    batch = ctx.new_dynamic_size()
    return NMSResults(
        torch.empty((batch, max_detections, 4)),
        torch.empty((batch, max_detections)),
        torch.empty((batch, max_detections), dtype=torch.int64),
        torch.empty((batch, 1), dtype=torch.int64)
    )    # yapf: disable


def _multiclass_nms_impl(boxes: Union[Tensor, np.ndarray], scores: Union[Tensor, np.ndarray], score_threshold: float,
                         iou_threshold: float, max_detections: int) -> NMSResults:
    """ See multiclass_nms """
    # this is needed for onnxruntime implementation
    if not isinstance(boxes, Tensor):
        boxes = Tensor(boxes)
    if not isinstance(scores, Tensor):
        scores = Tensor(scores)

    if not 0 <= score_threshold <= 1:
        raise ValueError(f'Invalid score_threshold {score_threshold} not in range [0, 1]')
    if not 0 <= iou_threshold <= 1:
        raise ValueError(f'Invalid iou_threshold {iou_threshold} not in range [0, 1]')
    if max_detections <= 0:
        raise ValueError(f'Invalid non-positive max_detections {max_detections}')

    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f'Invalid input boxes shape {boxes.shape}. Expected shape (batch, n_boxes, 4).')
    if scores.ndim != 3:
        raise ValueError(f'Invalid input scores shape {scores.shape}. Expected shape (batch, n_boxes, n_classes).')
    if boxes.shape[-2] != scores.shape[-2]:
        raise ValueError(f'Mismatch in the number of boxes between input boxes ({boxes.shape[-2]}) '
                         f'and scores ({scores.shape[-2]})')

    batch = boxes.shape[0]
    res = torch.zeros((batch, max_detections, 6), device=boxes.device)
    valid_dets = torch.zeros((batch, 1), device=boxes.device)
    for i in range(batch):
        res[i], valid_dets[i] = _image_multiclass_nms(boxes[i],
                                                      scores[i],
                                                      score_threshold=score_threshold,
                                                      iou_threshold=iou_threshold,
                                                      max_detections=max_detections)

    return NMSResults(boxes=res[..., :4],
                      scores=res[..., 4],
                      labels=res[..., 5].to(torch.int64),
                      n_valid=valid_dets.to(torch.int64))


def _image_multiclass_nms(boxes: Tensor, scores: Tensor, score_threshold: float, iou_threshold: float,
                          max_detections: int) -> Tuple[Tensor, int]:
    """
    Performs multi-class non-maximum suppression on a single image
    Args:
        boxes: input boxes of shape [n_boxes, 4]
        scores: input scores of shape [n_boxes, n_classes]
        score_threshold: score threshold
        iou_threshold: intersection over union threshold
        max_detections: fixed number of detections to return

    Returns:
        A tensor of shape [max_detections, 6] and the number of valid detections.
        out[:, :4] contains the selected boxes
        out[:, 4] and out[:, 5] contain the scores and labels for the selected boxes

    """
    x = _convert_inputs(boxes, scores, score_threshold)
    idxs = _nms_with_class_offsets(x, iou_threshold=iou_threshold)
    idxs = idxs[:max_detections]
    valid_dets = idxs.numel()
    out = torch.zeros(max_detections, 6, device=boxes.device)
    out[:valid_dets] = x[idxs]
    return out, valid_dets


def _convert_inputs(boxes: Tensor, scores: Tensor, score_threshold: float) -> Tensor:
    """
    Converts inputs and filters out boxes with score below the threshold.
    Args:
        boxes: input boxes of shape [n_boxes, 4]
        scores: input scores of shape [n_boxes, n_classes]
        score_threshold: score threshold for nms candidates

    Returns:
        A tensor of shape [m, 6] containing m nms candidates above the score threshold.
        x[:, :4] contains the boxes with replication for different labels
        x[:, 4] contains the scores
        x[:, 5] contains the labels indices (label i corresponds to input scores[:, i])
    """
    n_boxes, n_classes = scores.shape
    scores_mask = scores > score_threshold
    box_indices = torch.arange(n_boxes, device=boxes.device).unsqueeze(1).expand(-1, n_classes)[scores_mask]
    x = torch.empty((box_indices.numel(), 6), device=boxes.device)
    x[:, :4] = boxes[box_indices]
    x[:, 4] = scores[scores_mask]
    x[:, 5] = torch.arange(n_classes, device=boxes.device).unsqueeze(0).expand(n_boxes, -1)[scores_mask]
    return x


def _nms_with_class_offsets(x: Tensor, iou_threshold: float) -> Tensor:
    """
    Args:
        x: nms candidates of shape [n, 6] ([:,:4] boxes, [:, 4] scores, [:, 5] labels)
        iou_threshold: intersection over union threshold

    Returns:
        Indices of the selected candidates
    """
    # shift boxes of each class to prevent intersection between boxes of different classes, and use single-class nms
    # (similar to torchvision batched_nms trick)
    offsets = x[:, 5:] * (x[:, :4].max() + 1)
    shifted_boxes = x[:, :4] + offsets
    return torch.ops.torchvision.nms(shifted_boxes, x[:, 4], iou_threshold)
