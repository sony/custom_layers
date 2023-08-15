# -----------------------------------------------------------------------------
# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Sequence, Tuple
import dataclasses

import tensorflow as tf
import numpy as np

from custom_layers.tf.object_detection import FasterRCNNBoxDecode, ScoreConverter
from custom_layers.tf.custom_objects import register_layer


@dataclasses.dataclass
class SSDPostProcessCfg:
    anchors: np.ndarray | tf.Tensor
    scale_factors: Sequence[float | int]
    img_size: Sequence[float | int]
    score_converter: ScoreConverter | str
    score_threshold: float
    iou_threshold: float
    max_detections: float
    remove_background: bool

    def as_dict(self):
        return dataclasses.asdict(self)


@register_layer
class SSDPostProcess(tf.keras.layers.Layer):

    def __init__(self,
                 anchors: np.ndarray | tf.Tensor,
                 scale_factors: Sequence[int | float],
                 img_size: Sequence[int | float],
                 score_converter: ScoreConverter | str,
                 score_threshold: float,
                 iou_threshold: float,
                 max_detections: int,
                 remove_background: bool = False,
                 **kwargs):
        """
        SSD Post Processing

        Args:
            anchors: anchors of shape (n_boxes, 4) in corners coordinates (y_min, x_min, y_max, x_max).
            scale_factors: box decoding scaling factors in format (y, x, height, width).
            img_size: image size for clipping of decoded boxes in format (height, width).
                      Decoded boxes coordinates will be clipped into the range y=[0, height], x=[0, width].
            score_converter: conversion to apply on input logits (sigmoid, softmax or linear).
            score_threshold: score threshold for non-maximum suppression.
            iou_threshold: intersection over union threshold for non-maximum suppression.
            max_detections: number of detections to return.
            remove_background: if True, first class is sliced out from inputs scores (after score_converter is applied)

        """
        super().__init__(**kwargs)
        self._cfg = SSDPostProcessCfg(anchors=anchors,
                                      scale_factors=scale_factors,
                                      img_size=img_size,
                                      score_converter=score_converter,
                                      score_threshold=score_threshold,
                                      iou_threshold=iou_threshold,
                                      max_detections=max_detections,
                                      remove_background=remove_background)
        self._box_decode = FasterRCNNBoxDecode(anchors, scale_factors, (0, 0, *img_size))

    def call(self, inputs: Sequence[tf.Tensor], *args, **kwargs) -> Tuple[tf.Tensor]:
        """
        Args:
            inputs: a list/tuple of (rel_codes, scores)
                    0: encoded offsets of shape (batch, n_boxes, 4) in centroids coordinates (y_center, x_center, w, h)
                    1: scores/logits of shape (batch, n_boxes, n_labels)

        Returns:
            0: selected boxes sorted by scores in decreasing order, of shape (batch, max_detections, 4),
                in corners coordinates (y_min, x_min, y_max, x_max)
            1: scores corresponding to the selected boxes, of shape (batch, max_detection)
            2: labels corresponding to the selected boxes, of shape (batch, max_detections)
            3: the number of valid detections out of max_detections

        Raises:
            ValueError if receives input tensors with unexpected or non-matching shapes
        """

        rel_codes, scores = inputs
        if len(rel_codes.shape) != 3 and rel_codes.shape[-1] != 4:
            raise ValueError(f'Invalid input codes shape {rel_codes.shape}. Expected shape (batch, n_boxes, 4).')
        if len(scores.shape) != 3:
            raise ValueError(f'Invalid input scores shape {scores.shape}. Expected shape (batch, n_boxes, n_labels).')
        if rel_codes.shape[-2] != scores.shape[-2]:
            raise ValueError(f'Mismatch in the number of boxes between input codes ({rel_codes.shape[-2]}) '
                             f'and input scores ({scores.shape[-2]}).')

        if self._cfg.score_converter != ScoreConverter.LINEAR:
            scores = tf.keras.layers.Activation(self._cfg.score_converter)(scores)

        if self._cfg.remove_background:
            scores = tf.slice(scores, begin=[0, 0, 1], size=[-1, -1, -1])

        decoded_boxes = self._box_decode(rel_codes)
        # when decoded_boxes.shape[-2]==1, nms uses same boxes for all classes
        decoded_boxes = tf.expand_dims(decoded_boxes, axis=-2)

        outputs = tf.image.combined_non_max_suppression(decoded_boxes,
                                                        scores,
                                                        max_output_size_per_class=self._cfg.max_detections,
                                                        max_total_size=self._cfg.max_detections,
                                                        iou_threshold=self._cfg.iou_threshold,
                                                        score_threshold=self._cfg.score_threshold,
                                                        pad_per_class=False,
                                                        clip_boxes=False)
        return outputs

    def get_config(self) -> dict:
        return self._cfg.as_dict()
