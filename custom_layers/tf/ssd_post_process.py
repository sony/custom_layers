# -------------------------------------------------------------------------------
# (c) Copyright 2023 Sony Semiconductor Israel, Ltd. All rights reserved.
#
#      This software, in source or object form (the "Software"), is the
#      property of Sony Semiconductor Israel Ltd. (the "Company") and/or its
#      licensors, which have all right, title and interest therein, You
#      may use the Software only in accordance with the terms of written
#      license agreement between you and the Company (the "License").
#      Except as expressly stated in the License, the Company grants no
#      licenses by implication, estoppel, or otherwise. If you are not
#      aware of or do not agree to the License terms, you may not use,
#      copy or modify the Software. You may use the source code of the
#      Software only for your internal purposes and may not distribute the
#      source code of the Software, any part thereof, or any derivative work
#      thereof, to any third party, except pursuant to the Company's prior
#      written consent.
#      The Software is the confidential information of the Company.
# -------------------------------------------------------------------------------
"""
Created on 7/30/23

@author: irenab
"""
from typing import Sequence, Tuple
import dataclasses

import tensorflow as tf
import numpy as np

from . import BoxDecode, ScoreConverter
from .custom_objects import register_layer


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
        self._box_decode = BoxDecode(anchors, scale_factors, (0, 0, *img_size))

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
