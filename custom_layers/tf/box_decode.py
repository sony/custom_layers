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
Created on 6/8/23

@author: irenab
"""
from typing import Sequence, Any

import tensorflow as tf
import numpy as np

from custom_layers.tf.box_utils import corners_to_centroids, centroids_to_corners
from custom_layers.tf.custom_objects import register_layer


@register_layer
class BoxDecode(tf.keras.layers.Layer):

    def __init__(self, anchors: np.ndarray, scale_factors: Sequence[float | int], clip_window: Sequence[float | int],
                 **kwargs: Any):
        """
        Box decoding per Faster R-CNN, with clipping to image size.

        Args:
            anchors: anchors of shape (n_boxes, 4) in corners coordinates (y_min, x_min, y_max, x_max)
            scale_factors: scaling factors in format (y, x, height, width)
            clip_window: clipping window in format (y_min, x_min, y_max, x_max)

        Raises:
            ValueError if receives invalid parameters

        """
        super().__init__(**kwargs)
        anchors = tf.constant(anchors)
        if not (len(anchors.shape) == 2 and anchors.shape[-1] == 4):
            raise ValueError(f'Invalid anchors shape {anchors.shape}. Expected shape (n_boxes, 4).')
        self.anchors = anchors

        if len(scale_factors) != 4:
            raise ValueError(f'Invalid scale factors {scale_factors}. Expected 4 values for (y, x, height, width).')
        self.scale_factors = scale_factors

        if len(clip_window) != 4:
            raise ValueError(f'Invalid clip window {clip_window}. Expected 4 values for (y_min, x_min, y_max, x_max).')
        self.clip_window = clip_window

    def call(self, rel_codes: tf.Tensor) -> tf.Tensor:
        """
        Args:
            rel_codes: encoded offsets of shape (batch, n_boxes, 4) in centroids coordinates (y_center, x_center, h, w)

        Returns:
            decoded boxes of shape (batch, n_boxes, 4) in corners coordiantes (y_min, x_min, y_max, x_max)

        Raises:
            ValurError if receives input tensor with unexpected shape
        """
        if len(rel_codes.shape) != 3 or rel_codes.shape[-1] != 4:
            raise ValueError(f'Invalid input tensor shape {rel_codes.shape}. Expected shape (batch, n_boxes, 4).')
        if rel_codes.shape[-2] != self.anchors.shape[-2]:
            raise ValueError(f'Mismatch in the number of boxes between input tensor ({rel_codes.shape[-2]}) '
                             f'and anchors ({self.anchors.shape[-2]})')

        scaled_codes = rel_codes / tf.constant(self.scale_factors, dtype=rel_codes.dtype)

        a_y_min, a_x_min, a_y_max, a_x_max = tf.unstack(self.anchors, axis=-1)
        a_y_center, a_x_center, a_h, a_w = corners_to_centroids(a_y_min, a_x_min, a_y_max, a_x_max)

        box_y_center = scaled_codes[..., 0] * a_h + a_y_center
        box_x_center = scaled_codes[..., 1] * a_w + a_x_center
        box_h = tf.exp(scaled_codes[..., 2]) * a_h
        box_w = tf.exp(scaled_codes[..., 3]) * a_w
        box_y_min, box_x_min, box_y_max, box_x_max = centroids_to_corners(box_y_center, box_x_center, box_h, box_w)
        boxes = tf.stack([box_y_min, box_x_min, box_y_max, box_x_max], axis=-1)

        y_low, x_low, y_high, x_high = self.clip_window
        boxes = tf.clip_by_value(boxes, [y_low, x_low, y_low, x_low], [y_high, x_high, y_high, x_high])
        return boxes

    def get_config(self) -> dict:
        return {
            'anchors': self.anchors.numpy(),
            'scale_factors': self.scale_factors,
            'clip_window': self.clip_window,
        }
