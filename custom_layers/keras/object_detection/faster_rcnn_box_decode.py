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

from typing import Sequence, Union

import tensorflow as tf
import numpy as np

from custom_layers.keras.object_detection.box_utils import corners_to_centroids, centroids_to_corners
from custom_layers.keras.custom_objects import register_layer


@register_layer
class FasterRCNNBoxDecode(tf.keras.layers.Layer):

    def __init__(self, anchors: Union[np.ndarray, tf.Tensor], scale_factors: Sequence[Union[float, int]],
                 clip_window: Sequence[Union[float, int]], **kwargs):
        """
        Box decoding per Faster R-CNN with clipping

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

    def call(self, rel_codes: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Args:
            rel_codes: encoded offsets of shape (batch, n_boxes, 4) in centroids coordinates (y_center, x_center, h, w)

        Returns:
            decoded boxes of shape (batch, n_boxes, 4) in corners coordinates (y_min, x_min, y_max, x_max)

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
