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

from typing import Sequence, Tuple, Union, List
import dataclasses

import tensorflow as tf
import numpy as np

from sony_custom_layers.keras.base_custom_layer import CustomLayer
from sony_custom_layers.keras.object_detection import FasterRCNNBoxDecode, ScoreConverter
from sony_custom_layers.keras.custom_objects import register_layer


@dataclasses.dataclass
class SSDPostProcessCfg:
    anchors: Union[np.ndarray, tf.Tensor, List[List[float]]]
    scale_factors: Sequence[Union[float, int]]
    clip_size: Sequence[Union[float, int]]
    score_converter: Union[ScoreConverter, str]
    score_threshold: float
    iou_threshold: float
    max_detections: float
    remove_background: bool

    def as_dict(self):
        return dataclasses.asdict(self)


@register_layer
class SSDPostProcess(CustomLayer):

    def __init__(self,
                 anchors: Union[np.ndarray, tf.Tensor, List[List[float]]],
                 scale_factors: Sequence[Union[int, float]],
                 clip_size: Sequence[Union[int, float]],
                 score_converter: Union[ScoreConverter, str],
                 score_threshold: float,
                 iou_threshold: float,
                 max_detections: int,
                 remove_background: bool = False,
                 **kwargs):
        """
        SSD Post Processing, based on https://arxiv.org/abs/1512.02325.

        Args:
            anchors: Anchors with a shape of (n_boxes, 4) in corner coordinates (y_min, x_min, y_max, x_max).
            scale_factors: Box decoding scaling factors in the format (y, x, height, width).
            clip_size: Clipping size in the format (height, width). The decoded boxes are clipped to the
                       range y=[0, height] and x=[0, width]. Typically, the clipping size is (1, 1) for normalized boxes
                       and the image size for boxes in pixel coordinates.
            score_converter: Conversion to apply to the input logits (sigmoid, softmax, or linear).
            score_threshold: Score threshold for non-maximum suppression.
            iou_threshold: Intersection over union threshold for non-maximum suppression.
            max_detections: The number of detections to return.
            remove_background: If True, the first class is removed from the input scores (after the score_converter is
                               applied).
        """
        super().__init__(**kwargs)
        self.cfg = SSDPostProcessCfg(anchors=anchors,
                                     scale_factors=scale_factors,
                                     clip_size=clip_size,
                                     score_converter=score_converter,
                                     score_threshold=score_threshold,
                                     iou_threshold=iou_threshold,
                                     max_detections=max_detections,
                                     remove_background=remove_background)
        self._box_decode = FasterRCNNBoxDecode(anchors, scale_factors, (0, 0, *clip_size))

    def call(self, inputs: Sequence[tf.Tensor], *args, **kwargs) -> Tuple[tf.Tensor]:
        """
        Args:
            inputs: A list or tuple consisting of (rel_codes, scores).
              0: Relative codes (encoded offsets) with a shape of (batch, n_boxes, 4) in centroid coordinates
                 (y_center, x_center, w, h).
              1: Scores or logits with a shape of (batch, n_boxes, n_labels).

        Returns:
            0: Selected boxes sorted by scores in descending order, with a shape of (batch, max_detections, 4),
               in corner coordinates (y_min, x_min, y_max, x_max).
            1: Scores corresponding to the selected boxes, with a shape of (batch, max_detections).
            2: Labels corresponding to the selected boxes, with a shape of (batch, max_detections).
               Each label corresponds to the class index of the selected score in the input scores.
            3: The number of valid detections out of max_detections.

        Raises:
            ValueError: If provided input tensors have unexpected or non-matching shapes.
        """

        rel_codes, scores = inputs
        if len(rel_codes.shape) != 3 and rel_codes.shape[-1] != 4:
            raise ValueError(f'Invalid input offsets shape {rel_codes.shape}. '
                             f'Expected shape (batch, n_boxes, 4).')
        if len(scores.shape) != 3:
            raise ValueError(f'Invalid input scores shape {scores.shape}. '
                             f'Expected shape (batch, n_boxes, n_labels).')
        if rel_codes.shape[-2] != scores.shape[-2]:
            raise ValueError(f'Mismatch in the number of boxes between input codes ({rel_codes.shape[-2]}) '
                             f'and input scores ({scores.shape[-2]}).')

        if self.cfg.score_converter != ScoreConverter.LINEAR:
            scores = tf.keras.layers.Activation(self.cfg.score_converter)(scores)

        if self.cfg.remove_background:
            scores = tf.slice(scores, begin=[0, 0, 1], size=[-1, -1, -1])

        decoded_boxes = self._box_decode(rel_codes)
        # when decoded_boxes.shape[-2]==1, nms uses same boxes for all classes
        decoded_boxes = tf.expand_dims(decoded_boxes, axis=-2)

        outputs = tf.image.combined_non_max_suppression(decoded_boxes,
                                                        scores,
                                                        max_output_size_per_class=self.cfg.max_detections,
                                                        max_total_size=self.cfg.max_detections,
                                                        iou_threshold=self.cfg.iou_threshold,
                                                        score_threshold=self.cfg.score_threshold,
                                                        pad_per_class=False,
                                                        clip_boxes=False)
        return outputs

    def get_config(self) -> dict:
        config = super().get_config()
        d = self.cfg.as_dict()
        d['anchors'] = tf.constant(d['anchors']).numpy().tolist()
        config.update(d)
        return config
