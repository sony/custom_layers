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
Created on 7/19/23

@author: irenab
"""
import numpy as np
import tensorflow as tf

import pytest


class TestBoxDecode:
    @pytest.mark.parametrize('clip_size',  [(100, 200), None])
    def test_load_custom(self, tmp_path, clip_size):
        from custom_layers.tf import custom_objects
        shape = (10, 4)
        path = tmp_path / 'model.h5'
        anchors = np.random.uniform(size=shape).astype(np.float32)
        scale_factors = (1, 2.5, 3., 4)
        self._build_box_decode(anchors, scale_factors, clip_size, path)
        with pytest.raises(ValueError, match='Unknown layer.*BoxDecode'):
            tf.keras.models.load_model(path)

        model = tf.keras.models.load_model(path, custom_objects=custom_objects)

        cfg = model.layers[-1].get_config()
        assert np.array_equal(cfg['anchors'], anchors)
        assert tuple(cfg['scale_factors']) == scale_factors
        if clip_size:
            assert tuple(cfg['clip_size']) == clip_size
        else:
            assert cfg['clip_size'] is None
        # is inferable
        model(np.random.uniform(size=(1, *shape)).astype(np.float32))

    def test_zero_offsets(self):
        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, seed=1)
        model = self._build_box_decode(anchors, (1, 2, 3, 4), (1, 1))
        out = model(np.zeros((2, n_boxes, 4)).astype(np.float32))
        assert np.allclose(out, anchors)

    @pytest.mark.parametrize('scale_factors', [(1., 1., 1., 1.), (1, 2, 3, 4), (1.1, 2.2, 0.5, 3.3)])
    def test_box_decode(self, scale_factors):
        n_boxes = 100
        anchors = self._generate_random_anchors(n_boxes, img_size=(100, 200), seed=1)

        v0, v1, v2, v3 = .5, 1., .2, 1.2
        offsets = np.empty((2, n_boxes, 4)).astype(np.float32)
        # we define encoded offsets that will yield boxes such that:
        # np.log(boxes_h / anchors_h) = v2
        # np.log(boxes_w / anchors_w) = v3
        # (boxes_center_y - anchors_center_y) / anchors_h = v0
        # (boxes_center_x - anchors_center_x) / anchors_w = v1
        offsets[:, :, 0] = v0 * scale_factors[0]
        offsets[:, :, 1] = v1 * scale_factors[1]
        offsets[:, :, 2] = v2 * scale_factors[2]
        offsets[:, :, 3] = v3 * scale_factors[3]

        # disable clipping
        model = self._build_box_decode(anchors, scale_factors, img_size=None)
        boxes = model(offsets)
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        assert np.allclose(boxes_hw[..., 0] / anchors_hw[..., 0], np.exp(v2))
        assert np.allclose(boxes_hw[..., 1] / anchors_hw[..., 1], np.exp(v3))
        boxes_center = boxes[..., :2] + 0.5 * boxes_hw
        anchors_center = anchors[..., :2] + 0.5 * anchors_hw
        t = (boxes_center - anchors_center) / anchors_hw
        assert np.allclose(t[..., 0], v0, atol=1e-5)
        assert np.allclose(t[..., 1], v1, atol=1e-5)

    @pytest.mark.parametrize('img_size, normalize', [((90, 110), False), ((.9, 1.1), True)])
    def test_clipping(self, img_size, normalize):
        scale_factors = (1, 2, 3, 4)
        n_boxes = 3
        anchors = self._generate_random_anchors(n_anchors=n_boxes, seed=1)
        mul = 0.01 if normalize else 1
        # (2, n_boxes, 4)
        boxes = mul * np.array([[[-5, 5, 1, 12],          # clip y_min
                               [85, -4, 90, 2],           # clip x_min
                               [85, 95, 95, 100]],        # clip y_max
                              [[0, 85, 2, 115],           # clip x_max
                               [-10, 115, -5, 120],       # y_min, y_max < 0, x_min, x_max > x_size
                               [95, -10, 100, -5]]        # y_min, y_max > y_size, x_min, x_max < 0
                              ]).astype(np.float32)

        rel_codes = self._encode_offsets(boxes, anchors, scale_factors=scale_factors)
        model = self._build_box_decode(anchors, scale_factors=scale_factors, img_size=img_size)
        out = model(rel_codes)
        exp_boxes = mul * np.array([[[0, 5, 1, 12],       # clip y_min
                                   [85, 0, 90, 2],        # clip x_min
                                   [85, 95, 90, 100]],    # clip y_max
                                  [[0, 85, 2, 110],       # clip x_max
                                   [0, 110, 0, 110],      # y_min, y_max < 0, x_min, x_max > x_size
                                   [90, 0, 90, 0]]        # y_min, y_max > y_size, x_min, x_max < 0
                                  ]).astype(np.float32)
        assert np.allclose(out, exp_boxes)

    @staticmethod
    def _generate_random_boxes(n_batches, n_boxes, seed=None):
        if seed:
            np.random.seed(seed)
        boxes = np.empty((n_batches, n_boxes, 4))
        boxes[..., :2] = np.random.uniform(low=0, high=.9, size=(n_batches, n_boxes, 2))
        boxes[..., 2:] = np.random.uniform(low=boxes[..., :2], high=1., size=(n_batches, n_boxes, 2))
        return boxes.astype(np.float32)

    @classmethod
    def _generate_random_anchors(cls, n_anchors, img_size: tuple = None, seed=None):
        anchors = cls._generate_random_boxes(1, n_anchors, seed)[0]
        if img_size:
            anchors = anchors * np.array(img_size+img_size, dtype=np.float32)
        return anchors

    @staticmethod
    def _encode_offsets(boxes, anchors, scale_factors):
        anchors_hw = anchors[..., 2:] - anchors[..., :2]
        boxes_hw = boxes[..., 2:] - boxes[..., :2]
        boxes_center = boxes[..., :2] + boxes_hw / 2
        anchors_center = anchors[..., :2] + anchors_hw / 2
        thw = np.log(boxes_hw / anchors_hw)
        tyx = (boxes_center - anchors_center)/anchors_hw
        t = np.concatenate([tyx, thw], axis=-1)
        return t * np.asarray(scale_factors)

    @staticmethod
    def _build_box_decode(anchors, scale_factors, img_size, path=None):
        from custom_layers.tf.box_decode import BoxDecode
        box_decode = BoxDecode(anchors=anchors, scale_factors=scale_factors, clip_size=img_size)
        x = tf.keras.layers.Input(anchors.shape)
        model = tf.keras.Model(x, box_decode(x))
        if path:
            model.save(path)
        return model
