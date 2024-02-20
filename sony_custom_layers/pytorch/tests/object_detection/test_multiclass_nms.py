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
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from sony_custom_layers.pytorch.object_detection import multiclass_nms


class TestMultiClassNMS:

    def test_flatten_image_inputs(self):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.11, 0.21, 0.31, 0.41],
                        [0.12, 0.22, 0.32, 0.42]])    # yapf: disable
        scores = Tensor([[0.15, 0.25],
                         [0.16, 0.26],
                         [0.17, 0.27]])    # yapf: disable
        flat_boxes, flat_scores, labels = multiclass_nms._flatten_image_inputs(boxes, scores)
        assert flat_boxes.shape == (6, 4)
        assert flat_scores.shape == labels.shape == (6, )
        assert torch.equal(labels, Tensor([0, 1, 0, 1, 0, 1]))
        assert torch.equal(flat_boxes[::2], boxes)
        assert torch.equal(flat_boxes[1::2], boxes)
        assert torch.equal(flat_scores, Tensor([0.15, 0.25, 0.16, 0.26, 0.17, 0.27]))

    @pytest.mark.parametrize('max_detections', [3, 6, 10])
    # mock is to test our logic, and no mock is for integration sanity
    @pytest.mark.parametrize('mock_tv_op', [True, False])
    def test_image_multiclass_nms(self, mocker, max_detections, mock_tv_op):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8]])    # yapf: disable
        scores = Tensor([[0.2, 0.109, 0.3, 0.111],
                         [0.11, 0.5, 0.05, 0.4]])    # yapf: disable
        score_threshold = 0.11
        iou_threshold = 0.61
        if mock_tv_op:
            tv_nms_mock = mocker.patch('torchvision.ops.batched_nms',
                                       Mock(return_value=Tensor([4, 5, 1, 0, 2, 3]).to(torch.int64)))
        ret = multiclass_nms._image_multiclass_nms(boxes,
                                                   scores,
                                                   score_threshold=score_threshold,
                                                   iou_threshold=iou_threshold,
                                                   max_detections=max_detections)
        if mock_tv_op:
            assert torch.equal(tv_nms_mock.call_args.args[0],
                               Tensor([[0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8]]))    # yapf: disable
            assert torch.equal(tv_nms_mock.call_args.args[1], Tensor([0.2, 0.3, 0.111, 0.11, 0.5, 0.4]))
            assert torch.equal(tv_nms_mock.call_args.args[2], Tensor([0, 2, 3, 0, 1, 3]))
            assert tv_nms_mock.call_args.kwargs == {'iou_threshold': iou_threshold}

        assert ret.boxes.shape == (max_detections, 4)
        assert ret.scores.shape == ret.labels.shape == (max_detections, )
        valid_dets = min(6, max_detections)
        assert torch.equal(ret.boxes[:valid_dets],
                           Tensor([[0.5, 0.6, 0.7, 0.8],
                                   [0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])[:valid_dets])  # yapf: disable
        assert torch.all(ret.boxes[valid_dets:] == 0)
        assert torch.equal(ret.scores[:valid_dets], Tensor([0.5, 0.4, 0.3, 0.2, 0.111, 0.11])[:valid_dets])
        assert torch.all(ret.scores[valid_dets:] == 0)
        assert torch.equal(ret.labels[:valid_dets], Tensor([1, 3, 2, 0, 3, 0])[:valid_dets])
        assert torch.all(ret.labels[valid_dets:] == 0)
        assert torch.equal(ret.valid_detections, Tensor([valid_dets]).to(torch.int64))

    def test_batch_multiclass_nms(self, mocker):
        input_boxes, input_scores = self._generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        max_dets = 5

        # these numbers don't really make sense as nms outputs, but we don't really care, we only want to test
        # that outputs are combined correctly
        ret_boxes = torch.rand(3, max_dets, 4)
        ret_scores = torch.rand(3, max_dets)
        ret_labels = torch.randint(0, 10, (3, max_dets))
        ret_valid_dets = Tensor([[5], [4], [3]])
        # each time the function is called, next value in the list returned
        images_ret = [
            multiclass_nms.NMSResults(ret_boxes[i], ret_scores[i], ret_labels[i], ret_valid_dets[i]) for i in range(3)
        ]
        mock = mocker.patch('sony_custom_layers.pytorch.object_detection.multiclass_nms._image_multiclass_nms',
                            Mock(side_effect=lambda *args, **kwargs: images_ret.pop(0)))

        ret = multiclass_nms.multiclass_nms_impl(input_boxes,
                                                 input_scores,
                                                 score_threshold=0.1,
                                                 iou_threshold=0.6,
                                                 max_detections=5)

        # check each invocation
        for i, call_args in enumerate(mock.call_args_list):
            assert torch.equal(call_args.args[0], input_boxes[i]), i
            assert torch.equal(call_args.args[1], input_scores[i]), i
            assert call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5), i

        assert torch.equal(ret.boxes, ret_boxes)
        assert torch.equal(ret.scores, ret_scores)
        assert torch.equal(ret.labels, ret_labels)
        assert torch.equal(ret.valid_detections, ret_valid_dets)

    def test_torch_op(self, mocker):
        mock = mocker.patch('sony_custom_layers.pytorch.object_detection.multiclass_nms.multiclass_nms_impl',
                            Mock(return_value=(torch.rand(3, 5, 4), torch.rand(3, 5))))
        boxes, scores = self._generate_random_inputs(batch=3, n_boxes=10, n_classes=5)
        ret = torch.ops.sony.multiclass_nms(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert ret == mock.return_value

    def test_torch_module(self, mocker):
        mock = mocker.patch('sony_custom_layers.pytorch.object_detection.multiclass_nms.multiclass_nms_impl',
                            Mock(return_value=(torch.rand(3, 5, 4), torch.rand(3, 5))))
        boxes, scores = self._generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        nms = multiclass_nms.MultiClassNMS(score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        ret = nms(boxes, scores)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert ret == mock.return_value

    @staticmethod
    def _generate_random_inputs(batch: Optional[int], n_boxes, n_classes):
        boxes_shape = (batch, n_boxes, 4) if batch else (n_boxes, 4)
        scores_shape = (batch, n_boxes, n_classes) if batch else (n_boxes, n_classes)
        boxes = torch.rand(*boxes_shape)
        scores = torch.rand(*scores_shape)
        return boxes, scores
