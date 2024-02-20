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
import numpy as np
import torch
from torch import Tensor
import onnx
import onnxruntime as ort
from onnxruntime_extensions import get_library_path

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

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_onnx_export(self, dynamic_batch, tmpdir_factory):
        score_thresh = 0.1
        iou_thresh = 0.6
        n_boxes = 10
        n_classes = 5
        max_dets = 7
        nms = multiclass_nms.MultiClassNMS(score_threshold=score_thresh,
                                           iou_threshold=iou_thresh,
                                           max_detections=max_dets)

        path = str(tmpdir_factory.mktemp('nms').join('nms.onnx'))
        self._export_onnx(nms, n_boxes, n_classes, path, dynamic_batch)

        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        opset_info = list(model.opset_import)[1]
        assert opset_info.domain == 'Sony' and opset_info.version == 1

        nms_node = list(model.graph.node)[0]
        assert nms_node.domain == 'Sony'
        assert nms_node.op_type == 'MultiClassNMS'
        attrs = sorted(nms_node.attribute, key=lambda a: a.name)
        assert attrs[0].name == 'iou_threshold'
        np.isclose(attrs[0].f, iou_thresh)
        assert attrs[1].name == 'max_detections'
        assert attrs[1].i == max_dets
        assert attrs[2].name == 'score_threshold'
        np.isclose(attrs[2].f, score_thresh)
        assert len(nms_node.input) == 2
        assert len(nms_node.output) == 4
        assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][1:] == [10, 4]
        assert [d.dim_value for d in model.graph.input[1].type.tensor_type.shape.dim][1:] == [10, 5]
        # this tests shape inference that is defined as part of onnx op
        assert [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim][1:] == [max_dets, 4]
        assert model.graph.output[0].type.tensor_type.elem_type == torch.onnx.TensorProtoDataType.FLOAT
        assert [d.dim_value for d in model.graph.output[1].type.tensor_type.shape.dim][1:] == [max_dets]
        assert model.graph.output[1].type.tensor_type.elem_type == torch.onnx.TensorProtoDataType.FLOAT
        assert [d.dim_value for d in model.graph.output[2].type.tensor_type.shape.dim][1:] == [max_dets]
        assert model.graph.output[2].type.tensor_type.elem_type == torch.onnx.TensorProtoDataType.INT32
        assert [d.dim_value for d in model.graph.output[3].type.tensor_type.shape.dim][1:] == [1]
        assert model.graph.output[3].type.tensor_type.elem_type == torch.onnx.TensorProtoDataType.INT32

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_ort(self, dynamic_batch, tmpdir_factory):
        nms = multiclass_nms.MultiClassNMS(score_threshold=0.5, iou_threshold=0.3, max_detections=1000)
        n_boxes = 500
        n_classes = 20
        path = str(tmpdir_factory.mktemp('nms').join('nms.onnx'))
        self._export_onnx(nms, n_boxes, n_classes, path, dynamic_batch)

        batch = 5 if dynamic_batch else 1
        boxes, scores = self._generate_random_inputs(batch=batch, n_boxes=n_boxes, n_classes=n_classes, seed=42)
        torch_res = nms(boxes, scores)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        session = ort.InferenceSession(path, so)
        ort_res = session.run(output_names=None, input_feed={'boxes': boxes.numpy(), 'scores': scores.numpy()})
        # this is just a sanity test on random data
        for i in range(len(torch_res)):
            assert np.allclose(torch_res[i], ort_res[i]), i

    @staticmethod
    def _generate_random_inputs(batch: Optional[int], n_boxes, n_classes, seed=None):
        boxes_shape = (batch, n_boxes, 4) if batch else (n_boxes, 4)
        scores_shape = (batch, n_boxes, n_classes) if batch else (n_boxes, n_classes)
        if seed:
            torch.random.manual_seed(seed)
        boxes = torch.rand(*boxes_shape)
        scores = torch.rand(*scores_shape)
        return boxes, scores

    def _export_onnx(self, nms_model, n_boxes, n_classes, path, dynamic_batch: bool):
        input_names = ['boxes', 'scores']
        output_names = ['det_boxes', 'det_scores', 'det_labels', 'valid_dets']
        kwargs = dict(dynamic_axes={k: {0: 'batch'} for k in input_names + output_names}) if dynamic_batch else {}
        torch.onnx.export(nms_model,
                          args=(torch.ones(1, n_boxes, 4), torch.ones(1, n_boxes, n_classes)),
                          f=path,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs)
