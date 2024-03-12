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

from sony_custom_layers.pytorch.object_detection import nms
from sony_custom_layers.pytorch import load_custom_ops
from sony_custom_layers.util.test_util import exec_in_clean_process


class NMS(torch.nn.Module):

    def __init__(self, score_threshold, iou_threshold, max_detections):
        super().__init__()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, boxes, scores):
        return nms.multiclass_nms(boxes,
                                  scores,
                                  score_threshold=self.score_threshold,
                                  iou_threshold=self.iou_threshold,
                                  max_detections=self.max_detections)


class TestMultiClassNMS:

    def test_flatten_image_inputs(self):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.11, 0.21, 0.31, 0.41],
                        [0.12, 0.22, 0.32, 0.42]])    # yapf: disable
        scores = Tensor([[0.15, 0.25, 0.35, 0.45],
                         [0.16, 0.26, 0.11, 0.46],
                         [0.1, 0.27, 0.37, 0.47]])    # yapf: disable
        x = nms._convert_inputs(boxes, scores, score_threshold=0.11)
        flat_boxes, flat_scores, labels = x[:, :4], x[:, 4], x[:, 5]
        assert flat_boxes.shape == (10, 4)
        assert flat_scores.shape == labels.shape == (10, )
        assert torch.equal(labels, Tensor([0, 1, 2, 3, 0, 1, 3, 1, 2, 3]))
        for i in range(4):
            assert torch.equal(flat_boxes[i], boxes[0]), i
        for i in range(4, 7):
            assert torch.equal(flat_boxes[i], boxes[1]), i
        for i in range(7, 10):
            assert torch.equal(flat_boxes[i], boxes[2]), i
        assert torch.equal(flat_scores, Tensor([0.15, 0.25, 0.35, 0.45, 0.16, 0.26, 0.46, 0.27, 0.37, 0.47]))

    def test_nms_with_class_offsets(self):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4]])  # yapf: disable
        scores = Tensor([0.25, 0.15, 0.3, 0.45, 0.5, 0.4])
        labels = Tensor([1, 0, 1, 2, 2, 1])
        x = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
        iou_threshold = 0.5
        ret_idxs = nms._nms_with_class_offsets(x, iou_threshold)
        assert torch.equal(ret_idxs, Tensor([4, 3, 5, 2, 1]))

    @pytest.mark.parametrize('max_detections', [3, 6, 10])
    # mock is to test our logic, and no mock is for integration sanity
    @pytest.mark.parametrize('mock_tv_op', [True, False])
    def test_image_multiclass_nms(self, mocker, max_detections, mock_tv_op):
        boxes = Tensor([[0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8]])    # yapf: disable
        scores = Tensor([[0.2, 0.109, 0.3, 0.12],
                         [0.111, 0.5, 0.05, 0.4]])    # yapf: disable
        score_threshold = 0.11
        iou_threshold = 0.61
        if mock_tv_op:
            nms_mock = mocker.patch('sony_custom_layers.pytorch.object_detection.nms._nms_with_class_offsets',
                                    Mock(return_value=Tensor([4, 5, 1, 0, 2, 3]).to(torch.int64)))
        ret, ret_valid_dets = nms._image_multiclass_nms(boxes,
                                                        scores,
                                                        score_threshold=score_threshold,
                                                        iou_threshold=iou_threshold,
                                                        max_detections=max_detections)
        if mock_tv_op:
            assert torch.equal(nms_mock.call_args.args[0][:, :4],
                               Tensor([[0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.1, 0.2, 0.3, 0.4],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8],
                                       [0.5, 0.6, 0.7, 0.8]]))    # yapf: disable
            assert torch.equal(nms_mock.call_args.args[0][:, 4], Tensor([0.2, 0.3, 0.12, 0.111, 0.5, 0.4]))
            assert torch.equal(nms_mock.call_args.args[0][:, 5], Tensor([0, 2, 3, 0, 1, 3]))
            assert nms_mock.call_args.kwargs == {'iou_threshold': iou_threshold}

        assert ret.shape == (max_detections, 6)
        exp_valid_dets = min(6, max_detections)
        assert torch.equal(ret[:, :4][:exp_valid_dets],
                           Tensor([[0.5, 0.6, 0.7, 0.8],
                                   [0.5, 0.6, 0.7, 0.8],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.5, 0.6, 0.7, 0.8]])[:exp_valid_dets])  # yapf: disable
        assert torch.all(ret[:, :4][exp_valid_dets:] == 0)
        assert torch.equal(ret[:, 4][:exp_valid_dets], Tensor([0.5, 0.4, 0.3, 0.2, 0.12, 0.111])[:exp_valid_dets])
        assert torch.all(ret[:, 4][exp_valid_dets:] == 0)
        assert torch.equal(ret[:, 5][:exp_valid_dets], Tensor([1, 3, 2, 0, 3, 0])[:exp_valid_dets])
        assert torch.all(ret[:, 5][exp_valid_dets:] == 0)
        assert ret_valid_dets == exp_valid_dets

    def test_batch_multiclass_nms(self, mocker):
        input_boxes, input_scores = self._generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        max_dets = 5

        # these numbers don't really make sense as nms outputs, but we don't really care, we only want to test
        # that outputs are combined correctly
        img_nms_ret = torch.rand(3, max_dets, 6)
        img_nms_ret[..., 5] = torch.randint(0, 10, (3, max_dets), dtype=torch.float32)
        ret_valid_dets = Tensor([5, 4, 3])
        # each time the function is called, next value in the list returned
        images_ret = [(img_nms_ret[i], ret_valid_dets[i]) for i in range(3)]
        mock = mocker.patch('sony_custom_layers.pytorch.object_detection.nms._image_multiclass_nms',
                            Mock(side_effect=lambda *args, **kwargs: images_ret.pop(0)))

        ret = nms._multiclass_nms_impl(input_boxes,
                                       input_scores,
                                       score_threshold=0.1,
                                       iou_threshold=0.6,
                                       max_detections=5)

        # check each invocation
        for i, call_args in enumerate(mock.call_args_list):
            assert torch.equal(call_args.args[0], input_boxes[i]), i
            assert torch.equal(call_args.args[1], input_scores[i]), i
            assert call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5), i

        assert torch.equal(ret.boxes, img_nms_ret[:, :, :4])
        assert torch.equal(ret.scores, img_nms_ret[:, :, 4])
        assert torch.equal(ret.labels, img_nms_ret[:, :, 5])
        assert ret.labels.dtype == torch.int64
        assert torch.equal(ret.n_valid, ret_valid_dets)
        assert ret.n_valid.dtype == torch.int64

    def test_torch_op(self, mocker):
        mock = mocker.patch(
            'sony_custom_layers.pytorch.object_detection.nms._multiclass_nms_impl',
            Mock(return_value=(torch.rand(3, 5, 4), torch.rand(3, 5), torch.rand(3, 5), torch.rand(3, 1))))
        boxes, scores = self._generate_random_inputs(batch=3, n_boxes=10, n_classes=5)
        ret = torch.ops.sony.multiclass_nms(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5.)
        assert ret == mock.return_value

    def test_torch_op_wrapper(self, mocker):
        mock = mocker.patch(
            'sony_custom_layers.pytorch.object_detection.nms._multiclass_nms_impl',
            Mock(return_value=(torch.rand(3, 5, 4), torch.rand(3, 5), torch.rand(3, 5), torch.rand(3, 1))))
        boxes, scores = self._generate_random_inputs(batch=3, n_boxes=20, n_classes=10)
        ret = nms.multiclass_nms(boxes, scores, score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert torch.equal(mock.call_args.args[0], boxes)
        assert torch.equal(mock.call_args.args[1], scores)
        assert mock.call_args.kwargs == dict(score_threshold=0.1, iou_threshold=0.6, max_detections=5)
        assert isinstance(ret, nms.NMSResults)
        assert torch.equal(ret.boxes, mock.return_value[0])
        assert torch.equal(ret.scores, mock.return_value[1])
        assert torch.equal(ret.labels, mock.return_value[2])
        assert torch.equal(ret.n_valid, mock.return_value[3])

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_onnx_export(self, dynamic_batch, tmpdir_factory):
        score_thresh = 0.1
        iou_thresh = 0.6
        n_boxes = 10
        n_classes = 5
        max_dets = 7

        onnx_model = NMS(score_thresh, iou_thresh, max_dets)

        path = str(tmpdir_factory.mktemp('nms').join('nms.onnx'))
        self._export_onnx(onnx_model, n_boxes, n_classes, path, dynamic_batch=dynamic_batch)

        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model, full_check=True)
        opset_info = list(onnx_model.opset_import)[1]
        assert opset_info.domain == 'Sony' and opset_info.version == 1

        nms_node = list(onnx_model.graph.node)[0]
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

        def check_tensor(onnx_tensor, exp_shape, exp_type):
            tensor_type = onnx_tensor.type.tensor_type
            shape = [d.dim_value if d.dim_value else d.dim_param for d in tensor_type.shape.dim]
            exp_shape = ['batch' if dynamic_batch else 1] + exp_shape
            assert shape == exp_shape
            assert tensor_type.elem_type == exp_type

        check_tensor(onnx_model.graph.input[0], [10, 4], torch.onnx.TensorProtoDataType.FLOAT)
        check_tensor(onnx_model.graph.input[1], [10, 5], torch.onnx.TensorProtoDataType.FLOAT)
        # test shape inference that is defined as part of onnx op
        check_tensor(onnx_model.graph.output[0], [max_dets, 4], torch.onnx.TensorProtoDataType.FLOAT)
        check_tensor(onnx_model.graph.output[1], [max_dets], torch.onnx.TensorProtoDataType.FLOAT)
        check_tensor(onnx_model.graph.output[2], [max_dets], torch.onnx.TensorProtoDataType.INT32)
        check_tensor(onnx_model.graph.output[3], [1], torch.onnx.TensorProtoDataType.INT32)

    @pytest.mark.parametrize('dynamic_batch', [True, False])
    def test_ort(self, dynamic_batch, tmpdir_factory):
        model = NMS(0.5, 0.3, 1000)
        n_boxes = 500
        n_classes = 20
        path = str(tmpdir_factory.mktemp('nms').join('nms.onnx'))
        self._export_onnx(model, n_boxes, n_classes, path, dynamic_batch)

        batch = 5 if dynamic_batch else 1
        boxes, scores = self._generate_random_inputs(batch=batch, n_boxes=n_boxes, n_classes=n_classes, seed=42)
        torch_res = model(boxes, scores)

        so = load_custom_ops(load_ort=True)
        session = ort.InferenceSession(path, so)
        ort_res = session.run(output_names=None, input_feed={'boxes': boxes.numpy(), 'scores': scores.numpy()})
        # this is just a sanity test on random data
        for i in range(len(torch_res)):
            assert np.allclose(torch_res[i], ort_res[i]), i

    def test_pt2_export(self, tmpdir_factory):

        def f(boxes, scores):
            return nms.multiclass_nms(boxes, scores, 0.5, 0.3, 100)

        prog = torch.export.export(f, args=(torch.rand(1, 10, 4), torch.rand(1, 10, 5)))
        nms_node = list(prog.graph.nodes)[2]
        assert nms_node.target == torch.ops.sony.multiclass_nms.default
        val = nms_node.meta['val']
        assert val[0].shape[1:] == (100, 4)
        assert val[1].shape[1:] == val[2].shape[1:] == (100, )
        assert val[2].dtype == torch.int64
        assert val[3].shape[1:] == ()
        assert val[3].dtype == torch.int64

        boxes, scores = self._generate_random_inputs(1, 10, 5)
        torch_out = f(boxes, scores)
        prog_out = prog.module()(boxes, scores)
        for i in range(len(torch_out)):
            assert torch.allclose(torch_out[i], prog_out[i]), i

        path = str(tmpdir_factory.mktemp('nms').join('nms.pt2'))
        torch.export.save(prog, path)
        # check that exported program can be loaded in a clean env
        code = f"""
import torch
import sony_custom_layers.pytorch
prog = torch.export.load('{path}')
boxes = torch.rand(1, 10, 4)
boxes[..., 0], boxes[..., 2] = torch.aminmax(boxes[..., (0, 2)], dim=-1)
boxes[..., 1], boxes[..., 3] = torch.aminmax(boxes[..., (1, 3)], dim=-1)
prog.module()(boxes, torch.rand(1, 10, 5))
        """
        exec_in_clean_process(code, check=True)

    @staticmethod
    def _generate_random_inputs(batch: Optional[int], n_boxes, n_classes, seed=None):
        boxes_shape = (batch, n_boxes, 4) if batch else (n_boxes, 4)
        scores_shape = (batch, n_boxes, n_classes) if batch else (n_boxes, n_classes)
        if seed:
            torch.random.manual_seed(seed)
        boxes = torch.rand(*boxes_shape)
        boxes[..., 0], boxes[..., 2] = torch.aminmax(boxes[..., (0, 2)], dim=-1)
        boxes[..., 1], boxes[..., 3] = torch.aminmax(boxes[..., (1, 3)], dim=-1)
        scores = torch.rand(*scores_shape)
        return boxes, scores

    def _export_onnx(self, nms_model, n_boxes, n_classes, path, dynamic_batch: bool):
        input_names = ['boxes', 'scores']
        output_names = ['det_boxes', 'det_scores', 'det_labels', 'valid_dets']
        kwargs = {'dynamic_axes': {k: {0: 'batch'} for k in input_names + output_names}} if dynamic_batch else {}
        torch.onnx.export(nms_model,
                          args=(torch.ones(1, n_boxes, 4), torch.ones(1, n_boxes, n_classes)),
                          f=path,
                          input_names=input_names,
                          output_names=output_names,
                          **kwargs)
