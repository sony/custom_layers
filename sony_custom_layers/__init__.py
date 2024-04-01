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

# minimal requirements for dynamic validation in sony_custom_layers.{keras, pytorch}.__init__
requirements = {
    'tf': ['tensorflow>=2.10'],
    'torch': ['torch>=2.0', 'torchvision>=0.15'],
    'torch_ort': ['onnx', 'onnxruntime', 'onnxruntime_extensions>=0.8.0'],
}

# pinned requirements of latest tested versions for extra_requires
pinned_requirements = {
    'tf': ['tensorflow==2.15.*'],
    'torch': ['torch==2.2.*', 'torchvision==0.17.*'],
    'torch_ort': ['onnx==1.15.*', 'onnxruntime==1.17.*', 'onnxruntime_extensions==0.10.*']
}
