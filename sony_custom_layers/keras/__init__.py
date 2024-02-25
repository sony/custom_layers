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
from packaging.version import parse

try:
    import tensorflow
    if not parse('2.10') <= parse(tensorflow.__version__) < parse('2.16'):
        raise RuntimeError(f'Found unsupported TensorFlow version {tensorflow.__version__}.'
                           f'Supported versions >=2.10,<2.16')
except ImportError:
    raise RuntimeError('TensorFlow package not found, please install it. Supported versions >=2.10,<2.16')

from .object_detection import FasterRCNNBoxDecode, SSDPostProcess, ScoreConverter
from .custom_objects import custom_layers_scope
