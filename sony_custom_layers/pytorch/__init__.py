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
from typing import Optional, TYPE_CHECKING

from sony_custom_layers.util.import_util import check_pip_requirements

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = ['multiclass_nms', 'load_ort_custom_ops']

# required if this package is imported
requirements = ['torch~=2.2.0', 'torchvision~=0.17.0']
# required only for load_ort_custom_ops
ort_requirement = ['onnxruntime', 'onnxruntime_extensions']

check_pip_requirements(requirements)

from .object_detection import multiclass_nms    # noqa: E402


def load_ort_custom_ops(ort_session_ops: Optional['ort.SessionOptions'] = None) -> 'ort.SessionOptions':
    """
    Registers custom ops implementation for onnxruntime and sets up the SessionObject.

    Args:
        ort_session_ops: SessionOptions object to register the custom library, or None to return a new one
    Returns:
        ort_session_ops with registered custom library

    Usage:
        so = load_ort_custom_ops()
        session = ort.InferenceSession(model_path, so)
        session.run(...)
    """
    check_pip_requirements(ort_requirement)

    # trigger onnxruntime op registration
    from .object_detection import nms_ort

    from onnxruntime_extensions import get_library_path
    from onnxruntime import SessionOptions
    ort_session_ops = ort_session_ops or SessionOptions()
    ort_session_ops.register_custom_ops_library(get_library_path())
    return ort_session_ops
