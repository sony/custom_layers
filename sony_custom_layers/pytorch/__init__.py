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
from sony_custom_layers import requirements

if TYPE_CHECKING:
    import onnxruntime as ort

__all__ = ['multiclass_nms', 'NMSResults', 'load_custom_ops']

check_pip_requirements(requirements['torch'])

from .object_detection import multiclass_nms, NMSResults    # noqa: E402


def load_custom_ops(load_ort: bool = False,
                    ort_session_ops: Optional['ort.SessionOptions'] = None) -> Optional['ort.SessionOptions']:
    """
    Load custom ops for torch and, optionally, for onnxruntime.
    If 'load_ort' is True or 'ort_session_ops' is passed, registers the custom ops implementation for onnxruntime, and
    sets up the SessionOptions object for onnxruntime session.

    Note: this is a must for onnxruntime. To trigger torch ops registration any import from sony_custom_layers.pytorch
    is technically sufficient. This is just a dummy api to prevent unused import (e.g. when loading exported pt2 model)

    Usage:
        # for onnxruntime
        so = load_custom_ops(load_ort=True)
        session = ort.InferenceSession(model_path, sess_options=so)
        session.run(...)

    Args:
        load_ort: whether to register the custom ops for onnxruntime.
        ort_session_ops: SessionOptions object to register the custom ops library on. If None (and 'load_ort' is True),
                        creates a new object.

    Returns:
        SessionOptions object if ort registration was requested, otherwise None
    """
    if load_ort or ort_session_ops:
        check_pip_requirements(requirements['torch_ort'])

        # trigger onnxruntime op registration
        from .object_detection import nms_ort

        from onnxruntime_extensions import get_library_path
        from onnxruntime import SessionOptions
        ort_session_ops = ort_session_ops or SessionOptions()
        ort_session_ops.register_custom_ops_library(get_library_path())
        return ort_session_ops
    else:
        # nothing really to do after import was triggered
        return None
