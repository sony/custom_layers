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

from sony_custom_layers.util.import_util import check_pip_requirements
from sony_custom_layers import requirements

check_pip_requirements(requirements['tf'])

from .object_detection import FasterRCNNBoxDecode, SSDPostProcess, ScoreConverter    # noqa: E402
from .custom_objects import custom_layers_scope    # noqa: E402
