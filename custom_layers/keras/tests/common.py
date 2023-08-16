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
import os
import sys


class CustomOpTesterBase:

    @staticmethod
    def _test_clean_custom_objects(op: str):
        """ test in a clean process that op registration to custom_objects is triggered upon import """
        cmd = f"from custom_layers.keras import custom_objects;" \
              f"assert '{op}' in custom_objects"
        ret = os.system(f'{sys.executable} -c "{cmd}"')
        assert ret == 0

    @staticmethod
    def _test_clean_load_model_with_custom_objects(path):
        cmd = f"import tensorflow as tf;" \
              f"from custom_layers.keras import custom_objects;" \
              f"tf.keras.models.load_model('{path}', custom_objects=custom_objects)"
        ret = os.system(f'{sys.executable} -c "{cmd}"')
        assert ret == 0

    @staticmethod
    def _test_load_model(path):
        cmd = f"import tensorflow as tf;" \
              f"tf.keras.models.load_model('{path}')"
        ret = os.system(f'{sys.executable} -c "{cmd}"')
        assert ret == 0
