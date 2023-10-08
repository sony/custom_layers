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
Created on 10/8/23

@author: irenab
"""
import abc

import tensorflow as tf

from sony_custom_layers.version import __version__


class CustomLayer(tf.keras.layers.Layer, abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_version: str = __version__

    def get_config(self):
        config = super().get_config()
        config['custom_version'] = self.custom_version
        return config

    @classmethod
    def from_config(cls, config):
        custom_version = config.pop('custom_version')
        layer = cls(**config)
        layer.custom_version = custom_version
        return layer
