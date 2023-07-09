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
Created on 6/8/23

@author: irenab
"""


def corners_to_centroids(y_min, x_min, y_max, x_max):
    """ Converts corners coordinates to centroid coordinates """
    height = y_max - y_min
    width = x_max - x_min
    y_center = y_min + 0.5 * height
    x_center = x_min + 0.5 * width
    return y_center, x_center, height, width


def centroids_to_corners(y_center, x_center, height, width):
    """ Converts centroid coordinates to corners coordinates """
    y_min = y_center - 0.5 * height
    x_min = x_center - 0.5 * width
    y_max = y_min + height
    x_max = x_min + width
    return y_min, x_min, y_max, x_max
