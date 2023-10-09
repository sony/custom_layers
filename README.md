# Sony Custom Layers (SCL)

Sony Custom Layers (SCL) is an open-source project implementing detection post process NN layers not supported by the TensorFlow Keras API or Torch's torch.nn for the easy integration of those layers into pretrained models.

## Table of Contents

- [Getting Started](#getting-started)
- [Implemented Layers](#implemented-layers)
- [License](#license)


## Getting Started

This section provides an installation and a quick starting guide.

### Installation

To install the latest stable release of SCL, run the following command:
```
pip install sony-custom-layers
```

### Supported Versions

Currently, SCL is being tested on a matrix of Python and TensorFlow versions:
| **Framework** | **Tested FW versions** | **Tested Python version** | **Serialization** |
|---------------|------------------------|---------------------------|-------------------|
| TensorFlow    | 2.10                   | 3.8-3.10                  | .h5               |
| TensorFlow    | 2.11                   | 3.8-3.10                  | .h5               |
| TensorFlow    | 2.12                   | 3.8-3.11                  | .h5  .keras       |
| TensorFlow    | 2.13                   | 3.8-3.11                  | .keras            |
| TensorFlow    | 2.14                   | 3.8-3.11                  | .keras            |

## Implemented Layers
SCL currently includes implementations of the following layers:
### TensorFlow

| **Layer Name** | **Description**                                      | **API documentation**     |
|-------------------------|---------------------------------------------|---------------------------|
|  FasterRCNNBoxDecode    | Box decoding per [Faster R-CNN](https://arxiv.org/abs/1506.01497) with clipping |  [doc](./sony_custom_layers/keras/object_detection/ssd_pp.md)              |            
|  SSDPostProcess    | Post process as described in [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)  |[doc](./sony_custom_layers/keras/object_detection/faster_rcnn_box_decode.md)                | 

### Torch
* SCL aims to implement torch layers at a later stage

## Loading the model
### TensorFlow
```
with sony_custom_layers.keras.custom_layers_scope():
    model = tf.keras.models.load_model(path)
```
See [source](sony_custom_layers/keras/custom_objects.py) for further details.

## License
[Apache License 2.0](LICENSE.md).


