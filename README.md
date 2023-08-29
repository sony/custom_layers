# Sony Custom Layers (SCL)

Sony Custom Layers (SCL) is an open-source project implementing NN layers not supported by the TensorFlow Keras API or Torch's torch.nn, but are nevertheless supported by the SDSP converter.
The aim of this project is to extend the catalog of layers that can be optimally run on the IMX500 to enable better performance for models utilizing the implemented custom layers.
SCL is developed by researchers and engineers working at Sony Semiconductor Israel.

## Table of Contents

- [Getting Started](#getting-started)
- [Implemented Layers](#implemented-layers)
- [Results](#results)
- [Contributions](#contributions)
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
| TensorFlow    | 2.12                   | 3.8-3.10                  | .h5  .keras       |           

## Implemented Layers
SCL currently includes implementations of the following layers:
### TensorFlow

| **Layer Name** | **Description**                                      | **API documentation**     |
|-------------------------|---------------------------------------------|---------------------------|
|  Faster_RCNN_Box_Decode    | Box decoding per [Faster R-CNN](https://arxiv.org/abs/1506.01497) with clipping |  [doc](./sony_custom_layers/keras/object_detection/ssd_pp.md)              |            
|  SSD_Post_Process    | Post process as described in [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)  |[doc](./sony_custom_layers/keras/object_detection/faster_rcnn_box_decode.md)                | 

### Torch
* SCL aims to implement torch layers at a later stage

## Results
### TensorFlow

## Contributions
SCL aims at keeping a more up-to-date fork and welcomes contributions from anyone.

*You will find more information about contributions in the [Contribution guide](CONTRIBUTING.md).


## License
[Apache License 2.0](LICENSE.md).


