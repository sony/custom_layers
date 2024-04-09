# Sony Custom Layers (SCL)

Sony Custom Layers (SCL) is an open-source project implementing detection post process NN layers not supported by the TensorFlow Keras API or Torch's torch.nn for the easy integration of those layers into pretrained models.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Supported Versions](#supported-versions)
- [API](#api)
  - [TensorFlow API](#tensorflow-api)
  - [PyTorch API](#pytorch-api)
- [License](#license)


## Getting Started

This section provides an installation and a quick starting guide.

### Installation

To install the latest stable release of SCL, run the following command:
```
pip install sony-custom-layers
```
By default, no framework dependencies are installed.
To install SCL including the latest tested dependencies (up to patch version) for TensorFlow:
```
pip install sony-custom-layers[tf]
```
To install SCL including the latest tested dependencies (up to patch version) for PyTorch/ONNX/OnnxRuntime:
```
pip install sony-custom-layers[torch]
```
### Supported Versions

#### TensorFlow

| **Tested FW versions** | **Tested Python version** | **Serialization** |
|------------------------|---------------------------|-------------------|
| 2.10                   | 3.8-3.10                  | .h5               |
| 2.11                   | 3.8-3.10                  | .h5               |
| 2.12                   | 3.8-3.11                  | .h5  .keras       |
| 2.13                   | 3.8-3.11                  | .keras            |
| 2.14                   | 3.9-3.11                  | .keras            |
| 2.15                   | 3.9-3.11                  | .keras            |

#### PyTorch

| **Tested FW versions**                                                                                                   | **Tested Python version** | **Serialization**                                                               |
|--------------------------------------------------------------------------------------------------------------------------|---------------------------|---------------------------------------------------------------------------------|
| torch 2.0-2.2<br/>torchvision 0.15-0.17<br/>onnxruntime 1.15-1.17<br/>onnxruntime_extensions 0.8-0.10<br/>onnx 1.14-1.15 | 3.8-3.11                  | .onnx (via torch.onnx.export)<br/>.pt2 (via torch.export.export, torch2.2 only) |

## API
For sony-custom-layers API see https://sony.github.io/custom_layers

### TensorFlow API
For TensorFlow layers see
[KerasAPI](https://sony.github.io/custom_layers/sony_custom_layers/keras.html)

To load a model with custom layers in TensorFlow, see [custom_layers_scope](https://sony.github.io/custom_layers/sony_custom_layers/keras.html#custom_layers_scope)

### PyTorch API
For PyTorch layers see
[PyTorchAPI](https://sony.github.io/custom_layers/sony_custom_layers/pytorch.html)

No special handling is required for torch.onnx.export and onnx.load.

For OnnxRuntime / PT2 support see [load_custom_ops](https://sony.github.io/custom_layers/sony_custom_layers/pytorch.html#load_custom_ops) 

## License
[Apache License 2.0](LICENSE.md).


