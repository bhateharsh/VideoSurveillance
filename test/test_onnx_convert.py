#!/usr/bin/python3
"""
Convert model to ONNX
TODO: Add test framework
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries
import torch

from torchvision.models import vgg11, VGG11_Weights

from modules.model.ONNXConverter import ConvertONNX

def test_onnx_conversion(filepath, dynanmic=True):
    # model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model = vgg11(weights = VGG11_Weights.DEFAULT)
    if torch.cuda.is_available():
        model.cuda()
    converter = ConvertONNX(model=model, 
                            input_shape=(1,3,224,224), 
                            input_dtype=torch.float32)
    converter.convert(dynamic_shapes=dynanmic)
    converter.save(filepath=filepath)
    print("Saved model at: ", filepath)

if __name__=="__main__":
    test_onnx_conversion("weights/VGG16ONNX.onnx", dynanmic=False)
