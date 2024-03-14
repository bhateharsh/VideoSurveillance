#!/usr/bin/python3
"""
Convert model to ONNX
TODO: Add logger
TODO: Support older ONNX formats for devices such as TDA4VM
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

from torchsummary import summary

class ConvertONNX:
    """
    Class to convert pytorch model to ONNX,
    assumes latest version of ONNX compatible with 
    torchdynamo based onnx exporter
    """
    def __init__(self, model, input_shape, input_dtype):
        """
        Initializer to check the model and produce summary
        """
        self.model = model
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.onnx_program = None
        print("Model Summary: ")
        # summary(model, input_shape)

    def convert(self, dynamic_shapes=False):
        """
        Convert model to ONNX
        """
        if dynamic_shapes:
            export_options = torch.onnx.ExportOptions(dynamic_shapes=dynamic_shapes)
            self.onnx_program = torch.onnx.dynamo_export(
                self.model, 
                export_options = export_options
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = torch.rand(self.input_shape, 
                                      dtype=self.input_dtype, 
                                      device=device)
            self.onnx_program = torch.onnx.dynamo_export(self.model, 
                                                         input_tensor)
    
    def save(self, filepath):
        if self.onnx_program:
            self.onnx_program.save(filepath)
        else:
            raise ValueError("ONNX Program doesn't exist.\
                             Please convert model")
        