#!/usr/bin/python3
"""
Test model creation

TODO: Use google test or other framework and create unit tests
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

import os

import torch

from torchsummary import summary
from modules.model.VideoClassifier import VideoClassifier

def summarize_model(num_classes=18, pretrained=True):
    """
    Function to initialize the model, and present summary for
    reference class
    """
    model = VideoClassifier(num_video_classes=num_classes, 
                            use_pretrained=pretrained)
    if torch.cuda.is_available():
        model.cuda()
    summary(model, (3,30,112,112))

def test_model_save_and_load(filepath, num_classes=18, pretrained=True):
    """
    Function to load, save model for inference
    """
    model = VideoClassifier(num_video_classes=num_classes, 
                            use_pretrained=pretrained)
    if torch.cuda.is_available():
        model.cuda()
    # Save Model
    model.save_model(filepath=filepath)
    assert (os.path.exists(filepath)), "Model save failed"
    # Load model
    print("Load path: ", filepath)
    model.load_model(filepath=filepath)
    # model.load_state_dict(torch.load(filepath))
    model.eval()
    # Remove model
    os.remove(filepath)
if __name__=="__main__":
    summarize_model()
    test_model_save_and_load(filepath="weights/test.pth")