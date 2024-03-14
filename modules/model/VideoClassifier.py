#!/usr/bin/python3
"""
Class to describe PyTorch model for training.
TODO:s Add logger
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
import torch.nn as nn

from torchvision.models.video import r3d_18, R3D_18_Weights

class VideoClassifier(nn.Module):
    """
    Class that uses R3D_18 model as base model for creation of 
    multi-classification heads. The first head classifies the video into types 
    as mentioned in the dataset such as road accident, vandalism, etc. The 
    second head classifies the video into normal and abnormal. The base weight 
    are taken from PyTorch model zoo for fine-tuning.

    Reference for R3D_18 model: https://arxiv.org/abs/1711.11248
    """
    def __init__(self, num_video_classes, use_pretrained=True):
        """
        Initialize the VideoClassifier with R3D_18 base model and add custom 
        classification heads.
        """
        super(VideoClassifier, self).__init__()
        # Load model
        self.base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Modify the last fully connected layer for multi-head classification
        self.fc_video_class = nn.Linear(in_features = self.base_model.fc.out_features,
                                        out_features = num_video_classes)
        self.fc_abnormal_class = nn.Linear(in_features = self.base_model.fc.out_features, 
                                           out_features = 2)
        # Initializing weight for new FC layers
        nn.init.normal_(tensor=self.fc_video_class.weight, 
                        mean=0, 
                        std=0.001)
        nn.init.constant_(tensor=self.fc_video_class.bias, 
                          val=0)
        nn.init.normal_(tensor=self.fc_abnormal_class.weight, 
                        mean=0, 
                        std=0.001)
        nn.init.constant_(tensor=self.fc_abnormal_class.bias, 
                          val=0)
        
    def forward(self, x):
        """
        Describe the forward pass
        """
        # Get features from base model
        features = self.base_model(x)
        # Pass features to classification heads
        video_out = self.fc_video_class(features)
        abnormal_out = self.fc_abnormal_class(features)

        return video_out, abnormal_out
    
    def save_model(self, filepath):
        """
        Save model in pth format
        """
        torch.save(self.state_dict(), filepath)
        print ("Model saved to ", filepath)

    def load_model(self, filepath):
        """
        Load saved model
        """
        self.load_state_dict(torch.load(filepath))
        print ("Model loaded from ", filepath)