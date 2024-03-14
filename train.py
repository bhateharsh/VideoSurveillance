#!/usr/bin/python3
"""
Script to train the video classifier network
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Import Libraries
import os

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.video import R3D_18_Weights
from modules.training.VideoDataset import VideoDataset
from modules.model.VideoClassifier import VideoClassifier
from modules.utilities.dataset_utility import *
from modules.utilities.model_utilities import *

# ***** Constants *****
BATCH_SIZE = 1

# ***** Load dataset *****
# Get Training set
meta_data = get_filenames()
test_meta_data, train_meta_data = get_split(meta_data=meta_data, 
                                            split=0.2)
# Get transforms
weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()
# Initialize dataset
v_dataset = VideoDataset(meta_data=train_meta_data, 
                            transform=preprocess)
dataloader = DataLoader(v_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True)

# ***** Load Model *****
model = VideoClassifier(num_video_classes=13, 
                            use_pretrained=True)

# ***** Training *****
# Get weight matrix
class_weights, anomaly_weights = get_weight_tensor(meta_data=train_meta_data)
# Define weighted loss functions
class_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
anomaly_criterion = torch.nn.CrossEntropyLoss(weight=anomaly_weights)
# Define Optimizer and other parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Resorting device to CPU as GPU is running out of memory
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1
# Train the model 
model.train_model(dataloader, 
                  class_criterion, 
                  anomaly_criterion, 
                  optimizer, 
                  device, 
                  num_epochs)
# Save the model
model.save_model("weights/first_train.pth")