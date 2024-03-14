#!/usr/bin/python3
"""
Class to create dataset in PyTorch for training.

"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries
import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

from modules.utilities.dataset_utility import *
# Constants
DATASET_DIR = "dataset"
LABELS_DIR = "Labels"
VIDEO_FORMAT = ".mp4"

class VideoDataset(Dataset):
    """
    Class to create dataset for training transformer

    Usage:
        # Get Training set
        meta_data = get_filenames()
        test_meta_data, train_meta_data = get_split(meta_data=meta_data, 
        # Initialize dataset
        v_dataset = VideoDataset(meta_data=train_meta_data, 
                                transform=None)
        dataloader = DataLoader(v_dataset, 
                                batch_size=32, 
                                shuffle=True)
        # Accessing a sample
        video, class_label, anomaly_label = v_dataset[0]
        print("Video shape: ", video.shape)
        print("Class Label: ", class_label)
        print("Anomaly Label: ", anomaly_label)

    """
    def __init__(self, meta_data, transform=None):
        self.meta_data = meta_data
        self.transform = transform
        self.class_data, self.anomaly_data = get_statistics(self.meta_data)
        self.class_list = list(self.class_data.keys())
        print(self.class_list)
        self._dataset_dir = "dataset"
        self._video_format = ".mp4"
        self.videos = self._load_videos()
        print (self.videos[0:5])

    def _load_videos(self):
        """
        Function that creates a list of video path with annotations
        """
        return list(self.meta_data.itertuples(index=False, name=None))
    
    def __len__(self):
        """
        Get length of dataset
        """
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path, class_label, anomaly_label = self.videos[idx]
        video, _, _ = read_video(video_path)
        if self.transform:
            video = self.transform(video).unsqueeze(0)
        return video, class_label, anomaly_label


# if __name__=="__main__":
#     # Get Training set
#     meta_data = get_filenames()
#     test_meta_data, train_meta_data = get_split(meta_data=meta_data, 
#                                                 split=0.2)
#     # Get transforms
#     weights = R3D_18_Weights.DEFAULT
#     model = r3d_18(weights=weights)
#     model.eval()
#     # Initialize inference transforms
#     preprocess = weights.transforms()

#     # Initialize dataset
#     v_dataset = VideoDataset(meta_data=train_meta_data, 
#                              transform=None)
#     dataloader = DataLoader(v_dataset, 
#                             batch_size=32, 
#                             shuffle=True)
#     # Accessing a sample
#     video, class_label, anomaly_label = v_dataset[0]
#     print("Video shape: ", video.shape)
#     print("Class Label: ", class_label)
#     print("Anomaly Label: ", anomaly_label)
