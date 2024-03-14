#!/usr/bin/python3
"""
Demonstration of dataset technologies

"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries 
import glob
import math
import os 

import pandas as pd

# Constants
DATASET_DIR = "dataset"
LABELS_DIR = "Labels"
VIDEO_FORMAT = ".mp4"
CLASS_LIST = ['Robbery', 
              'RoadAccidents', 
              'Stealing', 
              'Burglary', 
              'Abuse', 
              'Shooting', 
              'Vandalism', 
              'Shoplifting', 
              'Arrest', 
              'Assault', 
              'Explosion', 
              'Arson', 
              'Fighting']

def _extract_label(string):
    """
    Function to get the correct path for video
    """
    parts = string.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:2])
    else:
        return None 
    
def _clean_and_update_metadata(dataset):
    """
    Update the path and check if it is true.
    """
    clean_data = []
    for _, row in dataset.iterrows():
        file_name = row[0]
        class_label = row[1]
        anomaly_label = row[2]
        # Check Class Label
        assert class_label in CLASS_LIST, "Class not found"
        # Check Anomaly Label
        assert ~math.isnan(anomaly_label), "Anomaly label not found"
        # Update file name
        sub_dir = _extract_label(file_name)
        assert sub_dir != None, f"sub dir: {file_name} couldn't be extracted"
        sub_dir = sub_dir + VIDEO_FORMAT
        file_path = os.path.join(DATASET_DIR, 
                                 class_label, 
                                 sub_dir, 
                                 file_name + VIDEO_FORMAT)
        # Update anomaly label
        anomaly_label = int(anomaly_label)
        # Update row
        if (os.path.exists(file_path)):
            clean_data.append((file_path, class_label, anomaly_label))
    new_dataset = pd.DataFrame(clean_data)
    return new_dataset
    
def get_filenames():
    """
    Function to get all videos, annotation and path
    """
    label_files = glob.glob(os.path.join(DATASET_DIR, LABELS_DIR, "*.csv"))
    _label_file_data_list = []
    for file_name in label_files:
        label_dataframe = pd.read_csv(file_name, index_col=None, header=None)
        _label_file_data_list.append(label_dataframe)
    
    dataset = pd.concat(_label_file_data_list, axis=0, ignore_index=True)
    updated_dataset = _clean_and_update_metadata(dataset=dataset)
    return updated_dataset

def print_statistics(meta_data):
    """
    Print statistics for the meta-data such as num samples, class etc.
    """
    _class_idx = 1
    _anomaly_idx = 2
    class_counts = meta_data[_class_idx].value_counts()
    anomaly_counts = meta_data[_anomaly_idx].value_counts()
    print("Classes: ")
    print(class_counts)
    print("Anomalies: ")
    print (anomaly_counts)
    print ("Total counts: ", len(meta_data))

def get_statistics(meta_data):
    """
    Get statistics for the meta-data such as num samples, class etc.
    """
    _class_idx = 1
    _anomaly_idx = 2
    class_counts = meta_data[_class_idx].value_counts()
    anomaly_counts = meta_data[_anomaly_idx].value_counts()
    return class_counts.to_dict(), anomaly_counts.to_dict()


def get_split(meta_data, split=0.2):
    """
    Split meta-data for train, val and test
    """
    assert split < 1.0, "Split should be less than or equal to 1.0"
    assert split > 0.0, "Split should be positive"
    meta_data = meta_data.sample(frac=1).reset_index(drop=True)
    meta_data_len = len(meta_data)-1
    num_samples = min(meta_data_len, int(meta_data_len*split))
    # Get array indices
    test_end = num_samples 
    train_start = num_samples+1
    # Split
    test_split = meta_data.iloc[:test_end]
    train_split = meta_data.iloc[train_start:]
    return test_split, train_split

if __name__=="__main__":
    meta_data = get_filenames()
    print_statistics(meta_data=meta_data)
    test_meta_data, train_meta_data = get_split(meta_data=meta_data, 
                                                split=0.2)
    class_data_train, anomaly_data_train = get_statistics(train_meta_data)
    print(class_data_train)