

import os
import math
import glob
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LABELMAP = {
    0: 'person',
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck',
    80: 'cyclist'
    }

def load_img(img_path: str) -> np.ndarray:
    img = plt.imread(img_path) # Shape: (1080, 1440, 3)
    return img

def load_adc(adc_path: str) -> Dict[str, Any]:
    return spio.loadmat(adc_path)["adcData"] # Shape: (128, 255, 4, 2)

def load_label(label_path: str) -> List[dict]:
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            labels.append({
                "track_id": str(parts[0]),
                "class_id": int(float(parts[1])),
                "category": LABELMAP.get(int(float(parts[1])), "unknown"),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "w": float(parts[4]),
                "l": float(parts[5]),
            })

    return labels

def load_rec_data(data_dir: str, rec: str) -> pd.DataFrame:
    label_file_path = os.path.join(data_dir, 'ANNO_RA', f'{rec}.txt')
    adc_paths = glob.glob(os.path.join(data_dir, rec, 'radar_raw_frame', "*.mat"))
    timestamps = [int(os.path.basename(p).split('.')[0]) for p in adc_paths]


    table = pd.read_csv(label_file_path, 
                sep=r"\s+",   # split on any whitespace
                header=None,  # no header in your file
                names=["timestamp", "range", "azimuth", "class"])

    joint_timestamps = set(timestamps).intersection(set(table['timestamp']))

    table = table[table['timestamp'].isin(joint_timestamps)]

    # Filter adc_paths to only those in joint_timestamps
    adc_paths = [p for p in adc_paths if int(os.path.basename(p).split('.')[0]) in joint_timestamps]

    table = table.groupby('timestamp').agg({
        'range': list,
        'azimuth': list,
        'class': list
        }).reset_index()

    table['adc_path'] = adc_paths
    return table
