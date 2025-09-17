import os
import glob
from typing import Dict, List, Tuple, Any, Literal, Optional
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


CRUW_REC: Dict[str, List[str]] = {
    "2019_04_09": [
        '2019_04_09_BMS1000',
        '2019_04_09_BMS1001',
        '2019_04_09_BMS1002',
        '2019_04_09_CMS1002',
        '2019_04_09_PMS1000',
        '2019_04_09_PMS1001',
        '2019_04_09_PMS2000',
        '2019_04_09_PMS3000'
    ],
    "2019_04_30": [
        '2019_04_30_MLMS000',
        '2019_04_30_MLMS001',
        '2019_04_30_MLMS002',
        '2019_04_30_PBMS002',
        '2019_04_30_PBMS003',
        '2019_04_30_PCMS001',
        '2019_04_30_PM2S003',
        '2019_04_30_PM2S004'
    ],
    "2019_05_09": [
        '2019_05_09_BM1S008',
        '2019_05_09_CM1S004',
        '2019_05_09_MLMS003',
        '2019_05_09_PBMS004',
        '2019_05_09_PCMS002'
    ],
    "2019_05_23": [
        '2019_05_23_PM1S012',
        '2019_05_23_PM1S013',
        '2019_05_23_PM1S014',
        '2019_05_23_PM1S015',
        '2019_05_23_PM2S011'
    ],
    "2019_05_29": [
        '2019_05_29_BCMS000',
        '2019_05_29_BM1S016',
        '2019_05_29_BM1S017',
        '2019_05_29_MLMS006',
        '2019_05_29_PBMS007',
        '2019_05_29_PCMS005',
        '2019_05_29_PM2S015',
        '2019_05_29_PM3S000'
    ],
    "2019_09_29": [
        '2019_09_29_ONRD001',
        '2019_09_29_ONRD002',
        '2019_09_29_ONRD005',
        '2019_09_29_ONRD006',
        '2019_09_29_ONRD011',
        '2019_09_29_ONRD013'
    ]
}


RAMPCNN_REC: Dict[str, List[str]] = {
    "2019_04_09": [
        '2019_04_09_bms1000',
        # '2019_04_09_cms1000',
        # '2019_04_09_css1000',
        '2019_04_09_pms1000',
        '2019_04_09_pms2000',
        '2019_04_09_pms3000'
    ],
    "2019_04_30": [
        '2019_04_30_cm1s000',
        '2019_04_30_mlms000',
        '2019_04_30_mlms001',
        '2019_04_30_pbms002',
        '2019_04_30_pbss000',
        '2019_04_30_pcms001'
    ],
    "2019_05_09": [
        '2019_05_09_bm1s007',
        '2019_05_09_cm1s003',
        '2019_05_09_mlms003',
        '2019_05_09_pbms004',
        '2019_05_09_pcms002'
    ],
    "2019_05_29": [
        '2019_05_29_bcms000',
        '2019_05_29_cm1s014',
        '2019_05_29_mlms006',
        '2019_05_29_pbms007',
        '2019_05_29_pcms005'
    ]
}

@dataclass
class RadarConfig:
    numRangeBins: int = 128
    numDopplerBins: int = 128
    numAngleBins: int = 128

    resRange: float = 0.1
    resDoppler: float = 0.1
    resAngle: float = 0.1

    maxRange: float = 25.0
    maxVelocity: float = 5.0
    maxAngle: float = 90.0

    frameRate: float = 30.0
    crop_range_bins: int = 3

    chirpSlope: float = 21.0017e12
    startFrequency: float = 76.999999488e9 # center
    adcStartTime: float = 0.0
    chirpIdleTime: float = 5.0e-4

    numChirpsPerFrame: int = 255
    numSamplesPerChirp: int = 128
    adcSampleRate: int = 400000
    numVirtualAntennas: int = 8

@dataclass
class ConfMapConfig:
    classes: List[str] = field(default_factory=lambda: ['pedestrian', 'cyclist', 'car'])
    confmap_sigmas: Dict[str, float] = field(default_factory=lambda: {
        'pedestrian': 15, 
        'cyclist': 20, 
        'car': 30, 
        'van': 40, 
        'truck': 50
    })
    confmap_sigmas_interval: Dict[str, List[float]] = field(default_factory=lambda: {
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        'van': [15, 40],
        'truck': [20, 50],
    })
    confmap_length: Dict[str, int] = field(default_factory=lambda: {
        'pedestrian': 1, 
        'cyclist': 2, 
        'car': 3, 
        'van': 4, 
        'truck': 5
    })
    gaussian_thres: float = 36

    @property
    def n_class(self) -> int:
        return len(self.classes)


