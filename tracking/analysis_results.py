import os
import sys
import time

import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm



env_path = os.path.join(os.path.dirname(__file__), '../lib')
if env_path not in sys.path:
    sys.path.append(env_path)
sys.path.append('.')

from lib.test.evaluation import trackerlist, get_dataset
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.utils.load_text import load_text


dataset_name = 'uav'

trackers = []

trackers.extend(trackerlist(name='ecttrack', parameter_name='ecttrack_256_128x1_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='ecttrack'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True)