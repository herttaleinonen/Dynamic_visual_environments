#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:09:28 2025

@author: herttaleinonen
"""

# config.py
# Contains parameters for the experiments


import numpy as np
from datetime import datetime
import os

# ========== Grid Settings ==========
grid_size_x = 35
grid_size_y = 35
cell_size = 35
DIAGONAL_SCALE = round(4 / np.sqrt(2), 2)

# ========== Task & Timing Settings ==========
num_trials = 150
trial_duration = 5 # 5,000ms
feedback_duration = 1
timeout_feedback_text = "Too Slow! Try to respond faster!"
transition_steps = 8 # 8 = 525.0 px/s
movement_delay = 1.0 / 30.0 # divided with the frame rate 


# ========== Stimulus Settings ==========
orientations = (170, 80, 20, 120, 135) 
spatial_frequencies = [1 / (cell_size / 2)]
target_orientation = 45
target_sf = 1 / (cell_size / 2)

# ========== Output Settings ==========
exp_info = {"Participant ID": ""}
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
