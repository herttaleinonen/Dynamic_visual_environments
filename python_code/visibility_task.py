# visibility_task.py
# -*- coding: utf-8 -*-

"""
Created on Jul 30 12:20:11 2025

@author: herttaleinonen

Provides run_visibility_trials() for calling from main.py after EyeLink setup.
Generates a visibility map by presenting single Gabor stimuli on a static Gaussian noise background,
collects responses via arrow keys, logs results to CSV.
Preserves all original stimulus parameters and response logic.
"""
import os
import random
import math
import csv
import numpy as np
from psychopy import visual, core, event


def run_visibility_trials(win, el_tracker, screen_width, screen_height,
                          participant_id, timestamp):
    """
    Runs the visibility mapping experiment.

    Parameters:
    - win: PsychoPy Window object (configured for ViewPixx)
    - el_tracker: EyeLink tracker object or None
    - screen_width, screen_height: window size in pixels
    - participant_id, timestamp: ID and time for filename

    Returns:
    - path to the CSV file with results
    """
    # map to legacy names
    screen_width_pix = screen_width
    screen_height_pix = screen_height

    # prepare output file
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(
        output_dir,
        f"visibility_{participant_id}_{timestamp}.csv"
    )

    # stimulus parameters (unchanged)
    distances_deg = [3, 6, 12, 20]
    orientation_diffs = [-45, -25, 0, 25, 45]
    target_orientation = 45
    n_reps = 1  # should be 20 in full version
    fixation_duration = 0.5
    stimulus_duration = 0.2
    gabor_size_deg = 1.0
    noise_grain = 5

    # build trials
    trials = []
    for ecc in distances_deg:
        for diff in orientation_diffs:
            stim_type = 'target' if diff == 0 else 'distractor'
            trials.append({'ecc': ecc,
                           'ori': target_orientation + diff,
                           'type': stim_type})
    trials *= n_reps
    random.shuffle(trials)

    # setup stimuli
    # fixation cross
    fixation = visual.TextStim(win, text='+', height=1, color='black', units='deg')
    # Gabor (no explicit color override)
    gabor = visual.GratingStim(
        win,
        tex='sin', mask='gauss', size=gabor_size_deg,
        sf=1.0, ori=0, units='deg'
    )

    # generate one static noise image per trial
    def generate_noise():
        h_grains = int(math.ceil(screen_height_pix / noise_grain))
        w_grains = int(math.ceil(screen_width_pix  / noise_grain))
        small = np.clip(np.random.normal(0, 0.3, (h_grains, w_grains)), -1, 1)
        noise = np.repeat(np.repeat(small, noise_grain, axis=0), noise_grain, axis=1)
        return noise[:screen_height_pix, :screen_width_pix]

    # run experiment
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'trial', 'ecc_deg', 'angle_deg', 'orientation_deg',
            'stim_type', 'x_pos_deg', 'y_pos_deg',
            'response', 'correct', 'rt'
        ])

        for i, tr in enumerate(trials, start=1):
            # generate new noise for this trial
            noise_img = generate_noise()
            noise_stim = visual.ImageStim(
                win, image=noise_img,
                size=(screen_width_pix, screen_height_pix),
                units='pix', interpolate=False
            )

            # 1) fixation
            noise_stim.draw()
            fixation.draw()
            win.flip()
            core.wait(fixation_duration)

            # 2) select random position on circle
            angle = random.uniform(0, 360)
            x_deg = tr['ecc'] * math.cos(math.radians(angle))
            y_deg = tr['ecc'] * math.sin(math.radians(angle))

            # 3) stimulus display with cross
            clock = core.Clock()
            while clock.getTime() < stimulus_duration:
                noise_stim.draw()
                fixation.draw()
                gabor.ori = tr['ori']
                gabor.pos = (x_deg, y_deg)
                gabor.phase = 0.25
                gabor.draw()
                win.flip()

            # 4) infinite response window with question mark
            noise_stim.draw()
            question = visual.TextStim(win, text='?', height=1, color='black', units='deg')
            question.draw()
            win.flip()
            response = None
            rt = None
            while response is None:
                keys = event.waitKeys(
                    keyList=['left', 'right', 'escape'],
                    timeStamped=clock
                )
                if not keys:
                    continue
                key, key_time = keys[0]
                if key == 'escape':
                    return filename
                response = 'target' if key == 'right' else 'distractor'
                rt = key_time

            # log
            correct = int(response == tr['type'])
            writer.writerow([
                i, tr['ecc'], round(angle, 2), tr['ori'],
                tr['type'], round(x_deg,2), round(y_deg,2),
                response, correct, round(rt,4)
            ])

    return filename
