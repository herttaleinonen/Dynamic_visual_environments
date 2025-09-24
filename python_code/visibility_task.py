#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 5 12:34:21 2025

@author: herttaleinonen

    Run visibility mapping and record EyeLink only during the stimulus window.
    esponses are keyboard-only (RIGHT=target, LEFT=distractor).

"""
import os
import random
import math
import csv
import numpy as np
from psychopy import visual, core, event

def run_visibility_trials(win, el_tracker, screen_width, screen_height,
                          participant_id, timestamp):
    
    # Pixel dimensions for the window
    screen_width_pix  = screen_width
    screen_height_pix = screen_height

    # Output path
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"visibility_{participant_id}_{timestamp}.csv")

    # ---- Params from config (safe fallbacks if missing) ----
    try:
        from config import (
            cell_size,            # px/deg (1 cell ≈ 1 deg)
            target_orientation,   # e.g., 45
            target_sf,            # cycles/pixel
            num_trials            # total trials desired
        )
    except Exception:
        cell_size = 35.0
        target_orientation = 45
        target_sf = 1.0 / (cell_size / 2.0)  # ~2 cycles per cell
        num_trials = 150

    px_per_deg   = float(cell_size)
    sf_cpd       = float(target_sf) * px_per_deg  # cycles/deg
    gabor_size_d = 1.0  # ≈ one deg

    # Core task parameters
    distances_deg     = [3, 6, 12, 20]
    orientation_diffs = [-45, -25, 0, 25, 45]
    fixation_duration = 0.5
    stimulus_duration = 0.2
    noise_grain       = 3  # px

    # ---- Build balanced trials (blocks of all ecc×ori diffs) ----
    combos = []
    for ecc in distances_deg:
        for diff in orientation_diffs:
            combos.append({
                'ecc':  ecc,
                'ori':  target_orientation + diff,
                'type': ('target' if diff == 0 else 'distractor'),
            })
    conds_per_block = len(combos)  # 20
    full_blocks, remainder = divmod(int(num_trials), conds_per_block)

    trials = []
    rng = random.Random()
    for _ in range(full_blocks):
        block = combos[:]
        rng.shuffle(block)
        trials.extend(block)
    if remainder:
        partial = combos[:]
        rng.shuffle(partial)
        trials.extend(partial[:remainder])

    # ---- Stimuli ----
    fixation = visual.TextStim(win, text='+', height=1, color='black', units='deg')

    gabor = visual.GratingStim(
        win,
        tex='sin', mask='gauss',
        size=gabor_size_d,
        sf=sf_cpd,                    # cycles/deg
        ori=0,
        units='deg',
        phase=0.25,
        contrast=1.0
    )

    def generate_noise():
        h_grains = int(math.ceil(screen_height_pix / noise_grain))
        w_grains = int(math.ceil(screen_width_pix  / noise_grain))
        small = np.clip(np.random.normal(0, 0.3, (h_grains, w_grains)), -1, 1)
        noise = np.repeat(np.repeat(small, noise_grain, axis=0), noise_grain, axis=1)
        return noise[:screen_height_pix, :screen_width_pix]

    # ---- Instruction screen with examples (unchanged UI) ----
    inst = visual.TextStim(
        win,
        text=("In this task, a single Gabor object will briefly appear on static noise at different eccentricities.\n"
              "If it is the TARGET (45° tilt), press RIGHT ARROW.\n"
              "If it is a DISTRACTOR (any other tilt), or you did not see the object, press LEFT ARROW.\n"
              "Try to respond quickly and accurately.\n"
              "Between trials there is a cross in the middle of the screen, try to focus your eyes there.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=screen_width_pix * 0.85, units='pix',
        pos=(0, screen_height_pix * 0.24)
    )

    distractor_oris = [target_orientation + d for d in orientation_diffs if d != 0]

    row_y_deg  = -2.0
    gap_deg    = 1.8
    extra_deg  = 5.0
    n_d = len(distractor_oris)
    row_w_deg   = n_d * gabor_size_d + (n_d - 1) * gap_deg
    start_x_deg = -row_w_deg / 2 + gabor_size_d / 2

    example_stims = []
    for i, ori in enumerate(distractor_oris):
        g_ex = visual.GratingStim(
            win, tex='sin', mask='gauss', size=gabor_size_d,
            sf=gabor.sf, ori=ori, phase=0.25, units='deg', contrast=1.0
        )
        g_ex.pos = (start_x_deg + i * (gabor_size_d + gap_deg), row_y_deg)
        example_stims.append(g_ex)

    tgt_x = start_x_deg + (n_d - 1) * (gabor_size_d + gap_deg) + gabor_size_d/2 + extra_deg
    tgt_ex = visual.GratingStim(
        win, tex='sin', mask='gauss', size=gabor_size_d,
        sf=gabor.sf, ori=target_orientation, phase=0.25, units='deg', contrast=1.0
    )
    tgt_ex.pos = (tgt_x, row_y_deg)
    example_stims.append(tgt_ex)

    lab_d = visual.TextStim(
        win, text="Distractors", color='white', height=0.8, units='deg',
        pos=((start_x_deg + (start_x_deg + (n_d - 1) * (gabor_size_d + gap_deg))) / 2.0,
             row_y_deg - 1.8)
    )
    lab_t = visual.TextStim(
        win, text="Target (45°)", color='white', height=0.8, units='deg',
        pos=(tgt_x, row_y_deg - 1.8)
    )

    inst.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
    win.flip()
    event.waitKeys(keyList=['return', 'enter'])
    event.clearEvents(eventType='keyboard')

    # ---------------------------- Run trials ----------------------------
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'trial', 'ecc_deg', 'angle_deg', 'orientation_deg',
            'stim_type', 'x_pos_deg', 'y_pos_deg',
            'response', 'correct', 'rt'
        ])

        for i, tr in enumerate(trials, start=1):
            # New static noise per trial
            noise_img = generate_noise()
            noise_stim = visual.ImageStim(
                win, image=noise_img,
                size=(screen_width_pix, screen_height_pix),
                units='pix', interpolate=False
            )

            # 1) Fixation
            noise_stim.draw()
            fixation.draw()
            win.flip()
            core.wait(fixation_duration)

            # 2) Random polar position at desired eccentricity
            angle = random.uniform(0, 360)
            x_deg = tr['ecc'] * math.cos(math.radians(angle))
            y_deg = tr['ecc'] * math.sin(math.radians(angle))

            # 3) Stimulus on noise — EyeLink recording strictly over the stimulus window
            clock = core.Clock()

            if el_tracker:
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {i}')
                el_tracker.startRecording(1, 1, 1, 1)
                core.wait(0.1)

            first_frame = True
            while clock.getTime() < stimulus_duration:
                noise_stim.draw()
                fixation.draw()
                gabor.ori = tr['ori']
                gabor.pos = (x_deg, y_deg)
                gabor.draw()
                win.flip()

                if first_frame and el_tracker:
                    el_tracker.sendMessage('stimulus_onset')
                    first_frame = False

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # 4) Response screen (keyboard only)
            noise_stim.draw()
            question = visual.TextStim(win, text='?', height=1, color='black', units='deg')
            question.draw()
            win.flip()

            event.clearEvents(eventType='keyboard')
            response = None
            rt = None
            while response is None:
                keys = event.waitKeys(keyList=['left', 'right', 'escape'], timeStamped=clock)
                if not keys:
                    continue
                key, key_time = keys[0]
                if key == 'escape':
                    return filename
                response = 'target' if key == 'right' else 'distractor'
                rt = key_time

            correct = int(response == tr['type'])
            writer.writerow([
                i, tr['ecc'], round(angle, 2), tr['ori'],
                tr['type'], round(x_deg, 2), round(y_deg, 2),
                response, correct, round(rt, 4)
            ])

    return filename
