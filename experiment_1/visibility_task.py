#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 5 12:34:21 2025

@author: herttaleinonen

    Run visibility mapping and record EyeLink during the stimulus window.
    Responses are right/left arrow on keyboard or keys 1 and 3 on Cedrus.
"""

import os
import random
import math
import csv
import numpy as np
from psychopy import visual, core, event

# ---------- Optional Cedrus support ----------
try:
    import pyxid2
except Exception:
    pyxid2 = None

def _cedrus_open():
    """Return first Cedrus device or None. Never raises."""
    if pyxid2 is None:
        return None
    try:
        devs = pyxid2.get_xid_devices()
        if not devs:
            return None
        dev = devs[0]
        if hasattr(dev, "reset_timer"):
            dev.reset_timer()
        else:
            if hasattr(dev, "reset_base_timer"): dev.reset_base_timer()
            if hasattr(dev, "reset_rt_timer"):   dev.reset_rt_timer()
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return dev
    except Exception as e:
        print(f"[Cedrus] init failed: {e}")
        return None

def _cedrus_flush(dev, dur=0.12):
    """Drain any pending Cedrus events for ~dur seconds."""
    if not dev:
        return
    try:
        t0 = core.getTime()
        while (core.getTime() - t0) < dur:
            if hasattr(dev, "poll_for_response"):
                dev.poll_for_response()
            while dev.has_response():
                dev.get_next_response()
            core.wait(0.005)
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
    except Exception as e:
        print(f"[Cedrus] flush failed: {e}")

def _cedrus_any_pressed(dev):
    """True if *any* Cedrus key press event is available (drains one batch)."""
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        any_pressed = False
        while dev.has_response():
            r = dev.get_next_response()
            if r and r.get("pressed", False):
                any_pressed = True
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return any_pressed
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False

def _cedrus_get_choice(dev):
    """
    Returns 'target' for GREEN (key 3), 'distractor' for RED (key 1), or None.
    """
    if not dev:
        return None
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        choice = None
        while dev.has_response():
            r = dev.get_next_response()
            if not r or not r.get("pressed", False):
                continue
            k = r.get("key", None)
            if k == 3:
                choice = 'target'
            elif k == 1:
                choice = 'distractor'
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return choice
    except Exception as e:
        print(f"[Cedrus] read failed: {e}")
        return None
    
# --------------------------------------------

def run_visibility_trials(win, el_tracker, screen_width, screen_height,
                          participant_id, timestamp):
    screen_width_pix  = screen_width
    screen_height_pix = screen_height

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"visibility_{participant_id}_{timestamp}.csv")

    try:
        from config import (
            cell_size, target_orientation, target_sf, num_trials
        )
    except Exception:
        cell_size = 35.0
        target_orientation = 45
        target_sf = 1.0 / (cell_size / 2.0)
        num_trials = 150

    px_per_deg   = float(cell_size)
    sf_cpd       = float(target_sf) * px_per_deg
    gabor_size_d = 1.0
    # Gabir stimulus
    distances_deg     = [3, 6, 12, 20] # distance from the center of the screen (in degrees)
    orientation_diffs = [-45, -25, 0, 25, 45] # difference in orientation in comparison to the target (in degrees)
    fixation_duration = 0.5 # fixation cross duration between trials in s 
    stimulus_duration = 0.2 # stimulus duration in s
    noise_grain       = 3 # size of the background noise (pixel x pixel)
    
    # create ori + distance pairs
    combos = []
    for ecc in distances_deg:
        for diff in orientation_diffs:
            combos.append({
                'ecc':  ecc,
                'ori':  target_orientation + diff,
                'type': ('target' if diff == 0 else 'distractor'),
            })
    conds_per_block = len(combos)
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

    # write to console if cedrus is available
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. GREEN(key 3)=Yes, RED(key 1)=No.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    fixation = visual.TextStim(win, text='+', height=1, color='black', units='deg')
    
    gabor = visual.GratingStim(
        win, tex='sin', mask='gauss', size=gabor_size_d,
        sf=sf_cpd, ori=0, units='deg', phase=0.25, contrast=1.0
    )

    # Initialize background noise
    def generate_noise():
        h_grains = int(math.ceil(screen_height_pix / noise_grain))
        w_grains = int(math.ceil(screen_width_pix  / noise_grain))
        small = np.clip(np.random.normal(0, 0.3, (h_grains, w_grains)), -1, 1)
        noise = np.repeat(np.repeat(small, noise_grain, axis=0), noise_grain, axis=1)
        return noise[:screen_height_pix, :screen_width_pix]
    
    # Initialize instruction screen
    inst = visual.TextStim(
        win,
        text=("In this task, a single object will briefly appear on static noise at different eccentricities.\n"
              "If it is the TARGET object (45° tilt), press GREEN button.\n"
              "If it is a DISTRACTOR object (any other tilt), or you did not see the object, press RED button.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start." ),
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
    
    # Draw example Gabors to the instruction screen
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

    # Check if cedrus/keyboard response has been registered
    if cedrus:
        while True:
            if _cedrus_any_pressed(cedrus):
                break
            keys = event.getKeys(keyList=['return', 'enter', 'escape'])
            if 'escape' in keys:
                return filename
            if keys:
                break
            core.wait(0.01)
        _cedrus_flush(cedrus)
    else:
        event.waitKeys(keyList=['return', 'enter'])
    event.clearEvents(eventType='keyboard')
    
    # measure fixation accutracy during the fixation cross and write to csv 
    def measure_fixation_drift(trial_idx, duration=0.5, bg=None):
        if not el_tracker:
            t0 = core.Clock()
            while t0.getTime() < duration:
                if bg: bg.draw()
                fixation.draw()
                win.flip()
            return ""
        try:
            el_tracker.setOfflineMode()
            el_tracker.sendMessage(f'FIXCHECK_START {trial_idx}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)
        except Exception:
            t0 = core.Clock()
            while t0.getTime() < duration:
                if bg: bg.draw()
                fixation.draw()
                win.flip()
            return ""
        samples = []
        clk_fix = core.Clock()
        while clk_fix.getTime() < duration:
            if bg: bg.draw()
            fixation.draw()
            win.flip()
            s = el_tracker.getNewestSample()
            eye = s.getRightEye() if (s and s.isRightSample()) else (s.getLeftEye() if (s and s.isLeftSample()) else None)
            if eye is not None:
                rx, ry = eye.getGaze()
                if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                    gx = float(rx) - (screen_width_pix / 2.0)
                    gy = (screen_height_pix / 2.0) - float(ry)
                    dist_deg = float(np.hypot(gx, gy)) / float(px_per_deg)
                    samples.append(dist_deg)
            core.wait(0.005)
        try:
            el_tracker.sendMessage('FIXCHECK_END')
            el_tracker.stopRecording()
        except Exception:
            pass
        return round(float(np.median(samples)), 3) if samples else ""

    # ---------------------------- Run trials ----------------------------
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'trial', 'ecc_deg', 'angle_deg', 'orientation_deg',
            'stim_type', 'x_pos_deg', 'y_pos_deg',
            'response', 'correct', 'rt', 'CalibrationDrift(deg)'
        ])

        for i, tr in enumerate(trials, start=1):
            # ground-truth as numeric (1=target, 0=distractor)
            stim_num = 1 if tr['type'] == 'target' else 0

            noise_img = generate_noise()
            noise_stim = visual.ImageStim(
                win, image=noise_img,
                size=(screen_width_pix, screen_height_pix),
                units='pix', interpolate=False
            )

            # 1) Fixation + drift
            drift_deg = measure_fixation_drift(i, duration=fixation_duration, bg=noise_stim)
            event.clearEvents(eventType='keyboard')
            if cedrus:
                _cedrus_flush(cedrus)

            # 2) Stim position
            angle = random.uniform(0, 360)
            x_deg = tr['ecc'] * math.cos(math.radians(angle))
            y_deg = tr['ecc'] * math.sin(math.radians(angle))

            # 3) Stimulus (record EyeLink only during stimulus)
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

            # 4) Response screen (Cedrus or keyboard), RT on same clock
            noise_stim.draw()
            visual.TextStim(win, text='?', height=1, color='black', units='deg').draw()
            win.flip()

            if cedrus:
                _cedrus_flush(cedrus)

            response = None
            rt = None
            while response is None:
                if cedrus:
                    choice = _cedrus_get_choice(cedrus)
                    if choice in ('target', 'distractor'):
                        response = choice
                        rt = clock.getTime()
                        break
                keys = event.getKeys(keyList=['left', 'right', 'escape'], timeStamped=clock)
                if keys:
                    key, key_time = keys[0]
                    if key == 'escape':
                        return filename
                    response = 'target' if key == 'right' else 'distractor'
                    rt = key_time
                    break
                core.wait(0.005)

            # numeric response & correctness
            resp_num = 1 if response == 'target' else 0
            correct  = int(resp_num == stim_num)

            writer.writerow([
                i, tr['ecc'], round(angle, 2), tr['ori'],
                stim_num, round(x_deg, 2), round(y_deg, 2),
                resp_num, correct, round(rt, 4), drift_deg
            ])

    return filename
