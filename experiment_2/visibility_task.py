#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:34:21 2025

@author: herttaleinonen


Run visibility task and record EyeLink during the stimulus window (400ms).
Responses are right/left arrow on keyboard or keys 1 and 3 on Cedrus.

    UPDATED:
    - Dynamic Gaussian noise background during stimulus (gabor/motion window).
    - Static noise during fixation/drift and during response.
    
    - Eccentricities are: 3, 6, 12, 16, 20 deg.
    - No eccentricity clamping. Instead, gabor is auto-scaled per trial to prevent clipping.
"""

import os
import random
import math
import csv
import numpy as np
from psychopy import visual, core, event


# --- debug overlay toggle helper ---
DEBUG_ECC_OVERLAY = False
DEBUG_TOGGLE_KEY  = 'd'

def _maybe_toggle_debug():
    """Press 'd' to toggle eccentricity overlay on/off during testing."""
    global DEBUG_ECC_OVERLAY
    keys = event.getKeys(keyList=[DEBUG_TOGGLE_KEY])
    if keys:
        DEBUG_ECC_OVERLAY = not DEBUG_ECC_OVERLAY
        print(f"[DEBUG] Eccentricity overlay {'ON' if DEBUG_ECC_OVERLAY else 'OFF'}")


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
    screen_width_pix  = int(screen_width)
    screen_height_pix = int(screen_height)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"visibility_{participant_id}_{timestamp}.csv")

    try:
        from config import (
            cell_size, target_orientation, target_sf, num_trials, orientations
        )
    except Exception:
        cell_size = 35.0
        target_orientation = 45
        target_sf = 1.0 / (cell_size / 2.0)
        num_trials = 150
        orientations = (170, 135, 120, 80, 20)

    px_per_deg   = float(cell_size)

    distances_deg = [3, 6, 12, 16, 20]

    distractor_oris = list(orientations)

    fixation_duration = 0.5  # s
    stimulus_duration = 0.4  # 400 ms flash
    noise_grain       = 3    # px

    # Dynamic noise speed control (only applied during stimulus)
    NOISE_UPDATE_INTERVAL = 4  # ~15 Hz updates on a 60 Hz monitor

    def deg2px(x_deg, y_deg, px_per_deg):
        return (x_deg * px_per_deg, y_deg * px_per_deg)

    # ---- exact 50/50 target vs distractor schedule ----
    n_present = int(num_trials) // 2
    present_schedule = [1] * n_present + [0] * (int(num_trials) - n_present)
    random.shuffle(present_schedule)

    trials = []
    for tp in present_schedule:
        ecc = random.choice(distances_deg)
        if tp:
            ori = int(target_orientation)
            stim_type = 'target'
        else:
            ori = int(random.choice(distractor_oris))
            stim_type = 'distractor'
        trials.append({'ecc': ecc, 'ori': ori, 'type': stim_type})

    # Cedrus
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. GREEN(key 3)=Yes, RED(key 1)=No.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    fixation = visual.TextStim(win, text='+', height=1, color='black', units='deg')

    # Base gabor (size will be auto-scaled per trial if needed)
    gabor = visual.GratingStim(
        win, tex='sin', mask='gauss',
        size=int(cell_size),
        sf=target_sf,
        ori=0,
        units='pix',
        phase=0.25, contrast=1.0
    )

    ecc_circle_intended = visual.Circle(
        win, radius=1.0, edges=180, units='deg',
        lineColor='white', fillColor=None, lineWidth=1.5, opacity=0.18
    )
    ecc_circle_actual = visual.Circle(
        win, radius=1.0, edges=180, units='deg',
        lineColor='white', fillColor=None, lineWidth=1.5, opacity=0.35
    )

    # ---------------- Noise bank (static except during stimulus) ----------------
    def generate_noise(screen_width, screen_height, grain_size=noise_grain):
        h_grains = int(np.ceil(screen_height / grain_size))
        w_grains = int(np.ceil(screen_width / grain_size))
        small = np.random.normal(loc=0, scale=0.3, size=(h_grains, w_grains))
        small = np.clip(small, -1, 1)
        noise = np.repeat(np.repeat(small, grain_size, axis=0), grain_size, axis=1)
        return noise[:screen_height, :screen_width]

    noise_bank = [
        generate_noise(screen_width_pix, screen_height_pix, grain_size=noise_grain)
        for _ in range(30)
    ]

    noise_frames = [
        visual.ImageStim(
            win, image=img,
            size=(screen_width_pix, screen_height_pix),
            units="pix", interpolate=False
        )
        for img in noise_bank
    ]

    bank_i = 0
    noise_frame_counter = 0

    def draw_static_noise():
        """Draw the current noise frame (no animation)."""
        noise_frames[bank_i].draw()

    def draw_dynamic_noise():
        """
        Draw dynamic noise BUT slow it down by updating the bank index
        only every NOISE_UPDATE_INTERVAL frames.
        Called during the gabor flash.
        """
        nonlocal bank_i, noise_frame_counter
        noise_frames[bank_i].draw()
        noise_frame_counter += 1
        if noise_frame_counter >= NOISE_UPDATE_INTERVAL:
            bank_i = (bank_i + 1) % len(noise_frames)
            noise_frame_counter = 0
    # ---------------------------------------------------------------------------

    # Instructions (STATIC noise)
    inst = visual.TextStim(
        win,
        text=("In this task, a single object will briefly appear on noise at different eccentricities.\n"
              "If it is the TARGET object (45° tilt), press GREEN button.\n"
              "If it is a DISTRACTOR object (any other tilt), or you did not see the object, press RED button.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start."),
        color='white', height=30, wrapWidth=screen_width_pix * 0.85, units='pix',
        pos=(0, screen_height_pix * 0.24)
    )

    # Example row
    row_y_deg  = -2.0
    gap_deg    = 1.8
    extra_deg  = 5.0

    example_distractors = list(distractor_oris)
    n_d = len(example_distractors)
    row_w_deg   = n_d * (cell_size/px_per_deg) + (n_d - 1) * gap_deg
    start_x_deg = -row_w_deg / 2 + (cell_size/px_per_deg) / 2

    example_stims = []
    for j, ori in enumerate(example_distractors):
        g_ex = visual.GratingStim(
            win, tex='sin', mask='gauss',
            size=int(cell_size),
            sf=target_sf,
            ori=ori, phase=0.25, units='pix', contrast=1.0
        )
        x_deg_j = start_x_deg + j * ((cell_size/px_per_deg) + gap_deg)
        g_ex.pos = deg2px(x_deg_j, row_y_deg, px_per_deg)
        example_stims.append(g_ex)

    tgt_x_deg = start_x_deg + (n_d - 1) * ((cell_size/px_per_deg) + gap_deg) + (cell_size/px_per_deg)/2 + extra_deg
    tgt_ex = visual.GratingStim(
        win, tex='sin', mask='gauss',
        size=int(cell_size),
        sf=target_sf,
        ori=int(target_orientation), phase=0.25, units='pix', contrast=1.0
    )
    tgt_ex.pos = deg2px(tgt_x_deg, row_y_deg, px_per_deg)
    example_stims.append(tgt_ex)

    lab_d = visual.TextStim(
        win, text="Distractors", color='white', height=0.8, units='deg',
        pos=((start_x_deg + (start_x_deg + (n_d - 1) * ((cell_size/px_per_deg) + gap_deg))) / 2.0,
             row_y_deg - 1.8)
    )
    lab_t = visual.TextStim(
        win, text="Target (45°)", color='white', height=0.8, units='deg',
        pos=(tgt_x_deg, row_y_deg - 1.8)
    )

    draw_static_noise()
    inst.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
    win.flip()

    # Start key
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

    # fixation drift check during fixation period
    def measure_fixation_drift(trial_idx, duration=0.5):
        if not el_tracker:
            t0 = core.Clock()
            while t0.getTime() < duration:
                draw_static_noise()
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
                draw_static_noise()
                fixation.draw()
                win.flip()
            return ""
        samples = []
        clk_fix = core.Clock()
        while clk_fix.getTime() < duration:
            draw_static_noise()
            fixation.draw()
            _maybe_toggle_debug()
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
            stim_num = 1 if tr['type'] == 'target' else 0

            # 1) Fixation + drift (static noise)
            drift_deg = measure_fixation_drift(i, duration=fixation_duration)
            event.clearEvents(eventType='keyboard')
            if cedrus:
                _cedrus_flush(cedrus)

            # 2) Intended ecc and random polar angle 
            angle = random.uniform(0, 360)
            rad = math.radians(angle)
            ecc_intended = float(tr['ecc'])
            ecc_actual = ecc_intended  # stays exact

            x_deg = ecc_actual * math.cos(rad)
            y_deg = ecc_actual * math.sin(rad)

            # ----------------- AUTO-SCALE GABOR TO PREVENT CLIPPING -----------------
            ecc_px = ecc_actual * px_per_deg
            stim_px_base = float(cell_size)
            half_stim_px_base = stim_px_base / 2.0

            half_w = screen_width_pix / 2.0
            half_h = screen_height_pix / 2.0

            cosr = abs(math.cos(rad))
            sinr = abs(math.sin(rad))
            eps = 1e-6

            max_r_x = (half_w - 1) / max(cosr, eps)
            max_r_y = (half_h - 1) / max(sinr, eps)
            max_r   = min(max_r_x, max_r_y)

            allowed_half_size_px = max_r - ecc_px
            allowed_half_size_px = max(allowed_half_size_px, 5.0)

            scale_factor = min(1.0, allowed_half_size_px / half_stim_px_base)
            stim_px_actual = stim_px_base * scale_factor

            gabor.size = stim_px_actual
            # -----------------------------------------------------------------------

            # 3) Stimulus (EyeLink only during stimulus + dynamic noise)
            clock = core.Clock()
            if el_tracker:
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {i}')
                el_tracker.startRecording(1, 1, 1, 1)
                core.wait(0.1)

            first_frame = True
            while clock.getTime() < stimulus_duration:
                draw_dynamic_noise()
                fixation.draw()
                gabor.ori = tr['ori']
                gabor.pos = deg2px(x_deg, y_deg, px_per_deg)
                gabor.draw()

                if DEBUG_ECC_OVERLAY:
                    ecc_circle_intended.radius = float(ecc_intended)
                    ecc_circle_intended.pos = (0, 0)
                    ecc_circle_intended.draw()
                    ecc_circle_actual.radius = float(ecc_actual)
                    ecc_circle_actual.pos = (0, 0)
                    ecc_circle_actual.draw()

                _maybe_toggle_debug()
                win.flip()

                if first_frame and el_tracker:
                    el_tracker.sendMessage('stimulus_onset')
                    first_frame = False

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # 4) Response screen (STATIC noise)
            draw_static_noise()
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

            resp_num = 1 if response == 'target' else 0
            correct  = int(resp_num == stim_num)

            writer.writerow([
                i, ecc_intended, round(angle, 2), tr['ori'],
                stim_num, round(x_deg, 2), round(y_deg, 2),
                resp_num, correct, round(rt, 4), drift_deg
            ])

    return filename
