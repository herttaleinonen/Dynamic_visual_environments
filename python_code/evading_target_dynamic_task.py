#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:15:16 2025

@author: herttaleinonen
"""

import os
import csv
import random
import math
import numpy as np
from psychopy import visual, core, event
from config import (
    grid_size_x, grid_size_y, cell_size, DIAGONAL_SCALE,
    num_trials, trial_duration, feedback_duration, timeout_feedback_text,
    orientations, spatial_frequencies, target_orientation,
    target_sf, transition_steps, movement_delay
)


# -------- Helper functions --------

# Generate Gaussian noise
noise_grain = 3 #pixel x pixel

def generate_noise(screen_width, screen_height, grain_size=noise_grain):
    #how many grains fit vertically/horizontally
    h_grains = int(np.ceil(screen_height / grain_size))
    w_grains = int(np.ceil(screen_width / grain_size))
    
    #generate small map and clip
    small = np.random.normal(loc=0, scale=0.3, size=(h_grains, w_grains))
    small = np.clip(small, -1, 1)
    
    #upsample by block-replication
    noise = np.repeat(np.repeat(small, grain_size, axis=0),
                    grain_size, axis=1)
    
    #crop to exact screen dimensions
    return noise[:screen_height, :screen_width]

# Center the grid
def grid_to_pixel(x, y, offset_x, offset_y, cell_size):
    return (offset_x + x * cell_size, offset_y + y * cell_size)
 
def get_valid_moves(x, y, last_move):

    all_moves = [
        (4, 0), (-4, 0), (0, 4), (0, -4),
        (DIAGONAL_SCALE, DIAGONAL_SCALE), (-DIAGONAL_SCALE, DIAGONAL_SCALE),
        (DIAGONAL_SCALE, -DIAGONAL_SCALE), (-DIAGONAL_SCALE, -DIAGONAL_SCALE),
    ]
    return [
        (dx, dy) for dx, dy in all_moves
        if (dx, dy) != (-last_move[0], -last_move[1])
        and 2 <= x + dx < grid_size_x - 2
        and 2 <= y + dy < grid_size_y - 2
    ]

# ----- TRIAL LOOP -----
def run_evading_target_dynamic_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")
    
    # Center the grid
    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2 
    grid_offset_y = -grid_pixel_height / 2

    # ---------- Gaze / fixation detection params (from static) ----------
    prev_gx, prev_gy   = screen_width/2.0, screen_height/2.0  # EMA state
    ema_alpha          = 0.55
    fix_radius_px      = cell_size            # ~35 px circle
    min_fix_frames     = 4                    # ~65 ms @ 60 Hz
    capture_radius_px  = 1.5 * cell_size      # snap a fixation to a Gabor if within this pixel radius
    min_gaze_sep_px    = 1.25 * cell_size     # avoid landing right under gaze
    holdoff_by_k       = { 1: 0.25, 2: 0.10, 4: 0.00, 8: 0.00 }  # seconds
    appear_delay_s     = 0.5                  # target appears at 500 ms
    # -------------------------------------------------------------------
    
    # Pre-compute a bank of 30 noise frames
    noise_bank = [generate_noise(screen_width, screen_height, grain_size=3)
                  for _ in range(30)]
    
    # Initialize Noise frames
    noise_frames = [
        visual.ImageStim(
            win,
            image=img,
            size=(screen_width, screen_height),
            units="pix",
            interpolate=False
        )
        for img in noise_bank
    ]
    bank_i = 0

    # For CSV: keep a simple nominal speed (same as your earlier formula)
    speed_px_per_sec = (4 * cell_size) / (transition_steps * movement_delay)
    
    # Initialize instructions screen
    instruction_text = visual.TextStim(win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press 'right arrow key ' if you see the target, 'left arrow key' if not.\n"
              "Each trial you have 5 seconds to decide, try to make the decision as fast as possible.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=screen_width * 0.8, units='pix'
    )
    instruction_text.draw(); 
    win.flip(); 
    event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')  # keep buffer clean

    # Open a CSV file for behavioural results
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # original columns preserved; two extras appended at end
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory", "Speed (px/s)",
                         "NBackK", "NBackUsedSeq"])
                         
        # ---- Balanced present/absent & balanced NBackK among present trials ----
        present_ratio = 0.5
        num_present_raw = int(round(num_trials * present_ratio))  # keep 50/50 overall
    
        def make_balanced_k_sequence(n_present):
            """
            Return a length-n_present list with counts for k in {1,2,4,8}
            as equal as mathematically possible. If n_present isn't divisible
            by 4, the remainder is distributed randomly across the set.
            """
            base = n_present // 4
            rem  = n_present % 4
            ks = []
            for k in (1, 2, 4, 8):
                ks.extend([k] * base)
            if rem:
                extras = [1, 2, 4, 8]
                random.shuffle(extras)
                ks.extend(extras[:rem])
            random.shuffle(ks)  # randomize order across present trials
            return ks
    
        # Build trial flags and per-present-trial K sequence
        present_flags = [1] * num_present_raw + [0] * (num_trials - num_present_raw)
        random.shuffle(present_flags)
        k_seq = make_balanced_k_sequence(num_present_raw)
        k_idx = 0  # cursor through k_seq
        # -------------------------------------------------------------------------

        # helper to mask a reveal/jump by re-randomizing all distractors
        def refresh_all_distractors(gabors, exclude_idx=None):
            for j, g in enumerate(gabors):
                if exclude_idx is not None and j == exclude_idx:
                    continue
                # (keep same object, just re-randomize features)
                g.ori = random.choice(orientations)
                g.sf  = random.choice(spatial_frequencies)

        for trial in range(num_trials):
            # 500 ms fixation cross
            fix_cross = visual.TextStim(win, text='+', color='black', height=40)
            fix_cross.draw()
            win.flip()
            core.wait(0.5)
            event.clearEvents(eventType='keyboard')

            # training notice unchanged
            if trial == 3:
                ready_text = visual.TextStim(win, text="Training phase is over! The experiment begins now. Press Enter to continue.",
                                             color="white", height=30)
                ready_text.draw()
                win.flip()
                event.waitKeys(keyList=["return"])
                event.clearEvents(eventType='keyboard')

            # Number of Gabors defined here
            num_gabors = random.choice([10])

            # Balanced present/absent + balanced K for present trials
            tp = bool(present_flags[trial])            # 1 -> True, 0 -> False
            trial_k = (k_seq[k_idx] if tp else '')     # K only for present trials
            if tp:
                k_idx += 1

            target_present = tp  # keep your existing variable name used below

            # Generate the positions (grid coords)
            positions = [(random.randint(2, grid_size_x - 3), random.randint(2, grid_size_y - 3))
                         for _ in range(num_gabors)]

            # Create Gabors (all start as distractors; target revealed later at 0.5 s)
            gabors = []
            for _ in range(num_gabors):
                sf = random.choice(spatial_frequencies)
                ori = random.choice(orientations)
                gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                           sf=sf, ori=ori, phase=0.25, units='pix')
                gabors.append(gabor)

            for i, (x, y) in enumerate(positions):
                gabors[i].pos = grid_to_pixel(x, y, grid_offset_x, grid_offset_y, cell_size)

            # start with NO target; will appear at 0.5 s if target_present
            target_index = None

            current_steps = [random.randint(0, transition_steps // 2) for _ in range(num_gabors)]
            targets = positions[:]
            last_moves = [random.choice([(4, 0), (0, 4), (0, -4),
                                         (DIAGONAL_SCALE, DIAGONAL_SCALE),
                                         (DIAGONAL_SCALE, -DIAGONAL_SCALE)]) for _ in range(num_gabors)]
            for i in range(num_gabors):
                x, y = positions[i]
                valid = get_valid_moves(x, y, last_moves[i])
                move = random.choice(valid) if valid else (0, 0)
                targets[i] = (x + move[0], y + move[1])
                last_moves[i] = move

            trial_clock = core.Clock()
            response = None
            rt = None
            gabor_trajectory = []
            target_trajectory = []  # list of (grid-x, grid-y) per frame for target

            # --- Gaze-driven state (like static) ---
            inspected_indices = []   # distractor indices visited (debounced)
            inspected_times   = []   # timestamps for those visits (trial_clock time)
            nback_used_seq    = []   # NEW: log which k actually fired
            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # NEW: appearance gate
            arm_at = trial_clock.getTime() + appear_delay_s
            armed  = False

            # Put tracker in idle mode before recording
            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')

            # Send message "TRIALID" to mark the start of a trial
            el_tracker.sendMessage(f'TRIALID {trial + 1}')
            
            # Start recording
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)

            # Log a message to mark the onset of the stimulus
            el_tracker.sendMessage('stimulus_onset')
            
            while trial_clock.getTime() < trial_duration:
                now_t = trial_clock.getTime()

                # --- REVEAL target at 0.5 s (only if target-present) ---
                if target_present and (not armed) and (now_t >= arm_at):
                    target_index = random.randrange(num_gabors)
                    gabors[target_index].ori = target_orientation
                    gabors[target_index].sf  = target_sf
                    # optional: mask reveal by refreshing all distractors' features
                    refresh_all_distractors(gabors, exclude_idx=target_index)
                    armed = True

                # update noise
                noise_frames[bank_i].draw()
                bank_i = (bank_i + 1) % len(noise_frames)

                # --- update random-walk targets & interpolate positions ---
                frame_positions = []
                for i in range(num_gabors):
                    if current_steps[i] >= transition_steps:
                        x, y = positions[i]
                        valid = get_valid_moves(x, y, last_moves[i])
                        move = random.choice(valid) if valid else (0, 0)
                        targets[i] = (x + move[0], y + move[1])
                        last_moves[i] = move
                        current_steps[i] = 0

                    t = current_steps[i] / transition_steps
                    interp_x = positions[i][0] + (targets[i][0] - positions[i][0]) * t
                    interp_y = positions[i][1] + (targets[i][1] - positions[i][1]) * t
                    gabors[i].pos = grid_to_pixel(interp_x, interp_y, grid_offset_x, grid_offset_y, cell_size)
                    frame_positions.append((interp_x, interp_y))

                    current_steps[i] += 1
                    if current_steps[i] >= transition_steps:
                        positions[i] = targets[i]

                gabor_trajectory.append([(round(x, 2), round(y, 2)) for (x, y) in frame_positions])

                # track target per frame (grid coords like before)
                if target_index is not None:
                    tx, ty = frame_positions[target_index]
                    target_trajectory.append((round(tx, 2), round(ty, 2)))

                # --- read & smooth gaze ---
                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()
                if eye:
                    rx, ry = eye.getGaze()
                    prev_gx = float(np.clip(ema_alpha*rx + (1-ema_alpha)*prev_gx, 0, screen_width-1))
                    prev_gy = float(np.clip(ema_alpha*ry + (1-ema_alpha)*prev_gy, 0, screen_height-1))

                    # fixation clustering
                    dx = prev_gx - cluster_cx
                    dy = prev_gy - cluster_cy
                    inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                    if inside:
                        cluster_len += 1
                        w = 1.0 / max(1, cluster_len)
                        cluster_cx = (1 - w)*cluster_cx + w*prev_gx
                        cluster_cy = (1 - w)*cluster_cy + w*prev_gy

                        if (not committed_this_cluster) and (cluster_len == min_fix_frames):
                            # commit fixation -> map to nearest current Gabor
                            centers_pix = np.array([g.pos for g in gabors], dtype=float)
                            d2 = (centers_pix[:,0] - cluster_cx)**2 + (centers_pix[:,1] - cluster_cy)**2
                            current_idx = None
                            if d2.size > 0:
                                j = int(np.argmin(d2))
                                if math.sqrt(d2[j]) <= capture_radius_px:
                                    current_idx = j

                            # if no capture, impute to nearest distractor (exclude target)
                            if current_idx is None:
                                cand_d2 = d2.copy()
                                if target_index is not None:
                                    cand_d2[target_index] = np.inf
                                j2 = int(np.argmin(cand_d2))
                                current_idx = None if np.isinf(cand_d2[j2]) else j2

                            # record inspection history from trial start (pre- and post-reveal), distractors only
                            if current_idx is not None and (target_index is None or current_idx != target_index):
                                if not inspected_indices or inspected_indices[-1] != current_idx:
                                    inspected_indices.append(current_idx)
                                    inspected_times.append(now_t)

                            # strict n-back retargeting: ONLY after reveal
                            if armed and target_present and (trial_k != '') and (target_index is not None):
                                k = int(trial_k)
                                # Use history excluding the *current* fixation to avoid 0-back
                                prior_len = len(inspected_indices) - 1
                                if prior_len >= k:
                                    cand_hist_idx = prior_len - k
                                    cand_idx      = inspected_indices[cand_hist_idx]
                                    cand_time     = inspected_times[cand_hist_idx]

                                    if cand_idx != target_index:
                                        # safeguards: SOA since that fixation + separation from current gaze
                                        age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)
                                        gx = cluster_cx; gy = cluster_cy
                                        dxg = gabors[cand_idx].pos[0] - gx
                                        dyg = gabors[cand_idx].pos[1] - gy
                                        sep_req = (1.5 if k == 1 else 1.0) * min_gaze_sep_px
                                        sep_ok  = (dxg*dxg + dyg*dyg) >= (sep_req * sep_req)

                                        if age_ok and sep_ok:
                                            # relabel: make cand the target; old target becomes distractor
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            if target_index is not None:
                                                gabors[target_index].ori = random.choice(orientations)
                                                gabors[target_index].sf  = random.choice(spatial_frequencies)
                                            target_index = cand_idx
                                            nback_used_seq.append(k)
                                            # optional: mask each jump
                                            refresh_all_distractors(gabors, exclude_idx=target_index)

                            committed_this_cluster = True
                    else:
                        # leaving cluster: reset cluster (keep it lean)
                        if (cluster_len >= min_fix_frames) and (not committed_this_cluster):
                            pass
                        cluster_cx, cluster_cy = prev_gx, prev_gy
                        cluster_len = 1
                        committed_this_cluster = False

                # draw scene
                for g in gabors: g.draw()
                win.flip()

                keys = event.getKeys(keyList=["right", "left", "escape"], timeStamped=trial_clock)
                if keys:
                    response, rt = keys[0]
                    break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()

            if response:
                is_correct = (response == "right" and target_present) or (response == "left" and not target_present)
                feedback_text = "Correct" if is_correct else "Incorrect"
            else:
                response = "None"
                rt = ""
                is_correct = False
                feedback_text = timeout_feedback_text

            feedback = visual.TextStim(win, text=feedback_text, color="white", height=40)
            feedback.draw()
            win.flip()
            core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')

            response_num = 1 if response == "right" else 0 if response == "left" else -1
            writer.writerow(["dynamic task", participant_id, trial + 1, int(target_present),
                             response_num, int(is_correct), rt,
                             num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2),
                             (trial_k if target_present else ''),   # NBackK
                             nback_used_seq                          # NBackUsedSeq
                             ])

"""
# -------- Trial loop --------
def run_evading_target_dynamic_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")
    
    # Center the grid
    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2 
    grid_offset_y = -grid_pixel_height / 2

    # ---------- Gaze / fixation detection params (from static) ----------
    prev_gx, prev_gy   = screen_width/2.0, screen_height/2.0  # EMA state
    ema_alpha          = 0.55
    fix_radius_px      = cell_size            # ~35 px circle
    min_fix_frames     = 4                    # ~65 ms @ 60 Hz
    capture_radius_px  = 1.5 * cell_size      # snap a fixation to a Gabor if within this pixel radius  
    holdoff_by_k       = { 1: 0.12, 2: 0.08, 4: 0.00, 8: 0.00 }  # seconds
    min_gaze_sep_px    = 1.0 * cell_size      # avoid landing right under gaze

    # -------------------------------------------------------------------
    
    # Pre-compute a bank of 30 noise frames
    noise_bank = [generate_noise(screen_width, screen_height, grain_size=3)
                for _ in range(30)]
    bank_i = 0
    
    
    # Initialize Noise
    noise_frames = [
        visual.ImageStim(
            win,
            image=img,
            size=(screen_width, screen_height),
            units="pix",
            interpolate=False
        )
        for img in noise_bank
    ]
    
    bank_i = 0

    
    # Initialize instructions screen
    instruction_text = visual.TextStim(win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press 'right arrow key ' if you see the target, 'left arrow key' if not.\n"
              "Each trial you have 5 seconds to decide, try to make the decision as fast as possible.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=screen_width * 0.8, units='pix'
    )
    instruction_text.draw(); 
    win.flip(); 
    event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')  # keep buffer clean

    # Open a CSV file for behavioural results
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # original columns preserved; two extras appended at end
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory", "Speed (px/s)",
                         "NBackK", "NBackUsedSeq"])
                         
        #calculate speed of Gabors in pixel/second
        #speed_px_per_sec = (4 * cell_size) / (transition_steps * movement_delay)
    
        # ---- Balanced present/absent & balanced NBackK among present trials ----
        present_ratio = 0.5
        num_present_raw = int(round(num_trials * present_ratio))  # keep 50/50 overall
    
        def make_balanced_k_sequence(n_present):
            
            # Return a length-n_present list with counts for k in {1,2,4,8}
            # as equal as mathematically possible. If n_present isn't divisible
            # by 4, the remainder is distributed randomly across the set.
            
            base = n_present // 4
            rem  = n_present % 4
            ks = []
            for k in (1, 2, 4, 8):
                ks.extend([k] * base)
            if rem:
                extras = [1, 2, 4, 8]
                random.shuffle(extras)
                ks.extend(extras[:rem])
            random.shuffle(ks)  # randomize order across present trials
            return ks
    
        # Build trial flags and per-present-trial K sequence
        present_flags = [1] * num_present_raw + [0] * (num_trials - num_present_raw)
        random.shuffle(present_flags)
        k_seq = make_balanced_k_sequence(num_present_raw)
        k_idx = 0  # cursor through k_seq
        # -------------------------------------------------------------------------

        
        for trial in range(num_trials):
            # 500 ms fixation cross
            fix_cross = visual.TextStim(win, text='+', color='black', height=40)
            fix_cross.draw()
            win.flip()
            core.wait(0.5)
            event.clearEvents(eventType='keyboard')

            # training notice unchanged
            if trial == 3:
                ready_text = visual.TextStim(win, text="Training phase is over! The experiment begins now. Press Enter to continue.",
                                             color="white", height=30)
                ready_text.draw()
                win.flip()
                event.waitKeys(keyList=["return"])
                event.clearEvents(eventType='keyboard')

            # Number of Gabors defined here
            num_gabors = random.choice([10])

            # Balanced present/absent + balanced K for present trials
            tp = bool(present_flags[trial])            # 1 -> True, 0 -> False
            trial_k = (k_seq[k_idx] if tp else '')     # K only for present trials
            if tp:
                k_idx += 1

            target_present = tp  # keep your existing variable name used below

            # Generate the positions (grid coords) for distractors and target
            positions = [(random.randint(2, grid_size_x - 3), random.randint(2, grid_size_y - 3)) for _ in range(num_gabors)]
            target_index = random.randint(0, num_gabors - 1) if target_present else None

            # Create Gabors (explicitly in pix units to match gaze)
            gabors = []
            d_oris, d_sfs = [], []
            for i in range(num_gabors):
                if i == target_index:
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=target_sf, ori=target_orientation, phase=0.25, units='pix')
                else:
                    sf = random.choice(spatial_frequencies)
                    ori = random.choice(orientations)
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=sf, ori=ori, phase=0.25, units='pix')
                    d_oris.append(ori); d_sfs.append(sf)
                gabors.append(gabor)

            for i, (x, y) in enumerate(positions):
                gabors[i].pos = grid_to_pixel(x, y, grid_offset_x, grid_offset_y, cell_size)
                
            
            # --- choose a stable distractor to measure speed (avoid target to ignore jumps) ---
            if target_present and target_index is not None and num_gabors > 1:
                speed_probe_idx = 0 if 0 != target_index else 1
            else:
                speed_probe_idx = 0
            speed_prev_px = gabors[speed_probe_idx].pos  # (x, y) in pixels
            speed_dist_px = 0.0


            current_steps = [random.randint(0, transition_steps // 2) for _ in range(num_gabors)]
            targets = positions[:]
            last_moves = [random.choice([(4, 0), (0, 4), (0, -4),
                                         (DIAGONAL_SCALE, DIAGONAL_SCALE),
                                         (DIAGONAL_SCALE, -DIAGONAL_SCALE)]) for _ in range(num_gabors)]
            for i in range(num_gabors):
                x, y = positions[i]
                valid = get_valid_moves(x, y, last_moves[i])
                move = random.choice(valid) if valid else (0, 0)
                targets[i] = (x + move[0], y + move[1])
                last_moves[i] = move

            trial_clock = core.Clock()
            response = None
            rt = None
            gabor_trajectory = []
            target_trajectory = []  # list of (grid-x, grid-y) per frame for target

            # --- Gaze-driven state (like static) ---
            inspected_indices = []   # distractor indices visited (debounced)
            inspected_times   = []   # timestamps for those visits (trial_clock time)
            nback_used_seq    = []   # track target jumps
            
            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # Put tracker in idle mode before recording
            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')

            # Send message "TRIALID" to mark the start of a trial
            el_tracker.sendMessage(f'TRIALID {trial + 1}')
            
            # Start recording
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)

            # Log a message to mark the onset of the stimulus
            el_tracker.sendMessage('stimulus_onset')          
            
            while trial_clock.getTime() < trial_duration:
                # update noise
                noise_frames[bank_i].draw()
                bank_i = (bank_i + 1) % len(noise_frames)

                # --- update random-walk targets & interpolate positions ---
                frame_positions = []
                for i in range(num_gabors):
                    if current_steps[i] >= transition_steps:
                        x, y = positions[i]
                        valid = get_valid_moves(x, y, last_moves[i])
                        move = random.choice(valid) if valid else (0, 0)
                        targets[i] = (x + move[0], y + move[1])
                        last_moves[i] = move
                        current_steps[i] = 0

                    t = current_steps[i] / transition_steps
                    interp_x = positions[i][0] + (targets[i][0] - positions[i][0]) * t
                    interp_y = positions[i][1] + (targets[i][1] - positions[i][1]) * t
                    gabors[i].pos = grid_to_pixel(interp_x, interp_y, grid_offset_x, grid_offset_y, cell_size)
                    frame_positions.append((interp_x, interp_y))

                    current_steps[i] += 1
                    if current_steps[i] >= transition_steps:
                        positions[i] = targets[i]

                gabor_trajectory.append([(round(x, 2), round(y, 2)) for (x, y) in frame_positions])

                # track target per frame (grid coords like before)
                if target_index is not None:
                    tx, ty = frame_positions[target_index]
                    target_trajectory.append((round(tx, 2), round(ty, 2)))
                
                # --- measure speed using the probe gabor ---
                probe_px = gabors[speed_probe_idx].pos
                dx = probe_px[0] - speed_prev_px[0]
                dy = probe_px[1] - speed_prev_px[1]
                speed_dist_px += math.hypot(dx, dy)
                speed_prev_px = probe_px

                # --- read & smooth gaze ---
                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()
                if eye:
                    rx, ry = eye.getGaze()
                    prev_gx = float(np.clip(ema_alpha*rx + (1-ema_alpha)*prev_gx, 0, screen_width-1))
                    prev_gy = float(np.clip(ema_alpha*ry + (1-ema_alpha)*prev_gy, 0, screen_height-1))

                    # fixation clustering
                    dx = prev_gx - cluster_cx
                    dy = prev_gy - cluster_cy
                    inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                    if inside:
                        cluster_len += 1
                        w = 1.0 / max(1, cluster_len)
                        cluster_cx = (1 - w)*cluster_cx + w*prev_gx
                        cluster_cy = (1 - w)*cluster_cy + w*prev_gy

                        if (not committed_this_cluster) and (cluster_len == min_fix_frames):
                            # commit fixation -> map to nearest current Gabor (prefer distractors)
                            # compute pixel distances to current centers
                            centers_pix = np.array([g.pos for g in gabors], dtype=float)
                            d2 = (centers_pix[:,0] - cluster_cx)**2 + (centers_pix[:,1] - cluster_cy)**2
                            current_idx = None
                            if d2.size > 0:
                                j = int(np.argmin(d2))
                                if math.sqrt(d2[j]) <= capture_radius_px:
                                    current_idx = j

                            # if no capture, impute to nearest distractor (exclude target)
                            if current_idx is None:
                                # exclude current target from consideration
                                cand_d2 = d2.copy()
                                if target_index is not None:
                                    cand_d2[target_index] = np.inf
                                j2 = int(np.argmin(cand_d2))
                                current_idx = None if np.isinf(cand_d2[j2]) else j2

                            # append inspection (debounced) for distractors only
                            if (target_present and target_index is not None
                                and current_idx is not None and current_idx != target_index):
                                if not inspected_indices or inspected_indices[-1] != current_idx:
                                    inspected_indices.append(current_idx)
                                    inspected_times.append(trial_clock.getTime())

                                # strict n-back jump (only when we have enough prior distractor fixations)
                                if trial_k != '':
                                    k = int(trial_k)
                                    prior_len = len(inspected_indices) - 1   # exclude current fixation to avoid 0-back
                                    if prior_len >= k:
                                        cand_hist_idx = prior_len - k
                                        cand_idx      = inspected_indices[cand_hist_idx]
                                        cand_time     = inspected_times[cand_hist_idx]

                                        if cand_idx != target_index:
                                            # safeguards: SOA since that fixation + separation from current gaze
                                            age_ok = (trial_clock.getTime() - cand_time) >= holdoff_by_k.get(k, 0.0)
                                            gx = cluster_cx; gy = cluster_cy
                                            dxg = gabors[cand_idx].pos[0] - gx
                                            dyg = gabors[cand_idx].pos[1] - gy
                                            sep_req = (1.5 if k == 1 else 1.0) * min_gaze_sep_px
                                            sep_ok  = (dxg*dxg + dyg*dyg) >= (sep_req * sep_req)

                                            if age_ok and sep_ok:
                                                # relabel: make cand the target; old target becomes distractor
                                                gabors[cand_idx].ori = target_orientation
                                                gabors[cand_idx].sf  = target_sf
                                                # re-randomize the old target's features
                                                if target_index is not None:
                                                    gabors[target_index].ori = random.choice(orientations)
                                                    gabors[target_index].sf  = random.choice(spatial_frequencies)
                                                target_index = cand_idx
                                                nback_used_seq.append(k)
                                                
                            committed_this_cluster = True
                    
                    else:   
                        # leaving cluster: commit if long enough and not yet committed
                        if (cluster_len >= min_fix_frames) and (not committed_this_cluster):
                            # map fixation center
                            centers_pix = np.array([g.pos for g in gabors], dtype=float)
                            d2 = (centers_pix[:,0] - cluster_cx)**2 + (centers_pix[:,1] - cluster_cy)**2
                            current_idx = None
                            if d2.size > 0:
                                j = int(np.argmin(d2))
                                if math.sqrt(d2[j]) <= capture_radius_px:
                                    current_idx = j
                    
                            # append distractor visit (debounced)
                            if target_present and (target_index is not None):
                                if current_idx is None:
                                    # impute to nearest distractor (exclude target)
                                    cand_d2 = d2.copy()
                                    cand_d2[target_index] = np.inf
                                    j2 = int(np.argmin(cand_d2))
                                    current_idx = None if np.isinf(cand_d2[j2]) else j2
                    
                                if (current_idx is not None) and (current_idx != target_index):
                                    if not inspected_indices or inspected_indices[-1] != current_idx:
                                        inspected_indices.append(current_idx)
                                        inspected_times.append(trial_clock.getTime())
                    
                                    # strict n-back jump
                                    if trial_k != '':
                                        k = int(trial_k)
                                        prior_len = len(inspected_indices) - 1  # exclude current fixation to avoid 0-back
                                        if prior_len >= k:
                                            cand_hist_idx = prior_len - k
                                            cand_idx      = inspected_indices[cand_hist_idx]
                                            cand_time     = inspected_times[cand_hist_idx]
                    
                                            if cand_idx != target_index:
                                                age_ok = (trial_clock.getTime() - cand_time) >= holdoff_by_k.get(k, 0.0)
                                                gx, gy = cluster_cx, cluster_cy
                                                dxg = gabors[cand_idx].pos[0] - gx
                                                dyg = gabors[cand_idx].pos[1] - gy
                                                # simpler separation rule
                                                sep_ok = (dxg*dxg + dyg*dyg) >= (min_gaze_sep_px * min_gaze_sep_px)
                    
                                                if age_ok and sep_ok:
                                                    # relabel: make cand the target; old target becomes distractor
                                                    gabors[cand_idx].ori = target_orientation
                                                    gabors[cand_idx].sf  = target_sf
                                                    if target_index is not None:
                                                        gabors[target_index].ori = random.choice(orientations)
                                                        gabors[target_index].sf  = random.choice(spatial_frequencies)
                                                    target_index = cand_idx
                                                    nback_used_seq.append(k)  # <-- log it
                    
                        # reset cluster for next fixation
                        cluster_cx, cluster_cy = prev_gx, prev_gy
                        cluster_len = 1
                        committed_this_cluster = False
                  

                # draw scene
                for g in gabors: g.draw()
                win.flip()

                keys = event.getKeys(keyList=["right", "left", "escape"], timeStamped=trial_clock)
                if keys:
                    response, rt = keys[0]
                    break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()
            
            # --- compute measured speed (px/s) over the stimulus interval ---
            trial_elapsed_s = trial_clock.getTime()  # time from trial start to stimulus end
            speed_px_per_sec = (speed_dist_px / trial_elapsed_s) if trial_elapsed_s > 0 else 0.0

            # Feedback
            if response:
                is_correct = (response == "right" and target_present) or (response == "left" and not target_present)
                feedback_text = "Correct" if is_correct else "Incorrect"
            else:
                response = "None"
                rt = ""
                is_correct = False
                feedback_text = timeout_feedback_text

            feedback = visual.TextStim(win, text=feedback_text, color="white", height=40)
            feedback.draw()
            win.flip()
            core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')

            response_num = 1 if response == "right" else 0 if response == "left" else -1
            # Append two extra columns: NBackK and the used sequence (for QA)
            writer.writerow(["dynamic task", participant_id, trial + 1, int(target_present),
                             response_num, int(is_correct), rt,
                             num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2),
                             (trial_k if target_present else ''),  # NBackK
                             nback_used_seq                        # NBackUsedSeq (strict -> always [k] per jump; optional to fill)
                             ])
"""