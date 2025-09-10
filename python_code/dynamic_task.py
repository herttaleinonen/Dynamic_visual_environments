#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:28:41 2025

@author: herttaleinonen

dynamic_task.py

Provides run_dynamic_trials() for calling from main.py after EyeLink setup.
Displays dynamic Gabor arrays on a Gaussian noise background,
flashes a fixation cross between trials for 0.5s,
collects responses during the trial duration window,
provides feedback between trials, incorrect/correct based on gaze coordinates
"""

import os
import csv
import random
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


# map gaze (pixels) to grid coords (float), for optional logging
def gaze_pix_to_grid(px, py, grid_offset_x, grid_offset_y, cell_size):
    gx = (px - grid_offset_x) / cell_size
    gy = (py - grid_offset_y) / cell_size
    return gx, gy


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

# -------- Trial loop --------
def run_dynamic_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")
    
    # Center the grid
    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2 
    grid_offset_y = -grid_pixel_height / 2
    
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
    

    # gaze/fixation parameters (tune to your setup)
    ema_alpha = 0.5
    fix_radius_px = max(80, int(cell_size * 1.2))  # larger fixation bubble
    min_fix_frames = 1                             # commit very quickly
    capture_radius_px = max(int(cell_size * 3.0), 160)  # large acceptance circle baseline
    gaze_to_press_max_lag = trial_duration         # effectively no time limit
    gauss_k = 6.0                                  # widen radius derived from Gaussian mask
 

    gauss_k = 3.0  # ~3σ covers most of a Gaussian; increase to be more lenient
    def target_accept_radius_px():
        # PsychoPy 'gauss' mask: σ ≈ size/6
        sigma = cell_size / 6.0
        return max(capture_radius_px, gauss_k * sigma)


    
    # Initialize instructions screen
    instruction_text = visual.TextStim(win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press 'SPACE' as soon as you see the target.\n"
              "If you do not press 'SPACE' in time, you'll see a timeout message.\n"
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
        # added columns for gaze-based scoring/debug
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory",
                         "Speed (px/s)", "FixOnTargetTime(s)", "LastFixIndex"])

        
# ---------------------------------------------------------------

        for trial in range(num_trials):
            # 500 ms fixation cross
            fix_cross = visual.TextStim(win, text='+', color='black', height=40)
            fix_cross.draw()
            win.flip()
            core.wait(0.5)
            event.clearEvents()

            # Number of Gabors defined here
            num_gabors = random.choice([10])
            # always target present
            target_present = True

            # Generate the positions for distractors and target
            positions = [(random.randint(2, grid_size_x - 3), random.randint(2, grid_size_y - 3)) for _ in range(num_gabors)]
            # always pick a target index (since target is always present)
            target_index = random.randint(0, num_gabors - 1)

            # Create the distractor Gabor positions and target presence
            distractor_oris = random.choices(orientations, k=num_gabors - 1)
            distractor_sfs = random.choices(spatial_frequencies, k=num_gabors - 1)

            gabors = []
            d_idx = 0
            for i in range(num_gabors):
                if i == target_index:
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=target_sf, ori=target_orientation, phase=0.25)
                else:
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=distractor_sfs[d_idx], ori=distractor_oris[d_idx], phase=0.25)
                    d_idx += 1
                gabors.append(gabor)

            for i, (x, y) in enumerate(positions):
                gabors[i].pos = grid_to_pixel(x, y, grid_offset_x, grid_offset_y, cell_size)

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
            target_trajectory = []  # Separate list to track the target's movement

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
            
            # Initialize speed measuring
            dist_px = 0.0
            prev_px = None
            
            # gaze/cluster state per trial
            # gaze/cluster state per trial (PsychoPy-centered coords)
            cluster_cx = 0.0
            cluster_cy = 0.0
            cluster_len = 0
            committed_this_cluster = False
            last_committed_fix_idx = None
            last_fix_on_target_time = None
            fix_log = []
            inspected_indices = []
            inspected_times = []
            # keep simple buffers for raw EMA in screen coords
            ema_x = screen_width / 2.0
            ema_y = screen_height / 2.0


            
            while trial_clock.getTime() < trial_duration:
                # draw noise
                noise_frames[bank_i].draw()
                bank_i = (bank_i + 1) % len(noise_frames)
                
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
                    frame_positions.append((round(interp_x, 2), round(interp_y, 2)))

                    current_steps[i] += 1
                    if current_steps[i] >= transition_steps:
                        positions[i] = targets[i]

                gabor_trajectory.append(frame_positions)

                # Separate target tracking (if target is present, track it separately)
                if target_index is not None:
                    target_x, target_y = frame_positions[target_index]
                    target_trajectory.append((target_x, target_y))

                    # convert target grid coords -> pixels
                    tx = target_x * cell_size + grid_offset_x
                    ty = target_y * cell_size + grid_offset_y
                
                if prev_px is not None:
                    dx = tx - prev_px[0]
                    dy = ty - prev_px[1]
                    dist_px += (dx*dx + dy*dy) ** 0.5
                
                prev_px = (tx, ty)
                
                # --- read & smooth gaze ---
                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()
                
                now_t = trial_clock.getTime()
                
                if eye is not None:
                    rx, ry = eye.getGaze()  # EyeLink screen coords: (0,0)=top-left, +y down
                    # guard against invalid/sentinel values
                    if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                        # EMA in screen coords for stability
                        ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, screen_width-1))
                        ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, screen_height-1))
                        # CONVERT to PsychoPy-centered coords: (0,0)=center, +y up
                        gx = ema_x - (screen_width  / 2.0)
                        gy = (screen_height / 2.0) - ema_y
                        # ----- fixation clustering in centered coords -----
                        dx = gx - cluster_cx
                        dy = gy - cluster_cy
                        inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                
                        if inside:
                            cluster_len += 1
                            w = 1.0 / max(1, cluster_len)
                            cluster_cx = (1 - w)*cluster_cx + w*gx
                            cluster_cy = (1 - w)*cluster_cy + w*gy
                
                            # commit when we've reached OR exceeded min_fix_frames
                            if (not committed_this_cluster) and (cluster_len >= min_fix_frames):
                                centers_pix = np.array([g.pos for g in gabors], dtype=float)  # Gabors are in centered coords
                                if centers_pix.size > 0:
                                    d2 = (centers_pix[:,0] - cluster_cx)**2 + (centers_pix[:,1] - cluster_cy)**2
                                    j = int(np.argmin(d2))
                                    current_idx = None
                                    if np.sqrt(d2[j]) <= 1.25 * target_accept_radius_px():  
                                        current_idx = j

                                    # log fixation location (grid cell) and add to inspection history (optional)
                                    ix, iy = gaze_pix_to_grid(cluster_cx, cluster_cy, grid_offset_x, grid_offset_y, cell_size)
                                    fix_log.append((ix, iy))
                                    if current_idx is not None:
                                        last_committed_fix_idx = current_idx
                                        if target_index is not None and current_idx == target_index:
                                            last_fix_on_target_time = now_t
                                        else:
                                            inspected_indices.append(current_idx)
                                            inspected_times.append(now_t)
                                committed_this_cluster = True
                        else:
                            # just LEFT the fixation cluster
                            cluster_len = 1
                            cluster_cx = gx
                            cluster_cy = gy
                            committed_this_cluster = False

                for g in gabors:
                    g.draw()
                win.flip()

                # only listen for space (detect target) and escape
                keys = event.getKeys(keyList=["space", "escape"], timeStamped=trial_clock)
                if keys:
                    response, rt = keys[0]
                    break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()
            
            # Get speed measurement
            elapsed = trial_clock.getTime()
            speed_px_per_sec = dist_px / elapsed if elapsed > 0 else 0.0

            # Feedback
            if response:
                # recent committed fixation?
                recently_fixated_target = (last_fix_on_target_time is not None) and \
                                          ((rt - last_fix_on_target_time) <= gaze_to_press_max_lag)
                
                # on-keypress fallback — was gaze near the target at the press moment?
                on_keypress_fixated_target = False
                if response == "space" and target_index is not None:
                    tgx, tgy = gabors[target_index].pos  # centered coords
                    # USE centered gaze at press; if no fresh sample this frame, fall back to cluster center
                    gaze_x_at_press = cluster_cx if 'gx' not in locals() else gx
                    gaze_y_at_press = cluster_cy if 'gy' not in locals() else gy
                    d_key = ((gaze_x_at_press - tgx)**2 + (gaze_y_at_press - tgy)**2) ** 0.5
                    
                    if d_key <= 1.25 * target_accept_radius_px(): 
                        on_keypress_fixated_target = True
                        if last_fix_on_target_time is None:
                            last_fix_on_target_time = rt
                            last_committed_fix_idx = target_index
                
                is_correct = (response == "space") and (recently_fixated_target or on_keypress_fixated_target)
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

            response_num = 1 if response == "space" else -1
        
            writer.writerow(["dynamic task", participant_id, trial + 1, int(target_present),
                             response_num, int(is_correct), rt,
                             num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2),
                             round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                             last_committed_fix_idx if last_committed_fix_idx is not None else ""])
