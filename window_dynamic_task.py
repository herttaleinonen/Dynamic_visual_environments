#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: herttaleinonen

"""
Dynamic task with gaze-contingent Gaussian peephole (optimized).
- Underlying moving noise + moving Gabors unchanged.
- Full-screen alpha mask is computed at reduced resolution and reused for speed.
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
noise_grain = 3  # pixel x pixel

def generate_noise(screen_width, screen_height, grain_size=noise_grain):
    h_grains = int(np.ceil(screen_height / grain_size))
    w_grains = int(np.ceil(screen_width / grain_size))
    small = np.random.normal(loc=0, scale=0.3, size=(h_grains, w_grains))
    small = np.clip(small, -1, 1)
    noise = np.repeat(np.repeat(small, grain_size, axis=0),
                      grain_size, axis=1)
    return noise[:screen_height, :screen_width]

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
def run_window_dynamic_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")

    # Center the grid
    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2
    grid_offset_y = -grid_pixel_height / 2

    # ---------------- NEW: peephole setup (fast) ----------------
    prev_gx, prev_gy = screen_width / 2.0, screen_height / 2.0  # start centered
    hole_radius_ratio = 0.30   # 15% of min dimension
    sigma_ratio       = 0.50   # σ = 0.5 * radius for edge softness
    mask_scale        = 0.50   # compute mask at 50% resolution (big speedup)
    update_every      = 1      # update mask every frame (set 2 for half-rate)

    hole_radius = int(min(screen_width, screen_height) * hole_radius_ratio)
    sigma       = hole_radius * sigma_ratio

    # Downsampled mask resolution
    h_s = int(screen_height * mask_scale)
    w_s = int(screen_width  * mask_scale)

    # Small pixel grids (top-left origin)
    ys_s, xs_s = np.ogrid[0:h_s, 0:w_s]

    # Scaled params
    hole_radius_s   = hole_radius * mask_scale
    sigma_s         = sigma * mask_scale
    two_sigma2_s    = 2.0 * (sigma_s * sigma_s)

    # Preallocate alpha + RGBA (float32)
    alpha_small = np.ones((h_s, w_s), dtype=np.float32)    # opaque by default
    mask_small  = np.zeros((h_s, w_s, 4), dtype=np.float32)
    mask_small[..., 3] = alpha_small                       # only alpha varies

    # One ImageStim reused; draw scaled to full screen
    mask_im = visual.ImageStim(
        win,
        image=mask_small,
        size=(screen_width, screen_height),
        units='pix',
        interpolate=True  # smooth Gaussian when upscaled
    )
    frame_i = 0
    # -----------------------------------------------------------

    # Initialize Noise and Instructions
    noise_stim = visual.ImageStim(
        win,
        image=generate_noise(screen_width, screen_height),
        size=(screen_width, screen_height),
        units="pix",
        interpolate=False
    )
    noise_bank = [generate_noise(screen_width, screen_height, grain_size=3)
                  for _ in range(30)]
    bank_i = 0

    instruction_text = visual.TextStim(
        win,
        text=("In the following experiment, you will see moving objects.\n"
              "Among them is a target, that is tilted 45 degrees like this: /.\n"
              "Press '>' if you see the target.\n"
              "Press '<' if you do not.\n"
              "Press Enter to start."),
        color="white", height=30, wrapWidth=screen_width * 0.8
    )
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["return"])

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory", "Speed (px/s)"])

        speed_px_per_sec = (4 * cell_size) / (transition_steps * movement_delay)

        for trial in range(num_trials):
            # 500 ms fixation cross
            fix_cross = visual.TextStim(win, text='+', color='black', height=40)
            fix_cross.draw()
            win.flip()
            core.wait(0.5)

            if trial == 3:
                ready_text = visual.TextStim(
                    win,
                    text="Training phase is over! The experiment begins now. Press Enter to continue.",
                    color="white", height=30
                )
                ready_text.draw()
                win.flip()
                event.waitKeys(keyList=["return"])

            event.clearEvents()

            # Number of Gabors / target presence
            num_gabors = random.choice([5, 10, 15])
            target_present = random.choice([True, False])
            positions = [(random.randint(2, grid_size_x - 3), random.randint(2, grid_size_y - 3))
                         for _ in range(num_gabors)]
            target_index = random.randint(0, num_gabors - 1) if target_present else None

            # Distractor params
            distractor_oris = random.choices(orientations, k=num_gabors - int(target_present))
            distractor_sfs  = random.choices(spatial_frequencies, k=num_gabors - int(target_present))

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
            target_trajectory = []

            # EyeLink record
            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')
            el_tracker.sendMessage(f'TRIALID {trial + 1}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)
            el_tracker.sendMessage('stimulus_onset')

            while trial_clock.getTime() < trial_duration:
                # sample gaze + EMA smooth
                if el_tracker:
                    samp = el_tracker.getNewestSample()
                    if samp and samp.isRightSample():
                        eye = samp.getRightEye()
                    elif samp and samp.isLeftSample():
                        eye = samp.getLeftEye()
                    else:
                        eye = None
                    if eye:
                        rx, ry = eye.getGaze()
                        prev_gx = float(np.clip((prev_gx + rx) / 2.0, 0, screen_width  - 1))
                        prev_gy = float(np.clip((prev_gy + ry) / 2.0, 0, screen_height - 1))

                # noise + movement
                noise_stim.image = noise_bank[bank_i]
                bank_i = (bank_i + 1) % len(noise_bank)
                noise_stim.draw()

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
                if target_index is not None:
                    target_x, target_y = frame_positions[target_index]
                    target_trajectory.append((target_x, target_y))

                for g in gabors:
                    g.draw()

                # ---------- NEW: fast mask update & draw ----------
                if frame_i % update_every == 0:
                    gx_s = prev_gx * mask_scale
                    gy_s = (screen_height - prev_gy) * mask_scale  # top-left origin
                    dx = xs_s - gx_s
                    dy = ys_s - gy_s
                    dist2 = dx*dx + dy*dy

                    alpha_small.fill(1.0)  # opaque by default
                    inside = dist2 <= (hole_radius_s * hole_radius_s)
                    # 1 - exp(-r^2 / (2*sigma^2)) inside the hole
                    alpha_small[inside] = 1.0 - np.exp(-dist2[inside] / two_sigma2_s)

                    # update alpha channel in-place
                    mask_small[..., 3] = alpha_small
                    mask_im.image = mask_small

                mask_im.draw()
                frame_i += 1
                # --------------------------------------------------

                win.flip()

                keys = event.getKeys(keyList=["right", "left", "escape"], timeStamped=trial_clock)
                if keys:
                    response, rt = keys[0]
                    break

                core.wait(movement_delay)

            el_tracker.sendMessage('stimulus_offset')
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

            response_num = 1 if response == "right" else 0 if response == "left" else -1
            writer.writerow(["dynamic task", participant_id, trial + 1, int(target_present),
                             response_num, int(is_correct), rt,
                             num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2)])
