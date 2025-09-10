#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 12:15:16 2025

@author: herttaleinonen


    Dynamic ‘evading target’ task with dynamic distractor Gabor stimulus. The target’s position is dynamically manipulated 
    in relation to the participant’s eye movements. Shortly after the search begins the target appears at locations 
    corresponding to −8 to −1 prior fixations (n-back), by replacing distractor Gabors that have been previously inspected. 
    
      - On target-present trials, the target appears at n s by taking over a distractor (appear_delay).
      - On that frame, all other distractors re-randomize (masked reveal).
      - Fixations are tracked continuously, but n-back retargeting only activates after the target appears (>= n s).
         
"""

import os
import csv
import random
#import math
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
    
    # --- helper: screen-pixel gaze -> centered pix -> grid cell (ix, iy) ---
    def gaze_pix_to_grid(cx, cy):
        # EyeLink gaze is in screen pixels (origin top-left). Convert to centered PsychoPy pix:
        fx_c = cx - screen_width/2.0
        fy_c =  screen_height/2.0 - cy
        # then to grid (continuous), then snap+clamp to integer cell indices
        gx = (fx_c - grid_offset_x) / cell_size
        gy = (fy_c - grid_offset_y) / cell_size
        ix = int(np.clip(np.round(gx), 0, grid_size_x - 1))
        iy = int(np.clip(np.round(gy), 0, grid_size_y - 1))
        return ix, iy


    # ---------- Gaze / fixation detection params (from static) ----------
    prev_gx, prev_gy   = screen_width/2.0, screen_height/2.0  # EMA state
    ema_alpha          = 0.55
    fix_radius_px      = cell_size            # ~35 px circle
    min_fix_frames     = 4                    # ~65 ms @ 60 Hz
    capture_radius_px  = 1.5 * cell_size      # snap a fixation to a Gabor if within this pixel radius
    min_gaze_sep_px    = 1.25 * cell_size     # avoid landing right under gaze
    holdoff_by_k       = { 1: 0.45, 2: 0.10, 4: 0.00, 8: 0.00 }  # seconds
    appear_delay_s     = 0.5                  # target appears at 500 ms
    # -------------------------------------------------------------------
    
    # acceptance radius for correctness 
    def score_accept_radius_px():
        sigma = cell_size / 6.0  # 'gauss' mask σ
        return max(2.0 * capture_radius_px, 8.0 * sigma)  # bigger circle than before


    
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
              "Press SPACE as soon as you see the target.\n"  
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
                         "NBackK", "NBackUsedSeq", "Fixations",
                         "FixOnTargetTime(s)", "LastFixIndex"])  


    
        # ---- Balanced per-trial "mode": 1-back, 2-back, 4-back, or RANDOM target (¼ each) ----
        present_ratio   = 1.0
        num_present_raw = int(round(num_trials * present_ratio))    # here: == num_trials

        def make_balanced_mode_sequence(n_present):
            """
            Return a length-n_present list with equal counts of:
              1, 2, 4, 'RND'
            If n_present isn't divisible by 4, the remainder is distributed randomly.
            """
            modes = [1, 2, 4, 'RND']
            base = n_present // 4
            rem  = n_present % 4
            seq = []
            for m in modes:
                seq.extend([m] * base)
            if rem:
                extras = modes[:]
                random.shuffle(extras)
                seq.extend(extras[:rem])
            random.shuffle(seq)
            return seq

        # Build per-trial mode sequence (no absent trials here)
        mode_seq = make_balanced_mode_sequence(num_present_raw)
        mode_idx = 0  # cursor through mode_seq

        
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

            # Number of Gabors defined here
            num_gabors = random.choice([10])

            # Always present, and pick the mode (1/2/4 or 'RND') for this trial
            target_present = True
            trial_mode = mode_seq[mode_idx]   # trial_mode ∈ {1, 2, 4, 'RND'}
            mode_idx += 1

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
            fix_log = []  # list of (ix, iy) grid cell coords of each committed fixation

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
            
            # scoring-only gaze state 
            last_fix_on_target_time = None
            last_committed_fix_idx  = None
            score_inside_count      = 0      # dwell counter (frames) for scoring commits
            score_gx_c, score_gy_c  = 0.0, 0.0  # last centered gaze for scoring
            score_prev_target_pos_c = None   # previous target position (centered) after a jump
            score_target_change_time = -1e9  # time of last target jump
            score_change_grace       = 0.8   # s: allow press near the *previous* target shortly after a jump
            
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
                
                if eye is not None:
                    rx, ry = eye.getGaze()
                    prev_gx = float(np.clip(ema_alpha*rx + (1-ema_alpha)*prev_gx, 0, screen_width-1))
                    prev_gy = float(np.clip(ema_alpha*ry + (1-ema_alpha)*prev_gy, 0, screen_height-1))
                    
                    # for scoring (correct/incorrect) convert screen → centered for scoring 
                    score_gx_c = prev_gx - (screen_width  / 2.0)
                    score_gy_c = (screen_height / 2.0) - prev_gy
                    
                    # If target exists, accept even a single-frame dwell inside a large radius
                    if target_index is not None:
                        tgx, tgy = gabors[target_index].pos  # centered coords
                        d = ((score_gx_c - tgx)**2 + (score_gy_c - tgy)**2) ** 0.5
                        if d <= score_accept_radius_px():
                            score_inside_count += 1
                            if (score_inside_count >= 1) and (last_fix_on_target_time is None):  
                                last_fix_on_target_time = trial_clock.getTime()
                                last_committed_fix_idx  = target_index
                        else:
                            score_inside_count = 0

                
                    # fixation clustering
                    dx = prev_gx - cluster_cx
                    dy = prev_gy - cluster_cy
                    inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                
                    if inside:
                        # still within this fixation
                        cluster_len += 1
                        w = 1.0 / max(1, cluster_len)
                        cluster_cx = (1 - w)*cluster_cx + w*prev_gx
                        cluster_cy = (1 - w)*cluster_cy + w*prev_gy
                
                        # commit ONCE when we first hit min_fix_frames
                        if (not committed_this_cluster) and (cluster_len == min_fix_frames):
                            # map fixation to nearest Gabor center (in pixels), with capture radius
                            centers_pix = np.array([g.pos for g in gabors], dtype=float)
                            d2 = (centers_pix[:,0] - cluster_cx)**2 + (centers_pix[:,1] - cluster_cy)**2
                            current_idx = None
                            if d2.size > 0:
                                j = int(np.argmin(d2))
                                if np.sqrt(d2[j]) <= capture_radius_px:
                                    current_idx = j
                
                            # if no capture, impute to nearest *distractor* (exclude target if known)
                            if current_idx is None:
                                cand_d2 = d2.copy()
                                if target_index is not None:
                                    cand_d2[target_index] = np.inf
                                j2 = int(np.argmin(cand_d2))
                                current_idx = None if np.isinf(cand_d2[j2]) else j2
                
                            # log fixation location (grid cell) and add to inspection history
                            ix, iy = gaze_pix_to_grid(cluster_cx, cluster_cy)
                            fix_log.append((ix, iy))
                            if current_idx is not None and (target_index is None or current_idx != target_index):
                                inspected_indices.append(current_idx)
                                inspected_times.append(now_t)
                
                            committed_this_cluster = True
                
                    else:
                        # just LEFT the fixation cluster → act now if the fixation was long enough
                        if cluster_len >= min_fix_frames:
                            # 1) n-back retargeting (only after reveal) — SKIP if mode is 'RND'
                            if armed and (trial_mode != 'RND') and (target_index is not None):
                                k = int(trial_mode)  # k ∈ {1,2,4}
                                prior_len = len(inspected_indices) - 1  # exclude the *current* (leaving) fixation
                                if prior_len >= k:
                                    cand_hist_idx = prior_len - k
                                    cand_idx      = inspected_indices[cand_hist_idx]
                                    cand_time     = inspected_times[cand_hist_idx]
                    
                                    if cand_idx != target_index:
                                        age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)
                                        gx, gy = cluster_cx, cluster_cy  # center of the fixation that just ended
                                        dxg = gabors[cand_idx].pos[0] - gx
                                        dyg = gabors[cand_idx].pos[1] - gy
                                        sep_mult_by_k = {1: 2.25, 2: 1.50, 4: 1.00}
                                        sep_req = sep_mult_by_k.get(k, 1.0) * min_gaze_sep_px
                                        sep_ok  = (dxg*dxg + dyg*dyg) >= (sep_req * sep_req)
                    
                                        if age_ok and sep_ok:
                                            # remember previous target position (centered coords) BEFORE swapping
                                            prev_target_pos_c = gabors[target_index].pos if target_index is not None else None
                                        
                                            # promote candidate to target; demote old target
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            if target_index is not None:
                                                gabors[target_index].ori = random.choice(orientations)
                                                gabors[target_index].sf  = random.choice(spatial_frequencies)
                                            target_index = cand_idx
                                            nback_used_seq.append(k)
                                        
                                            # log jump for forgiveness window
                                            if prev_target_pos_c is not None:
                                                score_prev_target_pos_c = prev_target_pos_c
                                                score_target_change_time = now_t

                    
                            # 2) global camo mask (re-randomize ALL distractors)
                            #    (Target keeps its signature features; exclude it here.)
                            refresh_all_distractors(gabors, exclude_idx=target_index)
                    
                        # reset cluster for next fixation
                        cluster_cx, cluster_cy = prev_gx, prev_gy
                        cluster_len = 1
                        committed_this_cluster = False

                
                # if there was no new eye sample this frame, we simply skip gaze logic


                # draw scene
                for g in gabors: g.draw()
                win.flip()

                keys = event.getKeys(keyList=["space", "escape"], timeStamped=trial_clock)  # space only
                if keys:
                    k, t0 = keys[0]
                    if k == "escape":
                        el_tracker.sendMessage('stimulus_offset')
                        el_tracker.stopRecording()
                        return filename
                    response, rt = k, t0
                    break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()

            # feedback (gaze-validated)
            if response == "space":
                # any committed target fixation during trial counts
                recently_fixated_target = (last_fix_on_target_time is not None)
            
                # keypress fallback: near *current* target at press?
                on_keypress_fixated_target = False
                if target_index is not None:
                    tgx, tgy = gabors[target_index].pos  # centered coords
                    d_curr = ((score_gx_c - tgx)**2 + (score_gy_c - tgy)**2) ** 0.5
                    if d_curr <= score_accept_radius_px():
                        on_keypress_fixated_target = True
            
                # jump forgiveness: if the target jumped recently, also accept proximity to the *previous* target
                near_prev_after_jump = False
                if (not on_keypress_fixated_target) and (score_prev_target_pos_c is not None):
                    if (rt - score_target_change_time) <= score_change_grace:
                        px, py = score_prev_target_pos_c
                        d_prev = ((score_gx_c - px)**2 + (score_gy_c - py)**2) ** 0.5
                        if d_prev <= score_accept_radius_px():
                            near_prev_after_jump = True
            
                is_correct = recently_fixated_target or on_keypress_fixated_target or near_prev_after_jump
            
                # if we accepted via a fallback and no time logged yet, fill it for CSV
                if is_correct and (last_fix_on_target_time is None):
                    last_fix_on_target_time = rt
                    last_committed_fix_idx  = target_index
            
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

            response_num = 1 if response == "space" else 0         
            writer.writerow([
                                "dynamic task", participant_id, trial + 1, int(target_present),
                                response_num, int(is_correct), rt,
                                num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2),
                                trial_mode,                 # NBackK: 1/2/4 or 'RND'
                                nback_used_seq,             # NBackUsedSeq (will stay [] for 'RND')
                                fix_log,
                                round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                                last_committed_fix_idx if last_committed_fix_idx is not None else ""
                            ])
