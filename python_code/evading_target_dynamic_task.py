#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 3 12:15:16 2025

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
import numpy as np
from psychopy import visual, core, event

from config import (
    grid_size_x, grid_size_y, cell_size, DIAGONAL_SCALE,
    num_trials, trial_duration, feedback_duration, timeout_feedback_text,
    orientations, spatial_frequencies, target_orientation,
    target_sf, transition_steps, movement_delay
)

# --------- Cedrus–response box setup (safe if pyxid2 missing) ---------
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
        if hasattr(dev, "reset_base_timer"): dev.reset_base_timer()
        if hasattr(dev, "reset_rt_timer"):   dev.reset_rt_timer()
        if hasattr(dev, "clear_response_queue"): dev.clear_response_queue()
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
                dev.get_next_response()  # pop everything
            core.wait(0.005)
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
    except Exception as e:
        print(f"[Cedrus] flush failed: {e}")


def _cedrus_any_pressed(dev) -> bool:
    """True if *any* Cedrus key press event is available (drains one batch)."""
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        if not dev.has_response():
            return False

        pressed_seen = False
        # Drain everything queued this instant and see if any was a press
        while dev.has_response():
            r = dev.get_next_response()
            if bool(r.get("pressed", False)):  # <-- default False
                pressed_seen = True

        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return pressed_seen
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False


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

# Balance diagonal movement and avoid corners 
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
    holdoff_by_k       = { 1: 0.45, 2: 0.10, 4: 0.00, 8: 0.00 }  # seconds, avoid ping-ponging
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

    # For CSV: keep track of Gabor speed (px per s)
    speed_px_per_sec = (4 * cell_size) / (transition_steps * movement_delay)
    
    # ----- Cedrus open (optional) -----
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/respond.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")
    
    # --- Instruction screen with example stimuli (text + icons) ---
    instruction_text = visual.TextStim(
        win,
        text=("In this task you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press the GREEN button as soon as you see the target.\n"
              "If you do not find the target, no not press anything.\n"
              "Each trial you have 7 seconds to respond.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start."),
        color='white', height=30, wrapWidth=screen_width * 0.8, units='pix'
    )
    
    # Layout params (row for distractors; target pushed further right)
    row_y  = -int(screen_height * 0.22)   # y-position for the icons row
    gap    = int(cell_size * 0.6)         # spacing between distractors
    extra  = int(cell_size * 5.0)         # extra separation before target
    tgt_dy = -int(0.00 * screen_height)   # small vertical nudge for target (set 0 for same row)
    
    ori_list = list(orientations)         # distractor orientations from config
    n_d = len(ori_list)
    
    # width of the distractor row (without the target)
    row_w_d = n_d * cell_size + (n_d - 1) * gap
    start_x = -row_w_d // 2 + cell_size // 2
    
    example_stims = []
    rng = np.random.default_rng(0)        # deterministic SF pick for a stable instruction screen
    
    # Distractors row
    for i, ori in enumerate(ori_list):
        sf = rng.choice(spatial_frequencies)
        g = visual.GratingStim(
            win, tex='sin', mask='gauss', size=cell_size,
            sf=sf, ori=ori, phase=0.25, units='pix'
        )
        x = start_x + i * (cell_size + gap)
        g.pos = (x, row_y)
        example_stims.append(g)
    
    # Target (45°), placed to the right with extra separation (and slight vertical offset)
    tgt_x = start_x + (n_d - 1) * (cell_size + gap) + cell_size // 2 + extra
    tgt_y = row_y + tgt_dy
    tgt = visual.GratingStim(
        win, tex='sin', mask='gauss', size=cell_size,
        sf=target_sf, ori=target_orientation, phase=0.25, units='pix'
    )
    tgt.pos = (tgt_x, tgt_y)
    example_stims.append(tgt)
    
    # Labels
    lab_d = visual.TextStim(
        win, text="Distractors", color='white', height=22, units='pix',
        pos=((start_x + (start_x + (n_d - 1) * (cell_size + gap))) / 2.0,
             row_y - int(0.12 * screen_height))
    )
    lab_t = visual.TextStim(
        win, text="Target (45°)", color='white', height=22, units='pix',
        pos=(tgt_x, tgt_y - int(0.12 * screen_height))
    )
    
    # Draw everything in one frame
    instruction_text.draw()
    for s in example_stims:
        s.draw()
    lab_d.draw()
    lab_t.draw()
    win.flip()

    # ---- Start gate: Cedrus (any button) OR keyboard (Enter), ESC abort ----
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
        if hasattr(cedrus, "clear_response_queue"):
            cedrus.clear_response_queue()
    else:
        event.waitKeys(keyList=['return', 'enter'])
    _cedrus_flush(cedrus)
    event.clearEvents(eventType='keyboard')  # keep buffer clean
    
    
    # Measure distance of gaze from the center of the fixation cross in degrees
    def measure_fixation_drift(trial_idx, duration=0.5):
        """
        Show a fixation cross for `duration` sec while recording from EyeLink,
        return median angular error (deg) from center. Empty string if unavailable.
        """
        cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    
        # No tracker: just show the cross for the same duration
        if not el_tracker:
            cross.draw(); win.flip(); core.wait(duration)
            return ""
    
        # Short recording for the fix-check
        try:
            el_tracker.setOfflineMode()
            el_tracker.sendMessage(f'FIXCHECK_START {trial_idx}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)  # settle
        except Exception:
            cross.draw(); win.flip(); core.wait(duration)
            return ""
    
        samples = []
        clk_fix = core.Clock()
        while clk_fix.getTime() < duration:
            cross.draw()
            win.flip()
    
            s = el_tracker.getNewestSample()
            eye = s.getRightEye() if (s and s.isRightSample()) else (s.getLeftEye() if (s and s.isLeftSample()) else None)
            if eye is not None:
                rx, ry = eye.getGaze()  # screen px (0,0)=TL, +y down
                if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                    # convert to centered pixels (+y up), then to deg using cell_size (px/deg)
                    gx = float(rx) - (screen_width / 2.0)
                    gy = (screen_height / 2.0) - float(ry)
                    dist_deg = np.hypot(gx, gy) / float(cell_size)
                    samples.append(dist_deg)
            core.wait(0.005)
    
        try:
            el_tracker.sendMessage('FIXCHECK_END')
            el_tracker.stopRecording()
        except Exception:
            pass
    
        return round(float(np.median(samples)), 3) if samples else ""


    # Open a CSV file for behavioural results
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory", "Speed (px/s)",
                         "NBackK", "NBackUsedSeq", "Fixations",
                         "FixOnTargetTime(s)", "LastFixIndex", 'CalibrationDrift(deg)' ]) 
        
        # --- Progress text shown during feedback ---
        progress_text = visual.TextStim(
            win,
            text="",
            color='white',
            height=28,
            pos=(0, -int(screen_height * 0.18)),  # a bit below the center; tweak if needed
            units='pix'
        )

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

        # helper to mask a reveal/jump by re-randomizing all distractors
        def refresh_all_distractors(gabors, exclude_idx=None):
            for j, g in enumerate(gabors):
                if exclude_idx is not None and j == exclude_idx:
                    continue
                g.ori = random.choice(orientations)
                g.sf  = random.choice(spatial_frequencies)

        for trial in range(num_trials):
            # 500 ms fixation + drift measurement (also draws the cross)
            drift_deg = measure_fixation_drift(trial + 1, duration=0.5)
            
            # clean input buffers after the fix-check
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')
            if cedrus and hasattr(cedrus, "clear_response_queue"):
                cedrus.clear_response_queue()

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
            nback_used_seq    = []   # log which k actually fired
            cluster_cx, cluster_cy = screen_width/2.0, screen_height/2.0
            cluster_len = 0
            committed_this_cluster = False

            # Appearance gate
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
            score_change_grace       = 0.8   # s


            while trial_clock.getTime() < trial_duration:
                now_t = trial_clock.getTime()

                # --- REVEAL target at 0.5 s (only if target-present) ---
                if target_present and (not armed) and (now_t >= arm_at):
                    target_index = random.randrange(num_gabors)
                    gabors[target_index].ori = target_orientation
                    gabors[target_index].sf  = target_sf
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

                    # fixation clustering (in screen coords for cluster center)
                    dx = prev_gx - cluster_cx
                    dy = prev_gy - cluster_cy
                    inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                
                    if inside:
                        cluster_len += 1
                        w = 1.0 / max(1, cluster_len)
                        cluster_cx = (1 - w)*cluster_cx + w*prev_gx
                        cluster_cy = (1 - w)*cluster_cy + w*prev_gy
                
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
                        if cluster_len >= min_fix_frames:
                            # n-back retargeting (only after reveal) — skip if mode is 'RND'
                            if armed and (trial_mode != 'RND') and (target_index is not None):
                                k = int(trial_mode)  # k ∈ {1,2,4}
                                prior_len = len(inspected_indices) - 1  # exclude current (leaving) fixation
                                if prior_len >= k:
                                    cand_hist_idx = prior_len - k
                                    cand_idx      = inspected_indices[cand_hist_idx]
                                    cand_time     = inspected_times[cand_hist_idx]
                    
                                    if cand_idx != target_index:
                                        age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)
                                        gx, gy = cluster_cx, cluster_cy
                                        dxg = gabors[cand_idx].pos[0] - gx
                                        dyg = gabors[cand_idx].pos[1] - gy
                                        sep_mult_by_k = {1: 2.25, 2: 1.50, 4: 1.00}
                                        sep_req = sep_mult_by_k.get(k, 1.0) * min_gaze_sep_px
                                        sep_ok  = (dxg*dxg + dyg*dyg) >= (sep_req * sep_req)
                    
                                        if age_ok and sep_ok:
                                            prev_target_pos_c = gabors[target_index].pos if target_index is not None else None
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            if target_index is not None:
                                                gabors[target_index].ori = random.choice(orientations)
                                                gabors[target_index].sf  = random.choice(spatial_frequencies)
                                            target_index = cand_idx
                                            nback_used_seq.append(k)
                                            if prev_target_pos_c is not None:
                                                score_prev_target_pos_c = prev_target_pos_c
                                                score_target_change_time = now_t

                            # global camo mask (re-randomize ALL distractors; keep target as-is)
                            refresh_all_distractors(gabors, exclude_idx=target_index)
                    
                        # reset cluster for next fixation
                        cluster_cx, cluster_cy = prev_gx, prev_gy
                        cluster_len = 1
                        committed_this_cluster = False

                # draw scene
                for g in gabors: g.draw()
                win.flip()

                # ----- responses: Cedrus first (ANY button = SPACE), then keyboard -----
                if cedrus and _cedrus_any_pressed(cedrus):
                    response = "space"
                    rt = trial_clock.getTime()
                    break

                keys = event.getKeys(keyList=["space", "escape"], timeStamped=trial_clock)
                if keys:
                    k, t0 = keys[0]
                    if k == "escape":
                        el_tracker.sendMessage('stimulus_offset')
                        el_tracker.stopRecording()
                        return filename
                    if k == "space":
                        response = "space"
                        rt = t0
                        break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()

            # Feedback (gaze-validated)
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

            feedback = visual.TextStim(win, text=feedback_text, color="white", height=40, units='pix')
            
            # Update "X/total" text
            progress_text.text = f"{trial + 1}/{num_trials}"
            
            # Draw feedback + progress count
            feedback.draw()
            progress_text.draw()
            win.flip()
            core.wait(feedback_duration)
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')

            response_num = 1 if response == "space" else 0         
            writer.writerow([
                "evading target dynamic task", participant_id, trial + 1, int(target_present),
                response_num, int(is_correct), rt,
                num_gabors, gabor_trajectory, target_trajectory, round(speed_px_per_sec, 2),
                trial_mode,                 # NBackK: 1/2/4 or 'RND'
                nback_used_seq,             # NBackUsedSeq (will stay [] for 'RND')
                fix_log,
                round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                last_committed_fix_idx if last_committed_fix_idx is not None else "", drift_deg
            ])
