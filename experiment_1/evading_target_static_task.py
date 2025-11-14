#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:22:50 2025

@author: herttaleinonen

    Static ‘evading target’ task with static distractor Gabor stimulus. The target’s position is dynamically manipulated 
    in relation to the participant’s eye movements. Shortly after the search begins the target appears at locations 
    corresponding to −8 to −1 prior fixations (n-back), by replacing distractor Gabors that have been previously inspected. 
    
      - per-trial fixed n-back and distractor camo:
      - On target-present trials, the target appears at n s by taking over a distractor (appear_delay).
      - On that frame, all other distractors re-randomize (masked reveal).
      - Fixations are tracked continuously, but n-back retargeting only activates after the target appears (>= n s).        
"""

import os
import csv
import random
import math
import numpy as np
from psychopy import visual, core, event

from config import (
    grid_size_x, grid_size_y, cell_size,
    num_trials, trial_duration, feedback_duration, timeout_feedback_text,
    orientations, spatial_frequencies, target_orientation,
    target_sf, movement_delay
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

# -------------------------------------------------------------------

def run_evading_target_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):

    sw, sh = screen_width, screen_height
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # ---------- Gaze & capture parameters ----------
    prev_gx, prev_gy   = sw/2.0, sh/2.0     # EMA state
    ema_alpha          = 0.55
    fix_radius_px      = cell_size          # ~35 px circle
    min_fix_frames     = 4                  # ~65 ms @ 60 Hz
    capture_radius_px  = 2.0 * cell_size    # snap a fixation to a Gabor if within this pixel radius
    appear_delay_s     = 0.5                # target appears at 500 ms
    min_gaze_sep_px    = 1.25 * cell_size   # avoid landing right under gaze

    holdoff_by_k  = {                       # seconds, to avoid ping-ponging
        1: 0.35,
        2: 0.20,
        4: 0.12
    }
    # ------------------------------------------------
    
    # acceptance radius for correctness 
    def score_accept_radius_px():
        sigma = cell_size / 6.0
        return max(2.0 * capture_radius_px, 8.0 * sigma)

    # grid offsets (centered PsychoPy pixels)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2
    
    # --- helper: screen-pixel gaze -> centered pix -> grid cell (ix, iy) ---
    def pix_to_grid(fx_c, fy_c):
        gx = (fx_c - off_x) / cell_size
        gy = (fy_c - off_y) / cell_size
        ix = int(np.clip(np.round(gx), 0, grid_size_x - 1))
        iy = int(np.clip(np.round(gy), 0, grid_size_y - 1))
        return ix, iy
   
    # ----- Cedrus open (optional) -----
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/respond.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    # --- Instruction screen with example stimuli (text + icons) ---
    inst = visual.TextStim(
        win,
        text=("In this task you will see moving objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press the GREEN button as soon as you see the target.\n"
              "If you do not find the target, no not press anything.\n"
              "Each trial you have 7 seconds to respond.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    
    # Draw example Gabors to the instruction screen
    example_stims = []
    row_y   = -int(sh * 0.22)
    gap     = int(cell_size * 0.6)
    extra   = int(cell_size * 5.0)
    tgt_dy  = 0

    ori_list = list(orientations)
    n_d      = len(ori_list)
    row_w_d  = n_d * cell_size + (n_d - 1) * gap
    start_x  = -row_w_d // 2 + cell_size // 2

    for i, ori in enumerate(ori_list):
        g = visual.GratingStim(
            win, tex='sin', mask='gauss', size=cell_size,
            sf=random.choice(spatial_frequencies),
            ori=ori, phase=0.25, units='pix'
        )
        x = start_x + i * (cell_size + gap)
        g.pos = (x, row_y)
        example_stims.append(g)

    tgt_x = start_x + (n_d - 1) * (cell_size + gap) + cell_size // 2 + extra
    tgt_y = row_y + tgt_dy
    tgt = visual.GratingStim(
        win, tex='sin', mask='gauss', size=cell_size,
        sf=target_sf, ori=target_orientation, phase=0.25, units='pix'
    )
    tgt.pos = (tgt_x, tgt_y)
    example_stims.append(tgt)

    lab_d = visual.TextStim(win, text="Distractors", color='white', height=22,
                            pos=((start_x + (start_x + (n_d - 1)*(cell_size + gap))) / 2.0,
                                 row_y - int(0.12 * sh)),
                            units='pix')
    lab_t = visual.TextStim(win, text="Target (45°)", color='white', height=22,
                            pos=(tgt_x, tgt_y - int(0.12 * sh)), units='pix')

    inst.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
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
    event.clearEvents(eventType='keyboard')

    # ---------- helpers ----------
    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    def nearest_distractor_index(centers_pix, fx_c, fy_c, exclude_idx=None):
        d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
        if exclude_idx is not None:
            d2[exclude_idx] = np.inf
        j = int(np.argmin(d2))
        return None if np.isinf(d2[j]) else j
    
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
            # Put tracker in idle mode before recording
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
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID',
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','Target Trajectory','NBackK','NBackUsedSeq', 'Fixations',
            'FixOnTargetTime(s)','LastFixIndex', 'CalibrationDrift(deg)'
        ])
        
        # --- Progress text shown during feedback ---
        progress_text = visual.TextStim(
            win, text="", color='white', height=28,
            pos=(0, -int(screen_height * 0.18)), units='pix'
        )
        
        # ---- Balanced per-trial "mode": 1-back, 2-back, 4-back, or RANDOM target (¼ each) ----
        def make_balanced_k_sequence(n_trials):
            ks = []
            base = n_trials // 4
            rem  = n_trials % 4
            for k in (1, 2, 4, 'RND'):
                ks.extend([k] * base)
            if rem:
                extras = [1, 2, 4, 'RND']
                random.shuffle(extras)
                ks.extend(extras[:rem])
            random.shuffle(ks)
            return ks

        k_seq = make_balanced_k_sequence(num_trials)
        k_idx = 0

        for t in range(num_trials):
            # 500ms fixation + drift measurement (also draws the cross)
            drift_deg = measure_fixation_drift(t+1, duration=0.5)
            
            # clean input buffers after the fix-check
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')
            if cedrus and hasattr(cedrus, "clear_response_queue"):
                cedrus.clear_response_queue()

            # per-trial k
            # Target always present, and pick the mode (1/2/4 or 'RND') for this trial
            tp = True
            trial_k = k_seq[k_idx]; k_idx += 1
            
            # Number of Gabors defined here
            n  = random.choice([10])
            
            # Generate the positions (grid coords)
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos:
                    pos.append((x,y))
            tgt_idx = None
            
            # Create Gabors (all start as distractors; target revealed later at 0.5 s)
            gabors = []
            centers_pix = []
            for i in range(n):
                g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                       sf=random.choice(spatial_frequencies),
                                       ori=random.choice(orientations), phase=0.25, units='pix')
                px = off_x + pos[i][0]*cell_size
                py = off_y + pos[i][1]*cell_size
                g.pos = (px, py)
                gabors.append(g)
                centers_pix.append((px, py))
            centers_pix = np.array(centers_pix, dtype=float)

            # update noise
            noise_img = generate_noise(sw, sh, noise_grain)
            noise_stim = visual.ImageStim(win, image=noise_img,
                                          size=(sw,sh), units='pix', interpolate=False)

            # EyeLink
            clk = core.Clock()
            if el_tracker:
                # Put tracker in idle mode before recording
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                
                # Send message "TRIALID" to mark the start of a trial
                el_tracker.sendMessage(f'TRIALID {t+1}')
                
                # Start recording
                el_tracker.startRecording(1,1,1,1)
                core.wait(0.1)
                
                # Log a message to mark the onset of the stimulus
                el_tracker.sendMessage('stimulus_onset')
            event.clearEvents(eventType='keyboard')
            
            # scoring-only gaze state 
            inspected_idxs   = []
            inspected_times  = []
            nback_used_seq   = []
            fix_log          = []
            target_traj      = []

            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            last_fix_on_target_time = None
            last_committed_fix_idx  = None
            score_inside_count      = 0
            score_gx_c, score_gy_c  = 0.0, 0.0
            score_prev_target_pos_c = None
            score_target_change_time = -1e9
            score_change_grace       = 0.8

            # helper to mask a reveal/jump by re-randomizing all distractors
            def refresh_all_distractors(exclude_idx=None):
                for j in range(n):
                    if exclude_idx is not None and j == exclude_idx:
                        continue
                    gabors[j].ori = random.choice(orientations)
                    gabors[j].sf  = random.choice(spatial_frequencies)

            arm_at   = clk.getTime() + appear_delay_s
            armed    = False

            response, rt = None, None
            while clk.getTime() < trial_duration:
                now_t = clk.getTime()

                # reveal target
                if tp and (not armed) and (now_t >= arm_at):
                    tgt_idx = random.randrange(n)
                    gabors[tgt_idx].ori = target_orientation
                    gabors[tgt_idx].sf  = target_sf
                    refresh_all_distractors(exclude_idx=tgt_idx)
                    armed = True
                    target_traj.append(pos[tgt_idx])

                # --- read & smooth gaze ---
                if el_tracker:
                    s = el_tracker.getNewestSample()
                    if s and s.isRightSample():
                        eye = s.getRightEye()
                    elif s and s.isLeftSample():
                        eye = s.getLeftEye()
                    else:
                        eye = None
                    if eye:
                        rx, ry = eye.getGaze()
                        prev_gx = float(np.clip(ema_alpha*rx + (1-ema_alpha)*prev_gx, 0, sw-1))
                        prev_gy = float(np.clip(ema_alpha*ry + (1-ema_alpha)*prev_gy, 0, sh-1))
                        
                        # for scoring (correct/incorrect) convert screen → centered for scoring 
                        score_gx_c = prev_gx - (sw/2.0)
                        score_gy_c = (sh/2.0) - prev_gy
                        
                        # If target exists, accept even a single-frame dwell inside a large radius
                        if tgt_idx is not None:
                            tgx, tgy = gabors[tgt_idx].pos
                            d = ((score_gx_c - tgx)**2 + (score_gy_c - tgy)**2) ** 0.5
                            if d <= score_accept_radius_px():
                                score_inside_count += 1
                                if (score_inside_count >= 1) and (last_fix_on_target_time is None):
                                    last_fix_on_target_time = now_t
                                    last_committed_fix_idx  = tgt_idx
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
                            
                            # map fixation to nearest Gabor center (in pixels), with capture radius
                            if (not committed_this_cluster) and (cluster_len == min_fix_frames):
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                                
                                # log fixation location (grid cell) and add to inspection history
                                ix, iy = pix_to_grid(fx_c, fy_c)
                                fix_log.append((ix, iy))

                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j
                                        
                                # if no capture, impute to nearest *distractor* (exclude target if known)
                                if current_idx is None:
                                    imputed = nearest_distractor_index(centers_pix, fx_c, fy_c, exclude_idx=tgt_idx)
                                    if (imputed is not None) and (not inspected_idxs or inspected_idxs[-1] != imputed):
                                        inspected_idxs.append(imputed)
                                        inspected_times.append(now_t)
                                else:
                                    if (tgt_idx is None) or (current_idx != tgt_idx):
                                        if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                            inspected_idxs.append(current_idx)
                                            inspected_times.append(now_t)
                                committed_this_cluster = True
                        else:
                            if cluster_len >= min_fix_frames:
                                
                                # n-back retargeting (only after reveal) — skip if mode is 'RND'
                                if armed and (tgt_idx is not None):
                                    if trial_k == 'RND':
                                        pool = [i for i in range(n) if i != tgt_idx]
                                        if pool:
                                            prev_target_pos_c = gabors[tgt_idx].pos if tgt_idx is not None else None
                                            cand_idx = random.choice(pool)
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            nback_used_seq.append('RND')
                                            target_traj.append(pos[tgt_idx])
                                            if prev_target_pos_c is not None:
                                                score_prev_target_pos_c = prev_target_pos_c
                                                score_target_change_time = now_t
                                    else:
                                        k = int(trial_k)
                                        prior_len = len(inspected_idxs)
                                        if prior_len >= k:
                                            cand_hist_idx = prior_len - k
                                            cand_idx      = inspected_idxs[cand_hist_idx]
                                            cand_time     = inspected_times[cand_hist_idx]
                                            if cand_idx != tgt_idx:
                                                age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)
                                                gaze_x = cluster_cx - sw/2.0
                                                gaze_y =  sh/2.0 - cluster_cy
                                                dx = centers_pix[cand_idx][0] - gaze_x
                                                dy = centers_pix[cand_idx][1] - gaze_y
                                                sep_ok = (dx*dx + dy*dy) >= (min_gaze_sep_px * min_gaze_sep_px)
                                                if age_ok and sep_ok:
                                                    prev_target_pos_c = gabors[tgt_idx].pos if tgt_idx is not None else None
                                                    gabors[cand_idx].ori = target_orientation
                                                    gabors[cand_idx].sf  = target_sf
                                                    gabors[tgt_idx].ori  = random.choice(orientations)
                                                    gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                                    tgt_idx = cand_idx
                                                    nback_used_seq.append(k)
                                                    target_traj.append(pos[tgt_idx])
                                                    if prev_target_pos_c is not None:
                                                        score_prev_target_pos_c = prev_target_pos_c
                                                        score_target_change_time = now_t
                                
                                # global camo mask (re-randomize ALL distractors; keep target as-is)
                                refresh_all_distractors(exclude_idx=tgt_idx)
                            
                            # reset cluster for next fixation
                            cluster_cx, cluster_cy = prev_gx, prev_gy
                            cluster_len = 1
                            committed_this_cluster = False

                # Draw scene
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()

                # ----- responses: Cedrus first (ANY button = SPACE), then keyboard -----
                if cedrus and _cedrus_any_pressed(cedrus):
                    response = 1
                    rt = clk.getTime()
                    break

                keys = event.getKeys(keyList=['space','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()
                        return filename
                    if k == 'space':
                        response = 1
                        rt = t0
                        break

                core.wait(movement_delay)
            
            # Log a message to mark the offset of the stimulus
            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                
                # Stop recording
                el_tracker.stopRecording()

            # feedback (gaze-validated)
            if response is None:
                fb_text = timeout_feedback_text
                resp_csv = 0
                rt_str   = ''
                corr     = 0
            else:
                # any committed target fixation during trial counts
                recently_fixated_target = (last_fix_on_target_time is not None)
                
                # keypress fallback: near *current* target at press?
                on_keypress_fixated_target = False
                if tgt_idx is not None:
                    tgx, tgy = gabors[tgt_idx].pos
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
                    last_committed_fix_idx  = tgt_idx

                fb_text = 'Correct' if is_correct else 'Incorrect'
                resp_csv = 1
                rt_str   = rt
                corr     = int(is_correct)

            fb = visual.TextStim(win, text=fb_text, color='white', height=40, units='pix')
            
            # Update "X/total" text
            progress_text.text = f"{t+1}/{num_trials}"
            
            # Draw feedback + progress count
            fb.draw(); progress_text.draw()
            win.flip()
            core.wait(feedback_duration)
            
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')
            
            # Write to csv
            writer.writerow([
                'evading target static task',
                participant_id,
                t+1, int(tp), resp_csv, corr, rt_str,
                n, pos,
                (target_traj if len(target_traj)>0 else []),
                trial_k,
                nback_used_seq,
                fix_log,
                round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                last_committed_fix_idx if last_committed_fix_idx is not None else "", drift_deg
            ])

    return filename
