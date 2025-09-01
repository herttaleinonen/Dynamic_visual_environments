#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 26 12:22:50 2025

@author: herttaleinonen

    
    Static ‘evading target’ task with static distractor Gabor stimulus. The target’s position is dynamically manipulated 
    in relation to the participant’s eye movements. Shortly after the search begins the target appears at locations 
    corresponding to −5 to −1 prior fixations (n-back), by replacing distractor Gabors that have been previously inspected. 
    
      - Per-trial fixed n-back and distractor camo
      - On target-present trials, the target appears at n ms by taking over a distractor (appear_delay).
      - On that frame, all other distractors re-randomize.
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

def run_evading_target_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):


    sw, sh = screen_width, screen_height
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # ---------- Gaze & capture parameters ----------
    prev_gx, prev_gy   = sw/2.0, sh/2.0
    ema_alpha          = 0.55              # smoothing (higher = snappier)
    fix_radius_px      = 45                # fixation radius (px)
    min_fix_frames     = 4                 # frames to commit (~65 ms at 60 Hz)
    capture_radius_px  = 2.0 * cell_size   # generous capture to reduce misses
    appear_delay_s     = 0.5               # target reveals at 500 ms
    min_gaze_sep_px   = 1.25 * cell_size   # small separation from *current* gaze to avoid quasi 0-back landings
    holdoff_by_k  = {                      # SOA before a −k location is allowed (seconds)
                         1: 0.25,          # strongest delay for k=1
                         2: 0.20,
                         3: 0.15,
                         4: 0.10,
                         5: 0.08,
                         }
    # ------------------------------------------------

    # grid offsets (centered PsychoPy pixels)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # fixation cross screen
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    # instructions screen
    inst = visual.TextStim(win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press 'right arrow key ' if you see the target, 'left arrow key' if not.\n"
              "Each trial you have 5 seconds to decide, try to make the decision as fast as possible.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip(); event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')  # keep buffer clean

    # create Gaussian noise background
    # grain size defined in the function call
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

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','NBackK','NBackUsedSeq'
        ])

        # --------- Balanced present/absent and EXACTLY balanced per-trial NBackK among PRESENT trials ---------
        present_ratio = 0.5  # 50/50 split (adjust if you want)

        num_present_raw = int(round(num_trials * present_ratio))
        # Force present count to nearest LOWER multiple of 5 so k=1..5 can be perfectly balanced
        num_present = (num_present_raw // 5) * 5
        present_flags = [1] * num_present + [0] * (num_trials - num_present)
        random.shuffle(present_flags)

        def make_balanced_k_sequence(n_present):
            """Return a list with equal counts of 1..5 whose length == n_present."""
            blocks = n_present // 5
            ks = []
            for _ in range(blocks):
                block = [1, 2, 3, 4, 5]
                random.shuffle(block)   # local shuffle inside each block
                ks.extend(block)
            random.shuffle(ks)          # global shuffle across all blocks
            return ks

        k_seq = make_balanced_k_sequence(num_present)  # len == num_present
        k_idx = 0  # cursor into k_seq
        # ------------------------------------------------------------------------------------------------------

        for t in range(num_trials):
            # fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            event.clearEvents(eventType='keyboard')

            # --- balanced present/absent and per-trial k ---
            tp = bool(present_flags[t])  # 1 -> True, 0 -> False
            if tp:
                trial_k = k_seq[k_idx]
                k_idx += 1
            else:
                trial_k = ''  # absent trials carry no k

            # layout
            n  = random.choice([10])
            # unique positions (grid coords)
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos:
                    pos.append((x,y))
            # start with NO target; it will appear at 0.5 s if tp==True
            tgt_idx = None

            # stimuli + centers (all start as distractors)
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

            # static noise
            noise_img = generate_noise(sw, sh, noise_grain)
            noise_stim = visual.ImageStim(win, image=noise_img,
                                          size=(sw,sh), units='pix', interpolate=False)

            # EyeLink
            clk = core.Clock()
            if el_tracker:
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {t+1}')
                el_tracker.startRecording(1,1,1,1)
                core.wait(0.1)
                el_tracker.sendMessage('stimulus_onset')
            event.clearEvents(eventType='keyboard')

            # per-trial state
            inspected_idxs   = []               # distractor indices inspected (post-appearance)
            inspected_times  = []               # MATCHED timestamps (clk.getTime()) for each entry above
            nback_used_seq   = []

            # fixation clustering state
            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # camo timer (present: local swaps; absent: full refresh)
            next_swap_time = clk.getTime() + random.uniform(0.35, 0.85)

            def swap_two_distractors(exclude=None):
                nonlocal tgt_idx
                if exclude is None:
                    exclude = set()
                pool = [i for i in range(n) if (tgt_idx is None or i != tgt_idx) and i not in exclude]
                if len(pool) < 2:
                    return False
                i1, i2 = random.sample(pool, 2)
                gabors[i1].ori, gabors[i2].ori = gabors[i2].ori, gabors[i1].ori
                gabors[i1].sf,  gabors[i2].sf  = gabors[i2].sf,  gabors[i1].sf
                return True

            def refresh_all_distractors(exclude_idx=None):
                for j in range(n):
                    if exclude_idx is not None and j == exclude_idx:
                        continue
                    gabors[j].ori = random.choice(orientations)
                    gabors[j].sf  = random.choice(spatial_frequencies)

            # target appearance arming
            arm_at         = clk.getTime() + appear_delay_s
            armed          = False

            # main loop
            response, rt = None, None
            while clk.getTime() < trial_duration:
                now_t = clk.getTime()

                # time-based camo
                if now_t >= next_swap_time:
                    if tp and (tgt_idx is not None):
                        swap_two_distractors(exclude=None)
                        next_swap_time = now_t + random.uniform(0.25, 0.55)
                    else:
                        refresh_all_distractors(exclude_idx=None)
                        next_swap_time = now_t + random.uniform(0.35, 0.85)

                # reveal target at 0.5 s in target-present trials
                if tp and (not armed) and (now_t >= arm_at):
                    tgt_idx = random.randrange(n)
                    gabors[tgt_idx].ori = target_orientation
                    gabors[tgt_idx].sf  = target_sf
                    refresh_all_distractors(exclude_idx=tgt_idx)
                    armed = True

                # sample gaze + EMA
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
                                # --- commit fixation ---
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j

                                # ALWAYS record fixation into history from trial start (pre- and post-reveal)
                                if current_idx is None:
                                    # impute to nearest distractor, excluding target if present
                                    imputed = nearest_distractor_index(centers_pix, fx_c, fy_c, exclude_idx=tgt_idx)
                                    if (imputed is not None) and (not inspected_idxs or inspected_idxs[-1] != imputed):
                                        inspected_idxs.append(imputed)
                                        inspected_times.append(now_t)
                                else:
                                    # skip adding the target itself if it's already on screen
                                    if (tgt_idx is None) or (current_idx != tgt_idx):
                                        if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                            inspected_idxs.append(current_idx)
                                            inspected_times.append(now_t)
                                
                                # n-back retargeting: ONLY after reveal
                                if armed and tp and (trial_k != ''):
                                    k = int(trial_k)
                                    # Use history excluding the *current* fixation to avoid 0-back
                                    prior_len = len(inspected_idxs) - 1
                                    if prior_len >= k:
                                        cand_hist_idx = prior_len - k          # index in inspected_* arrays
                                        cand_idx      = inspected_idxs[cand_hist_idx]
                                        cand_time     = inspected_times[cand_hist_idx]

                                        if cand_idx != tgt_idx:
                                            # (a) stimulus onset asynchrony (SOA) since that fixation
                                            age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)

                                            # (b) small separation from *current gaze* so −1 doesn't land right next to you
                                            gaze_x = cluster_cx - sw/2.0
                                            gaze_y =  sh/2.0  - cluster_cy
                                            dx = centers_pix[cand_idx][0] - gaze_x
                                            dy = centers_pix[cand_idx][1] - gaze_y
                                            sep_ok = (dx*dx + dy*dy) >= (min_gaze_sep_px * min_gaze_sep_px)

                                            if age_ok and sep_ok:
                                                # jump + mask 
                                                gabors[cand_idx].ori = target_orientation
                                                gabors[cand_idx].sf  = target_sf
                                                gabors[tgt_idx].ori  = random.choice(orientations)
                                                gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                                tgt_idx = cand_idx
                                                nback_used_seq.append(k)
                                                refresh_all_distractors(exclude_idx=tgt_idx)


                                committed_this_cluster = True

                        else:
                            # leaving cluster: commit if long enough and not yet committed
                            if (cluster_len >= min_fix_frames) and (not committed_this_cluster):
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j

                                # record fixation history from trial start
                                if current_idx is None:
                                    imputed = nearest_distractor_index(centers_pix, fx_c, fy_c, exclude_idx=tgt_idx)
                                    if (imputed is not None) and (not inspected_idxs or inspected_idxs[-1] != imputed):
                                        inspected_idxs.append(imputed)
                                else:
                                    if (tgt_idx is None) or (current_idx != tgt_idx):
                                        if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                            inspected_idxs.append(current_idx)

                                # n-back retargeting: ONLY after reveal
                                if armed and tp and (trial_k != ''):
                                    k = int(trial_k)
                                    # Use history excluding the *current* fixation to avoid 0-back
                                    prior_len = len(inspected_idxs) - 1
                                    if prior_len >= k:
                                        cand_hist_idx = prior_len - k          # index in inspected_* arrays
                                        cand_idx      = inspected_idxs[cand_hist_idx]
                                        cand_time     = inspected_times[cand_hist_idx]

                                        if cand_idx != tgt_idx:
                                            # (a) SOA since that fixation
                                            age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)

                                            # (b) small separation from *current gaze* so −1 doesn't land right next to you
                                            gaze_x = cluster_cx - sw/2.0
                                            gaze_y =  sh/2.0  - cluster_cy
                                            dx = centers_pix[cand_idx][0] - gaze_x
                                            dy = centers_pix[cand_idx][1] - gaze_y
                                            sep_ok = (dx*dx + dy*dy) >= (min_gaze_sep_px * min_gaze_sep_px)

                                            if age_ok and sep_ok:
                                                # jump + mask 
                                                gabors[cand_idx].ori = target_orientation
                                                gabors[cand_idx].sf  = target_sf
                                                gabors[tgt_idx].ori  = random.choice(orientations)
                                                gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                                tgt_idx = cand_idx
                                                nback_used_seq.append(k)
                                                refresh_all_distractors(exclude_idx=tgt_idx)

                            # start new cluster
                            cluster_cx, cluster_cy = prev_gx, prev_gy
                            cluster_len = 1
                            committed_this_cluster = False

                # draw scene
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()

                keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()
                        return filename
                    # for CSV file "Response" column, 1 if pressed yes, 0 if no
                    response = 1 if k=='right' else 0
                    rt = t0
                    break

                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback
            if response is None:
                # for CSV file "Response" column
                fb_text = timeout_feedback_text
                resp_str = ''      # leave empty for timeouts
                rt_str   = ''      # leave empty for RT
                corr     = 0
            else:
                # response == 1 means “right arrow” (target-present), response == 0 means “left arrow” (target-absent)
                corr = int((response == 1 and tp) or (response == 0 and not tp))
                # feedback on screen for participant
                fb_text = 'Correct' if corr else 'Incorrect'
                resp_str = 'target' if response == 1 else 'distractor'
                rt_str = rt

            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard') # flush after feedback

            # log (NBackK = fixed per-trial k; NBackUsedSeq = ks actually used each jump, e.g. [-4, -4, -4, -4,])
            writer.writerow([
                t+1, int(tp), resp_str, corr, rt_str,
                n, pos, (pos[tgt_idx] if (tp and tgt_idx is not None) else None),
                (trial_k if tp else ''), nback_used_seq
            ])

    return filename
