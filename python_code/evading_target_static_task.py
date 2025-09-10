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
    fix_radius_px      = cell_size         # fixation radius is 35x35px
    min_fix_frames     = 4                 # frames to commit (~65 ms at 60 Hz)
    capture_radius_px  = 2.0 * cell_size   # generous capture to reduce misses
    appear_delay_s     = 0.5               # target reveals at 500 ms
    min_gaze_sep_px    = 1.25 * cell_size  # small separation from *current* gaze to avoid quasi 0-back landings

    holdoff_by_k  = {                      # SOA before a −k location is allowed (seconds)
        1: 0.35,                           
        2: 0.20,
        4: 0.12
    }
    # ------------------------------------------------

    # SCORE: lenient acceptance radius (scoring only; does not affect targeting)
    def score_accept_radius_px():
        sigma = cell_size / 6.0  # 'gauss' mask σ
        return max(2.0 * capture_radius_px, 8.0 * sigma)

    # grid offsets (centered PsychoPy pixels)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    def pix_to_grid(fx_c, fy_c):
        # convert centered PsychoPy pixels -> grid cell indices
        gx = (fx_c - off_x) / cell_size
        gy = (fy_c - off_y) / cell_size
        ix = int(np.clip(np.round(gx), 0, grid_size_x - 1))
        iy = int(np.clip(np.round(gy), 0, grid_size_y - 1))
        return ix, iy


    # fixation cross screen
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    # instructions screen
    inst = visual.TextStim(win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press SPACE as soon as you see the target.\n"
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
            'Task Type','Participant ID',
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','Target Trajectory','NBackK','NBackUsedSeq', 'Fixations',
            # SCORE:
            'FixOnTargetTime(s)','LastFixIndex'
        ])

        # (legacy present/absent code not used because target is always present)

        # --------- All trials are target-present; EXACTLY balanced per-trial NBackK across trials ---------
        def make_balanced_k_sequence(n_trials):
            """
            Return a length-n_trials list with counts for conditions in {1, 2, 4, 'RND'}
            as equal as mathematically possible. If n_trials isn't divisible by 4,
            the remainder is distributed randomly across the set.
            """ 
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

        k_seq = make_balanced_k_sequence(num_trials)  # len == num_trials
        k_idx = 0  # cursor into k_seq

        # ------------------------------------------------------------------------------------------------------

        for t in range(num_trials):
            # fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            event.clearEvents(eventType='keyboard')

            # --- per-trial k (always present) ---
            tp = True                   # target always present
            trial_k = k_seq[k_idx]      # pick the K for this trial (1,2,4,'RND')  
            k_idx += 1

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
            
            # will store [(x_px_centered, y_px_centered), ...]
            fix_log = []  
            
            # track all target landings (in grid coords) for this trial
            target_traj = []

            # fixation clustering state
            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # SCORE: scoring-only vars (independent of targeting)
            last_fix_on_target_time = None
            last_committed_fix_idx  = None
            score_inside_count      = 0
            score_gx_c, score_gy_c  = 0.0, 0.0
            score_prev_target_pos_c = None
            score_target_change_time = -1e9
            score_change_grace       = 0.8  # seconds

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

                # reveal target at 0.5 s in target-present trials
                if tp and (not armed) and (now_t >= arm_at):
                    tgt_idx = random.randrange(n)
                    gabors[tgt_idx].ori = target_orientation
                    gabors[tgt_idx].sf  = target_sf
                    refresh_all_distractors(exclude_idx=tgt_idx)
                    armed = True
                    # log first landing in grid coordinates
                    target_traj.append(pos[tgt_idx])

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

                        # SCORE: convert to centered coords for scoring checks
                        score_gx_c = prev_gx - (sw/2.0)
                        score_gy_c = (sh/2.0) - prev_gy
                        # if target exists, accept even a single-frame dwell within large radius
                        if tgt_idx is not None:
                            tgx, tgy = gabors[tgt_idx].pos  # centered coords
                            d = ((score_gx_c - tgx)**2 + (score_gy_c - tgy)**2) ** 0.5
                            if d <= score_accept_radius_px():
                                score_inside_count += 1
                                if (score_inside_count >= 1) and (last_fix_on_target_time is None):
                                    last_fix_on_target_time = now_t
                                    last_committed_fix_idx  = tgt_idx
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
                        
                            if (not committed_this_cluster) and (cluster_len == min_fix_frames):
                                # --- commit fixation (log + add to inspection history) ---
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                        
                                # convert to grid for logs
                                ix, iy = pix_to_grid(fx_c, fy_c)
                                fix_log.append((ix, iy))  # grid coords
                        
                                # map fixation to nearest gabor (capture), else impute to nearest distractor (exclude target)
                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j
                        
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
                        
                                # no retargeting on commit
                                committed_this_cluster = True

                        else:
                            # just LEFT the fixation cluster
                            if cluster_len >= min_fix_frames:
                                # --- (A) Retarget only after reveal ---
                                if armed and (tgt_idx is not None):
                                    if trial_k == 'RND':  
                                        # pick ANY distractor index at random (not the current target)
                                        pool = [i for i in range(n) if i != tgt_idx]
                                        if pool:
                                            # SCORE: remember previous target position before swap
                                            prev_target_pos_c = gabors[tgt_idx].pos if tgt_idx is not None else None

                                            cand_idx = random.choice(pool)
                                            # promote candidate to target; demote old target
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            nback_used_seq.append('RND')
                                            target_traj.append(pos[tgt_idx])

                                            # SCORE: log jump (for press-near-previous forgiveness)
                                            if prev_target_pos_c is not None:
                                                score_prev_target_pos_c = prev_target_pos_c
                                                score_target_change_time = now_t
                                    else:
                                        # n-back = 1,2,4
                                        k = int(trial_k)
                                        prior_len = len(inspected_idxs)  # history up to the last committed fixation
                                        if prior_len >= k:
                                            cand_hist_idx = prior_len - k
                                            cand_idx      = inspected_idxs[cand_hist_idx]
                                            cand_time     = inspected_times[cand_hist_idx]
                                            if cand_idx != tgt_idx:
                                                # SOA & separation checks
                                                age_ok = (now_t - cand_time) >= holdoff_by_k.get(k, 0.0)
                                                gaze_x = cluster_cx - sw/2.0
                                                gaze_y =  sh/2.0 - cluster_cy
                                                dx = centers_pix[cand_idx][0] - gaze_x
                                                dy = centers_pix[cand_idx][1] - gaze_y
                                                sep_ok = (dx*dx + dy*dy) >= (min_gaze_sep_px * min_gaze_sep_px)
                                                if age_ok and sep_ok:
                                                    # SCORE: remember previous target position before swap
                                                    prev_target_pos_c = gabors[tgt_idx].pos if tgt_idx is not None else None

                                                    gabors[cand_idx].ori = target_orientation
                                                    gabors[cand_idx].sf  = target_sf
                                                    gabors[tgt_idx].ori  = random.choice(orientations)
                                                    gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                                    tgt_idx = cand_idx
                                                    nback_used_seq.append(k)
                                                    target_traj.append(pos[tgt_idx])

                                                    # SCORE: log jump (for press-near-previous forgiveness)
                                                    if prev_target_pos_c is not None:
                                                        score_prev_target_pos_c = prev_target_pos_c
                                                        score_target_change_time = now_t

                                # --- (B) Global camo mask on every fixation departure ---
                                refresh_all_distractors(exclude_idx=tgt_idx)
                        
                            # reset cluster for next fixation
                            cluster_cx, cluster_cy = prev_gx, prev_gy
                            cluster_len = 1
                            committed_this_cluster = False


                # draw scene
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()

                keys = event.getKeys(keyList=['space','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()
                        return filename
                    if k == 'space':
                        response = 1    # pressed SPACE to indicate target seen
                        rt = t0
                        break

                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()


            # feedback (gaze-validated scoring overlay)
            if response is None:
                # timeout
                fb_text = timeout_feedback_text
                resp_csv = ''    # leave empty for timeouts
                rt_str   = ''    # leave empty for RT
                corr     = 0
            else:
                # any committed scoring fixation on target during trial?
                recently_fixated_target = (last_fix_on_target_time is not None)

                # keypress fallback: near current target at press?
                on_keypress_fixated_target = False
                if tgt_idx is not None:
                    tgx, tgy = gabors[tgt_idx].pos  # centered coords
                    d_curr = ((score_gx_c - tgx)**2 + (score_gy_c - tgy)**2) ** 0.5
                    if d_curr <= score_accept_radius_px():
                        on_keypress_fixated_target = True

                # jump forgiveness: shortly after a jump, accept proximity to the previous target
                near_prev_after_jump = False
                if (not on_keypress_fixated_target) and (score_prev_target_pos_c is not None):
                    if (rt - score_target_change_time) <= score_change_grace:
                        px, py = score_prev_target_pos_c
                        d_prev = ((score_gx_c - px)**2 + (score_gy_c - py)**2) ** 0.5
                        if d_prev <= score_accept_radius_px():
                            near_prev_after_jump = True

                is_correct = recently_fixated_target or on_keypress_fixated_target or near_prev_after_jump

                # fill CSV timing if accepted via fallback
                if is_correct and (last_fix_on_target_time is None):
                    last_fix_on_target_time = rt
                    last_committed_fix_idx  = tgt_idx

                fb_text = 'Correct' if is_correct else 'Incorrect'
                resp_csv = 1
                rt_str   = rt
                corr     = int(is_correct)

            
            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')  # flush after feedback


            writer.writerow([
                'evading target static task',
                participant_id,
                t+1, int(tp), resp_csv, corr, rt_str,
                n, pos,
                (target_traj if len(target_traj)>0 else []),
                trial_k,                 # per-trial n-back condition (1,2,4,'RND')  
                nback_used_seq,
                fix_log,
                # SCORE:
                round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                last_committed_fix_idx if last_committed_fix_idx is not None else ""
            ])


    return filename
