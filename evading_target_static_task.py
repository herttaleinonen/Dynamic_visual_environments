#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:22:50 2025

@author: herttaleinonen
"""
import os
import csv
import random
import math
from collections import deque
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
    """
    Static 'evading target' with per-trial fixed n-back and distractor camo:
      - On target-present trials, the target appears at 0.2 s by taking over a distractor.
             On that frame, all other distractors re-randomize (masked reveal).
      - Fixations are tracked continuously, but n-back retargeting only activates after the target appears (>=n s).
    """

    sw, sh = screen_width, screen_height
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # ---------- Gaze & capture parameters ----------
    prev_gx, prev_gy   = sw/2.0, sh/2.0
    ema_alpha          = 0.55           # smoothing (higher = snappier)
    fix_radius_px      = 45             # fixation radius (px)
    min_fix_frames     = 4              # frames to commit (~65 ms at 60 Hz)
    capture_radius_px  = 1.75 * cell_size
    min_grid_sep_cells = 2              # >=2 cells away from current target (avoid neighbors)
    min_gaze_sep_px    = 1.75 * cell_size # keep target away from current gaze
    recent_target_max  = 2              # anti-ping-pong
    appear_delay_s     = 0.2            # NEW: target appears at 200 ms into target-present trials
    # ------------------------------------------------

    # grid offsets (centered PsychoPy pixels)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # fixation cross & instruction
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    inst = visual.TextStim(win,
        text=("In the following experiment, you will see stationary Gabors on noise.\n"
              "Press '>' if you see the 45° target, '<' if not.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip(); event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')  # keep buffer clean

    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # helpers for constraints
    def cheb_dist_cells(idx_a, idx_b):
        (xa, ya) = pos[idx_a]
        (xb, yb) = pos[idx_b]
        return max(abs(xa - xb), abs(ya - yb))

    def pix_dist(a_xy, b_xy):
        dx = a_xy[0] - b_xy[0]
        dy = a_xy[1] - b_xy[1]
        return math.hypot(dx, dy)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','NBackK','NBackUsedSeq'
        ])

        for t in range(num_trials):
            # fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            event.clearEvents(eventType='keyboard')  # flush between trials

            # layout
            n  = random.choice([10])
            tp = random.choice([True, False])  # target-present flag for the trial
            # unique positions (grid coords)
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos:
                    pos.append((x,y))
            # NOTE: start with NO target on screen; it will appear at 0.2 s if tp==True
            tgt_idx = None

            # choose fixed k for this trial (logged)
            trial_k = random.randint(1, 5)

            # stimuli + centers (centered PsychoPy px) — all start as distractors
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
            event.clearEvents(eventType='keyboard')  # clean start of response window

            # per-trial state
            inspected_idxs   = []               # sequence of *distractor* indices inspected (post-appearance only)
            recent_targets   = deque(maxlen=recent_target_max)
            nback_used_seq   = []

            # fixation clustering state
            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # --- camo helpers ---
            next_swap_time = clk.getTime() + random.uniform(0.35, 0.85)

            def swap_two_distractors(exclude=None):
                """Swap ori/sf between two distractors not in `exclude`."""
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
                """Randomize ori/sf for all distractors. If exclude_idx is given, skip that index."""
                for j in range(n):
                    if exclude_idx is not None and j == exclude_idx:
                        continue
                    gabors[j].ori = random.choice(orientations)
                    gabors[j].sf  = random.choice(spatial_frequencies)

            # NEW: target appearance arming
            arm_at   = clk.getTime() + appear_delay_s
            armed    = False  # becomes True after the 0.2 s reveal (only if tp==True)

            # main loop
            response, rt = None, None
            while clk.getTime() < trial_duration:
                now_t = clk.getTime()

                # time-based camo (as before; harmless before target appears)
                if now_t >= next_swap_time:
                    swap_two_distractors(exclude=None)
                    next_swap_time = now_t + random.uniform(0.25, 0.55)

                # NEW: reveal target at 0.2 s in target-present trials
                if tp and (not armed) and (now_t >= arm_at):
                    # choose any current distractor to become target
                    tgt_idx = random.randrange(n)
                    gabors[tgt_idx].ori = target_orientation
                    gabors[tgt_idx].sf  = target_sf
                    # mask the reveal by refreshing all other distractors
                    refresh_all_distractors(exclude_idx=tgt_idx)
                    recent_targets.clear()
                    recent_targets.append(tgt_idx)
                    armed = True
                    # reset fixation cluster & inspected history so n-back starts clean AFTER reveal
                    inspected_idxs = []
                    cluster_cx, cluster_cy = prev_gx, prev_gy
                    cluster_len = 0
                    committed_this_cluster = False

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
                        if (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px):
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

                                # record inspected distractor (debounced) — ONLY after reveal
                                appended = False
                                if armed and tp and (tgt_idx is not None) and (current_idx is not None) and (current_idx != tgt_idx):
                                    if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                        inspected_idxs.append(current_idx)
                                        appended = True

                                # n-back retargeting — ONLY after reveal
                                if armed and tp and (tgt_idx is not None):
                                    prior  = inspected_idxs[:-1] if appended else inspected_idxs[:]
                                    usable = len(prior)
                                    if usable >= 1:
                                        # choose candidate respecting constraints
                                        max_k = min(5, usable)
                                        ks = [k for k in range(1, max_k+1)]
                                        if trial_k in ks:
                                            ks.remove(trial_k)
                                            ks = [trial_k] + random.sample(ks, k=len(ks))
                                        else:
                                            random.shuffle(ks)
                                        gaze_now_px = (cluster_cx - sw/2.0, sh/2.0 - cluster_cy)
                                        cand_idx, used_k = None, None
                                        for k in ks:
                                            cidx = prior[-k]
                                            if cidx == tgt_idx: continue
                                            if (current_idx is not None) and (cidx == current_idx): continue
                                            if cidx in recent_targets: continue
                                            if cheb_dist_cells(cidx, tgt_idx) < min_grid_sep_cells: continue
                                            if pix_dist(centers_pix[cidx], gaze_now_px) < min_gaze_sep_px: continue
                                            cand_idx, used_k = cidx, k
                                            break
                                        if cand_idx is not None:
                                            # relabel identities (target jump) + mask
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            recent_targets.append(tgt_idx)
                                            nback_used_seq.append(used_k)
                                            refresh_all_distractors(exclude_idx=tgt_idx)

                                committed_this_cluster = True
                        else:
                            # leaving cluster (secondary path)
                            if (cluster_len >= min_fix_frames) and (not committed_this_cluster):
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j

                                appended = False
                                if armed and tp and (tgt_idx is not None) and (current_idx is not None) and (current_idx != tgt_idx):
                                    if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                        inspected_idxs.append(current_idx)
                                        appended = True

                                if armed and tp and (tgt_idx is not None):
                                    prior  = inspected_idxs[:-1] if appended else inspected_idxs[:]
                                    usable = len(prior)
                                    if usable >= 1:
                                        # same candidate selection as above
                                        max_k = min(5, usable)
                                        ks = [k for k in range(1, max_k+1)]
                                        if trial_k in ks:
                                            ks.remove(trial_k)
                                            ks = [trial_k] + random.sample(ks, k=len(ks))
                                        else:
                                            random.shuffle(ks)
                                        gaze_now_px = (cluster_cx - sw/2.0, sh/2.0 - cluster_cy)
                                        cand_idx, used_k = None, None
                                        for k in ks:
                                            cidx = prior[-k]
                                            if cidx == tgt_idx: continue
                                            if (current_idx is not None) and (cidx == current_idx): continue
                                            if cidx in recent_targets: continue
                                            if cheb_dist_cells(cidx, tgt_idx) < min_grid_sep_cells: continue
                                            if pix_dist(centers_pix[cidx], gaze_now_px) < min_gaze_sep_px: continue
                                            cand_idx, used_k = cidx, k
                                            break
                                        if cand_idx is not None:
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            recent_targets.append(tgt_idx)
                                            nback_used_seq.append(used_k)
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
                    response = 'target' if k=='right' else 'distractor'
                    rt = t0
                    break

                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback
            if response is None:
                fb_text = timeout_feedback_text; resp_str='None'; rt_str=''
                corr = 0
            else:
                corr = int((response=='target')==tp)
                fb_text = 'Correct' if corr else 'Incorrect'
                resp_str = response; rt_str = rt
            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')  # flush after feedback

            # log (NBackK = fixed per-trial k; NBackUsedSeq = ks actually used each jump)
            writer.writerow([
                t+1, int(tp), resp_str, corr, rt_str,
                n, pos, (pos[tgt_idx] if (tp and tgt_idx is not None) else None),
                (trial_k if tp else ''), nback_used_seq
            ])

    return filename


"""
def run_evading_target_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):
    
    Static 'evading target' with per-trial fixed n-back and distractor camo:
      - Target retargeting logic unchanged (per-trial fixed k, −1..−5 prior, constraints).
      - Target-present: jump frames are masked by refreshing all distractors.
      - Target-absent: at intervals, all distractors refresh simultaneously.
    

    sw, sh = screen_width, screen_height
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # ---------- Gaze & capture parameters ----------
    prev_gx, prev_gy   = sw/2.0, sh/2.0
    ema_alpha          = 0.55           # smoothing (higher = snappier)
    fix_radius_px      = 45             # fixation radius (px)
    min_fix_frames     = 4              # frames to commit (~65 ms at 60 Hz)
    capture_radius_px  = 1.75 * cell_size
    min_grid_sep_cells = 2              # >=2 cells away from current target (avoid neighbors)
    min_gaze_sep_px    = 1.75 * cell_size# keep target away from current gaze
    recent_target_max  = 2              # anti-ping-pong
    # ------------------------------------------------

    # grid offsets (centered PsychoPy pixels)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # fixation cross & instruction
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    inst = visual.TextStim(win,
        text=("In the following experiment, you will see stationary Gabors on noise.\n"
              "Press '>' if you see the 45° target, '<' if not.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip(); event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')   # FLUSH

    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # helpers for constraints
    def cheb_dist_cells(idx_a, idx_b):
        (xa, ya) = pos[idx_a]
        (xb, yb) = pos[idx_b]
        return max(abs(xa - xb), abs(ya - yb))

    def pix_dist(a_xy, b_xy):
        dx = a_xy[0] - b_xy[0]
        dy = a_xy[1] - b_xy[1]
        return math.hypot(dx, dy)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','NBackK','NBackUsedSeq'
        ])

        for t in range(num_trials):
            # fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            event.clearEvents(eventType='keyboard')   # FLUSH

            # layout
            n  = random.choice([10])
            tp = random.choice([True, False])
            # unique positions (grid coords)
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos:
                    pos.append((x,y))
            tgt_idx = random.randint(0, n-1) if tp else None

            # choose fixed k for this trial (logged)
            trial_k = random.randint(1, 5)

            # stimuli + centers (centered PsychoPy px)
            gabors = []
            centers_pix = []
            for i in range(n):
                if i == tgt_idx:
                    g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                           sf=target_sf, ori=target_orientation, phase=0.25, units='pix')
                else:
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
            
            # start trial with a clean keyboard buffer
            event.clearEvents(eventType='keyboard')   # FLUSH
            
            # per-trial state
            inspected_idxs   = []               # debounced sequence of *distractor* indices inspected
            recent_targets   = deque(maxlen=recent_target_max)
            if tp and (tgt_idx is not None):
                recent_targets.append(tgt_idx)
            nback_used_seq   = []

            cluster_cx, cluster_cy = prev_gx, prev_gy
            cluster_len = 0
            committed_this_cluster = False

            # --- camo helpers ---
            next_swap_time = clk.getTime() + random.uniform(0.35, 0.85)

            def swap_two_distractors(exclude=None):
                #Swap ori/sf between two distractors not in `exclude`.
                nonlocal tgt_idx
                if exclude is None:
                    exclude = set()
                pool = [i for i in range(n) if i != tgt_idx and i not in exclude]
                if len(pool) < 2:
                    return False
                i1, i2 = random.sample(pool, 2)
                gabors[i1].ori, gabors[i2].ori = gabors[i2].ori, gabors[i1].ori
                gabors[i1].sf,  gabors[i2].sf  = gabors[i2].sf,  gabors[i1].sf
                return True

            def refresh_all_distractors(exclude_idx=None):
                #Randomize ori/sf for all distractors. If exclude_idx is given, skip that index.
                for j in range(n):
                    if exclude_idx is not None and j == exclude_idx:
                        continue
                    # (All are distractors when target is absent; in present trials we skip the target)
                    gabors[j].ori = random.choice(orientations)
                    gabors[j].sf  = random.choice(spatial_frequencies)

            # choose candidate respecting constraints; returns (chosen_idx, used_k) or (None, None)
            def choose_candidate(prior_indices, current_idx, tgt_idx, usable_len):
                max_k = min(5, usable_len)
                ks = [k for k in range(1, max_k+1)]
                if trial_k in ks:
                    ks.remove(trial_k)
                    ks = [trial_k] + random.sample(ks, k=len(ks))
                else:
                    random.shuffle(ks)
                gaze_now_px = (cluster_cx - sw/2.0, sh/2.0 - cluster_cy)
                for k in ks:
                    cand_idx = prior_indices[-k]
                    if cand_idx == tgt_idx:
                        continue
                    if current_idx is not None and cand_idx == current_idx:
                        continue
                    if cand_idx in recent_targets:
                        continue
                    # far enough from current target?
                    if tgt_idx is not None and cheb_dist_cells(cand_idx, tgt_idx) < min_grid_sep_cells:
                        continue
                    # far enough from current gaze?
                    if pix_dist(centers_pix[cand_idx], gaze_now_px) < min_gaze_sep_px:
                        continue
                    return cand_idx, k
                return None, None

            # main loop
            response, rt = None, None
            while clk.getTime() < trial_duration:
                now_t = clk.getTime()

                # time-based camo
                if now_t >= next_swap_time:
                    if tp:
                        # target present -> small pairwise camo swap (as before)
                        swap_two_distractors(exclude=None)
                        next_swap_time = now_t + random.uniform(0.25, 0.55)
                    else:
                        # target absent -> synchronous refresh of ALL distractors
                        refresh_all_distractors(exclude_idx=None)
                        next_swap_time = now_t + random.uniform(0.35, 0.85)

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
                        if (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px):
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

                                # record inspected distractor (debounced)
                                appended = False
                                if (tp and tgt_idx is not None and current_idx is not None and current_idx != tgt_idx):
                                    if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                        inspected_idxs.append(current_idx)
                                        appended = True

                                # build prior (exclude current append if it was a distractor)
                                did_jump = False
                                if tp and (tgt_idx is not None):
                                    prior = inspected_idxs[:-1] if appended else inspected_idxs[:]
                                    usable = len(prior)
                                    if usable >= 1:
                                        cand_idx, used_k = choose_candidate(prior, current_idx, tgt_idx, usable)
                                        if cand_idx is not None:
                                            # relabel identities (target jump)
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            recent_targets.append(tgt_idx)
                                            nback_used_seq.append(used_k)
                                            did_jump = True
                                            # mask jump by changing ALL distractors this frame
                                            refresh_all_distractors(exclude_idx=tgt_idx)

                                committed_this_cluster = True
                        else:
                            # leaving cluster (secondary path)
                            if (cluster_len >= min_fix_frames) and (not committed_this_cluster):
                                fx_c = cluster_cx - sw/2.0
                                fy_c =  sh/2.0 - cluster_cy
                                d2 = (centers_pix[:,0] - fx_c)**2 + (centers_pix[:,1] - fy_c)**2
                                current_idx = None
                                if d2.size > 0:
                                    j = int(np.argmin(d2))
                                    if math.sqrt(d2[j]) <= capture_radius_px:
                                        current_idx = j

                                appended = False
                                if (tp and tgt_idx is not None and current_idx is not None and current_idx != tgt_idx):
                                    if not inspected_idxs or inspected_idxs[-1] != current_idx:
                                        inspected_idxs.append(current_idx)
                                        appended = True

                                did_jump = False
                                if tp and (tgt_idx is not None):
                                    prior = inspected_idxs[:-1] if appended else inspected_idxs[:]
                                    usable = len(prior)
                                    if usable >= 1:
                                        cand_idx, used_k = choose_candidate(prior, current_idx, tgt_idx, usable)
                                        if cand_idx is not None:
                                            gabors[cand_idx].ori = target_orientation
                                            gabors[cand_idx].sf  = target_sf
                                            gabors[tgt_idx].ori  = random.choice(orientations)
                                            gabors[tgt_idx].sf   = random.choice(spatial_frequencies)
                                            tgt_idx = cand_idx
                                            recent_targets.append(tgt_idx)
                                            nback_used_seq.append(used_k)
                                            did_jump = True
                                            # mask jump by changing ALL distractors this frame
                                            refresh_all_distractors(exclude_idx=tgt_idx)

                                if did_jump:
                                    exclude = {tgt_idx}
                                    if current_idx is not None:
                                        exclude.add(current_idx)
                                    swap_two_distractors(exclude=exclude)
                                    if random.random() < 0.5:
                                        swap_two_distractors(exclude=exclude)

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
                    response = 'target' if k=='right' else 'distractor'
                    rt = t0
                    break

                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback
            if response is None:
                fb_text = timeout_feedback_text; resp_str='None'; rt_str=''
                corr = 0
            else:
                corr = int((response=='target')==tp)
                fb_text = 'Correct' if corr else 'Incorrect'
                resp_str = response; rt_str = rt
            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')   # FLUSH

            # log (NBackK = fixed per-trial k; NBackUsedSeq = ks actually used each jump)
            writer.writerow([
                t+1, int(tp), resp_str, corr, rt_str,
                n, pos, (pos[tgt_idx] if (tp and tgt_idx is not None) else None),
                trial_k, nback_used_seq
            ])

    return filename
"""
