#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40:52 2025

@author: herttaleinonen

static_task.py
Keyboard + optional Cedrus (any button) support:
 - Any Cedrus button starts from the instruction screen
 - Any Cedrus button counts as SPACE during trials
 - Keyboard fallback: ENTER to start, SPACE to respond, ESC to abort
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

def _cedrus_any_pressed(dev) -> bool:
    """True if *any* Cedrus key press event is available (drains queue)."""
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        has_resp = getattr(dev, "has_response", lambda: False)()
        if not has_resp:
            return False
        r = dev.get_next_response()
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        # treat any response as a press; many firmwares set r["pressed"]=True
        return bool(r.get("pressed", True))
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False

# --------------------------------------------------------------------

def run_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):
    """
    Static Gabor detection task:
      - Target present every trial.
      - Placement rule k∈{0..4} with quotas and anti-streak.
      - Response: SPACE (keyboard) or any Cedrus button.
      - Correctness: lenient gaze-on-target recently or at keypress.
    """

    sw, sh = screen_width, screen_height
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")

    # grid offsets (PsychoPy-centered px)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # gaze/fixation params
    ema_alpha = 0.5
    fix_radius_px = max(80, int(cell_size * 1.2))
    min_fix_frames = 1
    capture_radius_px = max(int(cell_size * 3.0), 160)
    gaze_to_press_max_lag = trial_duration
    gauss_k = 6.0

    def target_accept_radius_px():
        # PsychoPy 'gauss' mask: σ ≈ size/6
        sigma = cell_size / 6.0
        return max(capture_radius_px, gauss_k * sigma)

    # ----- Cedrus open (optional) -----
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/respond.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    # fixation cross
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')

    # ---------- Instruction screen ----------
    inst = visual.TextStim(
        win,
        text=("In the following task, you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press 'SPACE' as soon as you see the target.\n"
              "Each trial you have 7 seconds to respond.\n"
              "Between trials there is a cross in the middle of the screen, try to focus your eyes there.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )

    # Example stims row: distractors + separated target
    row_y   = -int(sh * 0.22)
    gap     = int(cell_size * 0.6)
    extra   = int(cell_size * 5.0)
    tgt_dy  = 0

    ori_list = list(orientations)
    n_d      = len(ori_list)
    row_w_d  = n_d * cell_size + (n_d - 1) * gap
    start_x  = -row_w_d // 2 + cell_size // 2

    example_stims = []
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

    # Draw instructions + examples
    inst.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
    win.flip()

    # Start gate: Cedrus (any button) OR keyboard (Enter)
    if cedrus:
        while True:
            if _cedrus_any_pressed(cedrus):
                break
            keys = event.getKeys(keyList=['return','enter','escape'])
            if 'escape' in keys:
                return filename
            if keys:
                break
            core.wait(0.01)
        if hasattr(cedrus, "clear_response_queue"):
            cedrus.clear_response_queue()
    else:
        event.waitKeys(keyList=['return','enter'])
    event.clearEvents(eventType='keyboard')

    # ------- helper: Gaussian noise image -------
    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # quotas for rules (target placement)
    weights = (0.60, 0.10, 0.10, 0.10, 0.10) # weighted to avoid infinite repeats
    labels  = [0, 1, 2, 3, 4]
    raw     = [num_trials * w for w in weights]
    counts  = [int(math.floor(x)) for x in raw]
    remain  = num_trials - sum(counts)
    remaind = np.array(raw) - np.array(counts)
    for idx in np.argsort(remaind)[::-1][:remain]:
        counts[idx] += 1
    quota_left = {lab: cnt for lab, cnt in zip(labels, counts)}

    warmup_initial = 5
    block_size     = 24
    block_warmup   = 2
    max_run_same   = 1

    last_target = None
    run_same    = 0
    target_history = []   # list of (gx, gy)

    # -------------- CSV --------------
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID',
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','PlacementRule',
            'FixOnTargetTime(s)','LastFixIndex'
        ])

        progress_text = visual.TextStim(
            win, text="", color='white', height=28,
            pos=(0, -int(screen_height * 0.18)), units='pix'
        )

        # >>>>>>>>>>>>>>> trial loop <<<<<<<<<<<<<<<
        for t in range(num_trials):
            # fixation blink
            fix_cross.draw(); win.flip(); core.wait(0.5)
            event.clearEvents(eventType='keyboard')
            if cedrus and hasattr(cedrus, "clear_response_queue"):
                cedrus.clear_response_queue()

            tp = True
            n = 10  # fixed array size

            # --- choose rule_k with quotas + anti-streak ---
            in_block_warmup = (block_size and (t % block_size) < block_warmup)
            if (t < warmup_initial) or in_block_warmup:
                rule_k = 0
            else:
                valid = [0] + list(range(1, min(4, t) + 1))
                pool, wts = [], []
                for k in valid:
                    if quota_left.get(k, 0) <= 0:
                        continue
                    would_same = False
                    if k > 0 and last_target is not None and len(target_history) >= k:
                        would_same = (target_history[-k] == last_target)
                    if would_same and run_same >= max_run_same:
                        continue
                    pool.append(k); wts.append(max(1, quota_left.get(k, 0)))
                if not pool:
                    for k in valid:
                        would_same = False
                        if k > 0 and last_target is not None and len(target_history) >= k:
                            would_same = (target_history[-k] == last_target)
                        if would_same and run_same >= max_run_same:
                            continue
                        pool.append(k); wts.append(1)
                if not pool:
                    pool, wts = valid, [1]*len(valid)
                rule_k = random.choices(pool, weights=wts, k=1)[0]

            if quota_left.get(rule_k, 0) > 0:
                quota_left[rule_k] -= 1

            # positions (grid coords)
            pos = []
            if rule_k > 0 and len(target_history) >= rule_k:
                desired = target_history[-rule_k]
                pos.append(desired)
                seen = {desired}
                while len(pos) < n:
                    x = random.randint(2, grid_size_x - 3)
                    y = random.randint(2, grid_size_y - 3)
                    if (x, y) not in seen:
                        pos.append((x, y)); seen.add((x, y))
                tgt_idx = 0
            else:
                seen = set()
                while len(pos) < n:
                    x = random.randint(2, grid_size_x - 3)
                    y = random.randint(2, grid_size_y - 3)
                    if (x, y) not in seen:
                        pos.append((x, y)); seen.add((x, y))
                last = target_history[-1] if len(target_history) >= 1 else None
                allowed = [p for p in pos if (last is None or p != last)] or pos
                target_grid = random.choice(allowed)
                tgt_idx = pos.index(target_grid)

            # build Gabors in PsychoPy-centered px
            gabors = []
            for i in range(n):
                if i == tgt_idx:
                    g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                           sf=target_sf, ori=target_orientation,
                                           phase=0.25, units='pix')
                else:
                    g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                           sf=random.choice(spatial_frequencies),
                                           ori=random.choice(orientations),
                                           phase=0.25, units='pix')
                px = off_x + pos[i][0]*cell_size
                py = off_y + pos[i][1]*cell_size
                g.pos = (px, py)
                gabors.append(g)

            noise_img = generate_noise(sw, sh, noise_grain)
            noise_stim = visual.ImageStim(win, image=noise_img,
                                          size=(sw,sh), units='pix', interpolate=False)

            # EyeLink start
            clk = core.Clock()
            if el_tracker:
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {t+1}')
                el_tracker.startRecording(1,1,1,1)
                core.wait(0.1)
                el_tracker.sendMessage('stimulus_onset')

            # first frame
            noise_stim.draw()
            for g in gabors: g.draw()
            win.flip()
            event.clearEvents(eventType='keyboard')

            # per-trial gaze/cluster state
            cluster_cx = 0.0; cluster_cy = 0.0
            cluster_len = 0
            committed_this_cluster = False
            last_committed_fix_idx = None
            last_fix_on_target_time = None
            ema_x = sw/2.0; ema_y = sh/2.0

            # ---------- response loop ----------
            response, rt = None, None
            while clk.getTime() < trial_duration:
                # draw
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()

                # Cedrus first: ANY button = SPACE
                if cedrus and _cedrus_any_pressed(cedrus):
                    response = 'space'
                    rt = clk.getTime()
                    break

                # keyboard: SPACE / ESC
                keys = event.getKeys(keyList=['space','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()
                        return filename
                    if k == 'space':
                        response = 'space'; rt = t0; break

                # EyeLink sampling / fixation clustering
                if el_tracker:
                    s = el_tracker.getNewestSample()
                    eye = None
                    if s and s.isRightSample():
                        eye = s.getRightEye()
                    elif s and s.isLeftSample():
                        eye = s.getLeftEye()

                    now_t = clk.getTime()
                    if eye is not None:
                        rx, ry = eye.getGaze()  # EyeLink px, origin TL, +y down
                        if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                            ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, sw-1))
                            ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, sh-1))
                            # centered coordinates (+y up)
                            gx = ema_x - (sw/2.0)
                            gy = (sh/2.0) - ema_y

                            dx = gx - cluster_cx
                            dy = gy - cluster_cy
                            inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)

                            if inside:
                                cluster_len += 1
                                w = 1.0 / max(1, cluster_len)
                                cluster_cx = (1 - w)*cluster_cx + w*gx
                                cluster_cy = (1 - w)*cluster_cy + w*gy

                                if (not committed_this_cluster) and (cluster_len >= min_fix_frames):
                                    centers = np.array([gg.pos for gg in gabors], dtype=float)
                                    if centers.size > 0:
                                        d2 = (centers[:,0]-cluster_cx)**2 + (centers[:,1]-cluster_cy)**2
                                        j = int(np.argmin(d2))
                                        current_idx = None
                                        if np.sqrt(d2[j]) <= 1.25 * target_accept_radius_px():
                                            current_idx = j
                                        if current_idx is not None:
                                            last_committed_fix_idx = current_idx
                                            if current_idx == tgt_idx:
                                                last_fix_on_target_time = now_t
                                    committed_this_cluster = True
                            else:
                                cluster_len = 1
                                cluster_cx = gx
                                cluster_cy = gy
                                committed_this_cluster = False

                core.wait(movement_delay)
            # ---------- end while ----------

            # stimulus offset + stop recording
            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback & correctness
            if response is None:
                fb_text = timeout_feedback_text
                resp_str = 'None'; rt_str = ''
                corr = 0
            else:
                recently_fixated_target = (last_fix_on_target_time is not None) and \
                                          ((rt - last_fix_on_target_time) <= gaze_to_press_max_lag)
                # keypress fallback: was gaze near target at press?
                on_keypress_fixated_target = False
                tgx, tgy = gabors[tgt_idx].pos
                gaze_x_at_press = cluster_cx if 'gx' not in locals() else gx
                gaze_y_at_press = cluster_cy if 'gy' not in locals() else gy
                d_key = ((gaze_x_at_press - tgx)**2 + (gaze_y_at_press - tgy)**2) ** 0.5
                if d_key <= 1.25 * target_accept_radius_px():
                    on_keypress_fixated_target = True
                    if last_fix_on_target_time is None:
                        last_fix_on_target_time = rt
                        last_committed_fix_idx = tgt_idx

                is_correct = recently_fixated_target or on_keypress_fixated_target
                fb_text = 'Correct' if is_correct else 'Incorrect'
                resp_str = 'space'; rt_str = rt
                corr = int(is_correct)

            fb = visual.TextStim(win, text=fb_text, color='white', height=40, units='pix')
            progress_text.text = f"{t+1}/{num_trials}"
            fb.draw(); progress_text.draw()
            win.flip(); core.wait(feedback_duration)
            event.clearEvents(eventType='keyboard')

            target_grid = pos[tgt_idx]
            target_history.append(target_grid)
            same_outcome = (last_target is not None) and (target_grid == last_target)
            run_same     = (run_same + 1) if same_outcome else 0
            last_target  = target_grid

            writer.writerow([
                'static task', participant_id,
                t+1, int(tp),
                resp_str, corr, rt_str,
                n, pos, target_grid, rule_k,
                round(last_fix_on_target_time, 4) if last_fix_on_target_time is not None else "",
                last_committed_fix_idx if last_committed_fix_idx is not None else ""
            ])
        # <<<<<<<<<<<<< end trial loop <<<<<<<<<<<<<<<

    return filename
