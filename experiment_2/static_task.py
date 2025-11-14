#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40:52 2025

@author: herttaleinonen


    Static Gabor detection task:
      - Target present every trial.
      - Placement rule k∈{0..4} with quotas and anti-streak.
      - Response: SPACE (keyboard) or any Cedrus button.
      - Correctness: lenient gaze-on-target recently or at keypress.
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

# ---------- central fixation gate ----------
def wait_for_central_fixation(win, el_tracker, screen_width, screen_height,
                              deg_thresh=2.499, hold_ms=200,
                              cross_height=40, cross_color='black',
                              max_wait_s=None):
    """
    Blocks until gaze is within `deg_thresh` of screen center for >= `hold_ms`.
    Returns (ok, drift_deg) where drift_deg is the median distance (deg) during
    the final hold.

    If `max_wait_s` is None → wait indefinitely. If timed out → returns (False, "").
    If no tracker → shows cross for 0.5 s and returns (True, "").
    """
    cross = visual.TextStim(win, text='+', color=cross_color, height=cross_height, units='pix')

    # No tracker → just show briefly and continue
    if not el_tracker:
        cross.draw(); win.flip(); core.wait(0.5)
        return True, ""

    # Start a dedicated short recording for the gate
    try:
        el_tracker.setOfflineMode()
        el_tracker.sendMessage('FIXGATE_START')
        el_tracker.startRecording(1, 1, 1, 1)
        core.wait(0.1)
    except Exception:
        cross.draw(); win.flip(); core.wait(0.5)
        return True, ""

    ok = False
    hold_clock = core.Clock()
    total_clock = core.Clock()
    inside_since = None
    inside_samples_deg = []

    # px/deg from your config: 1 deg ≈ `cell_size` px
    px_per_deg = float(cell_size) if cell_size > 0 else 1.0

    try:
        while True:
            # Optional timeout
            if (max_wait_s is not None) and (total_clock.getTime() >= max_wait_s):
                break

            # Draw cross
            cross.draw()
            win.flip()

            # Read newest sample
            s = el_tracker.getNewestSample()
            eye = s.getRightEye() if (s and s.isRightSample()) else (
                  s.getLeftEye() if (s and s.isLeftSample()) else None)

            now = core.getTime()
            if eye is not None:
                rx, ry = eye.getGaze()  # screen px, origin TL
                if (rx is not None) and (ry is not None) and (rx > -1e5) and (ry > -1e5):
                    # Convert to centered px, then to deg
                    cx = float(rx) - (screen_width / 2.0)
                    cy = (screen_height / 2.0) - float(ry)
                    dist_deg = float(np.hypot(cx, cy) / px_per_deg)

                    if dist_deg <= deg_thresh:
                        if inside_since is None:
                            inside_since = now
                            inside_samples_deg = [dist_deg]
                            hold_clock.reset()
                        else:
                            inside_samples_deg.append(dist_deg)

                        # Enough dwell time?
                        if (hold_clock.getTime() * 1000.0) >= float(hold_ms):
                            ok = True
                            break
                    else:
                        inside_since = None
                        inside_samples_deg = []

            core.wait(0.005)
    finally:
        try:
            el_tracker.sendMessage('FIXGATE_END')
            el_tracker.stopRecording()
        except Exception:
            pass

    if not ok:
        return False, ""

    drift_deg = round(float(np.median(inside_samples_deg)), 3) if inside_samples_deg else ""
    return True, drift_deg


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
        while dev.has_response():
            r = dev.get_next_response()
            if bool(r.get("pressed", False)):
                pressed_seen = True

        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return pressed_seen
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False
    
    
def _cedrus_get_choice(dev):
    """
    Returns 'target' for GREEN (key 3), 'distractor' for RED (key 1), or None.
    """
    if not dev:
        return None
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        choice = None
        while dev.has_response():
            r = dev.get_next_response()
            if not r or not r.get("pressed", False):
                continue
            k = r.get("key", None)
            if k == 3:
                choice = 'target'
            elif k == 1:
                choice = 'distractor'
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return choice
    except Exception as e:
        print(f"[Cedrus] read failed: {e}")
        return None


# --------------------------------------------------------------------

def run_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):

    sw, sh = screen_width, screen_height
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")

    # grid offsets (PsychoPy-centered px)
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # gaze/fixation params (still used for clustering during the trial)
    ema_alpha = 0.5
    fix_radius_px = max(80, int(cell_size * 1.2))
    min_fix_frames = 1
    capture_radius_px = max(int(cell_size * 3.0), 160)
    gauss_k = 6.0
    
    def target_accept_radius_px():
        sigma = cell_size / 6.0
        return max(capture_radius_px, gauss_k * sigma)

    # ----- Cedrus open (optional) -----
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/respond.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    # fixation cross (used only in instructions)
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')

    # ---------- Instruction screen ----------
    inst = visual.TextStim(
        win,
        text=("In this task you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press the GREEN button as soon as you see the target.\n"
              "If you do not find the target, press the RED button.\n"
              "Each trial you have 7 seconds to respond.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start."),
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
        
    _cedrus_flush(cedrus)
    event.clearEvents(eventType='keyboard')
    
    # ------- helper: Gaussian noise image -------
    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # -------------- CSV --------------
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID',
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','PlacementRule',
            'FixOnTargetTime(s)','LastFixIndex','CalibrationDrift(deg)'
        ])

        progress_text = visual.TextStim(
            win, text="", color='white', height=28,
            pos=(0, -int(screen_height * 0.18)), units='pix'
        )
        
        
        # ---- exact 50/50 target-present schedule ----
        n_present = num_trials // 2
        present_schedule = [1]*n_present + [0]*(num_trials - n_present)
        random.shuffle(present_schedule)

        # >>>>>>>>>>>>>>> trial loop <<<<<<<<<<<<<<<
        for t in range(num_trials):
            # NEW: central fixation gate before each trial
            ok, drift_deg = wait_for_central_fixation(
                win, el_tracker, sw, sh,
                deg_thresh=2.499, hold_ms=200,
                cross_height=40, cross_color='black',
                max_wait_s=None   # wait indefinitely; set e.g. 5.0 for a timeout
            )
            if not ok:
                print(f"[FIXGATE] Trial {t+1}: timed out waiting for central fixation.")
                return filename
            
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')

            # --- target presence & array size ---
            tp = bool(present_schedule[t]) # 50% target present
            n  = 10                        # fixed array size
            
            # --- completely random unique grid positions ---
            seen = set()
            pos  = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x - 3)
                y = random.randint(2, grid_size_y - 3)
                if (x, y) not in seen:
                    pos.append((x, y))
                    seen.add((x, y))
            
            # --- pick target index randomly if present ---
            tgt_idx = random.randrange(n) if tp else None
            
            # for CSV compatibility, we keep a rule_k placeholder = 0
            rule_k = 0

            # build Gabors in PsychoPy-centered px
            gabors = []
            for i in range(n):
                if tp and i == tgt_idx:
                    # target gabor
                    g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                           sf=target_sf, ori=target_orientation,
                                           phase=0.25, units='pix')
                else:
                    # distractor gabor
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

            # per-trial gaze/cluster state (still logged but not used for correctness)
            cluster_cx = 0.0; cluster_cy = 0.0
            cluster_len = 0
            committed_this_cluster = False
            last_committed_fix_idx = None
            last_fix_on_target_time = None
            ema_x = sw/2.0; ema_y = sh/2.0

            # ---------- response loop ----------
            response, rt = None, None
            while clk.getTime() < trial_duration:
                noise_stim.draw()
                for g in gabors:
                    g.draw()
                win.flip()
            
                # Cedrus responses
                choice = _cedrus_get_choice(cedrus) if cedrus else None
                if choice is not None:
                    response = choice
                    rt = clk.getTime()
                    break
            
                # Keyboard fallback
                keys = event.getKeys(keyList=['g','r','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset')
                            el_tracker.stopRecording()
                        return filename
                    if k == 'g':
                        response = 'target'; rt = t0; break
                    if k == 'r':
                        response = 'distractor'; rt = t0; break
            
                # EyeLink sampling / fixation clustering (left as-is)
                if el_tracker:
                    s = el_tracker.getNewestSample()
                    eye = None
                    if s and s.isRightSample():
                        eye = s.getRightEye()
                    elif s and s.isLeftSample():
                        eye = s.getLeftEye()
            
                    now_t = clk.getTime()
                    if eye is not None:
                        rx, ry = eye.getGaze()
                        if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                            ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, sw-1))
                            ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, sh-1))
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
                                            if tp and (current_idx == tgt_idx):
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

            # feedback & correctness (response-only)
            if response is None:
                fb_text = timeout_feedback_text
                resp_str = 'timeout'; rt_str = ''
                corr = 0
            else:
                if tp:
                    is_correct = (response == 'target')
                else:
                    is_correct = (response == 'distractor')
                fb_text = 'Correct' if is_correct else 'Incorrect'
                resp_str = response
                rt_str = rt
                corr = int(is_correct)
            
            # clear gaze-derived outputs so CSV leaves blanks
            last_fix_on_target_time = None
            last_committed_fix_idx = None

            fb = visual.TextStim(win, text=fb_text, color='white', height=40, units='pix')
            progress_text.text = f"{t+1}/{num_trials}"
            fb.draw(); progress_text.draw()
            win.flip(); core.wait(feedback_duration)
            
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')

            if tp and tgt_idx is not None:
                target_grid = pos[tgt_idx]
            else:
                target_grid = ""
                
            # map textual response → numeric
            if resp_str == 'target':
                resp_num = 1
            elif resp_str == 'distractor':
                resp_num = 0
            else:
                resp_num = ""
            
            writer.writerow([
                'static task', participant_id,
                t+1, int(tp),
                resp_num,
                corr, rt_str,
                n, pos, target_grid, rule_k,
                round(last_fix_on_target_time, 4) if (tp and last_fix_on_target_time is not None) else "",
                last_committed_fix_idx if (tp and last_committed_fix_idx is not None) else "",
                drift_deg
            ])

        # <<<<<<<<<<<<< end trial loop <<<<<<<<<<<<<<<

    return filename


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
    
    
def _cedrus_get_choice(dev):

    if not dev:
        return None
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        choice = None
        while dev.has_response():
            r = dev.get_next_response()
            if not r or not r.get("pressed", False):
                continue
            k = r.get("key", None)
            if k == 3:
                choice = 'target'
            elif k == 1:
                choice = 'distractor'
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return choice
    except Exception as e:
        print(f"[Cedrus] read failed: {e}")
        return None


# --------------------------------------------------------------------

def run_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):

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
    #gaze_to_press_max_lag = trial_duration
    gauss_k = 6.0
    
    # for scoring
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
        text=("In this task you will see objects, and among them you have to find a target object.\n"
              "The target object is tilted 45°.\n"
              "Press the GREEN button as soon as you see the target.\n"
              "If you do not find the target, no not press anything.\n"
              "Each trial you have 7 seconds to respond.\n"
              "Between trials a cross is shown in the middle of the screen, try to focus your eyes there.\n"
              "\n"
              "Press any button to start."),
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
        
    _cedrus_flush(cedrus)
    event.clearEvents(eventType='keyboard')
    
    # --- helper: measure calibration drift during the fixation cross ---
    def measure_fixation_drift(trial_idx, duration=0.5):
       
        # If no tracker, just display the cross and wait
        if not el_tracker:
            fix_cross.draw(); win.flip(); core.wait(duration)
            return ""
    
        # Start a short recording just for the fixation check
        try:
            el_tracker.setOfflineMode()
            el_tracker.sendMessage(f'FIXCHECK_START {trial_idx}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)  # let it settle
        except Exception:
            # Fallback: still show fixation and wait, but no drift value
            fix_cross.draw(); win.flip(); core.wait(duration)
            return ""
    
        # Collect samples while showing the cross
        samples = []
        clk_fix = core.Clock()
        while clk_fix.getTime() < duration:
            fix_cross.draw()
            win.flip()
    
            s = el_tracker.getNewestSample()
            eye = None
            if s and s.isRightSample():
                eye = s.getRightEye()
            elif s and s.isLeftSample():
                eye = s.getLeftEye()
    
            if eye is not None:
                rx, ry = eye.getGaze()  # screen px, origin top-left, +y down
                if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                    # convert to PsychoPy-centered px (+y up)
                    gx = float(rx) - (sw / 2.0)
                    gy = (sh / 2.0) - float(ry)
                    # store distance in deg (cell_size ≈ px/deg)
                    dist_deg = math.hypot(gx, gy) / float(cell_size)
                    samples.append(dist_deg)
    
            core.wait(0.005)
    
        # Stop this short recording
        try:
            el_tracker.sendMessage('FIXCHECK_END')
            el_tracker.stopRecording()
        except Exception:
            pass
    
        # Robust summary
        if not samples:
            return ""
        return round(float(np.median(samples)), 3)

    # ------- helper: Gaussian noise image -------
    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # -------------- CSV --------------
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID',
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos','PlacementRule',
            'FixOnTargetTime(s)','LastFixIndex','CalibrationDrift(deg)'
        ])

        progress_text = visual.TextStim(
            win, text="", color='white', height=28,
            pos=(0, -int(screen_height * 0.18)), units='pix'
        )
        
        
        # ---- exact 50/50 target-present schedule ----
        n_present = num_trials // 2  # exact 50% if even; with odd, this is floor(50%)
        present_schedule = [1]*n_present + [0]*(num_trials - n_present)
        random.shuffle(present_schedule)  # randomize order


        # >>>>>>>>>>>>>>> trial loop <<<<<<<<<<<<<<<
        for t in range(num_trials):
            # fixation + drift measurement (also draws the cross)
            drift_deg = measure_fixation_drift(t+1, duration=0.5)
            
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')


            # --- target presence & array size ---
            tp = bool(present_schedule[t]) # 50% target present
            n  = 10                        # fixed array size
            
            # --- completely random unique grid positions ---
            seen = set()
            pos  = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x - 3)
                y = random.randint(2, grid_size_y - 3)
                if (x, y) not in seen:
                    pos.append((x, y))
                    seen.add((x, y))
            
            # --- pick target index randomly if present ---
            tgt_idx = random.randrange(n) if tp else None
            
            # for CSV compatibility, we keep a rule_k placeholder = 0
            rule_k = 0



            # build Gabors in PsychoPy-centered px
            gabors = []
            for i in range(n):
                if tp and i == tgt_idx:
                    # target gabor
                    g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                           sf=target_sf, ori=target_orientation,
                                           phase=0.25, units='pix')
                else:
                    # distractor gabor
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
                for g in gabors: 
                    g.draw()
                win.flip()
            
                # ----- Cedrus: GREEN (key 3) = 'target', RED (key 1) = 'distractor'
                choice = _cedrus_get_choice(cedrus) if cedrus else None
                if choice is not None:
                    response = choice
                    rt = clk.getTime()
                    break
            
                # ----- Keyboard fallback: 'g' = target present, 'r' = target absent, ESC = quit
                keys = event.getKeys(keyList=['g','r','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset')
                            el_tracker.stopRecording()
                        return filename
                    if k == 'g':
                        response = 'target'; rt = t0; break
                    if k == 'r':
                        response = 'distractor'; rt = t0; break
            
                # ----- EyeLink sampling / fixation clustering -----
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
                                            if tp and (current_idx == tgt_idx):
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


            # feedback & correctness (response-only)
            if response is None:
                fb_text = timeout_feedback_text
                resp_str = 'timeout'; rt_str = ''
                corr = 0
            else:
                if tp:
                    is_correct = (response == 'target')        # GREEN correct
                else:
                    is_correct = (response == 'distractor')    # RED correct
                fb_text = 'Correct' if is_correct else 'Incorrect'
                resp_str = response
                rt_str = rt
                corr = int(is_correct)
            
            # clear gaze-derived outputs so CSV leaves blanks
            last_fix_on_target_time = None
            last_committed_fix_idx = None



            fb = visual.TextStim(win, text=fb_text, color='white', height=40, units='pix')
            
            # Update "X/total" text
            progress_text.text = f"{t+1}/{num_trials}"
            
            # Draw feedback + progress count
            fb.draw(); progress_text.draw()
            win.flip(); core.wait(feedback_duration)
            
            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')

            if tp and tgt_idx is not None:
                target_grid = pos[tgt_idx]
            else:
                target_grid = ""
                

            # map textual response → numeric
            if resp_str == 'target':
                resp_num = 1
            elif resp_str == 'distractor':
                resp_num = 0
            else:
                resp_num = ""  # for timeout or missing
            
            writer.writerow([
                'static task', participant_id,
                t+1, int(tp),
                resp_num,               # numeric response
                corr, rt_str,
                n, pos, target_grid, rule_k,
                round(last_fix_on_target_time, 4) if (tp and last_fix_on_target_time is not None) else "",
                last_committed_fix_idx if (tp and last_committed_fix_idx is not None) else "",
                drift_deg
            ])


        # <<<<<<<<<<<<< end trial loop <<<<<<<<<<<<<<<

    return filename
"""
