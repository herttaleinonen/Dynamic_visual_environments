#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: herttaleinonen

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

def run_flicker_dynamic_trials(win, el_tracker,
                       screen_width, screen_height,
                       participant_id, timestamp,
                       noise_grain=3):
    """
    Dynamic Gabor search task:
      - Flash fixation (0.5s) between trials
      - During each trial, every 0.5s the screen goes black for 0.2s and
        ALL Gabors (target included) move to NEW random grid cells
      - Responses, feedback, logging remain identical to the baseline
    """
    sw, sh = screen_width, screen_height
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # grid offsets
    grid_w = grid_size_x * cell_size
    grid_h = grid_size_y * cell_size
    off_x = -grid_w / 2
    off_y = -grid_h / 2

    # fixation cross (pix units for clarity)
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')

    # instruction (unchanged)
    inst = visual.TextStim(win,
        text=("In the following experiment, you will see stationary Gabors on noise.\n"
              "Press '>' if you see the 45° target, '<' if not.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip()
    event.waitKeys(keyList=['return'])

    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        noise = np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)
        return noise[:h, :w]

    # helper: sample N unique grid-cell positions (grid coords, not pixels)
    def sample_unique_grid_positions(n):
        pos = []
        tries = 0
        while len(pos) < n:
            x = random.randint(2, grid_size_x-3)
            y = random.randint(2, grid_size_y-3)
            if (x, y) not in pos:
                pos.append((x, y))
            else:
                tries += 1
                if tries > 10000:
                    break
        return pos

    # helper: convert grid coords -> pixel coords
    def grid_to_pix(p):
        return (off_x + p[0]*cell_size, off_y + p[1]*cell_size)

    # full-screen black rectangle for blackout frames
    black_rect = visual.Rect(win, width=sw, height=sh,
                             fillColor='black', lineColor='black', units='pix')

    # open CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos'
        ])

        for t in range(num_trials):
            # flash fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)

            # trial setup
            n = random.choice([5,10,15])
            tp = random.choice([True, False])

            # initial unique positions (grid coords)
            pos = sample_unique_grid_positions(n)
            tgt_idx = random.randint(0, n-1) if tp else None

            # prepare Gabors at initial positions
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
                g.pos = grid_to_pix(pos[i])
                gabors.append(g)

            # static noise
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

            # blackout timing (same as your working task)
            blackout_interval = 0.5
            blackout_duration = 0.2
            next_blackout_time = blackout_interval
            in_blackout = False
            blackout_end = None

            # stimulus & response window (responses UNCHANGED: arrows via event.getKeys)
            response, rt = None, None
            while clk.getTime() < trial_duration:
                now = clk.getTime()

                # start a blackout if due
                if (not in_blackout) and (now >= next_blackout_time):
                    in_blackout = True
                    blackout_end = now + blackout_duration
                    if el_tracker:
                        el_tracker.sendMessage('blackout_onset')

                if in_blackout:
                    # keep screen black; still poll for keys (unchanged handling)
                    black_rect.draw(); win.flip()

                    keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                    if keys:
                        k, t0 = keys[0]
                        if k == 'escape':
                            if el_tracker:
                                el_tracker.sendMessage('stimulus_offset')
                                el_tracker.stopRecording()
                            return filename
                        response = 'target' if k == 'right' else 'distractor'
                        rt = t0
                        break

                    # end blackout?
                    if clk.getTime() >= blackout_end:
                        # choose brand NEW unique positions and move all Gabors there
                        new_pos = sample_unique_grid_positions(n)

                        # (Optional) ensure at least one position changed; if not, resample once
                        if new_pos == pos:
                            alt = sample_unique_grid_positions(n)
                            if len(alt) == n:
                                new_pos = alt

                        pos = new_pos  # record the new grid positions for next cycle
                        for i, g in enumerate(gabors):
                            g.pos = grid_to_pix(pos[i])

                        in_blackout = False
                        next_blackout_time += blackout_interval
                        if el_tracker:
                            el_tracker.sendMessage('blackout_offset')
                    else:
                        core.wait(min(0.005, blackout_end - clk.getTime()))
                    continue  # don't draw stimuli while black

                # visible frame: draw noise + gabors
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()

                # responses (UNCHANGED)
                keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k == 'escape':
                        if el_tracker:
                            el_tracker.sendMessage('stimulus_offset')
                            el_tracker.stopRecording()
                        return filename
                    response = 'target' if k == 'right' else 'distractor'
                    rt = t0
                    break

                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback (unchanged)
            if response is None:
                fb_text = timeout_feedback_text
                resp_str = 'None'; rt_str = ''
                corr = 0
            else:
                corr = int((response=='target')==tp)
                fb_text = 'Correct' if corr else 'Incorrect'
                resp_str = response; rt_str = rt
            fb = visual.TextStim(win, text=fb_text, color='white', height=40, units='pix')
            fb.draw(); win.flip(); core.wait(feedback_duration)

            # log (unchanged — logs initial positions and initial target pos)
            writer.writerow([
                t+1, int(tp), resp_str, corr, rt_str,
                n, pos, (pos[tgt_idx] if tgt_idx is not None else None)
            ])

    return filename
