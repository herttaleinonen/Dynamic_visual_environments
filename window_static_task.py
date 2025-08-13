#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author Hertta Leinonen

"""
window_static_task_gaussian_fullmask.py

Adds a gaze-contingent Gaussian-edged peephole mask covering the full screen.
- Underlying static scene (noise + Gabors) is unchanged.
- Each frame, a dynamic RGBA mask is computed full-screen: transparent in a Gaussian
  region around the gaze, opaque elsewhere (black).
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


def run_window_static_trials(win, el_tracker,
                              screen_width, screen_height,
                              participant_id, timestamp,
                              noise_grain=3,
                              hole_radius=None):
    sw, sh = screen_width, screen_height
    half_w, half_h = sw/2, sh/2
    prev_gx, prev_gy = half_w, half_h
    if hole_radius is None:
    # choose 15% of the smaller display dimension (width or height) for consistent hole size
    # "smaller screen dimension" refers to min(screen_width, screen_height)
        hole_radius = int(min(sw, sh) * 0.15)
        # increase default hole radius to 30% of screen smaller dimension
        hole_radius = int(min(sw, sh) * 0.30)
        # sigma controls falloff width
        sigma = hole_radius / 2.0

    # precompute pixel grid
    ys_full, xs_full = np.ogrid[0:sh, 0:sw]

    # fixation and instruction
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    inst = visual.TextStim(win,
        text=("In this experiment: stationary Gabors on noise.\n"
              "A Gaussian-edged peephole follows your gaze.\n"
              "Press '>' if you see the 45° target, '<' if not.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip(); event.waitKeys(keyList=['return'])

    def generate_noise(w, h, grain=noise_grain):
        hg, wg = math.ceil(h/grain), math.ceil(w/grain)
        small = np.clip(np.random.normal(0,0.3,(hg, wg)),-1,1)
        return np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)[:h,:w]

    out_dir = 'results'; os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos'
        ])

        # grid offsets
        grid_w = grid_size_x * cell_size
        grid_h = grid_size_y * cell_size
        off_x, off_y = -grid_w/2, -grid_h/2

        for t in range(num_trials):
            # flash fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            # trial setup
            n = random.choice([5,10,15]); tp = random.choice([True, False])
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos: pos.append((x,y))
            tgt_idx = random.randint(0, n-1) if tp else None

            # prepare static stimuli
            gabors = []
            for i in range(n):
                sf = target_sf if i==tgt_idx else random.choice(spatial_frequencies)
                ori = target_orientation if i==tgt_idx else random.choice(orientations)
                g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                       sf=sf, ori=ori, phase=0.25, units='pix')
                g.pos = (off_x + pos[i][0]*cell_size,
                         off_y + pos[i][1]*cell_size)
                gabors.append(g)
            noise_img = generate_noise(sw, sh, noise_grain)
            noise_stim = visual.ImageStim(win, image=noise_img,
                                          size=(sw,sh), units='pix', interpolate=False)

            # start EyeLink
            clk = core.Clock()
            if el_tracker:
                el_tracker.setOfflineMode(); el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {t+1}'); el_tracker.startRecording(1,1,1,1)
                core.wait(0.1); el_tracker.sendMessage('stimulus_onset')

            response, rt = None, None
            # trial loop
            while clk.getTime() < trial_duration:
                # sample gaze
                if el_tracker:
                    samp = el_tracker.getNewestSample()
                    if samp and samp.isRightSample(): eye = samp.getRightEye()
                    elif samp and samp.isLeftSample(): eye = samp.getLeftEye()
                    else: eye = None
                    if eye:
                        rx, ry = eye.getGaze()
                        prev_gx = np.clip((prev_gx + rx)/2, 0, sw)
                        prev_gy = np.clip((prev_gy + ry)/2, 0, sh)

                # draw static scene
                noise_stim.draw()
                for g in gabors: g.draw()

                # compute Gaussian alpha mask full-screen
                dx2 = (xs_full - prev_gx)**2
                dy2 = (ys_full - (sh - prev_gy))**2
                dist2 = dx2 + dy2
                gauss_full = np.exp(-dist2 / (2 * sigma**2))
                alpha_full = np.where(dist2 <= hole_radius**2,
                                      1.0 - gauss_full,
                                      1.0)
                # build and draw mask image each frame
                mask_full = np.zeros((sh, sw, 4), dtype=float)
                mask_full[..., 3] = alpha_full
                mask_im_full = visual.ImageStim(win,
                                    image=mask_full,
                                    size=(sw, sh),
                                    units='pix', interpolate=False)
                mask_im_full.draw()
                win.flip()

                keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k=='escape':
                        if el_tracker: el_tracker.stopRecording()
                        return filename
                    response, rt = ('target',t0) if k=='right' else ('distractor',t0)
                    break
                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()

            # feedback
            if response is None:
                fb_text, corr = timeout_feedback_text, 0
            else:
                corr = int((response=='target')==tp)
                fb_text = 'Correct' if corr else 'Incorrect'
            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)

            # log
            writer.writerow([
                t+1, int(tp), response or 'None', corr, rt or '',
                n, pos, (pos[tgt_idx] if tgt_idx is not None else None)
            ])

    return filename



"""
window_static_task_gaussian_mask.py

Adds a gaze-contingent circular peephole with Gaussian edges on top of the static_task.
- Underlying static scene (noise + Gabors) is unchanged.
- Uses a precomputed RGBA mask image (
- Masks full-screen with black, then draws the RGBA mask (transparent in center)
  at the gaze position to reveal the scene.


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

def run_window_static_trials(win, el_tracker,
                              screen_width, screen_height,
                              participant_id, timestamp,
                              noise_grain=3,
                              hole_radius=None):
    
    #Static Gabor task with a Gaussian-edged circular peephole.
    
    sw, sh = screen_width, screen_height
    half_w, half_h = sw/2, sh/2
    # initialize gaze smoothing state
    prev_gx, prev_gy = half_w, half_h
    # determine hole radius if not passed
    if hole_radius is None:
        hole_radius = int(min(sw, sh) * 0.15)
    sigma = hole_radius / 2.0

    # precompute RGBA mask array: transparent center, opaque edges with Gaussian falloff
    ys, xs = np.ogrid[-hole_radius:hole_radius, -hole_radius:hole_radius]
    gauss = np.exp(-(xs**2 + ys**2) / (2 * sigma**2))
    alpha = 1.0 - gauss  # 0 at center, 1 at border
    mask_rgba = np.zeros((2*hole_radius, 2*hole_radius, 4), dtype=float)
    mask_rgba[..., 3] = alpha
    # create mask image once
    mask_im = visual.ImageStim(win,
                                image=mask_rgba,
                                size=(2*hole_radius, 2*hole_radius),
                                units='pix', interpolate=False)
    # full-screen black mask
    black_mask = visual.Rect(win, width=sw, height=sh,
                              fillColor='black', lineColor='black', units='pix', pos=(0,0))

    # fixation cross
    fix_cross = visual.TextStim(win, text='+', color='black', height=40, units='pix')
    # instruction
    inst = visual.TextStim(win,
        text=("In this experiment, stationary Gabors on noise.\n"
              "A Gaussian-edged peephole follows your gaze.\n"
              "Press '>' if you see the 45° target, '<' if not.\n"
              "Press Enter to start."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    inst.draw(); win.flip(); event.waitKeys(keyList=['return'])

    def generate_noise(w, h, grain=noise_grain):
        hg = int(math.ceil(h/grain)); wg = int(math.ceil(w/grain))
        small = np.clip(np.random.normal(0,0.3,(hg,wg)),-1,1)
        return np.repeat(np.repeat(small, grain, axis=0), grain, axis=1)[:h,:w]

    # output file
    out_dir = 'results'; os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trial','TargetPresent','Response','Correct','RT',
            'NumGabors','GaborPos','TargetPos'
        ])

        # precompute grid offsets
        grid_w = grid_size_x * cell_size
        grid_h = grid_size_y * cell_size
        off_x, off_y = -grid_w/2, -grid_h/2

        for t in range(num_trials):
            # flash fixation
            fix_cross.draw(); win.flip(); core.wait(0.5)
            # trial setup
            n = random.choice([5,10,15]); tp = random.choice([True, False])
            pos = []
            while len(pos) < n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos: pos.append((x,y))
            tgt_idx = random.randint(0,n-1) if tp else None
            # prepare Gabors
            gabors = []
            for i in range(n):
                sf = target_sf if i==tgt_idx else random.choice(spatial_frequencies)
                ori = target_orientation if i==tgt_idx else random.choice(orientations)
                g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                                       sf=sf, ori=ori, phase=0.25, units='pix')
                g.pos = (off_x + pos[i][0]*cell_size,
                         off_y + pos[i][1]*cell_size)
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
            # trial loop
            response, rt = None, None
            while clk.getTime() < trial_duration:
                # sample and simple smooth gaze
                if el_tracker:
                    samp = el_tracker.getNewestSample()
                    if samp and samp.isRightSample(): eye = samp.getRightEye()
                    elif samp and samp.isLeftSample(): eye = samp.getLeftEye()
                    else: eye = None
                    if eye:
                        rx, ry = eye.getGaze()
                        px = (prev_gx + rx)/2; py = (prev_gy + ry)/2
                        prev_gx = np.clip(px, 0, sw)
                        prev_gy = np.clip(py, 0, sh)
                                # draw static scene
                noise_stim.draw()
                for g in gabors:
                    g.draw()

                # draw Gaussian-edged mask to cover everything except peephole
                mask_im.pos = (prev_gx - half_w, half_h - prev_gy)
                mask_im.draw()

                win.flip()
                keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                if keys:
                    k,t0 = keys[0]
                    if k=='escape':
                        if el_tracker: el_tracker.stopRecording()
                        return filename
                    response, rt = ('target',t0) if k=='right' else ('distractor',t0)
                    break
                core.wait(movement_delay)
            if el_tracker:
                el_tracker.sendMessage('stimulus_offset'); el_tracker.stopRecording()
            # feedback
            if response is None:
                fb_text, corr = timeout_feedback_text, 0
            else:
                corr = int((response=='target')==tp)
                fb_text = 'Correct' if corr else 'Incorrect'
            visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
            win.flip(); core.wait(feedback_duration)
            # log trial
            writer.writerow([
                t+1, int(tp), response or 'None', corr, rt or '',
                n, pos, (pos[tgt_idx] if tgt_idx is not None else None)
            ])
    return filename
"""