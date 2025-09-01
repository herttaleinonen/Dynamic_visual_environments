#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:40:52 2025

@author: herttaleinonen

static_task.py

Provides run_static_trials() for calling from main.py after EyeLink setup.
Displays stationary Gabor arrays on a static Gaussian noise background,
flashes a fixation cross between trials for 0.5s,
collects responses during the trial duration window,
provides feedback between trials.
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

def run_static_trials(win, el_tracker,
                      screen_width, screen_height,
                      participant_id, timestamp,
                      noise_grain=3):
    """
    Runs the static Gabor detection task:
      - Flash fixation (0.5s) between trials
      - Show static noise + Gabors for `trial_duration`, capture < and > responses
      - Provide feedback ('Correct', 'Incorrect', or timeout) for `feedback_duration`
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

    # instruction
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
            # unique positions
            pos = []
            while len(pos)<n:
                x = random.randint(2, grid_size_x-3)
                y = random.randint(2, grid_size_y-3)
                if (x,y) not in pos: pos.append((x,y))
            tgt_idx = random.randint(0,n-1) if tp else None

            # prepare Gabors
            gabors = []
            for i in range(n):
                if i==tgt_idx:
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

            # stimulus & response window
            response, rt = None, None
            while clk.getTime() < trial_duration:
                noise_stim.draw()
                for g in gabors: g.draw()
                win.flip()
                keys = event.getKeys(keyList=['right','left','escape'], timeStamped=clk)
                if keys:
                    k, t0 = keys[0]
                    if k=='escape': return filename
                    response = 'target' if k=='right' else 'distractor'
                    rt = t0
                    break
                core.wait(movement_delay)

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            # feedback between trials
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

            # log
            writer.writerow([
                t+1, int(tp), resp_str, corr, rt_str,
                n, pos, (pos[tgt_idx] if tgt_idx is not None else None)
            ])

    return filename


"""
import os
import csv
import random
import numpy as np
from psychopy import visual, core, event
from config import (
    grid_size_x, grid_size_y, cell_size, 
    num_trials, trial_duration, feedback_duration, timeout_feedback_text,
    orientations, spatial_frequencies, target_orientation,
    target_sf, movement_delay
)


# -------- Helper functions --------

# Generate Gaussian noise
def generate_noise(screen_width, screen_height):
    noise = np.random.normal(loc=0, scale=0.3, size=(screen_height, screen_width))
    return np.clip(noise, -1, 1)


# Center the grid
def grid_to_pixel(x, y, offset_x, offset_y, cell_size):

    #Converts grid coordinates (x, y) to pixel coordinates on the screen.

    # Correct calculation of screen pixel positions considering the offset and grid size
    pixel_x = offset_x + (x * cell_size)
    pixel_y = offset_y + (y * cell_size)
    
    return (pixel_x, pixel_y)


# Ensure unique positions for the Gabors to avoid overlap
def generate_unique_positions(num_gabors, grid_size_x, grid_size_y):

    #Generate a list of unique positions for the Gabors within the grid bounds.

    positions = set()  # Using a set to avoid duplicates
    while len(positions) < num_gabors:
        # Generate random positions within the grid
        x = random.randint(2, grid_size_x - 3)
        y = random.randint(2, grid_size_y - 3)
        positions.add((x, y))  # Add to the set (automatically handles duplicates)
    return list(positions)  # Convert back to a list for easy access


# -------- Trial loop --------
def run_static_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")
    
    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2 
    grid_offset_y = -grid_pixel_height / 2

    # Initialize Noise and Instructions
    noise_stim = visual.ImageStim(win, image=generate_noise(screen_width, screen_height),
                                  size=(screen_width, screen_height), units="pix", interpolate=True)

    instruction_text = visual.TextStim(win, text="In the following experiment, you will see stationary objects.\n"
                                                 "Among them is a target, that is tilted 45 degrees like this: /.\n"
                                                 "Press '>' if you see the target.\n"
                                                 "Press '<' if you do not.\n"
                                                 "Press Enter to start.",
                                       color="white", height=30, wrapWidth=screen_width * 0.8)
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=["return"])

    # Open a CSV file for behavioural results
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
                         "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Position"])

        for trial in range(num_trials):
            if trial == 3:
                ready_text = visual.TextStim(win, text="Training phase is over! The experiment begins now. Press Enter to continue.",
                                             color="white", height=30)
                ready_text.draw()
                win.flip()
                event.waitKeys(keyList=["return"])

            event.clearEvents()

            # Number of Gabors defined here
            num_gabors = random.choice([5, 10, 15])
            target_present = random.choice([True, False])

            # Generate unique positions for distractors and target (to avoid overlap)
            positions = generate_unique_positions(num_gabors, grid_size_x, grid_size_y)
            target_index = random.randint(0, num_gabors - 1) if target_present else None

            # Create the distractor Gabor positions and target presence
            distractor_oris = random.choices(orientations, k=num_gabors - int(target_present))
            distractor_sfs = random.choices(spatial_frequencies, k=num_gabors - int(target_present))

            gabors = []
            d_idx = 0
            target_position = None  # To store the target's position

            for i in range(num_gabors):
                if i == target_index:
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=target_sf, ori=target_orientation, phase=0.25)
                    target_position = positions[i]  # Store target's position here
                else:
                    gabor = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                               sf=distractor_sfs[d_idx], ori=distractor_oris[d_idx], phase=0.25)
                    d_idx += 1
                gabors.append(gabor)

            # Assign positions once (static)
            for i, (x, y) in enumerate(positions):
                gabors[i].pos = grid_to_pixel(x, y, grid_offset_x, grid_offset_y, cell_size)

            trial_clock = core.Clock()
            response = None
            rt = None

            # No need to update positions since they're static
            gabor_trajectory = [positions]  # Record the static positions just once

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

            while trial_clock.getTime() < trial_duration:
                noise_stim.image = generate_noise(screen_width, screen_height)
                noise_stim.draw()

                # We no longer need to update the positions since they're static
                for g in gabors:
                    g.draw()
                win.flip()

                keys = event.getKeys(keyList=["right", "left", "escape"], timeStamped=trial_clock)
                if keys:
                    response, rt = keys[0]
                    break

                core.wait(movement_delay)

            # Log a message to mark the offset of the stimulus
            el_tracker.sendMessage('stimulus_offset')

            # Stop recording
            el_tracker.stopRecording()

            if response:
                is_correct = (response == "right" and target_present) or (response == "left" and not target_present)
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

            response_num = 1 if response == "right" else 0 if response == "left" else -1
            writer.writerow(["static task", participant_id, trial + 1, int(target_present),
                             response_num, int(is_correct), rt,
                             num_gabors, gabor_trajectory, target_position])  # Recording target_position


"""
