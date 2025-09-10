#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:11:20 2025

@author: herttaleinonen

    Task for calibration check: 4 black squares on gray background.
    Participant moves a white '+' with their eyes; each square 'captures' with a short dwell.
    No timeout: runs until ESC. Completion is evaluated at ESC (all 4 captured or not).
    
"""

import os
import csv
import numpy as np
from psychopy import visual, core, event

def run_square_test(win, el_tracker,
                               screen_width, screen_height,
                               participant_id, timestamp):

    sw, sh = screen_width, screen_height
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # ---- visuals ----
    bg = visual.Rect(win, width=sw, height=sh, pos=(0, 0), units='pix',
                     fillColor=[0.2, 0.2, 0.2], lineColor=None)

    try:
        from config import cell_size
        sq_sz = int(max(2 * cell_size, 120))
    except Exception:
        sq_sz = 120

    half = sq_sz // 2
    offset_x = int(sw * 0.25)
    offset_y = int(sh * 0.25)
    square_positions = [(-offset_x,  offset_y),  # top-left
                        ( offset_x,  offset_y),  # top-right
                        (-offset_x, -offset_y),  # bottom-left
                        ( offset_x, -offset_y)]  # bottom-right

    squares = [
        visual.Rect(win, width=sq_sz, height=sq_sz, units='pix',
                    pos=pos, fillColor='black', lineColor=None)
        for pos in square_positions
    ]

    gaze_cross = visual.TextStim(win, text='+', color='white', height=40, units='pix')

    inst = visual.TextStim(win,
        text=("Gaze test:\n\n"
              "Move your eyes to place the white '+' on each of the four black squares.\n"
              "Hold briefly to capture—squares turn green when captured.\n"
              "Order doesn't matter.\n\n"
              "Press Enter to start. Press ESC to end at any time."),
        color='white', height=30, wrapWidth=sw*0.8, units='pix'
    )
    done_hint = visual.TextStim(win,
        text="All squares captured — press ESC to end.",
        color='white', height=28, pos=(0, -sh*0.35), units='pix'
    )

    inst.draw(); win.flip()
    event.waitKeys(keyList=['return'])
    event.clearEvents(eventType='keyboard')

    # ---- gaze params (EMA in screen coords) ----
    ema_alpha = 0.5
    ema_x, ema_y = sw/2.0, sh/2.0

    dwell_req_frames = 8  # ~130 ms @ 60 Hz
    in_counts = [0, 0, 0, 0]
    captured = [False, False, False, False]
    capture_times = [None, None, None, None]
    capture_order = []

    # ---- CSV header ----
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID','Completed','TotalTime(s)',
            'CaptureOrder','CaptureTimes(s)','SquarePositions(px,px)'
        ])

        clk = core.Clock()

        # EyeLink start
        if el_tracker:
            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')
            el_tracker.sendMessage('TRIALID 1')
            el_tracker.startRecording(1,1,1,1)
            core.wait(0.1)
            el_tracker.sendMessage('stimulus_onset')

        # ---- main loop: no timeout; ESC to end ----
        while True:
            # gaze sample & smoothing (screen coords)
            if el_tracker:
                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()

                if eye is not None:
                    rx, ry = eye.getGaze()
                    if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                        ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, sw-1))
                        ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, sh-1))

            # convert to centered coords for drawing/hit-testing
            gx = ema_x - (sw/2.0)
            gy = (sh/2.0) - ema_y
            gaze_cross.pos = (gx, gy)

            # hit test against squares; dwell to capture
            for i, (cx, cy) in enumerate(square_positions):
                if captured[i]:
                    continue
                inside = (abs(gx - cx) <= half) and (abs(gy - cy) <= half)
                if inside:
                    in_counts[i] += 1
                    if in_counts[i] >= dwell_req_frames:
                        captured[i] = True
                        capture_times[i] = clk.getTime()
                        capture_order.append(i)
                        squares[i].fillColor = 'green'
                else:
                    in_counts[i] = 0

            # draw
            bg.draw()
            for sq in squares:
                sq.draw()
            # show a subtle hint once all four are captured
            if all(captured):
                done_hint.draw()
            gaze_cross.draw()
            win.flip()

            # ESC to end at any time
            if event.getKeys(keyList=['escape']):
                break

        total_time = clk.getTime()

        # EyeLink stop
        if el_tracker:
            el_tracker.sendMessage('stimulus_offset')
            el_tracker.stopRecording()

        # feedback (short)
        completed = int(all(captured))
        fb_text = "All squares captured! Nice." if completed else "Test ended."
        visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
        win.flip(); core.wait(1.0)
        event.clearEvents(eventType='keyboard')

        # CSV row
        writer.writerow([
            'gaze cross test',
            participant_id,
            completed,
            round(total_time, 3),
            capture_order,
            [round(t, 3) if t is not None else '' for t in capture_times],
            square_positions
        ])

    return filename
