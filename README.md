# The experimental setup

The main.py runs a set of experiments developed to study dynamic visual search. 

1. Static Visual Search task (static_task.py)
Gabors remain in fixed positions throughout the trial.
The target appears either at a random location or at the same location as in one of the previous 1–4 trials.

2. Dynamic Visual Search task (dynamic_task.py)
Gabors move continuously along unpredictable paths during the trial.

3. Static Evading-Target Search task (evading_target_static_task.py)
The target’s location is dynamically manipulated based on the participant’s eye movements.
Shortly after the search begins, the target reappears at locations corresponding to 1–4 previous fixations, replacing distractors that have already been inspected.

4. Dynamic Evading-Target Search task (evading_target_dynamic_task.py)
Same as Task 3, but the Gabors move continuously.

In addition there is a visibility_task.py to measure the visibility of the search array, and a test_task.py to test the calibration.


# Running the experiment

Requires python3 (3.10.0) and PsychoPy (2024.2.4).

Requires EyeLink 1000 Plus + pylink to run with eye tracking. 

Optional Cedrus response box support (requires RB-530 or similar + pyxid2). 

To run main.py through PsychoPy Coder, place EyeLinkCoreGraphicsPsychoPy.py (from SR research) into the project folder. 



