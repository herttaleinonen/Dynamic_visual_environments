#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:52:42 2026

@author: herttaleinonen
"""

import numpy as np
import pandas as pd

from config import (
    MODEL_SAMPLING_RATE,
    GAZE_MODE,
    GLOBAL_TND_S,
)

from parsing_and_gaze import (
    debug_inventory,
    collect_visibility_files,
    collect_search_pairs,
)

from fitting import (
    make_synthetic_human_trials,
    fit_eta_theta_from_trials_fast,
    run_loss_surface_batch,
    run_full_replay_test_only,
    fit_eta_theta_weighted,
    fit_model_per_participant,
    run_model_recovery,
    fit_model_comparison_per_participant,
    fit_model_comparison_visibility,
    preprocess_participant_trials,
)

from visibility import build_dprime_splines_for_participant

from replay import (
    run_full_replay,
    build_saccade_prediction_table,
    run_replay_trial,
)



# =========================
# Main
# =========================
if __name__ == "__main__":
    SEARCH_DIR = "data/search"
    VIS_DIR = "data/visibility"
    debug_inventory(SEARCH_DIR, VIS_DIR)
    
    
    DO_FIT = True

    if DO_FIT:
        print("\n===================================")
        print("RUNNING PARAMETER FITTING AND REPLAY ")
        print("=====================================")
        SEED_OFFSETS = [42345]  
        ETA_GRID   = np.linspace(0.002, 0.20, 12)
        THETA_GRID = np.linspace(0.02, 1.0, 12)

        for offset in SEED_OFFSETS:
            print(f"\n--- seed {offset} ---")
            
            fit_model_per_participant(     # Runs the parameter fitting on the train-set of the participant data
                search_dir=SEARCH_DIR,
                visibility_dir=VIS_DIR,
                eta_grid=ETA_GRID,
                theta_grid=THETA_GRID,
                n_repeats_fit=5,              
                spline_s=0.5,
                out_csv="fitted_params.csv", #f"fitted_params_seed{offset}.csv",
                warmstart_csv=None,
                eta_lo=0.01, eta_hi=0.30,     
                theta_lo=0.10, theta_hi=4.0,  
                split_seed_offset=offset,
                dt_override_s=MODEL_SAMPLING_RATE,  
                use_group_visibility=False
            )
            
            run_full_replay_test_only(  # Runs the replay model on the held-out (test) participant data
                search_dir=SEARCH_DIR,
                visibility_dir=VIS_DIR,
                output_csv="replay_model_results_test.csv", #f"replay_model_results_full_seed{offset}.csv",
                fitted_params_csv= "fitted_params.csv", #f"fitted_params_seed{offset}.csv",
                n_model_repeats=1000,
                split_seed_offset=offset,
                dt_override_s=MODEL_SAMPLING_RATE,
                use_group_visibility=False
            )
            
            run_full_replay(            # Runs the replay model on the full participant data (train + test)
                search_dir=SEARCH_DIR,
                visibility_dir=VIS_DIR,
                output_csv="replay_model_results_full.csv", #f"replay_model_results_test_seed{offset}.csv",
                fitted_params_csv= "fitted_params.csv", #f"fitted_params_seed{offset}.csv",
                n_model_repeats=1000,
                dt_override_s=MODEL_SAMPLING_RATE,
                use_group_visibility=False
            )
        
    
    DO_SACCADE_PREDICTION = True  
    
    if DO_SACCADE_PREDICTION:
        
        print("\n==============================")
        print("RUNNING SACCADE PREDICTION")
        print("==============================")
        
        sac_df = build_saccade_prediction_table(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            fitted_params_csv="fitted_params.csv",
            output_csv="saccade_prediction_table.csv",
            spline_s=0.5,
            dt_override_s=MODEL_SAMPLING_RATE,
            gaze_mode=GAZE_MODE,
        )
 
    
    
    
    

    #=================================================================================
    #                       OPTIONAL DEBUG-RUNS BELOW:
    #=================================================================================

    
    DO_WEIGHT_SENSITIVITY = False
    
    if DO_WEIGHT_SENSITIVITY:
        ETA_GRID = np.linspace(0.002, 0.20, 12)
        THETA_GRID = np.linspace(0.02, 1.0, 12)
    
        weight_conditions = [
            ("equal (original)", 1.0, 1.0),
            ("2x choice weight", 2.0, 1.0),
            ("2x RT weight",     1.0, 2.0),
            ("0.5x choice weight", 0.5, 1.0),
            ("0.5x RT weight",     1.0, 0.5),
        ]
    
        vis_by_pp = collect_visibility_files(VIS_DIR)
        search_by_pp = collect_search_pairs(SEARCH_DIR)
    
        rows = []
        for pp in sorted(search_by_pp.keys()):
            dprime_model_pp = build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=0.5)
            trials_pp = preprocess_participant_trials(pp, search_by_pp[pp], dprime_model_pp)
            if len(trials_pp) < 10:
                continue
            for label, wc, wr in weight_conditions:
                eta_hat, theta_hat, loss = fit_eta_theta_weighted(
                    pp=pp, trials_pp=trials_pp, dprime_models_pp=dprime_model_pp,
                    eta_grid=ETA_GRID, theta_grid=THETA_GRID,
                    w_choice=wc, w_rt=wr, n_repeats_fit=10,
                )
                rows.append(dict(participant=pp, weighting=label, w_choice=wc, w_rt=wr,
                                  eta_hat=eta_hat, theta_hat=theta_hat, loss=loss))
            print(f"[WEIGHT-SENS] done {pp}")
    
        pd.DataFrame(rows).to_csv("weight_sensitivity.csv", index=False)
        print("[WEIGHT-SENS] wrote weight_sensitivity.csv")
        
    
    """
    DO_FIT = False  

    if DO_FIT:
        # parameter grid size 
        ETA_GRID   = np.linspace(0.002, 0.20, 12) 
        THETA_GRID = np.linspace(0.02, 1.0, 12)
    
        print("\n==============================")
        print("RUNNING PARAMETER GRID SEARCH")
        print("==============================")
    
        fit_df = fit_model_per_participant(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=ETA_GRID,
            theta_grid=THETA_GRID,
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="fitted_params_test.csv",
            warmstart_csv=None,
            eta_lo=0.01,
            eta_hi=0.30,
            theta_lo=0.10,
            theta_hi=4.0,
            dt_override_s=MODEL_SAMPLING_RATE,
            use_group_visibility=False,
        )
    
        print("Wrote fitted_params_test.csv")


    DO_REPLAY = False
    
    if DO_REPLAY:
        
        print("\n==============================")
        print("RUNNING DSFM")
        print("==============================")
        
        run_full_replay_test_only(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            output_csv="replay_model_results_test.csv",
            fitted_params_csv="fitted_params_test.csv",
            n_model_repeats=1000,
            spline_s=0.5,
            max_calib_drift_deg=None,
            dt_override_s=MODEL_SAMPLING_RATE,
            use_group_visibility=False,
        )
    
    """ 
    
    DO_VIS_COMPARISON = False # <-- change to True only when running a visibility null model comparisons 
    
    if DO_VIS_COMPARISON:
    
        print("\n==============================")
        print("RUNNING VISIBILITY NULL MODEL COMPARISON")
        print("==============================")

        print("\n--- Group visibility vs individual ---")
        fit_model_comparison_visibility(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=np.linspace(0.002, 0.20, 12),
            theta_grid=np.linspace(0.02, 1.0, 12),
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_comparison_visibility_group.csv",
            null_mode="group",
            eta_lo=0.01,   eta_hi=0.30,
            theta_lo=0.10, theta_hi=4.0,
            dt_override_s=MODEL_SAMPLING_RATE,
        )

        print("\n--- Constant visibility vs individual ---")
        fit_model_comparison_visibility(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=np.linspace(0.002, 0.20, 12),
            theta_grid=np.linspace(0.02, 1.0, 12),
            n_repeats_fit=5,
            spline_s=0.5,
            #fitted_params_csv="fitted_params_test.csv",
            out_csv="model_comparison_visibility_constant.csv",
            null_mode="constant",
            eta_lo=0.01,   eta_hi=0.30,
            theta_lo=0.10, theta_hi=4.0,
            dt_override_s=MODEL_SAMPLING_RATE,
        )
        
        
    DO_MODEL_COMPARISON = False # <-- change to True only when running a model comparison (1 vs. 2 params)
    
    if DO_MODEL_COMPARISON:
    
        # Pull group-mean fixed values from the full fit so the single-param
        # baselines are anchored at a sensible location, not arbitrary constants
        _full = pd.read_csv("fitted_params_test.csv")
        ETA_FIXED   = float(_full["eta"].mean())
        THETA_FIXED = float(_full["theta"].mean())
        print(f"\nFixed values for restricted models: eta={ETA_FIXED:.4f}, theta={THETA_FIXED:.4f}")
    
        ETA_GRID_C   = np.linspace(0.002, 0.20, 12)
        THETA_GRID_C = np.linspace(0.02,  1.0,  12)
    
        print("\n==============================")
        print("RUNNING MODEL COMPARISON (AIC)")
        print("==============================")
    
        cmp_df = fit_model_comparison_per_participant(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=ETA_GRID_C,
            theta_grid=THETA_GRID_C,
            eta_fixed=ETA_FIXED,
            theta_fixed=THETA_FIXED,
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_comparison.csv",
            eta_lo=0.01,   eta_hi=0.30,
            theta_lo=0.10, theta_hi=4.0,
            dt_override_s=MODEL_SAMPLING_RATE,
        )
    
        # Quick summary table printed to console
        cols = ["participant",
                "m2_AIC", "meta_AIC", "mtheta_AIC",
                "delta_AIC_vs_eta", "delta_AIC_vs_theta"]
        print("\n" + cmp_df[cols].to_string(index=False))
        

    DO_RECOVERY = False # <-- change to True only when running a model recovery analysis 
    
    if DO_RECOVERY:
        ETA_TRUE   = np.linspace(0.02, 0.18, 5)     # 5 evenly-spaced points within [0.002, 0.20]
        THETA_TRUE = np.linspace(0.1, 0.9, 5)       # 5 evenly-spaced points within [0.02, 1.0]

        ETA_GRID   = np.linspace(0.002, 0.20, 20)
        THETA_GRID = np.linspace(0.02, 1.0, 20)

        print("\n==============================")
        print("RUNNING MODEL RECOVERY")
        print("==============================")

        rec = run_model_recovery(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_true_grid=ETA_TRUE,
            theta_true_grid=THETA_TRUE,
            eta_fit_grid=ETA_GRID,
            theta_fit_grid=THETA_GRID,
            n_repeats_fit=10,
            spline_s=0.5,
            out_csv="model_recovery.csv",
        )
    
    DO_LOSS_SURFACE = False
    
    if DO_LOSS_SURFACE:
        ETA_GRID   = np.linspace(0.002, 0.20, 20)
        THETA_GRID = np.linspace(0.02, 1.0, 20)

        SURFACE_PARTICIPANTS = ["kh1", "kh7", "kh9"]   # for example
        SURFACE_CONDITIONS = [
            (0.05, 0.4),
            (0.12, 0.4),
            (0.12, 0.7),
        ]

        print("\n==============================")
        print("RUNNING LOSS-SURFACE SWEEP")
        print("==============================")

        run_loss_surface_batch(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            participants=SURFACE_PARTICIPANTS,
            conditions=SURFACE_CONDITIONS,
            eta_grid=ETA_GRID,
            theta_grid=THETA_GRID,
            n_repeats=15,
            out_npz="loss_surfaces.npz",
        )
    
    
    DO_ETA_BIAS_CHECK = False
    
    if DO_ETA_BIAS_CHECK:
        print("\n==============================")
        print("ETA BIAS CHECK: eta_true=0.15, n_repeats=5 vs 10")
        print("==============================")

        vis_by_pp = collect_visibility_files(VIS_DIR)
        search_by_pp = collect_search_pairs(SEARCH_DIR)

        ETA_GRID   = np.linspace(0.002, 0.20, 12)
        THETA_GRID = np.linspace(0.02, 1.0, 12)

        for pp in ["kh1", "kh7", "kh9"]:
            dprime_model_pp = build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=0.5)
            trials_real = preprocess_participant_trials(pp, search_by_pp[pp], dprime_model_pp)

            for theta_true in [0.15, 0.4, 0.7]:
                synth_trials = make_synthetic_human_trials(
                    trials_pp=trials_real,
                    dprime_models_pp=dprime_model_pp,
                    eta_true=0.15,
                    theta_true=theta_true,
                    tnd_s=GLOBAL_TND_S,
                )
                for n_rep in [5, 10]:
                    eta_hat, theta_hat, loss = fit_eta_theta_from_trials_fast(
                        pp=pp, trials_pp=synth_trials, dprime_models_pp=dprime_model_pp,
                        eta_grid=ETA_GRID, theta_grid=THETA_GRID, n_repeats_fit=n_rep,
                    )
                    print(f"{pp} theta_true={theta_true} n_repeats={n_rep}: "
                          f"eta_hat={eta_hat:.3f}, theta_hat={theta_hat:.3f}")
    
    
    DO_DV_DIAGNOSTIC = False
    
    if DO_DV_DIAGNOSTIC:
        print("\n==============================")
        print("D_t FINAL-VALUE DIAGNOSTIC (max-pooling bias check)")
        print("==============================")

        vis_by_pp = collect_visibility_files(VIS_DIR)
        search_by_pp = collect_search_pairs(SEARCH_DIR)
        pp = "kh1"
        dprime_model_pp = build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=0.5)
        trials_pp = preprocess_participant_trials(pp, search_by_pp[pp], dprime_model_pp)

        eta_fixed = 0.15
        tp_finals, ta_finals = [], []
        rng = np.random.default_rng(0)


        for tr in trials_pp[:200]:
            resp, rt, dv_final = run_replay_trial(
                obj_xy_cells=tr["obj"],
                gaze_xy_cells=tr["gaze_cells"],
                speed_px_s=tr["speed_px_s"],
                dprime_splines=dprime_model_pp,
                dt_s=tr["dt_s"],
                eta=eta_fixed,
                decision_theta_present=100.0,   # unreachable -> forces timeout, gives raw dv
                target_present=tr["target_present"],
                target_index=tr["target_index"],
                alpha_search=tr["alpha_trial"],
                rng=rng,
                return_dv=True,
            )
            if tr["target_present"] == 1:
                tp_finals.append(dv_final)
            else:
                ta_finals.append(dv_final)

        ta_arr = np.array(ta_finals)
        tp_arr = np.array(tp_finals)
        print(f"TA trials (n={len(ta_arr)}): min={ta_arr.min():.3f}, "
              f"median={np.median(ta_arr):.3f}, % negative={np.mean(ta_arr < 0):.3f}")
        print(f"TP trials (n={len(tp_arr)}): min={tp_arr.min():.3f}, "
              f"median={np.median(tp_arr):.3f}")
        
    
    
