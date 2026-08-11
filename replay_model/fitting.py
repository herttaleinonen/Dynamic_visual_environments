#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:49:12 2026

@author: herttaleinonen
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from numpy.random import SeedSequence
import zlib

from config import (
    GLOBAL_TND_S,
    SPEED_ORDER,
    DT_TO_SPEED,
    VISIBILITY_MODE,
    VIS_NULL_SOURCE_SPEED,
    VIS_CONSTANT_RULE,
    VIS_REF_ECC_DEG,
    GRID_SIZE_X,
    GRID_SIZE_Y,
    GAZE_MODE
)

from parsing_and_gaze import (
    safe_int,
    safe_float,
    safe_parse_positions,
    safe_parse_trajectory,
    parse_asc_events,
    estimate_dt_from_positions,
    dt_from_duration,
    gaze_series_cells_from_fixations,
    collect_visibility_files,
    collect_search_pairs,
    dt_from_name,
    stable_trial_seed,
    perturb_gaze,
)

from visibility import (
    build_dprime_splines_for_participant,
    make_visibility_null_model,
    rates_to_dprime,
    build_group_mean_dprime_splines,
)

from replay import (
    run_replay_trial,
    infer_target_index,
    ecc_deg_from_cells,
    min_target_eccentricity
)


# ------------------
# Fitting helpers
# ------------------
def make_refined_grid(center: float, half_width: float, n: int, lo: float, hi: float) -> np.ndarray:
    """n-point grid from center-half_width .. center+half_width, clipped to [lo, hi]."""
    g = np.linspace(center - half_width, center + half_width, n)
    return np.clip(g, lo, hi)


def safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))


def _human_scales(human_by_speed, eps: float = 1e-6) -> dict:
    """
    Compute SD across speeds for each summary metric (ignore NaNs).

    For RT metrics, compute SD in log-space (to match loss space).
    Returns: dict like {"hit": sd, "fa": sd, "rt_tp": sd, "rt_ta": sd}
    """
    scales = {}

    for key in ["hit", "fa", "rt_tp", "rt_ta"]:
        vals = []
        for s in SPEED_ORDER:
            v = human_by_speed[s].get(key, np.nan)
            if np.isnan(v):
                continue

            if key.startswith("rt_"):
                vals.append(float(safe_log(v)))  # match loss space
            else:
                vals.append(float(v))

        vals = np.asarray(vals, float)
        sd = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan

        # fallback: avoid division by tiny numbers
        if np.isnan(sd) or sd < eps:
            sd = 1.0

        scales[key] = sd

    return scales



# ------------------
# Pre-processing
# ------------------
def preprocess_participant_trials(
    pp: str,
    pairs: List[Tuple[str, str]],
    dprime_models_pp: Dict[int, UnivariateSpline],
) -> List[dict]:
    """
    Returns a list of dicts, each containing everything needed to simulate a trial.
    This is the expensive part (ASC parsing, gaze reconstruction, literal_eval of positions)
    and should be done ONCE per participant.
    """
    trials_out: List[dict] = []

    for csv_path, asc_path in pairs:
        dt = dt_from_name(csv_path)
        if dt is None:
            continue
        speed_px_s = DT_TO_SPEED[dt]
        if speed_px_s not in dprime_models_pp:
            continue

        # speed-dependent schedules (fixed during fit)
        alpha_use = 0.5 * (1 - 0.5 * (speed_px_s / 400))

        df = pd.read_csv(csv_path)
        needed = ["Trial", "Gabor Positions", "Target Present", "Target Trajectory",
          "Response", "Correct", "Reaction Time (s)"]

        if any(c not in df.columns for c in needed):
            continue

        (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)

        df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

        for _, r in df.iterrows():
            trial = safe_int(r.get("Trial"))
            target_present = safe_int(r.get("Target Present"))
            human_resp = safe_int(r.get("Response"))
            human_corr = safe_int(r.get("Correct"))
            human_rt = safe_float(r.get("Reaction Time (s)"))
            if (trial is None or target_present is None or
                human_resp is None or human_corr is None or human_rt is None):
                continue


            obj = r.get("gabor_pos")
            if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3 or obj.shape[0] < 2:
                continue

            # dt estimate
            dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
            if dt_s is None or dt_s <= 0:
                dt_s = dt_from_duration(obj, 3.5)

            # gaze series
            tr_eye = eye_trials.get(trial)
            if tr_eye is None:
                gaze_cells = np.full((obj.shape[0], 2), np.nan, dtype=float)
            else:
                gaze_cells = gaze_series_cells_from_fixations(tr_eye, obj.shape[0], dt_s, screen_w, screen_h)
                

            # target index if present
            target_index = None
            if target_present == 1:
                traj = safe_parse_trajectory(r.get("Target Trajectory"))
                if traj is not None:
                    target_index = infer_target_index(obj, traj)


            alpha_trial = alpha_use   # same for TP and TA

            # stable per-trial seed (important)
            seed_str = f"{pp}|{os.path.basename(csv_path)}|{trial}"
            trial_seed = zlib.crc32(seed_str.encode("utf-8")) & 0xFFFFFFFF

            trials_out.append(dict(
                speed_px_s=speed_px_s,
                obj=obj,
                gaze_cells=gaze_cells,
                dt_s=dt_s,
                target_present=target_present,
                target_index=target_index,
                alpha_trial=alpha_trial,
                trial_seed=trial_seed,
                
                
                # human fields for train/test summaries
                human_resp=human_resp,
                human_corr=human_corr,
                human_rt=human_rt,
            ))

    return trials_out


# ------------------
# Model recovery helpers
# ------------------
def information_criteria(loss, n_obs, k):
    """
    Computes AIC and BIC from SSE-like loss.
    """
    if loss <= 0:
        loss = 1e-12

    logL = -0.5 * n_obs * np.log(loss / n_obs)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(n_obs) - 2 * logL

    return AIC, BIC

def fit_eta_theta_weighted(pp, trials_pp, dprime_models_pp, eta_grid, theta_grid,
                            w_choice=1.0, w_rt=1.0, n_repeats_fit=10, tnd_s=GLOBAL_TND_S):
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)

    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp)
                 for i, tr in enumerate(train_trials)}

    best = {"loss": np.inf, "eta": None, "theta": None}
    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed_fast(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
                tnd_s=tnd_s, kin_cache=kin_cache,
            )
            L = loss_summary(human_train, model, w_choice=w_choice, w_rt=w_rt)
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}
    return best["eta"], best["theta"], best["loss"]

def compute_loss_surface(pp, synth_trials, dprime_models_pp, eta_grid, theta_grid,
                          n_repeats=15, tnd_s=GLOBAL_TND_S):
    """
    Computes the full loss surface over (eta_grid x theta_grid) for one
    synthetic dataset, instead of just returning the argmin. Reuses the
    same train split and kinematics cache as the fitter for consistency.
    Returns loss_grid of shape (len(eta_grid), len(theta_grid)).
    """
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(synth_trials))
    n_train = int(0.7 * len(idx))
    train_trials = [synth_trials[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)

    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp)
                 for i, tr in enumerate(train_trials)}

    loss_grid = np.full((len(eta_grid), len(theta_grid)), np.nan)
    for i, eta in enumerate(eta_grid):
        for j, theta in enumerate(theta_grid):
            model = simulate_model_summary_from_preprocessed_fast(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats,
                tnd_s=tnd_s, kin_cache=kin_cache,
            )
            loss_grid[i, j] = loss_summary(human_train, model)
    return loss_grid


def run_loss_surface_batch(search_dir, visibility_dir, participants, conditions,
                            eta_grid, theta_grid, n_repeats=15, spline_s=0.5,
                            out_npz="loss_surfaces.npz"):
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(visibility_dir if False else search_dir)
    dprime_models = {pp: build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=spline_s)
                      for pp in participants if pp in vis_by_pp}

    results = {}
    for pp in participants:
        if pp not in dprime_models or pp not in search_by_pp:
            print(f"[SKIP] {pp}: missing data")
            continue
        trials_real = preprocess_participant_trials(pp, search_by_pp[pp], dprime_models[pp])
        if len(trials_real) < 10:
            continue
        for (eta_true, theta_true) in conditions:
            synth_trials = make_synthetic_human_trials(
                trials_pp=trials_real, dprime_models_pp=dprime_models[pp],
                eta_true=float(eta_true), theta_true=float(theta_true), tnd_s=GLOBAL_TND_S,
            )
            print(f"[SURFACE] {pp} eta_true={eta_true} theta_true={theta_true} ...")
            loss_grid = compute_loss_surface(
                pp=pp, synth_trials=synth_trials, dprime_models_pp=dprime_models[pp],
                eta_grid=eta_grid, theta_grid=theta_grid, n_repeats=n_repeats,
            )
            key = f"{pp}_eta{eta_true}_theta{theta_true}"
            results[key] = loss_grid

    np.savez(out_npz, eta_grid=eta_grid, theta_grid=theta_grid, **results)
    print(f"[SURFACE] wrote {out_npz} with {len(results)} surfaces")
    return results

def make_synthetic_human_trials(
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_true: float,
    theta_true: float,
    tnd_s: float = GLOBAL_TND_S,
) -> List[dict]:
    """
    Returns a copy of trials_pp where the human_resp/human_rt/human_corr fields
    are replaced with simulated responses from the model at (eta_true, theta_true)
    for purposes of model recovery analysis. 
    """
    out = []
    for tr in trials_pp:
        tr2 = dict(tr)  # shallow copy

        # one RNG per trial (stable)
        rng = np.random.default_rng(tr["trial_seed"])

        resp, rt = run_replay_trial(
            obj_xy_cells=tr["obj"],
            gaze_xy_cells=tr["gaze_cells"],
            speed_px_s=tr["speed_px_s"],
            dprime_splines=dprime_models_pp,
            dt_s = tr["dt_s"],          
            dt_override_s = None,       
            eta=float(eta_true),
            decision_theta_present=float(theta_true),
            target_present=tr["target_present"],
            target_index=tr["target_index"],
            alpha_search=tr["alpha_trial"],
            rng=rng,
        )

        rt = float(rt + tnd_s)

        tr2["human_resp"] = int(resp)
        tr2["human_rt"] = float(rt)

        # correctness wrt ground truth (target_present)
        tr2["human_corr"] = int((resp == 1 and tr["target_present"] == 1) or (resp == 0 and tr["target_present"] == 0))

        out.append(tr2)

    return out


def count_obs(human_by_speed):
    n = 0
    for s in SPEED_ORDER:
        h = human_by_speed[s]
        for key in ["hit", "fa", "rt_tp", "rt_ta"]:
            if not np.isnan(h[key]):
                n += 1
    return n


# ------------------
# Loss function
# ------------------
def loss_summary(human_by_speed: Dict[int, Dict[str, float]],
                 model_by_speed: Dict[int, Dict[str, float]],
                 w_choice: float = 1.0,
                 w_rt: float = 1.0) -> float:

    loss = 0.0

    for s in SPEED_ORDER:
        h = human_by_speed[s]
        m = model_by_speed[s]

        # d'
        if not (np.isnan(h["hit"]) or np.isnan(h["fa"]) or
                np.isnan(m["hit"]) or np.isnan(m["fa"])):

            d_h = rates_to_dprime(h["hit"], h["fa"])
            d_m = rates_to_dprime(m["hit"], m["fa"])

            loss += w_choice * (d_h - d_m) ** 2

        # TP RT
        if not (np.isnan(h["rt_tp"]) or np.isnan(m["rt_tp"])):
            z = safe_log(h["rt_tp"]) - safe_log(m["rt_tp"])
            loss += w_rt * (z ** 2)

        # TA RT
        if not (np.isnan(h["rt_ta"]) or np.isnan(m["rt_ta"])):
            z = safe_log(h["rt_ta"]) - safe_log(m["rt_ta"])
            loss += w_rt * (z ** 2)

    return float(loss)


# ------------------
# Fitting loop
# ------------------
def fit_model_per_participant(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "fitted_params.csv",
    warmstart_csv: Optional[str] = None,
    eta_half_width: float = 0.15,
    theta_half_width: float = 6.0,
    theta_lo: float = 2.0,
    theta_hi: float = 12.0,
    theta_shift: float = 0.0,
    eta_lo: float = 0.05,
    eta_hi: float = 0.8,
    dt_override_s: Optional[float] = None,
    use_group_visibility: bool = False,
    split_seed_offset: int = 12345,
) -> pd.DataFrame:

    
    """
    Fits (eta, theta) per participant using a two-pass grid search (coarse + refine),
    with a stable 70/30 train/test split per participant for cross valucation.

    """

    # -----------------------------------------
    # Model parameter availability
    # -----------------------------------------
    # number of fitted parameters (eta, theta)
    K_PARAMS = 2

    # Fixed global NDT
    tnd = float(GLOBAL_TND_S)
    print(f"\n===== FIXED global NDT: tnd={tnd:.3f}s =====")

    # -----------------------------------------
    # Warmstart map (optional)
    # -----------------------------------------
    warm_map: Dict[str, Tuple[float, float]] = {}
    if warmstart_csv is not None:
        warm_df = pd.read_csv(warmstart_csv)
        if ("participant" in warm_df.columns) and ("eta" in warm_df.columns) and ("theta" in warm_df.columns):
            warm_map = {
                str(r["participant"]).strip(): (float(r["eta"]), float(r["theta"]))
                for _, r in warm_df.iterrows()
            }

    # -----------------------------------------
    # Collect files + build d' splines once
    # -----------------------------------------
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)
    """
    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base_model,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )
    """
    
    # build group splines once
    from visibility import build_group_mean_dprime_splines
    
    if use_group_visibility:
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        dprime_models = {pp: group_splines for pp in vis_by_pp.keys()}
    else:
        dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
        for pp, files in vis_by_pp.items():
            base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
            dprime_models[pp] = make_visibility_null_model(base_model,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )
        
    
    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[FIT SKIP] {pp}: no visibility splines")
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            print(f"[FIT SKIP] {pp}: too few preprocessed trials ({len(trials_pp)})")
            continue

        print(f"[FIT] {pp}: preprocessed {len(trials_pp)} trials")

        # -----------------------------------------
        # Train/test split (stable per participant)
        # -----------------------------------------
        #rng_split = np.random.default_rng(12345 + int(pp[2:]))
        rng_split = np.random.default_rng(split_seed_offset + int(pp[2:]))
        idx = rng_split.permutation(len(trials_pp))
        n_train = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials  = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test  = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)

        # -----------------------------------------
        # Per-participant grid (centered if warmstart)
        # -----------------------------------------
        if pp in warm_map:
            eta0, theta0_old = warm_map[pp]
            eta_grid_pp = make_refined_grid(eta0, eta_half_width, 9, lo=eta_lo, hi=eta_hi)

            theta_center = float(theta0_old + theta_shift)
            theta_grid_pp = make_refined_grid(theta_center, theta_half_width, 9, lo=theta_lo, hi=theta_hi)
        else:
            eta_grid_pp = np.asarray(eta_grid, dtype=float)
            theta_grid_pp = np.asarray(theta_grid, dtype=float)


        best = {"loss": np.inf, "eta": None, "theta": None}

        # ---------- PASS 1: coarse grid ----------
        for eta in eta_grid_pp:
            for theta in theta_grid_pp:
                model = simulate_model_summary_from_preprocessed(
                    trials_pp=train_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta=float(eta),
                    theta=float(theta),
                    n_repeats=n_repeats_fit,
                    tnd_s=tnd,
                    dt_override_s=dt_override_s,
                )
                L = loss_summary(human_train, model)
                if L < best["loss"]:
                    best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

        print(f"[FIT COARSE] {pp}: eta={best['eta']:.3f} theta={best['theta']:.3f} loss={best['loss']:.4f}")

        # ---------- PASS 2: refinement around winner ----------
        ETA_HALF_WIDTH_REFINE   = 0.15
        THETA_HALF_WIDTH_REFINE = 1.0
        N_REFINE = 12

        eta_ref = make_refined_grid(best["eta"], ETA_HALF_WIDTH_REFINE, N_REFINE, lo=eta_lo, hi=eta_hi)
        theta_ref = make_refined_grid(best["theta"], THETA_HALF_WIDTH_REFINE, N_REFINE, lo=theta_lo, hi=theta_hi)

        best2 = dict(best)

        for eta in eta_ref:
            for theta in theta_ref:
                model = simulate_model_summary_from_preprocessed(
                    trials_pp=train_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta=float(eta),
                    theta=float(theta),
                    n_repeats=n_repeats_fit,
                    tnd_s=tnd,
                    dt_override_s=dt_override_s,
                )
                L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0)
                if L < best2["loss"]:
                    best2 = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

        best = best2
        print(f"[FIT OK] {pp}: eta={best['eta']:.3f} theta={best['theta']:.3f} loss={best['loss']:.4f}")

        # information criteria 
        aic, bic = information_criteria(best["loss"], N_OBS, k=K_PARAMS)
        print(f"[FIT IC] {pp}: AIC={aic:.2f} BIC={bic:.2f}")

        # ===================== CROSS VALIDATION (TEST SET 30%) =====================
        model_test = simulate_model_summary_from_preprocessed(
            trials_pp=test_trials,
            dprime_models_pp=dprime_models[pp],
            eta=float(best["eta"]),
            theta=float(best["theta"]),
            n_repeats=n_repeats_fit,
            tnd_s=tnd,
            dt_override_s=dt_override_s,
        )

        test_loss = loss_summary(human_test, model_test, w_choice=1.0, w_rt=1.0)
        print(f"[FIT TEST] {pp}: test_loss={test_loss:.4f}")
        # ===============================================================

        rows.append({
            "participant": pp,
            "eta": best["eta"],
            "theta": best["theta"],
            "loss": best["loss"],
            "test_loss": test_loss,
            "AIC": aic,
            "BIC": bic,
            "tnd": tnd,  
        })

    if len(rows) == 0:
        raise RuntimeError("No participants were fit successfully (check file pairing / preprocessing).")

    fit_df = pd.DataFrame(
        rows,
        columns=["participant", "eta", "theta", "loss", "test_loss", "AIC", "BIC", "tnd"],
    )
    fit_df.to_csv(out_csv, index=False)
    print(f"[FIT] wrote {out_csv} (fixed global tnd={tnd:.3f}s) n_pp={len(fit_df)}")
    return fit_df


def fit_eta_theta_from_trials(
    pp: str,
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    tnd_s: float = GLOBAL_TND_S,
) -> Tuple[float, float, float]:
    """
    Fit eta, theta on a provided trial list (which already has human_* fields).
    Returns (eta_hat, theta_hat, best_loss) using the summary loss.
    """

    # stable split like the fitter
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]

    human_train = compute_human_summary_from_preprocessed(train_trials)

    best = {"loss": np.inf, "eta": None, "theta": None}

    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed(
                trials_pp=train_trials,
                dprime_models_pp=dprime_models_pp,
                eta=float(eta),
                theta=float(theta),
                n_repeats=n_repeats_fit,
                tnd_s=tnd_s,
            )
            L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0)
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

    return best["eta"], best["theta"], best["loss"]


def fit_eta_theta_from_trials_fast(pp, trials_pp, dprime_models_pp, eta_grid, theta_grid,
                                    n_repeats_fit=10, tnd_s=GLOBAL_TND_S):
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)

    # Precompute dstep ONCE per training trial -- reused for every grid point below
    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp)
                 for i, tr in enumerate(train_trials)}

    best = {"loss": np.inf, "eta": None, "theta": None}
    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed_fast(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
                tnd_s=tnd_s, kin_cache=kin_cache,
            )
            L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0)
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}
    return best["eta"], best["theta"], best["loss"]


# -----------------------------------------
# Functions for model recovery analysis run 
# -----------------------------------------
def run_model_recovery(
    search_dir: str,
    visibility_dir: str,
    eta_true_grid: np.ndarray,
    theta_true_grid: np.ndarray,
    eta_fit_grid: np.ndarray,
    theta_fit_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_recovery.csv",
) -> pd.DataFrame:

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    # build d' models
    dprime_models = {pp: build_dprime_splines_for_participant(files, spline_s=spline_s)
                     for pp, files in vis_by_pp.items()}

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            continue

        trials_real = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_real) < 10:
            continue

        for eta_true in eta_true_grid:
            for theta_true in theta_true_grid:

                synth_trials = make_synthetic_human_trials(
                    trials_pp=trials_real,
                    dprime_models_pp=dprime_models[pp],
                    eta_true=float(eta_true),
                    theta_true=float(theta_true),
                    tnd_s=GLOBAL_TND_S,
                )

                eta_hat, theta_hat, loss = fit_eta_theta_from_trials_fast( #fit_eta_theta_from_trials(
                    pp=pp,
                    trials_pp=synth_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta_grid=eta_fit_grid,
                    theta_grid=theta_fit_grid,
                    n_repeats_fit=n_repeats_fit,
                    tnd_s=GLOBAL_TND_S,
                )

                rows.append(dict(
                    participant=pp,
                    eta_true=float(eta_true),
                    theta_true=float(theta_true),
                    eta_hat=float(eta_hat),
                    theta_hat=float(theta_hat),
                    train_loss=float(loss),
                    n_trials=len(synth_trials),
                ))

        print(f"[RECOVERY] done {pp}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[RECOVERY] wrote {out_csv} rows={len(df)}")
    return df


def compute_human_summary_from_preprocessed(trials_subset: List[dict]) -> Dict[int, Dict[str, float]]:
    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_resp": [], "ta_resp": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}

    for tr in trials_subset:
        # skip trials with missing behavioral data
        if (tr.get("human_resp") is None) or (tr.get("human_corr") is None) or (tr.get("human_rt") is None):
            continue

        s = tr["speed_px_s"]
        if s not in tmp:
            continue

        tp = int(tr["target_present"]) # 1=present
        resp = int(tr["human_resp"])  # 1=present
        corr = int(tr["human_corr"]) # 1=correct
        rt = float(tr["human_rt"])

        if tp == 1:
            tmp[s]["tp_resp"].append(resp)
            if corr == 1:
                tmp[s]["tp_rt"].append(rt)
        else:
            tmp[s]["ta_resp"].append(resp)
            if corr == 1:
                tmp[s]["ta_rt"].append(rt)

    for s in SPEED_ORDER:
        tp_resp = np.asarray(tmp[s]["tp_resp"], float)
        ta_resp = np.asarray(tmp[s]["ta_resp"], float)
        tp_rt   = np.asarray(tmp[s]["tp_rt"], float)
        ta_rt   = np.asarray(tmp[s]["ta_rt"], float)

        by_speed[s]["hit"]   = float(np.mean(tp_resp)) if tp_resp.size else np.nan
        by_speed[s]["fa"]    = float(np.mean(ta_resp)) if ta_resp.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt))  if tp_rt.size   else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt))  if ta_rt.size   else np.nan

    return by_speed


def simulate_model_summary_from_preprocessed(
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta: float,
    theta: float,
    n_repeats: int = 10,
    tnd_s: float = GLOBAL_TND_S,
    dt_override_s: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:

    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_p": [], "ta_p": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}

    for tr in trials_pp:
        speed_px_s = tr["speed_px_s"]
        if speed_px_s not in tmp:
            continue

        ss = SeedSequence(tr["trial_seed"])
        child = ss.spawn(n_repeats)

        model_resp = np.empty(n_repeats, dtype=int)
        model_rt = np.empty(n_repeats, dtype=float)

        for k in range(n_repeats):
            rng_k = np.random.default_rng(child[k])

            resp, rt = run_replay_trial(
                obj_xy_cells=tr["obj"],
                gaze_xy_cells=tr["gaze_cells"],
                speed_px_s=speed_px_s,
                dprime_splines=dprime_models_pp,
                dt_s=tr["dt_s"],
                eta=eta,
                decision_theta_present=theta,
                target_present=tr["target_present"],
                target_index=tr["target_index"],
                alpha_search=tr["alpha_trial"],
                rng=rng_k,
                dt_override_s=dt_override_s,
            )

            model_resp[k] = resp
            model_rt[k] = rt + tnd_s

        p_present = float(np.mean(model_resp))
        present_mask = (model_resp == 1)
        absent_mask = (model_resp == 0)

        rt_present = float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan
        rt_absent = float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan

        if tr["target_present"] == 1:
            tmp[speed_px_s]["tp_p"].append(p_present)
            if not np.isnan(rt_present):
                tmp[speed_px_s]["tp_rt"].append(rt_present)
        else:
            tmp[speed_px_s]["ta_p"].append(p_present)
            if not np.isnan(rt_absent):
                tmp[speed_px_s]["ta_rt"].append(rt_absent)

    for s in SPEED_ORDER:
        tp_p = np.asarray(tmp[s]["tp_p"], float)
        ta_p = np.asarray(tmp[s]["ta_p"], float)
        tp_rt = np.asarray(tmp[s]["tp_rt"], float)
        ta_rt = np.asarray(tmp[s]["ta_rt"], float)

        by_speed[s]["hit"] = float(np.mean(tp_p)) if tp_p.size else np.nan
        by_speed[s]["fa"] = float(np.mean(ta_p)) if ta_p.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt)) if tp_rt.size else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt)) if ta_rt.size else np.nan

    return by_speed

def run_replay_trial_batch(kin, eta, decision_theta_present, target_present,
                            target_index, n_repeats, rng):
    """
    Simulates n_repeats replicates of ONE trial simultaneously, reusing a
    precomputed dstep. Returns (resp, rt) arrays of shape (n_repeats,).
    Numerically equivalent to calling run_replay_trial n_repeats times,
    but without the Python-level per-timestep, per-repeat loop.
    """
    dstep = kin["dstep"]                       # (T_use, N)
    dt_eff = kin["dt_eff"]
    T_use, N = dstep.shape

    theta_pos = float(decision_theta_present)
    theta_neg = -theta_pos
    noise_scale = np.sqrt(max(1e-6, float(eta)))

    x = rng.normal(0.0, noise_scale, size=(n_repeats, T_use, N))
    if target_present == 1 and target_index is not None:
        x[:, :, target_index] += dstep[np.newaxis, :, target_index]

    dllr = dstep[np.newaxis] * x - 0.5 * (dstep[np.newaxis] ** 2)
    logLR = np.cumsum(dllr, axis=1)             # (n_repeats, T_use, N)
    dv = np.max(logLR, axis=2)                  # (n_repeats, T_use)

    pos_cross = dv >= theta_pos
    neg_cross = dv <= theta_neg
    any_cross = pos_cross | neg_cross
    has_crossing = any_cross.any(axis=1)
    first_idx = np.argmax(any_cross, axis=1)     # 0 if no crossing (masked below)
    pos_at_first = np.take_along_axis(pos_cross, first_idx[:, None], axis=1)[:, 0]

    resp = np.where(has_crossing, pos_at_first.astype(int), (dv[:, -1] > 0).astype(int))
    stop_idx = np.where(has_crossing, first_idx, T_use - 1)
    rt = (stop_idx + 1) * dt_eff

    return resp.astype(int), rt.astype(float)

def precompute_trial_kinematics(tr, dprime_splines, max_time_s=3.5, dt_override_s=None):
    """
    Precomputes dstep (the deterministic per-object, per-timestep sensory
    drift) once per trial. dstep does NOT depend on eta or theta, so this
    result can be reused across the entire (eta, theta) grid search and
    across all repeats -- this is the main cost that was being redundantly
    recomputed before.
    """
    obj_xy_cells = tr["obj"]
    gaze_xy_cells = tr["gaze_cells"]
    speed_px_s = tr["speed_px_s"]
    dt_s = tr["dt_s"]
    alpha_search = tr["alpha_trial"]

    T, N, _ = obj_xy_cells.shape
    if dt_s <= 0:
        dt_s = max_time_s / max(1, T - 1)
    dt_eff = dt_s if dt_override_s is None else float(dt_override_s)
    T_use = min(T, int(np.floor(max_time_s / dt_s)))
    if T_use <= 0:
        return None
    if speed_px_s not in dprime_splines:
        raise ValueError(f"Missing d' spline for speed_px_s={speed_px_s}")
    spl = dprime_splines[speed_px_s]

    dstep = np.empty((T_use, N), dtype=float)
    for t in range(T_use):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            gaze = np.array([GRID_SIZE_X / 2.0, GRID_SIZE_Y / 2.0], dtype=float)
        ecc_deg = ecc_deg_from_cells(obj_xy_cells[t], gaze)
        d400 = np.maximum(alpha_search * spl(ecc_deg), 0.0)
        dstep[t] = 0.4 * d400 * np.sqrt(dt_eff / 0.4)

    return {"dstep": dstep, "dt_eff": dt_eff, "T_use": T_use, "N": N}

def simulate_model_summary_from_preprocessed_fast(
    trials_pp, dprime_models_pp, eta, theta, n_repeats=10,
    tnd_s=GLOBAL_TND_S, dt_override_s=None, kin_cache=None,
):
    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_p": [], "ta_p": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}

    for i, tr in enumerate(trials_pp):
        speed_px_s = tr["speed_px_s"]
        if speed_px_s not in tmp:
            continue
        if kin_cache is not None and i in kin_cache:
            kin = kin_cache[i]
        else:
            kin = precompute_trial_kinematics(tr, dprime_models_pp, dt_override_s=dt_override_s)
            if kin_cache is not None:
                kin_cache[i] = kin
        if kin is None:
            continue

        rng = np.random.default_rng(np.random.SeedSequence(tr["trial_seed"]))
        model_resp, model_rt = run_replay_trial_batch(
            kin=kin, eta=eta, decision_theta_present=theta,
            target_present=tr["target_present"], target_index=tr["target_index"],
            n_repeats=n_repeats, rng=rng,
        )
        model_rt = model_rt + tnd_s

        p_present = float(np.mean(model_resp))
        present_mask, absent_mask = model_resp == 1, model_resp == 0
        rt_present = float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan
        rt_absent = float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan

        bucket = tmp[speed_px_s]
        if tr["target_present"] == 1:
            bucket["tp_p"].append(p_present)
            if not np.isnan(rt_present): bucket["tp_rt"].append(rt_present)
        else:
            bucket["ta_p"].append(p_present)
            if not np.isnan(rt_absent): bucket["ta_rt"].append(rt_absent)

    for s in SPEED_ORDER:
        tp_p, ta_p = np.asarray(tmp[s]["tp_p"]), np.asarray(tmp[s]["ta_p"])
        tp_rt, ta_rt = np.asarray(tmp[s]["tp_rt"]), np.asarray(tmp[s]["ta_rt"])
        by_speed[s]["hit"] = float(np.mean(tp_p)) if tp_p.size else np.nan
        by_speed[s]["fa"] = float(np.mean(ta_p)) if ta_p.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt)) if tp_rt.size else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt)) if ta_rt.size else np.nan
    return by_speed

# for model comparisons
def fit_model_comparison_per_participant(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    eta_fixed: float,           # used when eta is NOT free (theta-only model)
    theta_fixed: float,         # used when theta is NOT free (eta-only model)
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_comparison.csv",
    eta_lo: float = 0.01,
    eta_hi: float = 0.30,
    theta_lo: float = 0.10,
    theta_hi: float = 4.0,
    dt_override_s: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fits three nested models per participant and compares via AIC/BIC:
      M2: eta + theta free  (k=2)
      M_eta: eta free, theta fixed  (k=1)
      M_theta: theta free, eta fixed  (k=1)
    
    eta_fixed / theta_fixed are the held values for the restricted models.
    Typically set these to group means from a prior full fit.
    """
    tnd = float(GLOBAL_TND_S)

    vis_by_pp   = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[COMPARE SKIP] {pp}: no visibility splines")
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            print(f"[COMPARE SKIP] {pp}: too few trials ({len(trials_pp)})")
            continue

        # Identical train split to main fit — same seed logic
        rng_split  = np.random.default_rng(12345 + int(pp[2:]))
        idx        = rng_split.permutation(len(trials_pp))
        n_train    = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials  = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test  = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)   # identical for all three models

        def _run_grid(etas, thetas):
            """Inner grid search; returns (best_eta, best_theta, best_loss)."""
            best = {"loss": np.inf, "eta": etas[0], "theta": thetas[0]}
            for eta in etas:
                for theta in thetas:
                    model = simulate_model_summary_from_preprocessed(
                        trials_pp=train_trials,
                        dprime_models_pp=dprime_models[pp],
                        eta=float(eta),
                        theta=float(theta),
                        n_repeats=n_repeats_fit,
                        tnd_s=tnd,
                        dt_override_s=dt_override_s,
                    )
                    L = loss_summary(human_train, model)
                    if L < best["loss"]:
                        best = {"loss": L, "eta": float(eta), "theta": float(theta)}
            return best["eta"], best["theta"], best["loss"]

        def _test_loss(eta, theta):
            model = simulate_model_summary_from_preprocessed(
                trials_pp=test_trials,
                dprime_models_pp=dprime_models[pp],
                eta=float(eta),
                theta=float(theta),
                n_repeats=n_repeats_fit,
                tnd_s=tnd,
                dt_override_s=dt_override_s,
            )
            return loss_summary(human_test, model)

        # ---- M2: both free (k=2) ----
        e2, t2, L2   = _run_grid(eta_grid, theta_grid)
        aic2, bic2   = information_criteria(L2, N_OBS, k=2)
        tl2          = _test_loss(e2, t2)

        # ---- M_eta: only eta free, theta pinned (k=1) ----
        e_eta, _, L_eta   = _run_grid(eta_grid, [theta_fixed])
        aic_eta, bic_eta  = information_criteria(L_eta, N_OBS, k=1)
        tl_eta            = _test_loss(e_eta, theta_fixed)

        # ---- M_theta: only theta free, eta pinned (k=1) ----
        _, t_th, L_th    = _run_grid([eta_fixed], theta_grid)
        aic_th, bic_th   = information_criteria(L_th, N_OBS, k=1)
        tl_th            = _test_loss(eta_fixed, t_th)

        print(
            f"[COMPARE] {pp} | "
            f"M2: eta={e2:.3f} theta={t2:.3f} AIC={aic2:.2f} | "
            f"M_eta: eta={e_eta:.3f} AIC={aic_eta:.2f} | "
            f"M_theta: theta={t_th:.3f} AIC={aic_th:.2f}"
        )

        rows.append(dict(
            participant=pp,
            # --- M2 ---
            m2_eta=e2,        m2_theta=t2,
            m2_train_loss=L2, m2_test_loss=tl2,
            m2_AIC=aic2,      m2_BIC=bic2,
            # --- M_eta (theta fixed) ---
            meta_eta=e_eta,         meta_theta_fixed=theta_fixed,
            meta_train_loss=L_eta,  meta_test_loss=tl_eta,
            meta_AIC=aic_eta,       meta_BIC=bic_eta,
            # --- M_theta (eta fixed) ---
            mtheta_eta_fixed=eta_fixed, mtheta_theta=t_th,
            mtheta_train_loss=L_th,     mtheta_test_loss=tl_th,
            mtheta_AIC=aic_th,          mtheta_BIC=bic_th,
            # --- ΔAIC (positive = M2 wins) ---
            delta_AIC_vs_eta=aic_eta - aic2,
            delta_AIC_vs_theta=aic_th  - aic2,
            N_obs=N_OBS,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[COMPARE] wrote {out_csv}  n_pp={len(df)}")
    return df

def fit_model_comparison_visibility(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_comparison_visibility.csv",
    null_mode: str = "constant",        # <-- "group" or "constant"
    eta_lo: float = 0.01,
    eta_hi: float = 0.30,
    theta_lo: float = 0.10,
    theta_hi: float = 4.0,
    dt_override_s: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compares two visibility models per participant:
      V_indiv : individual empirical d' splines  (the current model)
      V_group : shared group-mean d' splines     (null — no individual visibility)

    eta and theta are re-fit under each visibility model, so any difference
    in loss/AIC reflects the information carried by individual visibility, not
    just a parameter-count difference.
    """

    tnd = float(GLOBAL_TND_S)
    vis_by_pp    = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    # --- build individual splines ---
    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    # --- build null model depending on mode ---
    null_models: Dict[str, Dict[int, callable]] = {}

    if null_mode == "group":
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        print(f"[VIS COMPARE] null mode: group-mean splines for speeds: {sorted(group_splines.keys())}")
        for pp in vis_by_pp.keys():
            null_models[pp] = group_splines

    elif null_mode == "constant":
        #print(f"[VIS COMPARE] null mode: constant d' (no eccentricity dependence)")
        for pp, files in vis_by_pp.items():
            base = build_dprime_splines_for_participant(files, spline_s=spline_s)
            null_models[pp] = make_visibility_null_model(
                base,
                mode="constant_from_speed0",
                constant_rule="mean",
            )

    else:
        raise ValueError(f"Unknown null_mode: {null_mode!r}. Use 'group' or 'constant'.")

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            continue

        # same stable split as main fit
        rng_split  = np.random.default_rng(12345 + int(pp[2:]))
        idx        = rng_split.permutation(len(trials_pp))
        n_train    = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials  = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test  = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)
        
        def _grid(dprime_models, trials):
            best = {"loss": np.inf, "eta": eta_grid[0], "theta": theta_grid[0]}
            
            # coarse pass
            for eta in eta_grid:
                for theta in theta_grid:
                    model = simulate_model_summary_from_preprocessed(
                        trials_pp=trials,
                        dprime_models_pp=dprime_models,
                        eta=float(eta), theta=float(theta),
                        n_repeats=n_repeats_fit, tnd_s=tnd,
                        dt_override_s=dt_override_s,
                    )
                    L = loss_summary(human_train, model)
                    if L < best["loss"]:
                        best = {"loss": L, "eta": float(eta), "theta": float(theta)}
        
            # refinement pass around coarse winner
            eta_ref   = make_refined_grid(best["eta"],   0.15, 12, lo=eta_lo,   hi=eta_hi)
            theta_ref = make_refined_grid(best["theta"], 1.0,  12, lo=theta_lo, hi=theta_hi)
        
            for eta in eta_ref:
                for theta in theta_ref:
                    model = simulate_model_summary_from_preprocessed(
                        trials_pp=trials,
                        dprime_models_pp=dprime_models,
                        eta=float(eta), theta=float(theta),
                        n_repeats=n_repeats_fit, tnd_s=tnd,
                        dt_override_s=dt_override_s,
                    )
                    L = loss_summary(human_train, model)
                    if L < best["loss"]:
                        best = {"loss": L, "eta": float(eta), "theta": float(theta)}
        
            return best
       
        # V_indiv — refit eta/theta with individual visibility
        b_indiv = _grid(dprime_models[pp], train_trials)
        aic_indiv, bic_indiv = information_criteria(b_indiv["loss"], N_OBS, k=2)
        model_test_indiv = simulate_model_summary_from_preprocessed(
            trials_pp=test_trials, dprime_models_pp=dprime_models[pp],
            eta=b_indiv["eta"], theta=b_indiv["theta"],
            n_repeats=n_repeats_fit, tnd_s=tnd, dt_override_s=dt_override_s,
        )
        tl_indiv = loss_summary(human_test, model_test_indiv)

        # V_null — refit eta/theta with null visibility
        b_null = _grid(null_models[pp], train_trials)
        aic_null, bic_null = information_criteria(b_null["loss"], N_OBS, k=2)
        model_test_null = simulate_model_summary_from_preprocessed(
            trials_pp=test_trials, dprime_models_pp=null_models[pp],
            eta=b_null["eta"], theta=b_null["theta"],
            n_repeats=n_repeats_fit, tnd_s=tnd, dt_override_s=dt_override_s,
        )
        tl_null = loss_summary(human_test, model_test_null)

        print(
            f"[VIS COMPARE] {pp} | "
            f"V_indiv: eta={b_indiv['eta']:.3f} theta={b_indiv['theta']:.3f} AIC={aic_indiv:.2f} | "
            f"V_null: eta={b_null['eta']:.3f} theta={b_null['theta']:.3f} AIC={aic_null:.2f} | "
            f"ΔAIC={aic_null - aic_indiv:.2f}"
        )

        rows.append(dict(
            participant=pp,
            indiv_eta=b_indiv["eta"],         indiv_theta=b_indiv["theta"],
            indiv_train_loss=b_indiv["loss"], indiv_test_loss=tl_indiv,
            indiv_AIC=aic_indiv,              indiv_BIC=bic_indiv,
            null_eta=b_null["eta"],           null_theta=b_null["theta"],
            null_train_loss=b_null["loss"],   null_test_loss=tl_null,
            null_AIC=aic_null,                null_BIC=bic_null,
            delta_AIC=aic_null - aic_indiv,
            N_obs=N_OBS,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[VIS COMPARE] wrote {out_csv}  n_pp={len(df)}")
    return df


# =========================
# Test-only replay: regenerates model predictions using ONLY the
# held-out trials from each participant's train/test split
# =========================
def get_test_trial_seeds(pp, pairs, dprime_models_pp, spline_s=0.5, split_seed_offset=12345):
    """
    Reproduces the same 70/30 train/test split used in fit_model_per_participant
    for this participant, and returns the trial_seed values belonging to the
    held-out (test) trials. split_seed_offset must match whatever offset was
    used when fitting the parameters, or the split will disagree.
    """
    trials_pp = preprocess_participant_trials(pp, pairs, dprime_models_pp)
    rng_split = np.random.default_rng(split_seed_offset + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    test_idx = idx[n_train:]
    return set(trials_pp[i]["trial_seed"] for i in test_idx)


def run_full_replay_test_only(
    search_dir: str,
    visibility_dir: str,
    output_csv: str = "replay_model_results_test_only.csv",
    n_model_repeats: int = 50,
    spline_s: float = 0.5,
    tnd_s: float = GLOBAL_TND_S,
    max_calib_drift_deg: Optional[float] = None,
    fitted_params_csv: Optional[str] = None,
    dt_override_s: Optional[float] = None,
    debug: bool = False,
    use_group_visibility: bool = False,
    split_seed_offset: int = 12345,
) -> pd.DataFrame:

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    if fitted_params_csv is None:
        raise ValueError("fitted_params_csv must be provided to run per-participant fitted replay.")

    fit_df = pd.read_csv(fitted_params_csv)
    if "tnd" in fit_df.columns:
        tnd_s = float(np.nanmedian(fit_df["tnd"].values))
        print(f"[REPLAY-TEST] Using global tnd_s from fitted_params_csv: {tnd_s:.3f}s")

    fit_map = {
        str(r["participant"]).strip(): (float(r["eta"]), float(r["theta"]))
        for _, r in fit_df.iterrows()
    }

    # dprime models used for prediction (can optionally use group-mean visibility)
    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    if use_group_visibility:
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        for pp in vis_by_pp.keys():
            dprime_models[pp] = group_splines
    else:
        for pp, files in vis_by_pp.items():
            base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
            dprime_models[pp] = make_visibility_null_model(
                base_model, mode=VISIBILITY_MODE, source_speed=VIS_NULL_SOURCE_SPEED,
                constant_rule=VIS_CONSTANT_RULE, ref_ecc_deg=VIS_REF_ECC_DEG,
            )

    # dprime models used SPECIFICALLY to reproduce the fitting-time split
    # -- must match fit_model_per_participant's preprocessing exactly,
    # regardless of use_group_visibility above, since that's what
    # determines the train/test assignment during fitting.
    split_dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
        split_dprime_models[pp] = make_visibility_null_model(
            base_model, mode=VISIBILITY_MODE, source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE, ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    all_rows = []
    #printed_ecc_debug = set()
    n_test_total, n_train_skipped = 0, 0

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[SKIP] {pp}: no visibility files found")
            continue
        if pp not in fit_map:
            raise ValueError(f"No fitted parameters found for participant {pp}")

        eta_pp, theta_pp = fit_map[pp]

        test_seeds = get_test_trial_seeds(
            pp, pairs, split_dprime_models[pp],
            spline_s=spline_s, split_seed_offset=split_seed_offset,
        )
        print(f"[REPLAY-TEST] {pp}: {len(test_seeds)} held-out trials identified")

        for csv_path, asc_path in sorted(pairs):
            df = pd.read_csv(csv_path)

            if max_calib_drift_deg is not None and "CalibrationDrift(deg)" in df.columns:
                df = df[df["CalibrationDrift(deg)"] <= max_calib_drift_deg].copy()

            required = ["Trial", "Gabor Positions", "Target Present", "Response", "Correct", "Reaction Time (s)"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Missing column '{c}' in {csv_path}")

            dt = dt_from_name(csv_path)
            if dt is None:
                raise ValueError(f"Cannot infer dt# from filename: {csv_path}")

            speed_px_s = DT_TO_SPEED[dt]
            alpha_use = 0.5 * (1 - 0.5 * (speed_px_s / 400))
            raw_speed = float(df["Speed (px/s)"].iloc[0]) if "Speed (px/s)" in df.columns else float(speed_px_s)

            if speed_px_s not in dprime_models[pp]:
                print(f"[SKIP] {pp} speed {speed_px_s}: no visibility spline")
                continue

            (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)
            df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

            skipped, kept, skipped_train = 0, 0, 0

            for _, r in df.iterrows():
                trial = safe_int(r.get("Trial"))
                target_present = safe_int(r.get("Target Present"))
                human_resp = safe_int(r.get("Response"))
                human_corr = safe_int(r.get("Correct"))
                human_rt = safe_float(r.get("Reaction Time (s)"))

                if trial is None or target_present is None or human_resp is None or human_corr is None or human_rt is None:
                    skipped += 1
                    continue

                trial_seed = stable_trial_seed(pp, os.path.basename(csv_path), int(trial))

                # -- the key addition: skip any trial not in this participant's held-out set --
                if trial_seed not in test_seeds:
                    skipped_train += 1
                    continue

                obj = r.get("gabor_pos")
                if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3:
                    skipped += 1
                    continue

                Tstim = obj.shape[0]
                if Tstim < 2:
                    skipped += 1
                    continue

                dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
                if dt_s is None or dt_s <= 0:
                    dt_s = dt_from_duration(obj, 3.5)

                tr_eye = eye_trials.get(trial)
                if tr_eye is None:
                    gaze_cells = np.full((Tstim, 2), np.nan, dtype=float)
                else:
                    gaze_cells = gaze_series_cells_from_fixations(tr_eye, Tstim, dt_s, screen_w, screen_h)

                rng_trial = np.random.default_rng(trial_seed)
                gaze_cells_used = perturb_gaze(
                    gaze_cells, mode=GAZE_MODE, rng=rng_trial,
                    dt_s=dt_override_s if dt_override_s is not None else dt_s,
                )

                target_index = None
                if target_present == 1:
                    traj = safe_parse_trajectory(r.get("Target Trajectory"))
                    if traj is not None:
                        target_index = infer_target_index(obj, traj)

                min_target_ecc_real = min_target_eccentricity(obj, gaze_cells, target_index)
                min_target_ecc_used = min_target_eccentricity(obj, gaze_cells_used, target_index)

                model_resp = np.empty(n_model_repeats, dtype=int)
                model_rt = np.empty(n_model_repeats, dtype=float)

                ss = np.random.SeedSequence(trial_seed)
                child_seeds = ss.spawn(n_model_repeats)

                for k in range(n_model_repeats):
                    rng_k = np.random.default_rng(child_seeds[k])
                    resp, rt = run_replay_trial(
                        obj_xy_cells=obj, gaze_xy_cells=gaze_cells_used, speed_px_s=speed_px_s,
                        dprime_splines=dprime_models[pp], dt_s=dt_s, eta=eta_pp,
                        decision_theta_present=float(theta_pp), target_present=target_present,
                        target_index=target_index, alpha_search=alpha_use, rng=rng_k,
                        dt_override_s=dt_override_s,
                    )
                    model_resp[k] = resp
                    model_rt[k] = rt + float(tnd_s)

                present_mask = (model_resp == 1)
                absent_mask = (model_resp == 0)

                row = {
                    "Task Type": r.get("Task Type", "search"),
                    "Participant ID": r.get("Participant ID", pp),
                    "participant": pp,
                    "speed_px_s_raw": raw_speed,
                    "speed_px_s_used": speed_px_s,
                    "trial": trial,
                    "split": "test",
                    "dt_s_est": float(dt_s),
                    "human_target_present": target_present,
                    "human_response": human_resp,
                    "human_correct": human_corr,
                    "human_rt_s": float(human_rt),
                    "model_p_present": float(model_resp.mean()),
                    "model_rt_mean_s": float(model_rt.mean()),
                    "model_rt_median_s": float(np.median(model_rt)),
                    "model_rt_present_mean_s": float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan,
                    "model_rt_absent_mean_s": float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan,
                    "alpha_used": alpha_use,
                    "theta_present_used": float(theta_pp),
                    "eta_used": float(eta_pp),
                    "theta_fitted": float(theta_pp),
                    "tnd_used": float(tnd_s),
                    "min_target_ecc_deg_real": float(min_target_ecc_real),
                    "min_target_ecc_deg_used": float(min_target_ecc_used),
                    "search_csv": os.path.basename(csv_path),
                    "asc_file": os.path.basename(asc_path),
                }
                for extra in ["FixOnTargetTime(s)", "LastFixIndex", "CalibrationDrift(deg)"]:
                    if extra in r:
                        row[extra] = r[extra]

                all_rows.append(row)
                kept += 1

            n_test_total += kept
            n_train_skipped += skipped_train
            print(f"[OK] {pp} speed={speed_px_s}: {os.path.basename(csv_path)} "
                  f"kept(test)={kept} skipped(train)={skipped_train} skipped(invalid)={skipped}")

    out = pd.DataFrame(all_rows)
    out.to_csv(output_csv, index=False)
    print(f"\n[REPLAY-TEST] Wrote {output_csv} rows={len(out)} "
          f"(test trials kept={n_test_total}, train trials excluded={n_train_skipped})")
    return out
