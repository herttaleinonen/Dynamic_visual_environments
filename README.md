# The Dynamic Foveated Search Model 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/replay_model/images/dfsm.png" height="548" width="548" ></a>

Visibility calibration

Each search trial was approximated as a sequence of discrete time steps ($\Delta t = 50$ ms), even when the underlying evidence accumulation can be considered as continuous. For each object $i$ at time step $t$, retinal eccentricity was computed as

$$
e_{i,t} = \|x_{i,t} - g_t\|,
\tag{1}
$$

where $x_{i,t}$ and $g_t$ represent the object and gaze positions in 2D coordinates.

Orientation sensitivity was based on the visibility task data:

$$
S(e,v) = \Phi^{-1}\!\bigl(H(e,v)\bigr) - \Phi^{-1}\!\bigl(F(e,v)\bigr),
\tag{2}
$$

where $H(e,v)$ and $F(e,v)$ are hit and false-alarm rates, $e$ stands for retinal eccentricity, $v$ for object velocity, and $\Phi^{-1}$ is the inverse cumulative normal distribution. The resulting values were interpolated across eccentricity to obtain a continuous sensitivity function $S(e,v)$ for each velocity condition.

Foveated sensory signal

At each time step $t$, each object $i$ produces a noisy internal sensory sample $z_{i,t}$ which was modelled as a Gaussian random variable:

$$
z_{i,t} \sim
\begin{cases}
\mathcal N\!\left(S^*(e_{i,t},v), \eta\right), & \text{if object is the target},\\
\mathcal N(0,\eta), & \text{if object is a distractor}.
\end{cases}
\tag{3}
$$

where $\eta$ is the sensory noise variance. Thus, distractors generate noise-only activity, whereas targets generate signal-plus-noise activity.

Per-step sensitivity was derived from the calibrated $S(e,v)$, measured over a 400 ms observation window (duration of the visibility task), by assuming that discriminability grows with the square root of integration time:

$$
S^*(e_{i,t},v)
=
\alpha \, S(e_{i,t},v)
\sqrt{\frac{\Delta t}{0.4}},
\tag{4}
$$

where $\alpha$ is a fixed global gain parameter that maps sensitivity measured in the visibility task to the effective sensory signal during search, absorbing unmodeled factors such as crowding (Whitney & Levi, 2011).

Evidence accumulation

Momentary sensory samples were converted into object-wise log-likelihood ratio (LLR) increments:

$$
\Delta \mathrm{LLR}_{i,t}
=
\log
\frac{
p(z_{i,t}\mid \text{target})
}{
p(z_{i,t}\mid \text{distractor})
},
\tag{5}
$$

where $p(z\mid\cdot)$ represents the probability density of the sensory observation under the specified hypothesis.

Under the assumption of Gaussian sensory noise with unit variance, this simplifies to

$$
\Delta \mathrm{LLR}_{i,t}
=
S^*(e_{i,t},v)\,z_{i,t}
-
\frac{1}{2}S^*(e_{i,t},v)^2.
\tag{6}
$$

Object-specific evidence was accumulated over time:

$$
\mathrm{LLR}_{i,t}
=
\mathrm{LLR}_{i,t-1}
+
\Delta \mathrm{LLR}_{i,t}
=
\sum_{\tau=1}^{t}
\Delta \mathrm{LLR}_{i,\tau}.
\tag{7}
$$

Pooling across objects

To infer whether a target was present anywhere in the display, the model used the maximum object-wise accumulated evidence as the decision variable:

$$
D_t
=
\max_i \mathrm{LLR}_{i,t}.
\tag{8}
$$

This operation selects the object with the strongest accumulated evidence at each time step, effectively implementing a winner-take-all decision rule across objects.

Decision rule and response time

At each time step $t$, the decision variable $D_t$ was compared against two fixed decision bounds.

A positive upper bound $\Theta_{+}$ governed target-present responses:

$$
D_t \ge \Theta_{+}.
$$

If this condition was met, the model responded **target present**, and reaction time was recorded as

$$
RT=(t+1)\Delta t.
$$

Target-absent responses were governed by a fixed negative lower bound $\Theta_{-}$:

$$
D_t \le \Theta_{-}.
$$

If this condition was met, the model responded **target absent**, and reaction time was recorded as

$$
RT=(t+1)\Delta t.
$$

If no bound was crossed by stimulus offset (3.5 s), the model responded **target present** if the final decision variable was positive, and **target absent** if negative.

Parameter fitting and trial simulations

For each participant, the parameters $\eta$ (sensory noise) and $\Theta$ (decision criterion) were estimated via grid search by minimizing a composite loss function:

$$
L
=
\sum_v
\Big[
(d'_{h,v}-d'_{m,v})^2
+
(\log RT^{TP}_{h,v}-\log RT^{TP}_{m,v})^2
+
(\log RT^{TA}_{h,v}-\log RT^{TA}_{m,v})^2
\Big].
\tag{9}
$$

where the subscripts $h$ and $m$ denote human and model respectively, and the sum runs over velocity conditions $v$.

Target-present ($TP$) and target-absent ($TA$) reaction times are compared on a log scale to treat proportional differences equivalently across conditions. Trial-wise stimulus and gaze time series were precomputed once. Parameter estimation proceeded in two phases: $\eta$ was searched over $[0.002,\,0.20]$ (12 points) and $\Theta$ over $[0.02,\,1.0]$ (12 points), with a refined search of $\pm0.15$ and $\pm1.0$ respectively around the coarse optimum.


# Requirements
Python 3.10 or newer. <br/>

# Installation
Clone the `replay_model` -folder to your machine.

# Running 

To run the model, run 
```
python3 main.py
```
By default, this command runs parameter fitting, model simulations and saccade predictions.

To recreate model vs. human figures and analyses, run
```
python3 figures.py
```

