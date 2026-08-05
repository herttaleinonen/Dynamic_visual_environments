# The Dynamic Foveated Search Model 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/replay_model/images/dfsm.png" height="548" width="548" ></a>
Visibility calibration 
Each search trial was approximated as a sequence of discrete time steps (Δt = 50 ms), even when the underlying evidence accumulation can be considered as continuous. For each object i at time step t, retinal eccentricity was computed as:

█(e_(i,t)=∥x_(i,t)-g_t∥,#(1) )

where x_(i,t)  and g_(t  )represent the object and gaze positions in 2D coordinates. Orientation sensitivity was based on the visibility task data:

█(S(e,v)=Φ^(-1) (H(e,v))-Φ^(-1) (F(e,v)),#(2) )

where H(e,v)  and F(e,v)  are hit and false alarm rates, e standing for retinal eccentricity and v for object velocity, and Φ^(-1  )is the inverse cumulative normal distribution. The resulting values were interpolated across eccentricity to obtain a continuous sensitivity function S(e,v)   for each velocity condition.

Foveated sensory signal
At each time step t, each object i  produces a noisy internal sensory sample z_(i,t)  which was modelled as a Gaussian random variable:

█(z_(i,t)∼{■(N" " (S_step (e_(i,t),v)," " η),&"if object is the target" ,@N(0," " η),&"  if object is a distractor" ,)┤#(3) )

where η is the sensory noise variance. Thus, distractors generate noise-only activity, whereas targets generate signal-plus-noise activity. Per step sensitivity was derived from the calibrated S(e,v), measured over a 400 ms observation window (duration of the visibility task), by assuming that discriminability grows with the square root of integration time:

█(S_step (e_(i,t),v)=α" " S(e_(i,t),v)" " √(Δt/0.4),#(4) )

where α is a fixed global gain parameter that maps sensitivity measured in the visibility task to the effective sensory signal during search, absorbing unmodeled factors such as crowding (Whitney & Levi, 2011).
Evidence accumulation
Momentary sensory samples were converted into object wise log likelihood ratio (LLR) increments:
█(■(&ΔLLR_(i,t)=log⁡〖(p(z_(i,t)∣"target" ))/(p(z_(i,t)∣"distractor" ) )〗,&&)#(5) )
where p(z∣ ⋅) represents the probability density of the sensory observation under the specified hypothesis. Under the assumption of Gaussian sensory noise with unit variance simplifies to
█(■(&ΔLLR_(i,t)=S_step (e_(i,t),v)" " z_(i,t)-1/2 " " S_step (e_(i,t),v)^2.&&)#(6) )
Object specific evidence was accumulated over time:
█(■(&LLR_(i,t)=LLR_(i,t-1)+ΔLLR_(i,t)=∑_(τ=1)^t▒Δ LLR_(i,τ).&&)#(7) )

Pooling across objects
To infer whether a target was present anywhere in the display, the model used the maximum object-wise accumulated evidence as the decision variable:

█(D_t=(max )┬i  LLR_(i,t).#(8) )

This operation selects the object with the strongest accumulated evidence at each time step, effectively implementing a winner-take-all decision rule across objects.

Decision rule and response time
At each time step t, the decision variable D_t  was compared against two fixed decision bounds. A positive upper bound Θ_P  governed target-present responses: if D_t≥Θ_P, the model responded, “target present”, and reaction time was recorded as (t+1)Δt. Target-absent responses were governed by a fixed negative lower bound Θ_A: if D_t≤Θ_A, the model responded, “target absent”, and reaction time was recorded as (t+1)Δt. If no bound was crossed by stimulus offset (3.5 s), the model responded “target present” if the final decision variable was positive, and “target absent” if negative.

Parameter fitting and trial simulations
For each participant, the parameters η (sensory noise) and Θ (decision criterion) were estimated via grid search by minimizing a composite loss function:

█(L=∑_v▒〖├ [(〖d^'〗_(h,v)-〖d^'〗_(m,v) )^2+(〖log⁡RT〗_(h,v)^TP-〖log⁡RT〗_(m,v)^TP )^2+(〖log⁡RT〗_(h,v)^TA-〖log⁡RT〗_(m,v)^TA )^2 ┤],〗#(9) )

where the subscripts h and m denote human and model respectively, and the sum runs over velocity conditions v. Target-present TP, and target-absent TA reaction times are compared on a log scale to treat proportional differences equivalently across conditions. Trial-wise stimulus and gaze time series were precomputed once. Parameter estimation proceeded in two phases: η was searched over [0.002, 0.20] (12 points) and Θ over [0.02, 1.0] (12 points), with a refined search of ±0.15 and ±1.0 respectively around the coarse optimum. 


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

