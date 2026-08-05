# The Dynamic Foveated Search Model 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/replay_model/images/dfsm.png" height="548" width="548" ></a>


Each search trial was approximated as a sequence of discrete time steps (Δ𝑡 = 50 ms). For each object
𝑖 at time step 𝑡, retinal eccentricity was computed as: 

$$
e_{i,t} = \lVert x_{i,t} - g_t \rVert
$$

where $x_{i,t}$ and $g_t$ are the object and gaze positions in 2D coordinates. Orientation sensitivity was based on the visibility task data:

$$
S(e,v)=\Phi^{-1}(H(e,v))-\Phi^{-1}(F(e,v))
$$

where 𝐻(𝑒, 𝑣) and 𝐹(𝑒, 𝑣) are hit and false-alarm rates, 𝑒 standing for retinal eccentricity and
𝑣 for object velocity, and $\Phi^{-1}$ is the inverse cumulative normal distribution. The resulting
values were interpolated across eccentricity to obtain a continuous sensitivity function
𝑆(𝑒, 𝑣) for each velocity condition. At each time step 𝑡, each object 𝑖 produces a noisy internal sensory sample $z_{i,t}$ which was
modelled as a Gaussian random variable:

$$
z_{i,t}\sim
\begin{cases}
\mathcal{N}(0,\eta), & \text{if the object is a distractor},\\
\mathcal{N}(S_{\Delta t}(e_{i,t},v),\eta), & \text{if the object is the target}.
\end{cases}
$$


$$
S_{\Delta t}(e_{i,t},v)=
\alpha\,S(e_{i,t},v)\sqrt{\frac{\Delta t}{0.4}}
$$

$$
\Delta\mathrm{LLR}_{i,t}=
\log
\frac{p(z_{i,t}\mid\text{target})}
{p(z_{i,t}\mid\text{distractor})}
$$

$$
\Delta\mathrm{LLR}_{i,t}=S_{\Delta t}(e_{i,t},v)z_{i,t}-\frac{1}{2}S_{\Delta t}(e_{i,t},v)^2
$$

$$
\mathrm{LLR}_{i,t}=\mathrm{LLR}_{i,t-1}+\Delta\mathrm{LLR}_{i,t}=\sum_{\tau=1}^{t}\Delta\mathrm{LLR}_{i,\tau}
$$

$$
D_t=\max_i \mathrm{LLR}_{i,t}
$$

$$
D_t \ge \Theta_{p}
$$

$$
D_t \le \Theta_{A}
$$

$$
L=\sum_v\left[(d_h-d_m)^2+(\log RT_h^{TP}-\log RT_m^{TP})^2+(\log RT_h^{TA}-\log RT_m^{TA})^2\right]
$$


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

