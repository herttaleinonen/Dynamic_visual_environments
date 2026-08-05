# The Dynamic Foveated Search Model 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/replay_model/images/dfsm.png" height="548" width="548" ></a>

$$
e_{i,t} = \lVert x_{i,t} - g_t \rVert
$$

$$
S(e,v)=\Phi^{-1}(H(e,v))-\Phi^{-1}(F(e,v))
$$

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
D_t \ge \Theta_{+}
$$

$$
D_t \le \Theta_{-}
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

