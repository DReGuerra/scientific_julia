# scientific_julia
Exploring julia packages for data science and other numerical modeling applications.

André Guerra \
April, 2023 \
andre.guerra@mail.mcgill.ca  

---
Description: \
This repository contains a series of scripts that explore various julia packages that may be used for data science and developing numerical solutions to physical models. We start with a simple system of 3 ODEs solved using symbolics packages in `ode_sys_sym.jl`, then we introduce a parameter to the system of ODEs and solve it for a range of parameter values in `ode_sys_num_param.jl`.

---
## Core Contents
1. `ode_sys_sym.jl` $\rightarrow$ solve a system of 3 ODEs with symbolic packages
2. `ode_sys_num_param.jl` $\rightarrow$ solve a system of 3 ODEs with one parameter using numerical definition
3. `ode_sys_num_param_opt.jl` $\rightarrow$ solve for an optimal value of the parameter from (2) using an objective function

## References
1. [MATLAB example used for the ODE system](https://www3.nd.edu/~nancy/Math20750/Demos/3dplots/dim3system.html)
2. [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
3. [DifferentailEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
4. [Plots.jl](https://docs.juliaplots.org/stable/)

---
Below are quick descriptions of the main files in this repo and the problem statements to be solved. More detail is found as comments in the respective scripts files.

## `ode_sys_sym.jl`

The naming convention for this script indicates that we are solving an ODE (`ode`) system (`sys`) using symbolics (`sym`) packages in Julia. 

Reproduce the MATALB example of a simple system of ODEs[1] in Julia using [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) and [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

### Problem Statement
Consider the nonlinear system:<br>
$x' = -x + 3z$<br>
$y' = -y + 2z$<br>
$z' = x^2 - 2z$<br>
<br>
Initial conditions:<br>
$x(t=0) = 0$<br>
$y(t=0) = 1/2$<br>
$z(t=0) = 3$<br>

## `ode_sys_num_param.jl`

In this script, we represent the ODE system numerically (not using the symbolics packages).

### Problem Statement
Consider the nonlinear system (the same as above but with a parameter `p`):<br>
$x' = -x + pz$<br>
$y' = -y + 2z$<br>
$z' = x^2 - 2z$<br>
<br>
Initial conditions:<br>
$x(t=0) = 0$<br>
$y(t=0) = 1/2$<br>
$z(t=0) = 3$<br>
<br>
In this script, we solve the system with a parameter space `p` = [0,1,2].

## `ode_sys_num_opt_param.jl`

In this script we optimize the parameter `p` to find the value that satisfies the data available. In this script we also produce this data given a value of `p`=1.

### Problem Statement
The problem statement here is the same as in `ode_sys_num_param.jl` above. However, here we use an L2 loss function as the objective cost to be optimized in the search for the value of `p`.