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
1. [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
2. [DifferentailEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
3. [Plots.jl](https://docs.juliaplots.org/stable/)
4. [MATLAB example used for the ODE system](https://www3.nd.edu/~nancy/Math20750/Demos/3dplots/dim3system.html)
