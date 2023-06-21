# Jennifer Tram Su
# .jl file expansion on `ode_sys_sym_param_pinn.ipynb` to optimize multiple parameters

## Packages
using Pkg
Pkg.activate("../../PINN")

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Plots, DifferentialEquations, Zygote
import ModelingToolkit: Interval, infimum, supremum

## Defining Helper Functions 
function getData(sol, time)
    data = []
    us = hcat(sol(time).u...)     # hcat concatenates along dimension 2 (columns)
    ts_ = hcat(sol(time).t...)    # `...` "splats" the values of a container into arguments of a function
                                # --> each row vector turned into column vector
    return [us, ts_]
end

## Defining ODE system
@parameters t, a, b, c, d, e, f
@variables x(..), y(..), z(..)
Dt = Differential(t)

eqs = [Dt(x(t)) ~ a*x(t) + b*z(t),
    Dt(y(t)) ~ c*y(t) + d*z(t),
    Dt(z(t)) ~ e*x(t)^2 + f*z(t)]

bcs = [x(0) ~ 1, y(0) ~ 1/2, z(0) ~ 3]
domains = [t ∈ Interval(0.0, 5.0)]
dt = 0.01 

## Defining the Neural Network
input_ = length(domains)
n = 24

chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dx/dt
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dy/dt
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dz/dt

## Generating Training Data
function sys!(du, u, p, t)          
    du[1] = p[1]*u[1] + p[2]*u[3]
    du[2] = p[3]*u[2] + p[4]*u[3]
    du[3] = p[5]*u[1]^2 + p[6]*u[3]
end

u0 = [1; 1/2; 3]
tspan = (0.0, 5.0)
ps = [-1, 1, -1, 2, 1, -2]

prob = ODEProblem(sys!, u0, tspan, ps)
sol = solve(prob, Tsit5(), dt = 0.1)

ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]  # domains produces a vector
                                                                    # d.domain extracts the interval
                                                                    # infimum returns the greatest lower bound (0.0)
                                                                    # supremum returns the least upper bound (1.5)

data = getData(sol, ts)

(u_, t_) = data
len = length(data[2])

plot(t_', u_[1, :], linewidth = 2, label = "x", title = "p=$ps", xlabel = "t")
plot!(t_', u_[2, :], label = "y")
plot!(t_', u_[3, :], label = "z")
savefig("ode_sys_sym_opt_param_pinn.png")

## Creating Additional Loss Function
depvars = [:x, :y, :z] # : is the symbol operator, here we specify the dependent variables using the Symbol data type

function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, θ[depvars[i]]) .- u_[[i], :]) / len for i in 1:1:3)
end

## Defining PINN Interface
discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3], 
                                                NeuralPDE.GridTraining(dt), param_estim = true, 
                                                additional_loss = additional_loss)

@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)], [a, b, c, d, e, f], defaults = Dict([p .=> 0.1 for p in [a, b, c, d, e, f]]))

prob_ = NeuralPDE.discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

# Store callback frames in animation
a = Animation()

callback = function(p, l)
    # Print losses
    # println("loss: ", l)
    # println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    # println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    # println()

    # # Visualizing training
    weights = [p.depvar.x, p.depvar.y, p.depvar.z]
    state = [[discretization.phi[i]([t], weights[i])[1] for t in ts] for i in 1:3] 
    fig = plot(ts, state, xlabel = "t", label = ["x(t)" "y(t)" "z(t)"], xlims=(0, 5.0), ylims=(0, 3), legend=:outertopright)
    frame(a, fig)

    return false
end

res = Optimization.solve(prob_, BFGS(); callback = callback, maxiters = 5000)
p_ = res.u[end-5:end] # Last layer is optimized parameter

gif(a, "ode_sys_sym_opt_param_pinn_fitting.gif")
println(p_)

## Visualizing the PINN Prediction
minimizers = [res.u.depvar[depvars[i]] for i in 1:3]                                    # retrieving NN representation (weights and biases) for each dependent variable
ts = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domains][1]               # time domain
u_predict = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts] for i in 1:3] 
plot(sol, title = "Predicted vs Actual Sol")
plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])
savefig("ode_sys_sym_opt_param_pinn_pred.png")

## Extrapolation
# Remaking problem on tspan (0.0, 10.0)
newts = (0.0, 10.0)
newprob = remake(prob, tspan=newts)
newsol = solve(newprob, Tsit5(), dt = 0.1)

# Generating predictions for new tspan
domain_test = [t ∈ Interval(0, 10.0)]
ts_test = [infimum(d.domain):(dt / 10):supremum(d.domain) for d in domain_test][1]

# PINN Prediction
u_predict_test = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts_test] for i in 1:3]

plot(newsol, title = "Predicted vs Actual Sol", xlabel="t", label = ["x(t)" "y(t)" "z(t"])
plot!(ts_test, u_predict_test, label = ["x_pred" "y_pred" "z_pred"])
savefig("ode_sys_sym_opt_param_pinn_extrap.png")

# Extracting data from newsol
data = getData(newsol, ts_test)
newsol_, newts_ = data

# Comparing x(t)
plot(newts_', newsol_[1, :], linewidth = 2, label = "x_act", title = "p=$ps", xlabel = "t")
plot!(ts_test, u_predict_test[1, :], label = "x_pred")
savefig("ode_sys_sym_opt_param_pinn_x_extrap.png")

# Comparing y(t)
plot(newts_', newsol_[2, :], linewidth = 2, label = "y_act", title = "p=$ps", xlabel = "t")
plot!(ts_test, u_predict_test[2, :], label = "y_pred")
savefig("ode_sys_sym_opt_param_pinn_y_extrap.png")

# Comparing z(t)
plot(newts_', newsol_[3, :], linewidth = 2, label = "z_act", title = "p=$ps", xlabel = "t")
plot!(ts_test, u_predict_test[3, :], label = "z_pred")
savefig("ode_sys_sym_opt_param_pinn_z_extrap.png")

## Error Analysis
u_pred = hcat(u_predict_test...)'
error = broadcast(abs, (u_pred - newsol_))

plot(ts_test, error[1, :], linewidth = 2, label = "x", title = "Error for p=$ps", xlabel = "t")
plot!(ts_test, error[2, :], label = "y")
plot!(ts_test, error[3, :], label = "z")
savefig("ode_sys_sym_opt_param_pinn_error.png")

# Remaking problem with predicted parameter
prob_estim = remake(newprob, p=p_)
sol_estim = solve(newprob, Tsit5(), dt = 0.1)

plot(sol_estim, linewidth = 2, label = ["x_param" "y_param" "z_param"], title = "Plotting with Predicted Parameter p=$ps", xlabel = "t")
plot!(newsol, linewidth = 2, label = ["x(t)" "y(t)" "z(t)"], title = "Plotting with Predicted Parameter p=$ps", xlabel = "t")
savefig("ode_sys_sym_opt_param_pinn_param.png")