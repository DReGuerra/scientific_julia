# Jennifer Tram Su
# .jl file expansion on `ode_sys_sym_param_pinn.ipynb` to optimize multiple parameters

## Packages
using Pkg
Pkg.activate("../../PINN")

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Plots, DifferentialEquations, Zygote, StatsBase, Noise
using ArgParse
import ModelingToolkit: Interval, infimum, supremum

## Reading in Command-Line Arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--train", "-r"
            help = "Define end of training interval"
            arg_type = Float64
            default = 10.0
        "--nodes", "-n"
            help = "Define number of nodes in each hidden layers"
            arg_type = Int
            default = 40
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

train = parsed_args["train"]
n = parsed_args["nodes"]

## Generating Training Data
function sys!(du, u, p, t)          
    du[1] = p[1]*u[1] + p[2]*u[3]
    du[2] = p[3]*u[2] + p[4]*u[3]
    du[3] = p[5]*u[1]^2 + p[6]*u[3]
end

u0 = [1; 1/2; 3]
tspan = (0.0, train)
ps = [-1, 1, -1, 2, 1, -2]

prob = ODEProblem(sys!, u0, tspan, ps)
sol = solve(prob, Tsit5(), dt = 0.1)

## Defining ODE system
@parameters t
@variables x(..), y(..), z(..)
Dt = Differential(t)

eqs = [Dt(x(t)) ~ -x(t) + z(t),
    Dt(y(t)) ~ -y(t) + 2*z(t),
    Dt(z(t)) ~ x(t)^2 - 2*z(t)]
bcs = [x(0) ~ 1, y(0) ~ 1/2, z(0) ~ 3]

domains = [t ∈ Interval(0.0, train)]
dt = 0.001
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]
depvars = [:x, :y, :z]

## Defining the Neural Network
input_ = length(domains)

chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dx/dt
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dy/dt
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) # dz/dt

## Defining PINN Interface
discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3], NeuralPDE.GridTraining(dt))

@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)])

prob_ = NeuralPDE.discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

# Store callback frames in animation
a = Animation()

# Storing losses
losses = []
pde_losses = []
bcs_losses = []

callback = function(p, l)
    # Visualize losses
    push!(losses, l)
    push!(pde_losses, map(l_ -> l_(p), pde_inner_loss_functions))
    push!(bcs_losses, map(l_ -> l_(p), bcs_inner_loss_functions))

    # # Visualizing training
    weights = [p.depvar.x, p.depvar.y, p.depvar.z]
    state = [[discretization.phi[i]([t], weights[i])[1] for t in ts] for i in 1:3] 
    fig = plot(ts, state, xlabel = "t", label = ["x(t)" "y(t)" "z(t)"], xlims=(0, train), ylims=(0, 3), legend=:outertopright)
    frame(a, fig)

    return false
end

res = Optimization.solve(prob_, BFGS(); callback = callback, maxiters = 1000)
p_ = res.u

## Visualizing the PINN Prediction
minimizers = [res.u.depvar[depvars[i]] for i in 1:3]
u_predict = [[discretization.phi[i]([t], minimizers[i])[1] for t in ts] for i in 1:3] 
plot(sol, label = ["x(t)" "y(t)" "z(t)"])
plot!(ts, u_predict, label = ["x_pred" "y_pred" "z_pred"], title="Prediction")
savefig("ode_sys_sym_extrap_pinn_pred.png")

# ## Plotting losses
# pde_losses_x = hcat(pde_losses...)[1, :]
# pde_losses_y = hcat(pde_losses...)[2, :]
# pde_losses_z = hcat(pde_losses...)[3, :]

# bcs_losses_x = hcat(bcs_losses...)[1, :]
# bcs_losses_y = hcat(bcs_losses...)[2, :]
# bcs_losses_z = hcat(bcs_losses...)[3, :]

# # Plot loss on log scale
# plot(1:length(losses), log10.(losses), label = "Total Loss", xlabel="Iteration", ylabel="log10(Loss)")
# plot!(1:length(losses), log10.(pde_losses_x), label = "PDE Loss x")
# plot!(1:length(losses), log10.(pde_losses_y), label = "PDE Loss y")
# plot!(1:length(losses), log10.(pde_losses_z), label = "PDE Loss z")

# # Plot bcs Loss
# plot!(1:length(losses), log10.(bcs_losses_x), label = "BC Loss x")
# plot!(1:length(losses), log10.(bcs_losses_y), label = "BC Loss y")
# plot!(1:length(losses), log10.(bcs_losses_z), label = "BC Loss z")
# savefig("ode_sys_sym_extrap_pinn_loss.png")

gif(a, "ode_sys_sym_extrap_pinn_fitting.gif")