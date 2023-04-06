# Andre Guerra
# andre.guerra@mail.mcgill.ca
#
# This is an expansion on the file ode_sys_num_param.jl available in this repo.
# Like in the previous script, we define here the system using a function
# (not symbolic) which can still be used by the package 
# DifferentialEquations.jl

using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using  Plots#, Plots.savefig, Plots.plotlyjs
# plotting backend
plotlyjs()

# ode system with parameter p
function sys(du,u,p,t)
    du[1] = -u[1] + p[1]*u[3]
    du[2] = -u[2] + 2*u[3]
    du[3] = u[1]^2 - 2*u[3]
end

# initial conditions
u0 = [1,1/2,3]
# timespan
tspan = (0.0,1.5)

# first we solve the system with a unique parameter value, ps
# parameter space
ps = 1
prob = ODEProblem(sys,u0,tspan,ps)
sol = solve(prob,Tsit5())
solution = reduce(vcat,sol.u')
# visualize the solution
plot(sol.t, [solution[:,1] solution[:,2] solution[:,3]],
     linewidth=2, label=["x" "y" "z"], title="p=$ps", xlabel="t")
savefig("ode_sys_num_opt_param_solution_p_$ps")

# we generate synthetic data using this solution (p value above)
t = collect(range(0,stop=1.5,length=200))
randomized = VectorOfArray([(sol(t[i]) + .01randn(3)) for i in eachindex(t)])
data = convert(Array,randomized)

# redefine the prob with a new parameter
newprob = remake(prob, p=1.2)
newsol = solve(newprob,Tsit5())
# visualize the two solutions together
plot(sol, title="p=1 vs p=1.2")
plot!(newsol)
savefig("ode_sys_num_opt_param_newsol")

# define an objective (cost) function for optimization
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data),
                                     Optimization.AutoForwardDiff(),
                                     maxiters=10000,verbose=false)

# test the parameter space [0,2] with the cost_function
param_space = 0.0:0.01:2
cost = [cost_function(i) for i in param_space]
# visualize the 
plot(param_space, cost, yscale=:log10, xlabel="Parameter",
     ylabel="Cost", title="1-Parameter Cost Function", lw=2)
savefig("ode_sys_num_opt_param_costfunction")

# note: since the data was produced using ps=1, the cost_function provides us
# with the optimal value in the parameter space as 1. This procedure has
# optimized our ODE system with parameter to fit the synthetic data generated 