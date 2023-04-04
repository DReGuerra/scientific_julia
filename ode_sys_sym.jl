# In this script we solve a system of 3 ODEs using the symbolic packages
# available in Julia

using ModelingToolkit
using OrdinaryDiffEq, DifferentialEquations
import Plots.plot, Plots.savefig, Plots.plotlyjs
# plotting backend
plotlyjs()

# define parameters, variables, and d/dt operator
@parameters t
@variables x(..) y(..) z(..)
Dt = Differential(t)

# symbolic definition of the system of ODEs
eqs = [Dt(x(t)) ~ -x(t) + 3*z(t),
       Dt(y(t)) ~ -y(t) + 2*z(t),
       Dt(z(t)) ~ x(t)^2 - 2*z(t)]

# initial conditions
ics = [0,1/2,3]

# define the problem using DifferentialEquations.jl
@named odesystem = ODESystem(eqs,t,tspan=(0,1.5))
prob = ODEProblem(odesystem,ics)
# solve the system
sol = solve(prob,Tsit5())

# visualize sol
plot(sol, label=["x" "y" "z"], title="Numerical Solution",
     lw=2, xlabel="t");
savefig("ode_sys_sym_sol")

# sol is a vector of vectors, convert to matrix
solution = reduce(vcat,sol.u')

# use the matrix to plot solution
plot(sol.t, [solution[:,1] solution[:,2] solution[:,3]],
     lw=2, label=["x" "y" "z"], xlabel="t");
savefig("ode_sys_sym_matrix")

# subplots of x, y, and z solutions
p1 = plot(sol.t, solution[:,1], lw=2, lc=:blue, label="x");
p2 = plot(sol.t, solution[:,2], lw=2, lc=:orange, label="y");
p3 = plot(sol.t, solution[:,3], lw=2, lc=:green, label="z");
plot(p1, p2, p3, layout=(3,1));
savefig("ode_sys_sym_subplots")

# 3D plot
plot(solution[:,1], solution[:,2], solution[:,3],
     label="(x,y,z)", lw=2, title="3D space solution",
     xlabel="x", ylabel="y", zlabel="z");
savefig("ode_sys_sym_3D")
