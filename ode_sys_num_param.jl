# This is an expansion on the file ode_sys_sym.jl available in this repo.
# In this script we define the system using a function (not symbolic) which
# can still be used by the package DifferentialEquations.jl

using DifferentialEquations
import Plots.plot, Plots.savefig, Plots.plotlyjs
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
# parameter space
p = [0,1,2]

# solution_y_a = zeros(length(p))
for ps in p
    prob = ODEProblem(sys,u0,tspan,ps)
    sol = solve(prob,Tsit5())
    solution = reduce(vcat,sol.u')

    plot(sol.t, [solution[:,1] solution[:,2] solution[:,3]],
         linewidth=2, label=["x" "y" "z"], title="p=$ps", xlabel="t")
    savefig("ode_sys_num_param_solutions_p_$ps")
end