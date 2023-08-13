using Pkg
Pkg.activate("../PINN")

using Glob, Plots, ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--path", "-p"
            help = "Define path to files"
            arg_type = String
            default = "exploration/parameter"
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
path = parsed_args["path"]

train = "train_2_ts_10_chain_4_node_24"
files = Glob.glob(path * "/*/") # Glob only directories
files = [f[1:end-1] for f in files]
points = [basename(f) for f in files]
text = [Glob.glob(file * "/" * train * "/*.txt") for file in files]

#fig = Plots.Plot()
ps = [-1, 1, -1, 2, 1, -2]
i = 1

display(plot()) # Refresh plot

for f in text
    open(f[1], "r") do io
        arr = []

        while ! eof(io)
            ss = readline(io)
            push!(arr, parse(Float64, ss))
        end

        # Calculating error
        p_err = abs.((arr - ps) ./ ps * 100)
        err = abs.((arr - ps))
        p_str = "% Error for: " * points[i] * " is " * string(p_err)
        str = "Error for: " * points[i] * " is " * string(err)

        println(p_str)
        println(str)

        open(path * "/ode_sys_sym_opt_param_pinn__2_10_error.txt", "a") do io
            write(io, p_str * "\n")
            write(io, str * "\n")
        end

        display(plot!(err, label=points[i], xlabel=ps, ylabel="Error"))
        savefig(path * "/ode_sys_sym_opt_param_pinn_2_10_error.png")
        
    end
    i = i + 1
end