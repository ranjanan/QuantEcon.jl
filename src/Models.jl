module Models

# Import all QuantEcon names
using ..QuantEcon
using Compat

# 3rd party dependencies
using Distributions
using Optim: optimize
using Grid: CoordInterpGrid, BCnan, BCnearest, InterpLinear

abstract AbstractModel

"""
Generic function to solve model via value function iteration.

For this function to work the model must have implemented the following:

* `init_values(m::T)`, where `T <: AbstractModel` is the model's type
* `bellman_operator(m::T, a::S)`, where `S` is the type returned from
   init_values
"""

function solve_vf(m::AbstractModel; kwargs...)
    # make sure init_values is defined. If not provide a decent error message
    T_model = typeof(m)
    if !method_exists(init_values, (T_model,))
        e = "init_values(m::$(T_model) must be defined to use default solver"
        error(e)
    end
    init = init_values(m)

    # hand off to routine that takes initial values as argument
    solve_vf(m, init; kwargs...)
end

function solve_vf(m::AbstractModel, init; kwargs...)
    # make sure bellman_operator is defined

    T_model = typeof(m)
    T_init = typeof(init)
    if !method_exists(bellman_operator, (T_model, typeof(init)))
        e = "bellman_operator(m::$(T_model), x::$(T_init))"
        e *= " must be defined to use default solver"
        error(e)
    end

    # solve this thing!
    f(x) = bellman_operator(m, x)
    compute_fixed_point(f, init; kwargs...)
end

solve_pf(m::AbstractModel; kwargs...) = get_greedy(m, solve_vf(m; kwargs...))
function solve_pf(m::AbstractModel, init; kwargs...)
    get_greedy(m, init, solve_vf(m, init; kwargs...))
end

function solve_both(m::AbstractModel; kwargs...)
    vf = solve_vf(m; kwargs...)
    pf = get_greedy(m, vf)
    (vf, pf)
end

function solve_both(m::AbstractModel, init; kwargs...)
    vf = solve_vf(m, init; kwargs...)
    pf = get_greedy(m, vf)
    (vf, pf)
end

export
# types
    AbstractModel,
    AssetPrices,
    CareerWorkerProblem,
    ConsumerProblem,
    JvWorker,
    LucasTree,
    SearchProblem,
    GrowthModel,
    tree_price, consol_price, call_option,            # asset_pricing
    get_greedy, get_greedy!,                          # career, odu, optgrowth
    coleman_operator, coleman_operator!, init_values, # ifp
    compute_lt_price, lucas_operator,                 # lucastree
    res_wage_operator, res_wage_operator!,            # odu
    bellman_operator, bellman_operator!,              # many
    solve_vf, solve_pf, solve_both                    # many

____bellman_main_docstring = """
Apply the Bellman operator for a given model and initial value
"""

____greedy_main_docstring = """
Extract the greedy policy (policy function) of the model
"""

____see_methods_docstring = """
See the specific methods of the mutating function for more details on arguments
"""

____mutate_last_positional_docstring = """
The last positional argument passed to this function will be over-written
"""

____kwarg_note = """
There is also a version of this function that accepts keyword arguments for
each parameter
"""

include("models/asset_pricing.jl")
include("models/career.jl")
include("models/ifp.jl")
include("models/jv.jl")
include("models/lucastree.jl")
include("models/odu.jl")
include("models/optgrowth.jl")


"""
$(____bellman_main_docstring). $(____see_methods_docstring)
"""
bellman_operator

"""
$(____bellman_main_docstring). $(____see_methods_docstring)

$(____mutate_last_positional_docstring)
"""
bellman_operator!

"""
$(____greedy_main_docstring). $(____see_methods_docstring)
"""
get_greedy

"""
$(____greedy_main_docstring). $(____see_methods_docstring)

$(____mutate_last_positional_docstring)
"""
get_greedy!


end  # module
