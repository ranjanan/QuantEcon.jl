abstract AbstractModel

function Base.writemime{T<:AbstractModel}(io::IO, ::MIME"text/plain", m::T)
    msg = "This should be overwritten by subtype, but since you asked"
    msg *= " this is a Model of type $T"
    print(io, msg)
end

abstract AbstractSolution

function Base.writemime{T<:AbstractSolution}(io::IO, ::MIME"text/plain", ms::T)
    msg = "$T, solution for model type $(typeof(ms.model))\n"
    msg *= "Fields on this type are:\n  - " * join(fieldnames(ms), "\n  - ")
    print(io, msg)
end

function vfi{TM<:AbstractModel}(m::TM; kwargs...)
    # make sure init_values is defined. If not provide a decent error message
    if !method_exists(init_values, (TM,))
        error("init_values(m::$(TM) must be defined to use default solver")
    end
    init = init_values(m)

    # hand off to routine that takes initial values as argument
    vfi(m, init; kwargs...)
end


function vfi{TM<:AbstractModel, TI}(m::TM, init::TI; kwargs...)
    # make sure bellman_operator is defined

    if !method_exists(bellman_operator, (TM, TI))
        e = "bellman_operator(m::$(TM), x::$(TI))"
        e *= " must be defined to use default solver"
        error(e)
    end

    # solve this thing!
    f(x) = bellman_operator(m, x)
    compute_fixed_point(f, init; kwargs...)
end

solve_pf{TM<:AbstractModel}(m::TM; kwargs...) =
    get_greedy(m, vfi(m; kwargs...))

solve_pf{TM<:AbstractModel}(m::TM, init; kwargs...) =
    get_greedy(m, vfi(m, init; kwargs...))

function solve_both(m::AbstractModel; kwargs...)
    vf = vfi(m; kwargs...)
    pf = get_greedy(m, vf)
    (vf, pf)
end

function solve_both(m::AbstractModel, init; kwargs...)
    vf = vfi(m, init; kwargs...)
    pf = get_greedy(m, vf)
    (vf, pf)
end
