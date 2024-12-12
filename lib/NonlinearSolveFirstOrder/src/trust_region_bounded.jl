"""
    BoundedTrustRegion(;
        concrete_jac = nothing, linsolve = nothing,
        radius_update_scheme = RadiusUpdateSchemes.Simple, max_trust_radius::Real = 0 // 1,
        initial_trust_radius::Real = 0 // 1, step_threshold::Real = 1 // 10000,
        shrink_threshold::Real = 1 // 4, expand_threshold::Real = 3 // 4,
        shrink_factor::Real = 1 // 4, expand_factor::Real = 2 // 1,
        max_shrink_times::Int = 32,
        vjp_autodiff = nothing, autodiff = nothing, jvp_autodiff = nothing,
        lb = nothing, ub = nothing
    )

A modification of the advanced TrustRegion implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems, but adapted to handle lower and upper bounds on the parameter values.

### Keyword Arguments

  - `radius_update_scheme`: the scheme used to update the trust region radius. Defaults to
    `RadiusUpdateSchemes.WangYuan`. See [`RadiusUpdateSchemes`](@ref) for more details. For a
    review on trust region radius update schemes, see [yuan2015recent](@citet).

For the remaining arguments, see [`NonlinearSolveFirstOrder.GenericTrustRegionScheme`](@ref)
documentation.
"""
function BoundedTrustRegion(;
        concrete_jac = nothing, linsolve = nothing,
        radius_update_scheme = RadiusUpdateSchemes.Simple, max_trust_radius::Real = 0 // 1,
        initial_trust_radius::Real = 0 // 1, step_threshold::Real = 1 // 10000,
        shrink_threshold::Real = 1 // 4, expand_threshold::Real = 3 // 4,
        shrink_factor::Real = 1 // 4, expand_factor::Real = 2 // 1,
        max_shrink_times::Int = 32,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        lb = nothing, ub = nothing
)
    descent = Dogleg(; linsolve)
    trustregion = BoundedTrustRegionScheme(;
        method = radius_update_scheme, step_threshold, shrink_threshold, expand_threshold,
        shrink_factor, expand_factor, initial_trust_radius, max_trust_radius
    )
    return GeneralizedFirstOrderAlgorithm(;
        trustregion, descent, autodiff, vjp_autodiff, jvp_autodiff, max_shrink_times,
        concrete_jac, name = :TrustRegion
    )
end

include("trust_region.jl")

"""
    BoundedTrustRegionScheme(;
        method = RadiusUpdateSchemes.Simple,
        max_trust_radius = nothing, initial_trust_radius = nothing,
        step_threshold = nothing, shrink_threshold = nothing, expand_threshold = nothing,
        shrink_factor = nothing, expand_factor = nothing
    )

Trust Region Method that updates and stores the current trust region radius in
`trust_region`. For any of the keyword arguments, if the value is `nothing`, then we use
the value used in the respective paper.

### Keyword Arguments

  - `radius_update_scheme`: the choice of radius update scheme to be used. Defaults to
    `RadiusUpdateSchemes.Simple` which follows the conventional approach. Other available
    schemes are documented in [`RadiusUpdateSchemes`](@ref),. These schemes have the trust
    region radius converging to zero that is seen to improve convergence. For more details,
    see [1].
  - `max_trust_radius`: the maximal trust region radius. Defaults to
    `max(norm(fu), maximum(u) - minimum(u))`, except for `RadiusUpdateSchemes.NLsolve`
    where it defaults to `Inf`.
  - `initial_trust_radius`: the initial trust region radius. Defaults to
    `max_trust_radius / 11`, except for `RadiusUpdateSchemes.NLsolve` where it defaults
    to `u0_norm > 0 ? u0_norm : 1`.
  - `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
    compared with a value `r`, which is the actual reduction in the objective function
    divided by the predicted reduction. If `step_threshold > r` the model is not a good
    approximation, and the step is rejected. Defaults to `nothing`.
  - `shrink_threshold`: the threshold for shrinking the trust region radius. In every
    iteration, the threshold is compared with a value `r` which is the actual reduction in
    the objective function divided by the predicted reduction. If `shrink_threshold > r` the
    trust region radius is shrunk by `shrink_factor`. Defaults to `nothing`.
  - `expand_threshold`: the threshold for expanding the trust region radius. If a step is
    taken, i.e `step_threshold < r` (with `r` defined in `shrink_threshold`), a check is
    also made to see if `expand_threshold < r`. If that is true, the trust region radius is
    expanded by `expand_factor`. Defaults to `nothing`.
  - `shrink_factor`: the factor to shrink the trust region radius with if
    `shrink_threshold > r` (with `r` defined in `shrink_threshold`). Defaults to `0.25`.
  - `expand_factor`: the factor to expand the trust region radius with if
    `expand_threshold < r` (with `r` defined in `shrink_threshold`). Defaults to `2.0`.
"""
@kwdef @concrete struct BoundedTrustRegionScheme <: AbstractTrustRegionMethod
    method <: RUS.AbstractRadiusUpdateScheme = RUS.WangYuan
    step_threshold = nothing
    shrink_threshold = nothing
    shrink_factor = nothing
    expand_factor = nothing
    expand_threshold = nothing
    max_trust_radius = nothing
    initial_trust_radius = nothing

end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::BoundedTrustRegionScheme, f, fu, u, p,
        args...; stats, internalnorm::F = L2_NORM, vjp_autodiff = nothing,
        jvp_autodiff = nothing, kwargs...
) where {F}
    # In the bounded method, instead of solving the ordinary trust region problem, we solve
    # the problemm in the affine scale space. The affine scale space is defined by the 
    # variables ak, bk, tk, and Dk. The affine scale space is defined by the following
    # the bounds lb and ub should be defined in the problem


    T = promote_type(eltype(u), eltype(fu))
    u0_norm = internalnorm(u)
    fu_norm = internalnorm(fu)

    # Common Setup
    mtr = max_trust_radius(alg.max_trust_radius, T, alg.method, u, fu_norm)
    itr = initial_trust_radius(
        alg.initial_trust_radius, T, alg.method, mtr, u0_norm, fu_norm
    )
    stt = step_threshold(alg.step_threshold, T, alg.method)
    sht = shrink_threshold(alg.shrink_threshold, T, alg.method)
    shf = shrink_factor(alg.shrink_factor, T, alg.method)
    et = expand_threshold(alg.expand_threshold, T, alg.method)
    ef = expand_factor(alg.expand_factor, T, alg.method)

    # Scheme Specific Setup
    p1, p2, p3, p4 = get_parameters(T, alg.method)
    ϵ = T(1e-8)

    vjp_operator = alg.method isa RUS.__Yuan || alg.method isa RUS.__Bastin ?
                   VecJacOperator(prob, fu, u; autodiff = vjp_autodiff) : nothing

    jvp_operator = alg.method isa RUS.__Bastin ?
                   JacVecOperator(prob, fu, u; autodiff = jvp_autodiff) : nothing

    if alg.method isa RUS.__Yuan
        Jᵀfu_cache = StatefulJacobianOperator(vjp_operator, u, prob.p) * Utils.safe_vec(fu)
        itr = T(p1 * internalnorm(Jᵀfu_cache))
    elseif u isa Number
        Jᵀfu_cache = u
    else
        @bb Jᵀfu_cache = similar(u)
    end

    if alg.method isa RUS.__Bastin
        @bb δu_cache = similar(u)
    else
        δu_cache = nothing
    end

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    @bb Jδu_cache = similar(fu)

    return BoundedTrustRegionSchemeCache(
        alg.method, f, p, mtr, itr, itr, stt, sht, et, shf, ef,
        p1, p2, p3, p4, ϵ, T(0), vjp_operator, jvp_operator, Jᵀfu_cache, Jδu_cache,
        δu_cache, internalnorm, u_cache, fu_cache, false, 0, stats, alg
    )
end

@concrete mutable struct BoundedTrustRegionSchemeCache <: AbstractTrustRegionMethodCache
    method
    f
    p
    max_trust_radius
    initial_trust_radius
    trust_region
    step_threshold
    shrink_threshold
    expand_threshold
    shrink_factor
    expand_factor
    p1
    p2
    p3
    p4
    ϵ
    ρ
    vjp_operator
    jvp_operator
    Jᵀfu_cache
    Jδu_cache
    δu_cache
    internalnorm
    u_cache
    fu_cache
    last_step_accepted::Bool
    shrink_counter::Int
    stats::NLStats
    alg
    # Adding affine scale parameters (WangYuan 2010)
    ak
    bk
    tk
    Dk
end

function InternalAPI.reinit!(
        cache::BoundedTrustRegionSchemeCache; p = cache.p, u0 = nothing, kwargs...
)
    cache.p = p
    if u0 !== nothing
        u0_norm = cache.internalnorm(u0)
    end
    cache.last_step_accepted = false
    cache.shrink_counter = 0
end

# Defaults
for func in (
    :max_trust_radius, :initial_trust_radius, :step_threshold, :shrink_threshold,
    :shrink_factor, :expand_threshold, :expand_factor
)
    @eval function $(func)(val, ::Type{T}, args...) where {T}
        iszero(val) && return $(func)(nothing, T, args...)
        return T(val)
    end
end

function update_vectors_ak_bk!(ak, bk, u, lb, ub)
    ak .= u .- lb
    bk .= ub .- u
end

function calculate_tk(ak, bk, gk, Delta)
    tk = sqrt(sum(ak.*gk)+sum(bk.*abs.(gk)))/Delta
    return tk
end

function update_Dk!(Dk, tk, ak, bk, gk, Delta, epsilon)
    for i in axis(Dk, 1)
        if (ak[i] <= Delta) & (gk[i] >= epsilon*ak[i])
            Dk[i,i] = tk * sqrt(ak[i]/gk[i])
        elseif (bk[i] <= Delta) & (-gk[i] >= epsilon*bk[i])
            Dk[i,i] = tk * sqrt(bk[i]/abs(gk[i]))
        else
            Dk[i,i] = 1
        end
    end
end

max_trust_radius(::Nothing, ::Type{T}, method, u, fu_norm) where {T} = T(Inf)
function max_trust_radius(::Nothing, ::Type{T}, ::Union{RUS.__Simple, RUS.__NocedalWright},
        u, fu_norm) where {T}
    u_min, u_max = extrema(u)
    return max(T(fu_norm), u_max - u_min)
end

function initial_trust_radius(
        ::Nothing, ::Type{T}, method, max_tr, u0_norm, fu_norm
) where {T}
    method isa RUS.__NLsolve && return T(ifelse(u0_norm > 0, u0_norm, 1))
    (method isa RUS.__Hei || method isa RUS.__Bastin) && return T(1)
    method isa RUS.__Fan && return T((fu_norm^0.99) / 10)
    return T(max_tr / 11)
end

function step_threshold(::Nothing, ::Type{T}, method) where {T}
    method isa RUS.__Hei && return T(0)
    method isa RUS.__Yuan && return T(1 // 1000)
    method isa RUS.__Bastin && return T(1 // 20)
    return T(1 // 10000)
end

function shrink_threshold(::Nothing, ::Type{T}, method) where {T}
    method isa RUS.__Hei && return T(0)
    (method isa RUS.__NLsolve || method isa RUS.__Bastin) && return T(1 // 20)
    return T(1 // 4)
end

function expand_threshold(::Nothing, ::Type{T}, method) where {T}
    method isa RUS.__NLsolve && return T(9 // 10)
    method isa RUS.__Hei && return T(0)
    method isa RUS.__Bastin && return T(9 // 10)
    return T(3 // 4)
end

function shrink_factor(::Nothing, ::Type{T}, method) where {T}
    method isa RUS.__NLsolve && return T(1 // 2)
    method isa RUS.__Hei && return T(0)
    method isa RUS.__Bastin && return T(1 // 20)
    return T(1 // 4)
end

function get_parameters(::Type{T}, method) where {T}
    method isa RUS.__NLsolve && return (T(1 // 2), T(0), T(0), T(0))
    method isa RUS.__Hei && return (T(5), T(1 // 10), T(15 // 100), T(15 // 100))
    method isa RUS.__Yuan && return (T(2), T(1 // 6), T(6), T(0))
    method isa RUS.__Fan && return (T(1 // 10), T(1 // 4), T(12), T(1e18))
    method isa RUS.__Bastin && return (T(5 // 2), T(1 // 4), T(0), T(0))
    return (T(0), T(0), T(0), T(0))
end

expand_factor(::Nothing, ::Type{T}, method) where {T} = T(2)


function InternalAPI.solve!(
        cache::BoundedTrustRegionSchemeCache, J, fu, u, δu, descent_stats
)
    T = promote_type(eltype(u), eltype(fu))
    @bb @. cache.u_cache = u + δu
    cache.fu_cache = Utils.evaluate_f!!(cache.f, cache.fu_cache, cache.u_cache, cache.p)
    cache.stats.nf += 1

    if hasfield(typeof(descent_stats), :δuJᵀJδu) && !isnan(descent_stats.δuJᵀJδu)
        δuJᵀJδu = descent_stats.δuJᵀJδu
    else
        @bb cache.Jδu_cache = J × vec(δu)
        δuJᵀJδu = Utils.safe_dot(cache.Jδu_cache, cache.Jδu_cache)
    end
    @bb cache.Jᵀfu_cache = transpose(J) × vec(fu)
    num = (cache.internalnorm(cache.fu_cache)^2 - cache.internalnorm(fu)^2) / 2
    denom = Utils.safe_dot(δu, cache.Jᵀfu_cache) + δuJᵀJδu / 2
    cache.ρ = num / denom

    if cache.ρ > cache.step_threshold
        cache.last_step_accepted = true
    else
        cache.last_step_accepted = false
    end

    if cache.method isa RUS.__WangYuan


    if cache.method isa RUS.__Simple
        if cache.ρ < cache.shrink_threshold
            cache.trust_region *= cache.shrink_factor
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            if cache.ρ > cache.expand_threshold && cache.ρ > cache.step_threshold
                cache.trust_region = cache.expand_factor * cache.trust_region
            end
        end
    elseif cache.method isa RUS.__NLsolve
        if cache.ρ < cache.shrink_threshold
            cache.trust_region *= cache.shrink_factor
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            if cache.ρ ≥ cache.expand_threshold
                cache.trust_region = cache.expand_factor * cache.internalnorm(δu)
            elseif cache.ρ ≥ cache.p1
                cache.trust_region = max(
                    cache.trust_region, cache.expand_factor * cache.internalnorm(δu)
                )
            end
        end
    elseif cache.method isa RUS.__NocedalWright
        if cache.ρ < cache.shrink_threshold
            cache.trust_region = cache.shrink_factor * cache.internalnorm(δu)
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            if cache.ρ > cache.expand_threshold &&
               abs(cache.internalnorm(δu) - cache.trust_region) < 1e-6 * cache.trust_region
                cache.trust_region = cache.expand_factor * cache.trust_region
            end
        end
    elseif cache.method isa RUS.__Hei
        tr_new = rfunc_adaptive_trust_region(
            cache.ρ, cache.shrink_threshold, cache.p1, cache.p3, cache.p4, cache.p2
        ) * cache.internalnorm(δu)
        if tr_new < cache.trust_region
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
        end
        cache.trust_region = tr_new
    elseif cache.method isa RUS.__Yuan
        if cache.ρ < cache.shrink_threshold
            cache.p1 = cache.p2 * cache.p1
            cache.shrink_counter += 1
        else
            if cache.ρ ≥ cache.expand_threshold &&
               2 * cache.internalnorm(δu) > cache.trust_region
                cache.p1 = cache.p3 * cache.p1
            end
            cache.shrink_counter = 0
        end
        operator = StatefulJacobianOperator(cache.vjp_operator, cache.u_cache, cache.p)
        @bb cache.Jᵀfu_cache = operator × vec(cache.fu_cache)
        cache.trust_region = cache.p1 * cache.internalnorm(cache.Jᵀfu_cache)
    elseif cache.method isa RUS.__Fan
        if cache.ρ < cache.shrink_threshold
            cache.p1 *= cache.p2
            cache.shrink_counter += 1
        else
            cache.shrink_counter = 0
            cache.ρ > cache.expand_threshold &&
                (cache.p1 = min(cache.p1 * cache.p3, cache.p4))
        end
        cache.trust_region = cache.p1 * (cache.internalnorm(cache.fu_cache)^T(0.99))
    elseif cache.method isa RUS.__Bastin
        if cache.ρ > cache.step_threshold
            jvp_op = StatefulJacobianOperator(cache.jvp_operator, cache.u_cache, cache.p)
            vjp_op = StatefulJacobianOperator(cache.vjp_operator, cache.u_cache, cache.p)
            @bb cache.Jδu_cache = jvp_op × vec(cache.δu_cache)
            @bb cache.Jᵀfu_cache = vjp_op × vec(cache.fu_cache)
            denom_1 = dot(Utils.safe_vec(cache.Jᵀfu_cache), cache.Jᵀfu_cache)
            @bb cache.Jᵀfu_cache = vjp_op × vec(cache.Jδu_cache)
            denom_2 = dot(Utils.safe_vec(cache.Jᵀfu_cache), cache.Jᵀfu_cache)
            denom = denom_1 + denom_2 / 2
            ρ = num / denom
            if ρ ≥ cache.expand_threshold
                cache.trust_region = cache.p1 * cache.internalnorm(cache.δu_cache)
            end
            cache.shrink_counter = 0
        else
            cache.trust_region *= cache.p2
            cache.shrink_counter += 1
        end
    end

    cache.trust_region = min(cache.trust_region, cache.max_trust_radius)

    return cache.last_step_accepted, cache.u_cache, cache.fu_cache
end
