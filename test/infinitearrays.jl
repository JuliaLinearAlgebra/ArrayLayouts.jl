# Infinite Arrays implementation from
# https://github.com/JuliaLang/julia/blob/master/test/testhelpers/InfiniteArrays.jl
module InfiniteArrays
    using Infinities
    export OneToInf

    abstract type AbstractInfUnitRange{T<:Real} <: AbstractUnitRange{T} end
    Base.length(r::AbstractInfUnitRange) = ℵ₀
    Base.size(r::AbstractInfUnitRange) = (ℵ₀,)
    Base.last(r::AbstractInfUnitRange) = ℵ₀
    Base.axes(r::AbstractInfUnitRange) = (OneToInf(),)

    Base.IteratorSize(::Type{<:AbstractInfUnitRange}) = Base.IsInfinite()
    Base.sum(r::AbstractInfUnitRange) = last(r)

    """
        OneToInf(n)
    Define an `AbstractInfUnitRange` that behaves like `1:∞`, with the added
    distinction that the limits are guaranteed (by the type system) to
    be 1 and ∞.
    """
    struct OneToInf{T<:Integer} <: AbstractInfUnitRange{T} end

    OneToInf() = OneToInf{Int}()

    Base.axes(r::OneToInf) = (r,)
    Base.first(r::OneToInf{T}) where {T} = oneunit(T)
    Base.oneto(::InfiniteCardinal{0}) = OneToInf()


    struct InfUnitRange{T<:Real} <: AbstractInfUnitRange{T}
        start::T
    end
    Base.first(r::InfUnitRange) = r.start
    InfUnitRange(a::InfUnitRange) = a
    InfUnitRange{T}(a::AbstractInfUnitRange) where T<:Real = InfUnitRange{T}(first(a))
    InfUnitRange(a::AbstractInfUnitRange{T}) where T<:Real = InfUnitRange{T}(first(a))
    Base.:(:)(start::T, stop::InfiniteCardinal{0}) where {T<:Integer} = InfUnitRange{T}(start)
    function Base.getindex(v::InfUnitRange{T}, i::Integer) where T
        @boundscheck i > 0 || throw(BoundsError(v, i))
        convert(T, first(v) + i - 1)
    end
    function Base.getindex(v::InfUnitRange{T}, i::AbstractUnitRange{<:Integer}) where T
        @boundscheck checkbounds(v, i)
        v[first(i)]:v[last(i)]
    end
    function Base.getindex(v::InfUnitRange{T}, i::AbstractInfUnitRange{<:Integer}) where T
        @boundscheck checkbounds(v, first(i))
        v[first(i)]:ℵ₀
    end
end
