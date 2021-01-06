"""
    RangeCumsum(range)

represents the cumsum of a `AbstractRange`. 
"""
struct RangeCumsum{T, RR<:AbstractRange{T}} <: LayoutVector{T}
    range::RR
end

size(c::RangeCumsum) = size(c.range)

==(a::RangeCumsum, b::RangeCumsum) = a.range == b.range
BroadcastStyle(::Type{<:RangeCumsum{<:Any,RR}}) where RR = BroadcastStyle(RR)


Base.@propagate_inbounds function getindex(c::RangeCumsum{<:Any,<:AbstractRange}, k::Integer)
    @boundscheck checkbounds(c, k)
    r = c.range
    k * (first(r) + r[k]) ÷ 2
end
Base.@propagate_inbounds function getindex(c::RangeCumsum{<:Any,<:AbstractUnitRange}, k::Integer)
    @boundscheck checkbounds(c, k)
    r = c.range
    k * (2first(r) + k - 1) ÷ 2
end

Base.@propagate_inbounds getindex(c::RangeCumsum, kr::OneTo) = RangeCumsum(c.range[kr])

last(r::RangeCumsum) = sum(r.range)
diff(r::RangeCumsum) = r.range[2:end]

union(a::RangeCumsum{<:Any,<:Base.OneTo}, b::RangeCumsum{<:Any,<:Base.OneTo}) = 
    RangeCumsum(Base.OneTo(max(last(a.range),last(b.range))))

sort!(a::RangeCumsum{<:Any,<:AbstractUnitRange}) = a