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

first(r::RangeCumsum) = first(r.range)
last(r::RangeCumsum) = sum(r.range)
diff(r::RangeCumsum) = r.range[2:end]
isempty(r::RangeCumsum) = isempty(r.range)

union(a::RangeCumsum{<:Any,<:OneTo}, b::RangeCumsum{<:Any,<:OneTo}) =
    RangeCumsum(OneTo(max(last(a.range), last(b.range))))

sort!(a::RangeCumsum{<:Any,<:AbstractUnitRange}) = a

convert(::Type{RangeCumsum{T,R}}, r::RangeCumsum) where {T,R} = RangeCumsum{T,R}(convert(R, r.range))
