"""
    RangeCumsum(range)

represents the cumsum of a `AbstractRange`.
"""
struct RangeCumsum{T, RR<:AbstractRange{T}} <: LayoutVector{T}
    range::RR
end

size(c::RangeCumsum) = size(c.range)
axes(c::RangeCumsum) = axes(c.range)

==(a::RangeCumsum, b::RangeCumsum) = a.range == b.range
BroadcastStyle(::Type{<:RangeCumsum{<:Any,RR}}) where RR = BroadcastStyle(RR)

_getindex(r::AbstractUnitRange{<:Integer}, k) = k * (2first(r) + k - 1) ÷ 2
Base.@propagate_inbounds _getindex(r::AbstractRange, k) = sum(r[range(firstindex(r), length=k)])

Base.@propagate_inbounds function getindex(c::RangeCumsum{<:Any,<:AbstractRange}, k::Integer)
    @boundscheck checkbounds(c, k)
    r = c.range
    _getindex(r, k-firstindex(r)+1)
end

Base.@propagate_inbounds getindex(c::RangeCumsum, kr::OneTo) = RangeCumsum(c.range[kr])

Base.@propagate_inbounds view(c::RangeCumsum, kr::OneTo) = c[kr]

first(r::RangeCumsum) = first(r.range)
last(r::RangeCumsum) = sum(r.range)
diff(r::RangeCumsum) = r.range[firstindex(r)+1:end]
isempty(r::RangeCumsum) = isempty(r.range)

union(a::RangeCumsum{<:Any,<:OneTo}, b::RangeCumsum{<:Any,<:OneTo}) =
    RangeCumsum(OneTo(max(last(a.range), last(b.range))))

sort!(a::RangeCumsum{<:Any,<:Base.OneTo}) = a
sort(a::RangeCumsum{<:Any,<:Base.OneTo}) = a

convert(::Type{RangeCumsum{T,R}}, r::RangeCumsum) where {T,R} = RangeCumsum{T,R}(convert(R, r.range))

function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(*), x::Number, r::RangeCumsum)
    RangeCumsum(x * r.range)
end
function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(*), r::RangeCumsum, x::Number)
    RangeCumsum(r.range * x)
end
