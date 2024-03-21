"""
    RangeCumsum(range)

represents the cumsum of a `AbstractRange`.
"""
struct RangeCumsum{T, RR<:AbstractRange{T}} <: LayoutVector{T}
    range::RR
end

size(c::RangeCumsum) = size(c.range)
axes(c::RangeCumsum) = axes(c.range)

Base.parent(r::RangeCumsum) = r.range

==(a::RangeCumsum, b::RangeCumsum) = a.range == b.range
BroadcastStyle(::Type{<:RangeCumsum{<:Any,RR}}) where RR = BroadcastStyle(RR)

_half(x::Integer) = x ÷ 2
_half(x) = x / 2

_half_prod(a, b) = _half(a) * b
function _half_prod(a::Integer, b::Integer)
    iseven(a) ? (a ÷ 2) * b : a * (b ÷ 2)
end

function _getindex(r::AbstractRange{<:Real}, k)
    v = first(r)
    s = step(r)
    # avoid overflow, if possible
    k * v + s * _half_prod(k, k-1)
end
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

function _onethird_prod(a::Integer, b::Integer)
    mod(a, 3) == 0 ? (a÷3) * b : a * (b÷3)
end

function Base.sum(r::RangeCumsum{<:Real})
    N = length(r)
    v = first(r)
    s = step(r.range)
    # avoid overflow, if possible
    halfnnp1 = _half_prod(N, N+1)
    v * halfnnp1 + s * _onethird_prod(halfnnp1, N-1)
end

union(a::RangeCumsum{<:Any,<:OneTo}, b::RangeCumsum{<:Any,<:OneTo}) =
    RangeCumsum(OneTo(max(last(a.range), last(b.range))))

sort!(a::RangeCumsum{<:Any,<:Base.OneTo}) = a
sort(a::RangeCumsum{<:Any,<:Base.OneTo}) = a

convert(::Type{RangeCumsum{T,R}}, r::RangeCumsum) where {T,R} = RangeCumsum{T,R}(convert(R, r.range))

function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(-), r::RangeCumsum)
    RangeCumsum(.-r.range)
end
function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(*), x::Number, r::RangeCumsum)
    RangeCumsum(x * r.range)
end
function Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(*), r::RangeCumsum, x::Number)
    RangeCumsum(r.range * x)
end
