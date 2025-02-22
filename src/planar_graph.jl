struct HalfEdge{T<:Integer}
    src::T
    dst::T
end

src(he::HalfEdge) = he.src
dst(he::HalfEdge) = he.dst

new_edge(src::T, dst::T) where {T<:Integer} = (HalfEdge(src, dst), HalfEdge(dst, src))

function Base.:(==)(he1::HalfEdge, he2::HalfEdge)
    src(he1) == src(he2) && dst(he1) == dst(he2)
end

mutable struct PlanarMultigraph{T<:Integer}
    v2he::Dict{T,T} # v_id -> he_id
    half_edges::Dict{T,HalfEdge{T}} # he_id -> he

    f2he::Dict{T,T}  # f_id -> he_id
    he2f::Dict{T,T}    # he_id -> f_id, if cannot find, then it's a boundary

    next::Dict{T,T}    # he_id -> he_id, counter clockwise
    twin::Dict{T,T}    # he_id -> he_id


    v_max::T
    he_max::T
    f_max::T
    boundary::Vector{T} # f_id
end

PlanarMultigraph{T}() where {T<:Int} = PlanarMultigraph{T}(
    Dict{T,T}(),
    Dict{T,HalfEdge{T}}(),
    Dict{T,T}(),
    Dict{T,T}(),
    Dict{T,T}(),
    Dict{T,T}(),
    0,
    0,
    0,
    [0],
)

function PlanarMultigraph{T}(qubits::Int) where {T<:Integer}
    g = PlanarMultigraph{T}()
    for _ = 1:qubits
        vtxs = create_vertex!(g; mul = 2)
        hes_id, _ = create_edge!(g, vtxs[1], vtxs[2])
        g.he2f[hes_id[1]] = 0
        g.he2f[hes_id[2]] = 0
    end
    return g
end

Base.copy(g::PlanarMultigraph) = PlanarMultigraph(
    copy(g.v2he),
    copy(g.half_edges),
    copy(g.f2he),
    copy(g.he2f),
    copy(g.next),
    copy(g.twin),
    g.v_max,
    g.he_max,
    g.f_max,
    copy(g.boundary),
)

function Base.:(==)(pmg1::PlanarMultigraph{T}, pmg2::PlanarMultigraph{T}) where {T<:Integer}
    return pmg_equiv(pmg1, pmg2, false)
end



