// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
};

struct Cfg {
    uint d;
    uint m;
    uint r_rows;
    uint r_off;
};
typedef float type_2[1];
struct type_3 {
    float inner[2048];
};
struct type_4 {
    uint inner[1024];
};
struct type_6 {
    metal::float4 inner[512];
};
struct type_8 {
    metal::float2 inner[256];
};
struct type_10 {
    metal::float3 inner[128];
};
struct type_12 {
    int inner[512];
};
struct type_14 {
    metal::atomic_uint inner[256];
};
struct type_16 {
    metal::atomic_int inner[128];
};
struct type_40 {
    float inner[64];
};
struct type_41 {
    uint inner[16];
};
struct type_42 {
    uint inner[64];
};
struct type_43 {
    uint inner[128];
};
struct type_44 {
    bool inner[16];
};
struct type_45 {
    metal::float4 inner[16];
};
struct type_46 {
    metal::float2 inner[16];
};
struct type_47 {
    metal::float4 inner[64];
};
constant uint WG_X = 16u;
constant uint WG_Y = 16u;
constant uint KNOB_WCOL = 16u;
constant uint KNOB_WROW = 16u;
constant uint KNOB_BCOL = 4u;
constant uint KNOB_BROW = 4u;
constant uint KNOB_KTILE = 1u;
constant uint KNOB_THREADS = 256u;

int add_sat(
    int a,
    int b
) {
    int s_1 = as_type<int>(as_type<uint>(a) + as_type<uint>(b));
    bool ov = ((a ^ s_1) & (b ^ s_1)) < 0;
    return ov ? ((a < 0) ? (-2147483647 - 1) : 2147483647) : s_1;
}

bool at2_bool(
    metal::bool2 v,
    uint i
) {
    switch(i) {
        case 0u: {
            return v.x;
        }
        default: {
            return v.y;
        }
    }
}

half at2_f16_(
    metal::half2 v_1,
    uint i_1
) {
    switch(i_1) {
        case 0u: {
            return v_1.x;
        }
        default: {
            return v_1.y;
        }
    }
}

float at2_f32_(
    metal::float2 v_2,
    uint i_2
) {
    switch(i_2) {
        case 0u: {
            return v_2.x;
        }
        default: {
            return v_2.y;
        }
    }
}

int at2_i32_(
    metal::int2 v_3,
    uint i_3
) {
    switch(i_3) {
        case 0u: {
            return v_3.x;
        }
        default: {
            return v_3.y;
        }
    }
}

uint at2_u32_(
    metal::uint2 v_4,
    uint i_4
) {
    switch(i_4) {
        case 0u: {
            return v_4.x;
        }
        default: {
            return v_4.y;
        }
    }
}

bool at3_bool(
    metal::bool3 v_5,
    uint i_5
) {
    switch(i_5) {
        case 0u: {
            return v_5.x;
        }
        case 1u: {
            return v_5.y;
        }
        default: {
            return v_5.z;
        }
    }
}

half at3_f16_(
    metal::half3 v_6,
    uint i_6
) {
    switch(i_6) {
        case 0u: {
            return v_6.x;
        }
        case 1u: {
            return v_6.y;
        }
        default: {
            return v_6.z;
        }
    }
}

float at3_f32_(
    metal::float3 v_7,
    uint i_7
) {
    switch(i_7) {
        case 0u: {
            return v_7.x;
        }
        case 1u: {
            return v_7.y;
        }
        default: {
            return v_7.z;
        }
    }
}

int at3_i32_(
    metal::int3 v_8,
    uint i_8
) {
    switch(i_8) {
        case 0u: {
            return v_8.x;
        }
        case 1u: {
            return v_8.y;
        }
        default: {
            return v_8.z;
        }
    }
}

uint at3_u32_(
    metal::uint3 v_9,
    uint i_9
) {
    switch(i_9) {
        case 0u: {
            return v_9.x;
        }
        case 1u: {
            return v_9.y;
        }
        default: {
            return v_9.z;
        }
    }
}

bool at4_bool(
    metal::bool4 v_10,
    uint i_10
) {
    switch(i_10) {
        case 0u: {
            return v_10.x;
        }
        case 1u: {
            return v_10.y;
        }
        case 2u: {
            return v_10.z;
        }
        default: {
            return v_10.w;
        }
    }
}

half at4_f16_(
    metal::half4 v_11,
    uint i_11
) {
    switch(i_11) {
        case 0u: {
            return v_11.x;
        }
        case 1u: {
            return v_11.y;
        }
        case 2u: {
            return v_11.z;
        }
        default: {
            return v_11.w;
        }
    }
}

float at4_f32_(
    metal::float4 v_12,
    uint i_12
) {
    switch(i_12) {
        case 0u: {
            return v_12.x;
        }
        case 1u: {
            return v_12.y;
        }
        case 2u: {
            return v_12.z;
        }
        default: {
            return v_12.w;
        }
    }
}

int at4_i32_(
    metal::int4 v_13,
    uint i_13
) {
    switch(i_13) {
        case 0u: {
            return v_13.x;
        }
        case 1u: {
            return v_13.y;
        }
        case 2u: {
            return v_13.z;
        }
        default: {
            return v_13.w;
        }
    }
}

uint at4_u32_(
    metal::uint4 v_14,
    uint i_14
) {
    switch(i_14) {
        case 0u: {
            return v_14.x;
        }
        case 1u: {
            return v_14.y;
        }
        case 2u: {
            return v_14.z;
        }
        default: {
            return v_14.w;
        }
    }
}

float clamp_p(
    float x,
    metal::float2 p
) {
    return metal::clamp(x, p.x, p.y);
}

int clamp_pi(
    int x_1,
    metal::int2 p_1
) {
    return metal::clamp(x_1, p_1.x, p_1.y);
}

metal::float2 col_at2x2_(
    metal::float2x2 m,
    uint i_15
) {
    switch(i_15) {
        case 0u: {
            return m[0];
        }
        default: {
            return m[1];
        }
    }
}

metal::float3 col_at2x3_(
    metal::float2x3 m_1,
    uint i_16
) {
    switch(i_16) {
        case 0u: {
            return m_1[0];
        }
        default: {
            return m_1[1];
        }
    }
}

metal::float4 col_at2x4_(
    metal::float2x4 m_2,
    uint i_17
) {
    switch(i_17) {
        case 0u: {
            return m_2[0];
        }
        default: {
            return m_2[1];
        }
    }
}

metal::float2 col_at3x2_(
    metal::float3x2 m_3,
    uint i_18
) {
    switch(i_18) {
        case 0u: {
            return m_3[0];
        }
        case 1u: {
            return m_3[1];
        }
        default: {
            return m_3[2];
        }
    }
}

metal::float3 col_at3x3_(
    metal::float3x3 m_4,
    uint i_19
) {
    switch(i_19) {
        case 0u: {
            return m_4[0];
        }
        case 1u: {
            return m_4[1];
        }
        default: {
            return m_4[2];
        }
    }
}

metal::float4 col_at3x4_(
    metal::float3x4 m_5,
    uint i_20
) {
    switch(i_20) {
        case 0u: {
            return m_5[0];
        }
        case 1u: {
            return m_5[1];
        }
        default: {
            return m_5[2];
        }
    }
}

metal::float2 col_at4x2_(
    metal::float4x2 m_6,
    uint i_21
) {
    switch(i_21) {
        case 0u: {
            return m_6[0];
        }
        case 1u: {
            return m_6[1];
        }
        case 2u: {
            return m_6[2];
        }
        default: {
            return m_6[3];
        }
    }
}

metal::float3 col_at4x3_(
    metal::float4x3 m_7,
    uint i_22
) {
    switch(i_22) {
        case 0u: {
            return m_7[0];
        }
        case 1u: {
            return m_7[1];
        }
        case 2u: {
            return m_7[2];
        }
        default: {
            return m_7[3];
        }
    }
}

metal::float4 col_at4x4_(
    metal::float4x4 m_8,
    uint i_23
) {
    switch(i_23) {
        case 0u: {
            return m_8[0];
        }
        case 1u: {
            return m_8[1];
        }
        case 2u: {
            return m_8[2];
        }
        default: {
            return m_8[3];
        }
    }
}

uint fmix32_(
    uint h0_
) {
    uint h = {};
    h = h0_;
    uint _e2 = h;
    uint _e3 = h;
    h = _e2 ^ (_e3 >> 16u);
    uint _e7 = h;
    h = _e7 * 2246822507u;
    uint _e10 = h;
    uint _e11 = h;
    h = _e10 ^ (_e11 >> 13u);
    uint _e15 = h;
    h = _e15 * 3266489909u;
    uint _e18 = h;
    uint _e19 = h;
    h = _e18 ^ (_e19 >> 16u);
    uint _e23 = h;
    return _e23;
}

uint combine_hash(
    uint a_1,
    uint b_1
) {
    uint _e11 = fmix32_(a_1 ^ (((b_1 * 2654435769u) + (a_1 << 6u)) + (a_1 >> 2u)));
    return _e11;
}

metal::float2x2 inverse_2_(
    metal::float2x2 m_9
) {
    float id = 1.0 / metal::determinant(m_9);
    return metal::float2x2(metal::float2(m_9[1].y, -(m_9[0].y)) * id, metal::float2(-(m_9[1].x), m_9[0].x) * id);
}

metal::float3x3 inverse_3_(
    metal::float3x3 m_10
) {
    metal::float3 r0_ = metal::cross(m_10[1], m_10[2]);
    metal::float3 r1_ = metal::cross(m_10[2], m_10[0]);
    metal::float3 r2_ = metal::cross(m_10[0], m_10[1]);
    float id_1 = 1.0 / metal::dot(m_10[0], r0_);
    return metal::transpose(metal::float3x3(r0_, r1_, r2_)) * id_1;
}

metal::float4x4 inverse_4_(
    metal::float4x4 m_11
) {
    float a00_ = m_11[0].x;
    float a01_ = m_11[0].y;
    float a02_ = m_11[0].z;
    float a03_ = m_11[0].w;
    float a10_ = m_11[1].x;
    float a11_ = m_11[1].y;
    float a12_ = m_11[1].z;
    float a13_ = m_11[1].w;
    float a20_ = m_11[2].x;
    float a21_ = m_11[2].y;
    float a22_ = m_11[2].z;
    float a23_ = m_11[2].w;
    float a30_ = m_11[3].x;
    float a31_ = m_11[3].y;
    float a32_ = m_11[3].z;
    float a33_ = m_11[3].w;
    float b00_ = (a00_ * a11_) - (a01_ * a10_);
    float b01_ = (a00_ * a12_) - (a02_ * a10_);
    float b02_ = (a00_ * a13_) - (a03_ * a10_);
    float b03_ = (a01_ * a12_) - (a02_ * a11_);
    float b04_ = (a01_ * a13_) - (a03_ * a11_);
    float b05_ = (a02_ * a13_) - (a03_ * a12_);
    float b06_ = (a20_ * a31_) - (a21_ * a30_);
    float b07_ = (a20_ * a32_) - (a22_ * a30_);
    float b08_ = (a20_ * a33_) - (a23_ * a30_);
    float b09_ = (a21_ * a32_) - (a22_ * a31_);
    float b10_ = (a21_ * a33_) - (a23_ * a31_);
    float b11_ = (a22_ * a33_) - (a23_ * a32_);
    float id_2 = 1.0 / ((((((b00_ * b11_) - (b01_ * b10_)) + (b02_ * b09_)) + (b03_ * b08_)) - (b04_ * b07_)) + (b05_ * b06_));
    return metal::float4x4(metal::float4(((a11_ * b11_) - (a12_ * b10_)) + (a13_ * b09_), ((a02_ * b10_) - (a01_ * b11_)) - (a03_ * b09_), ((a31_ * b05_) - (a32_ * b04_)) + (a33_ * b03_), ((a22_ * b04_) - (a21_ * b05_)) - (a23_ * b03_)) * id_2, metal::float4(((a12_ * b08_) - (a10_ * b11_)) - (a13_ * b07_), ((a00_ * b11_) - (a02_ * b08_)) + (a03_ * b07_), ((a32_ * b02_) - (a30_ * b05_)) - (a33_ * b01_), ((a20_ * b05_) - (a22_ * b02_)) + (a23_ * b01_)) * id_2, metal::float4(((a10_ * b10_) - (a11_ * b08_)) + (a13_ * b06_), ((a01_ * b08_) - (a00_ * b10_)) - (a03_ * b06_), ((a30_ * b04_) - (a31_ * b02_)) + (a33_ * b00_), ((a21_ * b02_) - (a20_ * b04_)) - (a23_ * b00_)) * id_2, metal::float4(((a11_ * b07_) - (a10_ * b09_)) - (a12_ * b06_), ((a00_ * b09_) - (a01_ * b07_)) + (a02_ * b06_), ((a31_ * b01_) - (a30_ * b03_)) - (a32_ * b00_), ((a20_ * b03_) - (a21_ * b01_)) + (a22_ * b00_)) * id_2);
}

bool lit_bool(
    bool x_2
) {
    return x_2;
}

half lit_f16_(
    half x_3
) {
    return x_3;
}

float lit_f32_(
    float x_4
) {
    return x_4;
}

int lit_i32_(
    int x_5
) {
    return x_5;
}

uint lit_u32_(
    uint x_6
) {
    return x_6;
}

metal::float2 lo_width(
    float lo,
    float w
) {
    return metal::float2(lo, lo + w);
}

metal::int2 lo_width_i(
    int lo_1,
    int w_1
) {
    return metal::int2(lo_1, metal::max(lo_1, as_type<int>(as_type<uint>(lo_1) + as_type<uint>(w_1))));
}

metal::float4 load4_phi(
    uint k,
    device type_2 const& phi,
    constant _mslBufferSizes& _buffer_sizes
) {
    uint m_13 = (1 + (_buffer_sizes.size0 - 0 - 4) / 4) - 1u;
    float _e8 = phi[metal::min(k, m_13)];
    float _e14 = phi[metal::min(k + 1u, m_13)];
    float _e20 = phi[metal::min(k + 2u, m_13)];
    float _e26 = phi[metal::min(k + 3u, m_13)];
    return metal::float4(_e8, _e14, _e20, _e26);
}

uint sat_add_u32_(
    uint x_7,
    uint y
) {
    uint s_2 = x_7 + y;
    return (s_2 < x_7) ? 4294967295u : s_2;
}

int naga_neg(int val) {
    return as_type<int>(-as_type<uint>(val));
}

int mul_q(
    int a_2,
    int b_2
) {
    uint m_12 = 4294967295u;
    bool neg = (a_2 < 0) != (b_2 < 0);
    uint ua = (a_2 < 0) ? static_cast<uint>(naga_neg(a_2)) : static_cast<uint>(a_2);
    uint ub = (b_2 < 0) ? static_cast<uint>(naga_neg(b_2)) : static_cast<uint>(b_2);
    uint ah = ua >> 16u;
    uint al = ua & 65535u;
    uint bh = ub >> 16u;
    uint bl = ub & 65535u;
    uint hh = ah * bh;
    if (hh < 32768u) {
        uint _e35 = sat_add_u32_(hh << 16u, ah * bl);
        uint _e40 = sat_add_u32_(al * bh, (al * bl) >> 16u);
        uint _e41 = sat_add_u32_(_e35, _e40);
        m_12 = _e41;
    }
    if (neg) {
        uint _e42 = m_12;
        uint _e46 = m_12;
        return (_e46 > 2147483648u) ? (-2147483647 - 1) : naga_neg(static_cast<int>(_e42));
    }
    uint _e50 = m_12;
    uint _e53 = m_12;
    return (_e53 > 2147483647u) ? 2147483647 : static_cast<int>(_e50);
}

int naga_f2i32(float value) {
    return static_cast<int>(metal::clamp(value, -2147483600.0, 2147483500.0));
}

int nan0_i32_(
    float x_8
) {
    return naga_f2i32((x_8 != x_8) ? 0.0 : x_8);
}

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}

uint nan0_u32_(
    float x_9
) {
    return naga_f2u32((x_9 != x_9) ? 0.0 : x_9);
}

float negate_if(
    float x_10,
    bool s
) {
    return s ? x_10 : -(x_10);
}

metal::float2x2 outer_2x2_(
    metal::float2 a_3,
    metal::float2 b_3
) {
    return metal::float2x2(a_3 * b_3.x, a_3 * b_3.y);
}

metal::float2x3 outer_2x3_(
    metal::float3 a_4,
    metal::float2 b_4
) {
    return metal::float2x3(a_4 * b_4.x, a_4 * b_4.y);
}

metal::float2x4 outer_2x4_(
    metal::float4 a_5,
    metal::float2 b_5
) {
    return metal::float2x4(a_5 * b_5.x, a_5 * b_5.y);
}

metal::float3x2 outer_3x2_(
    metal::float2 a_6,
    metal::float3 b_6
) {
    return metal::float3x2(a_6 * b_6.x, a_6 * b_6.y, a_6 * b_6.z);
}

metal::float3x3 outer_3x3_(
    metal::float3 a_7,
    metal::float3 b_7
) {
    return metal::float3x3(a_7 * b_7.x, a_7 * b_7.y, a_7 * b_7.z);
}

metal::float3x4 outer_3x4_(
    metal::float4 a_8,
    metal::float3 b_8
) {
    return metal::float3x4(a_8 * b_8.x, a_8 * b_8.y, a_8 * b_8.z);
}

metal::float4x2 outer_4x2_(
    metal::float2 a_9,
    metal::float4 b_9
) {
    return metal::float4x2(a_9 * b_9.x, a_9 * b_9.y, a_9 * b_9.z, a_9 * b_9.w);
}

metal::float4x3 outer_4x3_(
    metal::float3 a_10,
    metal::float4 b_10
) {
    return metal::float4x3(a_10 * b_10.x, a_10 * b_10.y, a_10 * b_10.z, a_10 * b_10.w);
}

metal::float4x4 outer_4x4_(
    metal::float4 a_11,
    metal::float4 b_11
) {
    return metal::float4x4(a_11 * b_11.x, a_11 * b_11.y, a_11 * b_11.z, a_11 * b_11.w);
}

float own_get_wg_f_11_(
    uint i_24,
    threadgroup type_3& wg_f_11_
) {
    float _e5 = wg_f_11_.inner[metal::min(i_24, 2047u)];
    return (i_24 < 2048u) ? _e5 : 0.0;
}

int own_get_wg_i_9_(
    uint i_25,
    threadgroup type_12& wg_i_9_
) {
    int _e6 = wg_i_9_.inner[metal::min(i_25, 511u)];
    return (i_25 < 512u) ? _e6 : 0;
}

uint own_get_wg_u_10_(
    uint i_26,
    threadgroup type_4& wg_u_10_
) {
    uint _e6 = wg_u_10_.inner[metal::min(i_26, 1023u)];
    return (i_26 < 1024u) ? _e6 : 0u;
}

metal::float2 own_get_wg_v2_8_(
    uint i_27,
    threadgroup type_8& wg_v2_8_
) {
    metal::float2 _e6 = wg_v2_8_.inner[metal::min(i_27, 255u)];
    return (i_27 < 256u) ? _e6 : metal::float2 {};
}

metal::float3 own_get_wg_v3_7_(
    uint i_28,
    threadgroup type_10& wg_v3_7_
) {
    metal::float3 _e6 = wg_v3_7_.inner[metal::min(i_28, 127u)];
    return (i_28 < 128u) ? _e6 : metal::float3 {};
}

metal::float4 own_get_wg_v4_9_(
    uint i_29,
    threadgroup type_6& wg_v4_9_
) {
    metal::float4 _e6 = wg_v4_9_.inner[metal::min(i_29, 511u)];
    return (i_29 < 512u) ? _e6 : metal::float4 {};
}

float own_load_h(
    uint r,
    uint c,
    device type_2 const& h_1,
    constant Cfg& cfg,
    constant _mslBufferSizes& _buffer_sizes
) {
    uint _e5 = cfg.d;
    float _e14 = h_1[metal::min((r * _e5) + c, (1 + (_buffer_sizes.size1 - 0 - 4) / 4) - 1u)];
    return _e14;
}

float shuffle_down_f(
    float x_11,
    uint d,
    uint lane,
    uint size
) {
    uint dd = metal::min(d, size);
    float unnamed = metal::simd_shuffle_down(x_11, dd);
    return ((lane + dd) >= size) ? x_11 : unnamed;
}

uint shuffle_down_i(
    uint x_12,
    uint d_1,
    uint lane_1,
    uint size_1
) {
    uint dd_1 = metal::min(d_1, size_1);
    uint unnamed_1 = metal::simd_shuffle_down(x_12, dd_1);
    return ((lane_1 + dd_1) >= size_1) ? x_12 : unnamed_1;
}

float shuffle_up_f(
    float x_13,
    uint d_2,
    uint lane_2,
    uint size_2
) {
    uint dd_2 = metal::min(d_2, size_2);
    float unnamed_2 = metal::simd_shuffle_up(x_13, dd_2);
    return (lane_2 < dd_2) ? x_13 : unnamed_2;
}

uint shuffle_up_i(
    uint x_14,
    uint d_3,
    uint lane_3,
    uint size_3
) {
    uint dd_3 = metal::min(d_3, size_3);
    uint unnamed_3 = metal::simd_shuffle_up(x_14, dd_3);
    return (lane_3 < dd_3) ? x_14 : unnamed_3;
}

metal::float2 sorted(
    float a_12,
    float b_12
) {
    return metal::float2(metal::min(a_12, b_12), metal::max(a_12, b_12));
}

metal::int2 sorted_i(
    int a_13,
    int b_13
) {
    return metal::int2(metal::min(a_13, b_13), metal::max(a_13, b_13));
}

uint sub_floor(
    uint a_14,
    uint b_14
) {
    return a_14 - metal::min(a_14, b_14);
}

int sub_sat(
    int a_15,
    int b_15
) {
    int s_3 = as_type<int>(as_type<uint>(a_15) - as_type<uint>(b_15));
    bool ov_1 = ((a_15 ^ b_15) & (a_15 ^ s_3)) < 0;
    return ov_1 ? ((a_15 < 0) ? (-2147483647 - 1) : 2147483647) : s_3;
}

metal::float2x2 zero_mat2x2_(
) {
    return metal::float2x2 {};
}

metal::float2x3 zero_mat2x3_(
) {
    return metal::float2x3 {};
}

metal::float2x4 zero_mat2x4_(
) {
    return metal::float2x4 {};
}

metal::float3x2 zero_mat3x2_(
) {
    return metal::float3x2 {};
}

metal::float3x3 zero_mat3x3_(
) {
    return metal::float3x3 {};
}

metal::float3x4 zero_mat3x4_(
) {
    return metal::float3x4 {};
}

metal::float4x2 zero_mat4x2_(
) {
    return metal::float4x2 {};
}

metal::float4x3 zero_mat4x3_(
) {
    return metal::float4x3 {};
}

metal::float4x4 zero_mat4x4_(
) {
    return metal::float4x4 {};
}

metal::bool2 zero_vec2_bool(
) {
    return metal::bool2 {};
}

metal::half2 zero_vec2_f16_(
) {
    return metal::half2 {};
}

metal::float2 zero_vec2_f32_(
) {
    return metal::float2 {};
}

metal::int2 zero_vec2_i32_(
) {
    return metal::int2 {};
}

metal::uint2 zero_vec2_u32_(
) {
    return metal::uint2 {};
}

metal::bool3 zero_vec3_bool(
) {
    return metal::bool3 {};
}

metal::half3 zero_vec3_f16_(
) {
    return metal::half3 {};
}

metal::float3 zero_vec3_f32_(
) {
    return metal::float3 {};
}

metal::int3 zero_vec3_i32_(
) {
    return metal::int3 {};
}

metal::uint3 zero_vec3_u32_(
) {
    return metal::uint3 {};
}

metal::bool4 zero_vec4_bool(
) {
    return metal::bool4 {};
}

metal::half4 zero_vec4_f16_(
) {
    return metal::half4 {};
}

metal::float4 zero_vec4_f32_(
) {
    return metal::float4 {};
}

metal::int4 zero_vec4_i32_(
) {
    return metal::int4 {};
}

metal::uint4 zero_vec4_u32_(
) {
    return metal::uint4 {};
}
uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

metal::uint4 naga_div(metal::uint4 lhs, metal::uint4 rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint naga_dot_uint4(metal::uint4 a, metal::uint4 b) {
    return ( + a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}


struct main_Input {
};
[[max_total_threads_per_threadgroup(256)]] kernel void main_(
  metal::uint3 gid [[thread_position_in_grid]]
, metal::uint3 lid [[thread_position_in_threadgroup]]
, metal::uint3 wid [[threadgroup_position_in_grid]]
, uint lidx [[thread_index_in_threadgroup]]
, metal::uint3 nwg [[threadgroups_per_grid]]
, uint sg_size [[threads_per_simdgroup]]
, uint sg_num [[simdgroups_per_threadgroup]]
, device type_2& h_1 [[user(fake0)]]
, constant Cfg& cfg [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::float2 w8_ = metal::float2 {};
    type_46 pq2_ = {};
    uint n1_ = 0u;
    uint k1_ = 0u;
    uint nx1_ = 0u;
    uint kx1_ = 0u;
    bool local = {};
    uint sg_lane = lidx & (sg_size - 1u);
    uint sg_id = naga_div(lidx, sg_size);
    uint _e395 = lit_u32_(64u);
    n1_ = metal::min(32u, _e395);
    k1_ = 0u;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e398 = k1_;
        uint _e399 = n1_;
        if (_e398 < _e399) {
        } else {
            break;
        }
        {
        }
        k1_ = k1_ + 1u;
    }
    metal::float4x4 _e404 = zero_mat4x4_();
    float _e409 = lit_f32_(0.0);
    w8_ = metal::float2(metal::determinant(-1.0 * _e404) * _e409);
    uint _e413 = lit_u32_(1u);
    nx1_ = _e413;
    kx1_ = 0u;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        uint _e415 = kx1_;
        uint _e416 = nx1_;
        if (_e415 < _e416) {
        } else {
            break;
        }
        {
            metal::uint4 _e418 = zero_vec4_u32_();
            metal::uint4 _e419 = zero_vec4_u32_();
            metal::uint4 _e420 = zero_vec4_u32_();
            metal::uint4 _e421 = zero_vec4_u32_();
            metal::uint4 _e424 = zero_vec4_u32_();
            metal::uint4 _e425 = zero_vec4_u32_();
            metal::half4 _e428 = zero_vec4_f16_();
            metal::half4 _e429 = zero_vec4_f16_();
            metal::half4 _e432 = zero_vec4_f16_();
            metal::half4 _e435 = zero_vec4_f16_();
            metal::uint4 _e438 = zero_vec4_u32_();
            float _e447 = lit_f32_(0.0);
            pq2_.inner[naga_dot_uint4(metal::select(_e418, (_e419 + metal::max(_e420, _e421)) * naga_div(_e424, _e425), ((_e428 / _e429.xyxy) - metal::sin(_e432)) >= _e435), _e438) & 15u][sg_lane & 1u] = _e447;
            bool _e449 = lit_bool(true);
            if (_e449) {
                break;
            }
        }
        kx1_ = kx1_ + 1u;
    }
    {
        uint r_1 = gid.y * KNOB_BROW;
        uint c_1 = gid.x * KNOB_BCOL;
        uint _e461 = cfg.r_rows;
        if (r_1 < _e461) {
            uint _e467 = cfg.d;
            local = c_1 < _e467;
        } else {
            local = false;
        }
        bool _e470 = local;
        if (_e470) {
            uint _e474 = cfg.d;
            uint _e481 = cfg.d;
            float _e485 = h_1[(r_1 * _e481) + c_1];
            h_1[(r_1 * _e474) + c_1] = _e485;
            return;
        } else {
            return;
        }
    }
}

