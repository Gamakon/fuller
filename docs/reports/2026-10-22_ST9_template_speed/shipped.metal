// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct TimeUniform {
    float time;
    float delta;
    uint frame;
    uint _padding;
};
struct VolumeParams {
    float speed;
    float intensity;
    float color1_r;
    float color1_g;
    float color1_b;
    float color2_r;
    float color2_g;
    float color2_b;
    float color3_r;
    float color3_g;
    float color3_b;
    float gamma;
    float zoom;
    float _padding1_;
    float _padding2_;
    float _padding3_;
};
typedef metal::float3 v3_;
typedef metal::float2 v2_;
typedef metal::float4 v4_;
typedef metal::float2x2 m2_;

float n(
    v2_ p_1
) {
    return metal::sin((p_1.x * 3.0) + metal::sin(p_1.y * 2.7)) * metal::cos((p_1.y * 1.1) + metal::cos(p_1.x * 2.3));
}

float f(
    v3_ p_2
) {
    float v = 0.0;
    float a = 1.0;
    v3_ pos = {};
    int i_1 = 0;
    pos = p_2;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e31 = i_1;
            i_1 = as_type<int>(as_type<uint>(_e31) + as_type<uint>(1));
        }
        loop_init = false;
        int _e8 = i_1;
        if (_e8 < 7) {
        } else {
            break;
        }
        {
            float _e11 = v;
            v3_ _e12 = pos;
            float _e15 = pos.z;
            float _e20 = n(_e12.xy + metal::float2(_e15 * 0.5));
            float _e21 = a;
            v = _e11 + (_e20 * _e21);
            v3_ _e24 = pos;
            pos = _e24 * 2.0;
            float _e27 = a;
            a = _e27 * 0.5;
        }
    }
    float _e33 = v;
    return _e33;
}

m2_ mm2_(
    float a_1
) {
    float c_1 = metal::cos(a_1);
    float s = metal::sin(a_1);
    return metal::float2x2(metal::float2(c_1, -(s)), metal::float2(s, c_1));
}

v3_ aces_tonemap(
    v3_ color
) {
    metal::float3 v_1 = {};
    metal::float3 a_2 = {};
    metal::float3 b = {};
    v_1 = metal::float3x3(metal::float3(0.59719, 0.076, 0.0284), metal::float3(0.35458, 0.90834, 0.13383), metal::float3(0.04823, 0.01566, 0.83777)) * color;
    metal::float3 _e29 = v_1;
    metal::float3 _e30 = v_1;
    a_2 = (_e29 * (_e30 + metal::float3(0.0245786))) - metal::float3(0.000090537);
    metal::float3 _e39 = v_1;
    metal::float3 _e40 = v_1;
    b = (_e39 * ((0.983729 * _e40) + metal::float3(0.432951))) + metal::float3(0.238081);
    metal::float3 _e51 = a_2;
    metal::float3 _e52 = b;
    return metal::float3x3(metal::float3(1.60475, -0.10208, -0.00327), metal::float3(-0.53108, 1.10813, -0.07276), metal::float3(-0.07367, -0.00605, 1.07602)) * (_e51 / _e52);
}

float pmap(
    float value,
    float domain
) {
    float r = metal::fmod(value, domain);
    return ((r >= 0.0) ? r : (r + domain)) - (domain * 0.5);
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  metal::uint3 gid [[thread_position_in_grid]]
, constant TimeUniform& time_data [[user(fake0)]]
, metal::texture2d<float, metal::access::write> output [[user(fake0)]]
, constant VolumeParams& params [[user(fake0)]]
) {
    v2_ R = {};
    bool local = {};
    metal::float3 o = metal::float3(0.0);
    float i = 0.0;
    float d = 0.0;
    float z = 0.0;
    float N = 0.0;
    int j = 0;
    metal::float3 p = {};
    v3_ c = {};
    metal::float3 result = {};
    metal::uint2 dims = metal::uint2(output.get_width(), output.get_height());
    R = static_cast<metal::float2>(dims);
    if (!((gid.x >= dims.x))) {
        local = gid.y >= dims.y;
    } else {
        local = true;
    }
    bool _e15 = local;
    if (_e15) {
        return;
    }
    v2_ C = v2_(static_cast<float>(gid.x), static_cast<float>(gid.y));
    v2_ r_1 = R;
    float _e34 = time_data.time;
    float _e37 = params.speed;
    float t = _e34 * _e37;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            int _e186 = j;
            j = as_type<int>(as_type<uint>(_e186) + as_type<uint>(1));
        }
        loop_init_1 = false;
        int _e43 = j;
        if (_e43 < 50) {
        } else {
            break;
        }
        {
            float _e46 = i;
            i = _e46 + 1.0;
            float _e49 = z;
            float _e50 = d;
            z = _e49 + (_e50 * 0.6);
            float _e54 = z;
            float _e61 = params.zoom;
            p = _e54 * metal::normalize(v3_((C - (0.5 * r_1.xy)) / metal::float2(_e61), r_1.y));
            float _e70 = p.z;
            p.z = _e70 + t;
            float _e73 = p.z;
            m2_ _e76 = mm2_(_e73 * 1.1);
            metal::float3 _e77 = p;
            metal::float2 rotated_xy = _e76 * _e77.xy;
            p.x = rotated_xy.x;
            p.y = rotated_xy.y;
            metal::float3 _e84 = p;
            float _e89 = f(_e84 + metal::float3(t * 0.1));
            N = _e89;
            metal::float3 _e90 = p;
            float _e95 = N;
            d = (metal::length(_e90.xy) - 1.0) + (_e95 * 0.3);
            float _e101 = p.z;
            float _e103 = pmap(_e101, 4.0);
            p.z = _e103;
            float _e104 = d;
            d = metal::abs(_e104) + 0.01;
            metal::float3 _e108 = p;
            float _e110 = N;
            float den1_ = metal::max(metal::length(_e108.xy + metal::float2(_e110)) * 0.8, 0.005);
            metal::float3 _e118 = p;
            float _e120 = N;
            float den2_ = metal::max(metal::length(_e118.xz + metal::float2(_e120)) * 0.8, 0.005);
            float _e130 = params.color1_r;
            float _e133 = params.color1_g;
            float _e136 = params.color1_b;
            c = v3_(_e130, _e133, _e136) / metal::float3(den1_);
            v3_ _e141 = c;
            float _e144 = params.color2_r;
            float _e147 = params.color2_g;
            float _e150 = params.color2_b;
            c = _e141 + (v3_(_e144, _e147, _e150) / metal::float3(den2_));
            v3_ _e155 = c;
            float _e158 = params.color3_r;
            float _e161 = params.color3_g;
            float _e164 = params.color3_b;
            float _e166 = N;
            c = _e155 + (v3_(_e158, _e161, _e164) * (0.5 + (0.5 * metal::sin((_e166 * 1.0) + t))));
            metal::float3 _e177 = o;
            v3_ _e178 = c;
            float _e179 = d;
            o = _e177 + ((_e178 / metal::float3(_e179)) * 0.12);
        }
    }
    metal::float3 _e188 = o;
    float _e191 = params.intensity;
    result = metal::tanh(_e188 * _e191);
    metal::float3 _e195 = result;
    v3_ _e196 = aces_tonemap(_e195);
    result = _e196;
    metal::float3 _e197 = result;
    result = metal::max(_e197, metal::float3(0.0));
    metal::float3 _e201 = result;
    float _e204 = params.gamma;
    result = metal::pow(_e201, metal::float3(1.0 / _e204));
    metal::float3 _e211 = result;
    output.write(v4_(_e211, 1.0), metal::uint2(gid.xy));
    return;
}
