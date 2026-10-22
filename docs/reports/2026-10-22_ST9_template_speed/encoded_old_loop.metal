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
struct DispatchTile {
    metal::uint4 origin;
    metal::uint4 grid;
};
constant uint WG_X = 8u;
constant uint WG_Y = 8u;
constant uint KNOB_WCOL = 8u;
constant uint KNOB_WROW = 8u;
constant uint KNOB_BCOL = 1u;
constant uint KNOB_BROW = 1u;
constant uint KNOB_KTILE = 1u;
constant uint KNOB_THREADS = 64u;
constant bool DISPATCH_TILED = false;

float lit_f32_(
    float x
) {
    return x;
}

metal::float3 zero_vec3_f32_(
) {
    return metal::float3 {};
}

uint row_axis(
    metal::uint3 g
) {
    return g.y;
}

uint col_axis(
    metal::uint3 g_1
) {
    return g_1.x;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  metal::uint3 gid_tile [[thread_position_in_grid]]
, metal::uint3 lid [[thread_position_in_threadgroup]]
, metal::uint3 wid_tile [[threadgroup_position_in_grid]]
, uint lidx [[thread_index_in_threadgroup]]
, metal::uint3 nwg_tile [[threadgroups_per_grid]]
, constant TimeUniform& time_data [[user(fake0)]]
, metal::texture2d<float, metal::access::write> output [[user(fake0)]]
, constant VolumeParams& params [[user(fake0)]]
, constant DispatchTile& dispatch_tile [[user(fake0)]]
) {
    float rf0_ = 0.0;
    float rf1_ = 0.0;
    float rf2_ = 0.0;
    float rf3_ = 0.0;
    float rf4_ = 0.0;
    float rf5_ = 0.0;
    float rf6_ = 0.0;
    float rf7_ = 0.0;
    float rf8_ = 0.0;
    float rf9_ = 0.0;
    float rf10_ = 0.0;
    float rf11_ = 0.0;
    float rf12_ = 0.0;
    float rf13_ = 0.0;
    float rf14_ = 0.0;
    metal::float2 w0_ = metal::float2 {};
    metal::float2 w1_ = metal::float2 {};
    metal::float2 w2_ = metal::float2 {};
    metal::float2 w3_ = metal::float2 {};
    metal::float3 C0_ = metal::float3 {};
    metal::float3 C1_ = metal::float3 {};
    metal::float3 C2_ = metal::float3 {};
    metal::float3 C3_ = metal::float3 {};
    metal::float3 C4_ = metal::float3 {};
    metal::float3 C5_ = metal::float3 {};
    metal::float3 C6_ = metal::float3 {};
    metal::float3 C7_ = metal::float3 {};
    uint n1_ = 0u;
    uint k1_ = 0u;
    uint n2_ = 0u;
    uint k2_ = 0u;
    bool local = {};
    metal::uint4 _e7 = dispatch_tile.origin;
    metal::uint3 wid = DISPATCH_TILED ? (wid_tile + _e7.xyz) : wid_tile;
    metal::uint4 _e14 = dispatch_tile.origin;
    metal::uint3 gid = DISPATCH_TILED ? (gid_tile + (_e14.xyz * metal::uint3(8u, 8u, 1u))) : gid_tile;
    metal::uint4 _e26 = dispatch_tile.grid;
    metal::uint3 nwg = DISPATCH_TILED ? _e26.xyz : nwg_tile;
    w1_ = metal::float2(static_cast<float>(metal::uint2(output.get_width(), output.get_height()).x), static_cast<float>(metal::uint2(output.get_width(), output.get_height()).y));
    uint _e101 = col_axis(gid);
    uint _e103 = row_axis(gid);
    w0_ = metal::float2(static_cast<float>(_e101), static_cast<float>(_e103));
    metal::float3 _e106 = zero_vec3_f32_();
    C0_ = _e106;
    {
        rf3_ = 0.0;
        rf1_ = 0.0;
        rf0_ = 0.0;
        {
            float _e112 = time_data.time;
            float _e115 = params.speed;
            rf4_ = _e112 * _e115;
            rf2_ = 0.0;
            n1_ = 50u;
            k1_ = 4294967295u;
            uint2 loop_bound = uint2(4294967295u);
            while(true) {
                if (metal::all(loop_bound == uint2(0u))) { break; }
                loop_bound -= uint2(loop_bound.y == 0u, 1u);
                uint _e120 = k1_;
                k1_ = _e120 + 1u;
                uint _e123 = k1_;
                uint _e124 = n1_;
                if (_e123 >= _e124) {
                    break;
                }
                float _e126 = rf3_;
                rf3_ = _e126 + 1.0;
                float _e129 = rf0_;
                float _e130 = rf1_;
                rf0_ = _e129 + (_e130 * 0.6);
                float _e134 = rf0_;
                metal::float2 _e135 = w0_;
                metal::float2 _e136 = w1_;
                float _e142 = params.zoom;
                float _e146 = w1_.y;
                C1_ = _e134 * metal::normalize(metal::float3((_e135 - (0.5 * _e136)) / metal::float2(_e142), _e146));
                {
                    metal::float3 _e150 = C1_;
                    float _e153 = C1_.z;
                    float _e154 = rf4_;
                    C1_ = metal::float3(_e150.xy, _e153 + _e154);
                    float _e158 = C1_.z;
                    rf10_ = _e158 * 1.1;
                    float _e161 = rf10_;
                    rf8_ = metal::cos(_e161);
                    {
                        float _e163 = rf10_;
                        rf9_ = metal::sin(_e163);
                        float _e165 = rf8_;
                        float _e166 = rf9_;
                        float _e169 = rf9_;
                        float _e170 = rf8_;
                        metal::float3 _e173 = C1_;
                        w2_ = metal::float2x2(metal::float2(_e165, -(_e166)), metal::float2(_e169, _e170)) * _e173.xy;
                        metal::float2 _e176 = w2_;
                        float _e178 = C1_.z;
                        C1_ = metal::float3(_e176, _e178);
                        {
                            metal::float3 _e180 = C1_;
                            float _e181 = rf4_;
                            C2_ = _e180 + metal::float3(_e181 * 0.1);
                            rf6_ = 0.0;
                            rf7_ = 1.0;
                            {
                                n2_ = 7u;
                                k2_ = 4294967295u;
                                uint2 loop_bound_1 = uint2(4294967295u);
                                while(true) {
                                    if (metal::all(loop_bound_1 == uint2(0u))) { break; }
                                    loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                                    uint _e190 = k2_;
                                    k2_ = _e190 + 1u;
                                    uint _e193 = k2_;
                                    uint _e194 = n2_;
                                    if (_e193 >= _e194) {
                                        break;
                                    }
                                    metal::float3 _e196 = C2_;
                                    float _e199 = C2_.z;
                                    w3_ = _e196.xy + metal::float2(_e199 * 0.5);
                                    float _e205 = w3_.x;
                                    float _e209 = w3_.y;
                                    float _e216 = w3_.y;
                                    float _e220 = w3_.x;
                                    rf11_ = metal::sin((_e205 * 3.0) + metal::sin(_e209 * 2.7)) * metal::cos((_e216 * 1.1) + metal::cos(_e220 * 2.3));
                                    float _e227 = rf6_;
                                    float _e228 = rf11_;
                                    float _e229 = rf7_;
                                    rf6_ = _e227 + (_e228 * _e229);
                                    {
                                        metal::float3 _e232 = C2_;
                                        C2_ = 2.0 * _e232;
                                        float _e235 = rf7_;
                                        rf7_ = _e235 * 0.5;
                                    }
                                }
                                float _e238 = rf6_;
                                rf2_ = _e238;
                                metal::float3 _e239 = C1_;
                                float _e244 = rf2_;
                                rf1_ = (metal::length(_e239.xy) - 1.0) + (_e244 * 0.3);
                                {
                                    float _e249 = C1_.z;
                                    rf12_ = metal::fmod(_e249, 4.0);
                                    metal::float3 _e252 = C1_;
                                    float _e254 = rf12_;
                                    float _e257 = rf12_;
                                    float _e258 = rf12_;
                                    float _e263 = lit_f32_(0.5);
                                    C1_ = metal::float3(_e252.xy, ((_e258 >= 0.0) ? _e257 : (_e254 + 4.0)) - (4.0 * _e263));
                                    float _e268 = rf1_;
                                    rf1_ = metal::abs(_e268) + 0.01;
                                    {
                                        metal::float3 _e272 = C1_;
                                        float _e274 = rf2_;
                                        rf5_ = metal::max(metal::length(_e272.xy + metal::float2(_e274)) * 0.8, 0.005);
                                        metal::float3 _e282 = C1_;
                                        float _e284 = rf2_;
                                        rf13_ = metal::max(metal::length(_e282.xz + metal::float2(_e284)) * 0.8, 0.005);
                                        float _e294 = params.color1_r;
                                        float _e297 = params.color1_g;
                                        float _e301 = params.color1_b;
                                        float _e303 = rf5_;
                                        C3_ = metal::float3(metal::float2(_e294, _e297), _e301) / metal::float3(_e303);
                                        {
                                            metal::float3 _e306 = C3_;
                                            float _e309 = params.color2_r;
                                            float _e312 = params.color2_g;
                                            float _e316 = params.color2_b;
                                            float _e318 = rf13_;
                                            C3_ = _e306 + (metal::float3(metal::float2(_e309, _e312), _e316) / metal::float3(_e318));
                                            float _e322 = rf2_;
                                            float _e325 = rf4_;
                                            rf14_ = 0.5 + (0.5 * metal::sin((_e322 * 1.0) + _e325));
                                            metal::float3 _e332 = C3_;
                                            float _e333 = rf14_;
                                            float _e336 = params.color3_r;
                                            float _e339 = params.color3_g;
                                            float _e343 = params.color3_b;
                                            C3_ = _e332 + (_e333 * metal::float3(metal::float2(_e336, _e339), _e343));
                                            metal::float3 _e347 = C0_;
                                            metal::float3 _e348 = C3_;
                                            float _e349 = rf1_;
                                            C0_ = _e347 + (0.12 * (_e348 / metal::float3(_e349)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    {
        float _e357 = params.intensity;
        metal::float3 _e358 = C0_;
        C4_ = metal::tanh(_e357 * _e358);
        float _e362 = lit_f32_(0.076);
        float _e368 = lit_f32_(0.90834);
        float _e374 = lit_f32_(0.01566);
        metal::float3 _e380 = C4_;
        C5_ = metal::float3x3(metal::float3(metal::float2(0.59719, _e362), 0.0284), metal::float3(metal::float2(0.35458, _e368), 0.13383), metal::float3(metal::float2(0.04823, _e374), 0.83777)) * _e380;
        metal::float3 _e382 = C5_;
        metal::float3 _e383 = C5_;
        float _e385 = lit_f32_(0.0245786);
        float _e390 = lit_f32_(0.000090537);
        C6_ = (_e382 * (_e383 + metal::float3(_e385))) - metal::float3(_e390);
        {
            metal::float3 _e393 = C5_;
            metal::float3 _e394 = C5_;
            float _e398 = lit_f32_(0.432951);
            float _e403 = lit_f32_(0.238081);
            C7_ = (_e393 * ((0.983729 * _e394) + metal::float3(_e398))) + metal::float3(_e403);
            float _e407 = lit_f32_(-0.10208);
            float _e413 = lit_f32_(1.10813);
            float _e419 = lit_f32_(-0.00605);
            metal::float3 _e425 = C6_;
            metal::float3 _e426 = C7_;
            C4_ = metal::float3x3(metal::float3(metal::float2(1.60475, _e407), -0.00327), metal::float3(metal::float2(-0.53108, _e413), -0.07276), metal::float3(metal::float2(-0.07367, _e419), 1.07602)) * (_e425 / _e426);
        }
    }
    metal::float3 _e429 = C4_;
    metal::float3 _e430 = zero_vec3_f32_();
    C4_ = metal::max(_e429, _e430);
    metal::float3 _e432 = C4_;
    float _e435 = params.gamma;
    C4_ = metal::pow(_e432, metal::float3(1.0 / _e435));
    {
        uint _e440 = row_axis(gid);
        uint r = _e440 * KNOB_BROW;
        uint _e443 = col_axis(gid);
        uint c = _e443 * KNOB_BCOL;
        if (r < metal::uint2(output.get_width(), output.get_height()).y) {
            local = c < metal::uint2(output.get_width(), output.get_height()).x;
        } else {
            local = false;
        }
        bool _e457 = local;
        if (_e457) {
            metal::float3 _e460 = C4_;
            output.write(metal::float4(_e460, 1.0), metal::uint2(metal::uint2(c, r)));
            return;
        } else {
            return;
        }
    }
}
