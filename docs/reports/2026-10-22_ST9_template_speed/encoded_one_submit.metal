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
    uint kc1_ = 0u;
    uint n2_ = 0u;
    uint k2_ = 0u;
    uint kc2_ = 0u;
    bool local = {};
    metal::uint4 _e7 = dispatch_tile.origin;
    metal::uint3 wid = DISPATCH_TILED ? (wid_tile + _e7.xyz) : wid_tile;
    metal::uint4 _e14 = dispatch_tile.origin;
    metal::uint3 gid = DISPATCH_TILED ? (gid_tile + (_e14.xyz * metal::uint3(8u, 8u, 1u))) : gid_tile;
    metal::uint4 _e26 = dispatch_tile.grid;
    metal::uint3 nwg = DISPATCH_TILED ? _e26.xyz : nwg_tile;
    w1_ = metal::float2(static_cast<float>(metal::uint2(output.get_width(), output.get_height()).x), static_cast<float>(metal::uint2(output.get_width(), output.get_height()).y));
    uint _e105 = col_axis(gid);
    uint _e107 = row_axis(gid);
    w0_ = metal::float2(static_cast<float>(_e105), static_cast<float>(_e107));
    metal::float3 _e110 = zero_vec3_f32_();
    C0_ = _e110;
    {
        rf3_ = 0.0;
        rf1_ = 0.0;
        rf0_ = 0.0;
        {
            float _e116 = time_data.time;
            float _e119 = params.speed;
            rf4_ = _e116 * _e119;
            rf2_ = 0.0;
            n1_ = 50u;
            kc1_ = 0u;
            uint2 loop_bound = uint2(4294967295u);
            while(true) {
                if (metal::all(loop_bound == uint2(0u))) { break; }
                loop_bound -= uint2(loop_bound.y == 0u, 1u);
                uint _e124 = kc1_;
                k1_ = _e124;
                uint _e125 = k1_;
                uint _e126 = n1_;
                if (_e125 >= _e126) {
                    break;
                }
                uint _e128 = k1_;
                kc1_ = _e128 + 1u;
                float _e131 = rf3_;
                rf3_ = _e131 + 1.0;
                float _e134 = rf0_;
                float _e135 = rf1_;
                rf0_ = _e134 + (_e135 * 0.6);
                float _e139 = rf0_;
                metal::float2 _e140 = w0_;
                metal::float2 _e141 = w1_;
                float _e147 = params.zoom;
                float _e151 = w1_.y;
                C1_ = _e139 * metal::normalize(metal::float3((_e140 - (0.5 * _e141)) / metal::float2(_e147), _e151));
                {
                    metal::float3 _e155 = C1_;
                    float _e158 = C1_.z;
                    float _e159 = rf4_;
                    C1_ = metal::float3(_e155.xy, _e158 + _e159);
                    float _e163 = C1_.z;
                    rf10_ = _e163 * 1.1;
                    float _e166 = rf10_;
                    rf8_ = metal::cos(_e166);
                    {
                        float _e168 = rf10_;
                        rf9_ = metal::sin(_e168);
                        float _e170 = rf8_;
                        float _e171 = rf9_;
                        float _e174 = rf9_;
                        float _e175 = rf8_;
                        metal::float3 _e178 = C1_;
                        w2_ = metal::float2x2(metal::float2(_e170, -(_e171)), metal::float2(_e174, _e175)) * _e178.xy;
                        metal::float2 _e181 = w2_;
                        float _e183 = C1_.z;
                        C1_ = metal::float3(_e181, _e183);
                        {
                            metal::float3 _e185 = C1_;
                            float _e186 = rf4_;
                            C2_ = _e185 + metal::float3(_e186 * 0.1);
                            rf6_ = 0.0;
                            rf7_ = 1.0;
                            {
                                n2_ = 7u;
                                kc2_ = 0u;
                                uint2 loop_bound_1 = uint2(4294967295u);
                                while(true) {
                                    if (metal::all(loop_bound_1 == uint2(0u))) { break; }
                                    loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                                    uint _e195 = kc2_;
                                    k2_ = _e195;
                                    uint _e196 = k2_;
                                    uint _e197 = n2_;
                                    if (_e196 >= _e197) {
                                        break;
                                    }
                                    uint _e199 = k2_;
                                    kc2_ = _e199 + 1u;
                                    metal::float3 _e202 = C2_;
                                    float _e205 = C2_.z;
                                    w3_ = _e202.xy + metal::float2(_e205 * 0.5);
                                    float _e211 = w3_.x;
                                    float _e215 = w3_.y;
                                    float _e222 = w3_.y;
                                    float _e226 = w3_.x;
                                    rf11_ = metal::sin((_e211 * 3.0) + metal::sin(_e215 * 2.7)) * metal::cos((_e222 * 1.1) + metal::cos(_e226 * 2.3));
                                    float _e233 = rf6_;
                                    float _e234 = rf11_;
                                    float _e235 = rf7_;
                                    rf6_ = _e233 + (_e234 * _e235);
                                    {
                                        metal::float3 _e238 = C2_;
                                        C2_ = 2.0 * _e238;
                                        float _e241 = rf7_;
                                        rf7_ = _e241 * 0.5;
                                    }
                                }
                                float _e244 = rf6_;
                                rf2_ = _e244;
                                metal::float3 _e245 = C1_;
                                float _e250 = rf2_;
                                rf1_ = (metal::length(_e245.xy) - 1.0) + (_e250 * 0.3);
                                {
                                    float _e255 = C1_.z;
                                    rf12_ = metal::fmod(_e255, 4.0);
                                    metal::float3 _e258 = C1_;
                                    float _e260 = rf12_;
                                    float _e263 = rf12_;
                                    float _e264 = rf12_;
                                    float _e269 = lit_f32_(0.5);
                                    C1_ = metal::float3(_e258.xy, ((_e264 >= 0.0) ? _e263 : (_e260 + 4.0)) - (4.0 * _e269));
                                    float _e274 = rf1_;
                                    rf1_ = metal::abs(_e274) + 0.01;
                                    {
                                        metal::float3 _e278 = C1_;
                                        float _e280 = rf2_;
                                        rf5_ = metal::max(metal::length(_e278.xy + metal::float2(_e280)) * 0.8, 0.005);
                                        metal::float3 _e288 = C1_;
                                        float _e290 = rf2_;
                                        rf13_ = metal::max(metal::length(_e288.xz + metal::float2(_e290)) * 0.8, 0.005);
                                        float _e300 = params.color1_r;
                                        float _e303 = params.color1_g;
                                        float _e307 = params.color1_b;
                                        float _e309 = rf5_;
                                        C3_ = metal::float3(metal::float2(_e300, _e303), _e307) / metal::float3(_e309);
                                        {
                                            metal::float3 _e312 = C3_;
                                            float _e315 = params.color2_r;
                                            float _e318 = params.color2_g;
                                            float _e322 = params.color2_b;
                                            float _e324 = rf13_;
                                            C3_ = _e312 + (metal::float3(metal::float2(_e315, _e318), _e322) / metal::float3(_e324));
                                            float _e328 = rf2_;
                                            float _e331 = rf4_;
                                            rf14_ = 0.5 + (0.5 * metal::sin((_e328 * 1.0) + _e331));
                                            metal::float3 _e338 = C3_;
                                            float _e339 = rf14_;
                                            float _e342 = params.color3_r;
                                            float _e345 = params.color3_g;
                                            float _e349 = params.color3_b;
                                            C3_ = _e338 + (_e339 * metal::float3(metal::float2(_e342, _e345), _e349));
                                            metal::float3 _e353 = C0_;
                                            metal::float3 _e354 = C3_;
                                            float _e355 = rf1_;
                                            C0_ = _e353 + (0.12 * (_e354 / metal::float3(_e355)));
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
        float _e363 = params.intensity;
        metal::float3 _e364 = C0_;
        C4_ = metal::tanh(_e363 * _e364);
        float _e368 = lit_f32_(0.076);
        float _e374 = lit_f32_(0.90834);
        float _e380 = lit_f32_(0.01566);
        metal::float3 _e386 = C4_;
        C5_ = metal::float3x3(metal::float3(metal::float2(0.59719, _e368), 0.0284), metal::float3(metal::float2(0.35458, _e374), 0.13383), metal::float3(metal::float2(0.04823, _e380), 0.83777)) * _e386;
        metal::float3 _e388 = C5_;
        metal::float3 _e389 = C5_;
        float _e391 = lit_f32_(0.0245786);
        float _e396 = lit_f32_(0.000090537);
        C6_ = (_e388 * (_e389 + metal::float3(_e391))) - metal::float3(_e396);
        {
            metal::float3 _e399 = C5_;
            metal::float3 _e400 = C5_;
            float _e404 = lit_f32_(0.432951);
            float _e409 = lit_f32_(0.238081);
            C7_ = (_e399 * ((0.983729 * _e400) + metal::float3(_e404))) + metal::float3(_e409);
            float _e413 = lit_f32_(-0.10208);
            float _e419 = lit_f32_(1.10813);
            float _e425 = lit_f32_(-0.00605);
            metal::float3 _e431 = C6_;
            metal::float3 _e432 = C7_;
            C4_ = metal::float3x3(metal::float3(metal::float2(1.60475, _e413), -0.00327), metal::float3(metal::float2(-0.53108, _e419), -0.07276), metal::float3(metal::float2(-0.07367, _e425), 1.07602)) * (_e431 / _e432);
        }
    }
    metal::float3 _e435 = C4_;
    metal::float3 _e436 = zero_vec3_f32_();
    C4_ = metal::max(_e435, _e436);
    metal::float3 _e438 = C4_;
    float _e441 = params.gamma;
    C4_ = metal::pow(_e438, metal::float3(1.0 / _e441));
    {
        uint _e446 = row_axis(gid);
        uint r = _e446 * KNOB_BROW;
        uint _e449 = col_axis(gid);
        uint c = _e449 * KNOB_BCOL;
        if (r < metal::uint2(output.get_width(), output.get_height()).y) {
            local = c < metal::uint2(output.get_width(), output.get_height()).x;
        } else {
            local = false;
        }
        bool _e463 = local;
        if (_e463) {
            metal::float3 _e466 = C4_;
            output.write(metal::float4(_e466, 1.0), metal::uint2(metal::uint2(c, r)));
            return;
        } else {
            return;
        }
    }
}
