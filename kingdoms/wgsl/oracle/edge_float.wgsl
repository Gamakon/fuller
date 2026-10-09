// Float edges: division by zero, sqrt and log of non-positives, NaN
// through min, max and select, round to nearest even, sign of -0.0,
// overflow to infinity. One lane per case, computed by hand from IEEE 754
// and the WGSL spec; the sign and payload of a NaN are not specified, so
// the expectation says only "nan" for those lanes.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 8u; i = i + 1u) {
        switch (i) {
            case 0u: { out[i] = a[i] / b[i]; }
            case 1u: { out[i] = sqrt(a[i]); }
            case 2u: { out[i] = log(a[i]); }
            case 3u: { out[i] = min(a[i], b[i]); }
            case 4u: { out[i] = max(a[i], b[i]); }
            case 5u: { out[i] = select(a[i], b[i], a[i] < b[i]); }
            case 6u: { out[i] = round(a[i]) + sign(b[i]); }
            default: { out[i] = a[i] * b[i]; }
        }
    }
}
