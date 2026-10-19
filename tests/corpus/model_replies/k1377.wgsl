@group(0) @binding(0) var<storage, read_write> buffer_0_global: array<u32>;
@group(0) @binding(1) var<storage, read_write> buffer_1_global: array<f32>;
@group(0) @binding(2) var<storage, read_write> buffer_2_global: array<f32>;
@group(0) @binding(3) var<storage, read_write> buffer_3_global: array<f32>;
@group(0) @binding(4) var<storage, read_write> buffer_4_global: array<f32>;
@group(0) @binding(5) var<storage, read_write> buffer_5_global: array<u32>;
@group(0) @binding(6) var<storage, read_write> buffer_6_global: array<u32>;
@group(0) @binding(7) var<storage, read_write> buffer_7_global: array<f32>;
@group(0) @binding(8) var<storage, read_write> info: array<u32>;

const ROWS: u32 = 4u;

fn fmix32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 2246822507u;
    h = h ^ (h >> 13u);
    h = h * 3266489909u;
    h = h ^ (h >> 16u);
    return h;
}

fn safe_tanh(x: f32) -> f32 {
    if x > 43.0 {
        return 1.0;
    }
    return tanh(x);
}

@compute @workgroup_size(64, 1, 1)
fn generated_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = (global_id.z * num_workgroups.x * 64u * num_workgroups.y) + (global_id.y * num_workgroups.x * 64u) + global_id.x;
    let off = buffer_6_global[0u];
    let n = buffer_5_global[0u];
    let key_lo = buffer_5_global[1u];
    let key_hi = buffer_5_global[2u];
    let noise_lo = buffer_6_global[1u];
    let noise_hi = buffer_6_global[2u];
    let step_t = buffer_6_global[3u];
    let loc = buffer_7_global[0u];
    let scale = buffer_7_global[1u];
    let leak = buffer_7_global[2u];
    let sigma = buffer_7_global[4u];
    let one_m_leak = 1.0 - leak;
    let noise_k = noise_hi ^ step_t;

    let base = id * ROWS;
    for (var k = 0u; k < ROWS; k += 1u) {
        let r = base + k + off;
        if r < n {
            let hr = fmix32(key_lo ^ r);
            let rs = r * 10u;
            var acc_z = 0.0;
            var comp = 0.0;
            var acc_x = 0.0;
            for (var s = 0u; s < 10u; s += 1u) {
                let src = buffer_0_global[rs + s];
                let xv = buffer_1_global[src];
                let hh = fmix32(hr ^ (key_hi ^ s));
                let w = buffer_2_global[hh >> 22u];
                let term = w * xv;
                let y = term - comp;
                let t = acc_z + y;
                comp = (t - acc_z) - y;
                acc_z = t;
                acc_x = acc_x + xv;
            }
            let hn = fmix32(fmix32(noise_lo ^ r) ^ noise_k);
            let z = buffer_2_global[hn >> 22u];
            let pre = scale * acc_z + loc * acc_x + buffer_3_global[r] + sigma * z;
            buffer_4_global[r] = one_m_leak * buffer_1_global[r] + leak * safe_tanh(pre);
        }
    }
}
