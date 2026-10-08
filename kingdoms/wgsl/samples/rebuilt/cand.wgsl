struct Params {
    pop: u32,
    n_genes: u32,
    n_combinations: u32,
    wrappers: u32,
    typed: u32,
    width: u32,
    identity_only: u32,
    pad1_: u32,
}

const NAN_BITS: u32 = 2143289344u;

@group(0) @binding(0) 
var<uniform> tp: Params;
@group(0) @binding(1) 
var<storage> gene_depth: array<u32>;
@group(0) @binding(2) 
var<storage> gene_ok: array<u32>;
@group(0) @binding(3) 
var<storage> combinations: array<u32>;
@group(0) @binding(4) 
var<storage, read_write> scores: array<f32>;
@group(0) @binding(5) 
var<storage, read_write> tower: array<u32>;

@compute @workgroup_size(64, 1, 1) 
fn type_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var depth: u32 = 0u;
    var usable: bool = true;
    var g: u32 = 0u;

    let cid = gid.x;
    let _e7 = tp.n_combinations;
    let _e9 = tp.wrappers;
    let per = (_e7 * _e9);
    let _e13 = tp.pop;
    if (cid >= (_e13 * per)) {
        return;
    }
    let row = (cid / per);
    let rem = (cid % per);
    let _e20 = tp.wrappers;
    let combo = (rem / _e20);
    let _e24 = tp.wrappers;
    let w = (rem % _e24);
    let _e29 = combinations[combo];
    let genes = (_e29 & 16777215u);
    loop {
        let _e33 = g;
        let _e35 = tp.n_genes;
        if (_e33 < _e35) {
        } else {
            break;
        }
        {
            let _e40 = g;
            if (((genes >> _e40) & 1u) == 0u) {
                continue;
            }
            let _e47 = tp.n_genes;
            let _e49 = g;
            let id = ((row * _e47) + _e49);
            let _e54 = gene_ok[id];
            if (_e54 == 0u) {
                usable = false;
            } else {
                let _e60 = depth;
                let _e62 = gene_depth[id];
                depth = max(_e60, _e62);
            }
        }
        continuing {
            let _e66 = g;
            g = (_e66 + 1u);
        }
    }
    let _e71 = depth;
    tower[cid] = _e71;
    let _e79 = tp.typed;
    let _e81 = usable;
    let _e85 = depth;
    let _e89 = tp.identity_only;
    if (((((_e79 == 1u) && _e81) && (w != 0u)) && (_e85 >= 2u)) || ((_e89 == 1u) && (w != 0u))) {
        let _e98 = tp.width;
        scores[(cid * _e98)] = bitcast<f32>(NAN_BITS);
        return;
    } else {
        return;
    }
}
