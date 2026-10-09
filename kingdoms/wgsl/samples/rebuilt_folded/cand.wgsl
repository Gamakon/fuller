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
    let def_a = (w != 0u);
    let _e31 = combinations[combo];
    let genes = (_e31 & 16777215u);
    loop {
        let _e35 = g;
        let _e37 = tp.n_genes;
        if (_e35 < _e37) {
        } else {
            break;
        }
        {
            let _e42 = g;
            if (((genes >> _e42) & 1u) == 0u) {
                continue;
            }
            let _e49 = tp.n_genes;
            let _e51 = g;
            let id = ((row * _e49) + _e51);
            let _e56 = gene_ok[id];
            if (_e56 == 0u) {
                usable = false;
            } else {
                let _e62 = depth;
                let _e64 = gene_depth[id];
                depth = max(_e62, _e64);
            }
        }
        continuing {
            let _e68 = g;
            g = (_e68 + 1u);
        }
    }
    let _e73 = depth;
    tower[cid] = _e73;
    let _e80 = tp.typed;
    let _e82 = usable;
    let _e85 = depth;
    let _e89 = tp.identity_only;
    if (((((_e80 == 1u) && _e82) && def_a) && (_e85 >= 2u)) || ((_e89 == 1u) && def_a)) {
        let _e97 = tp.width;
        scores[(cid * _e97)] = bitcast<f32>(NAN_BITS);
        return;
    } else {
        return;
    }
}
