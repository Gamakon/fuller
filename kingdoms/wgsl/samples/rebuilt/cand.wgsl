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

    let _e7 = tp.pop;
    let _e9 = tp.n_combinations;
    let _e11 = tp.wrappers;
    if (gid.x >= (_e7 * (_e9 * _e11))) {
        return;
    }
    loop {
        let _e17 = g;
        let _e19 = tp.n_genes;
        if (_e17 < _e19) {
        } else {
            break;
        }
        {
            let _e30 = tp.n_combinations;
            let _e32 = tp.wrappers;
            let _e36 = tp.wrappers;
            let _e39 = combinations[((gid.x % (_e30 * _e32)) / _e36)];
            let _e41 = g;
            if ((((_e39 & 16777215u) >> _e41) & 1u) == 0u) {
                continue;
            }
            let _e52 = tp.n_combinations;
            let _e54 = tp.wrappers;
            let _e58 = tp.n_genes;
            let _e60 = g;
            let _e63 = gene_ok[(((gid.x / (_e52 * _e54)) * _e58) + _e60)];
            if (_e63 == 0u) {
                usable = false;
            } else {
                let _e72 = depth;
                let _e75 = tp.n_combinations;
                let _e77 = tp.wrappers;
                let _e81 = tp.n_genes;
                let _e83 = g;
                let _e86 = gene_depth[(((gid.x / (_e75 * _e77)) * _e81) + _e83)];
                depth = max(_e72, _e86);
            }
        }
        continuing {
            let _e90 = g;
            g = (_e90 + 1u);
        }
    }
    let _e97 = depth;
    tower[gid.x] = _e97;
    let _e106 = tp.typed;
    let _e108 = usable;
    let _e112 = tp.n_combinations;
    let _e114 = tp.wrappers;
    let _e118 = tp.wrappers;
    let _e122 = depth;
    let _e126 = tp.identity_only;
    let _e130 = tp.n_combinations;
    let _e132 = tp.wrappers;
    let _e136 = tp.wrappers;
    if (((((_e106 == 1u) && _e108) && (((gid.x % (_e112 * _e114)) % _e118) != 0u)) && (_e122 >= 2u)) || ((_e126 == 1u) && (((gid.x % (_e130 * _e132)) % _e136) != 0u))) {
        let _e147 = tp.width;
        scores[(gid.x * _e147)] = bitcast<f32>(NAN_BITS);
        return;
    } else {
        return;
    }
}
