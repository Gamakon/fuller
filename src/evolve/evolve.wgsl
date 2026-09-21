// The evolution engine's kernels. Step 1: population initialisation. Every
// draw is mix64 over (seed, stream, generation, row, slot) — mix64.wgsl, which
// is concatenated in front of this file — the SAME arithmetic as evolve/mod.rs,
// the CPU reference each kernel is checked against bit for bit.

struct Params {
    pop: u32,
    n_genes: u32,
    head: u32,
    tail: u32,
    n_rnc: u32,
    n_functions: u32,
    n_terminals: u32,
    seed: u32,
    generation: u32,
    rnc_lo: i32,
    rnc_span: u32,
    n_wrappers: u32,
    first_row: u32,
    n_rows: u32,
    // The virtual head, already resolved by the host (never 0 here): only the first
    // `vhead` head positions may hold a function.
    vhead: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sample_functions: array<u32>;
@group(0) @binding(2) var<storage, read> sample_terminals: array<u32>;
@group(0) @binding(3) var<storage, read_write> genome: array<u32>;
@group(0) @binding(4) var<storage, read_write> rnc: array<f32>;
@group(0) @binding(5) var<storage, read_write> wrapper_id: array<u32>;

const STREAM_KIND: u32 = 1u;
const STREAM_SYMBOL: u32 = 2u;
const STREAM_DC: u32 = 3u;
const STREAM_RNC: u32 = 4u;
const STREAM_WRAPPER: u32 = 5u;

// One draw: mix64_3(domain, generation, row, slot), the domain carrying the
// fit's seed and the stream. The same call as evolve/mod.rs :: draw.
const DOMAIN_LO: u32 = 0x79C19B14u;   // SHA-256("fuller_evolve_v1")[:8], big-endian,
const DOMAIN_HI: u32 = 0x6B86E41Au;   // = 0x6B86E41A79C19B14

fn domain(stream: u32) -> vec2<u32> {
    return vec2<u32>(DOMAIN_LO ^ params.seed, DOMAIN_HI ^ stream);
}

fn key(v: u32) -> vec2<u32> {
    return vec2<u32>(v, 0u);
}

// A draw on 0 .. n-1 (Lemire multiply-shift over the 64-bit hash).
fn below(row: u32, slot: u32, stream: u32, n: u32) -> u32 {
    return mix64_modp(domain(stream), key(params.generation), key(row), key(slot), n);
}

// A fair coin: the hash's top bit.
fn coin(row: u32, slot: u32, stream: u32) -> bool {
    return (mix64_3p(domain(stream), key(params.generation), key(row), key(slot)).y >> 31u) == 1u;
}

fn terminal(row: u32, slot: u32) -> u32 {
    return sample_terminals[below(row, slot, STREAM_SYMBOL, params.n_terminals)];
}

@compute @workgroup_size(64)
fn init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n_rows) {
        return;
    }
    let row = params.first_row + gid.x;
    let width = params.head + 2u * params.tail;
    for (var g = 0u; g < params.n_genes; g = g + 1u) {
        let base = (row * params.n_genes + g) * width;
        for (var p = 0u; p < width; p = p + 1u) {
            let slot = g * width + p;
            if (p < params.vhead) {
                // geppy: a head slot is a function or a terminal, equal odds.
                if (coin(row, slot, STREAM_KIND)) {
                    genome[base + p] = sample_functions[below(row, slot, STREAM_SYMBOL, params.n_functions)];
                } else {
                    genome[base + p] = terminal(row, slot);
                }
            } else if (p < params.head + params.tail) {
                genome[base + p] = terminal(row, slot);
            } else {
                genome[base + p] = below(row, slot, STREAM_DC, params.n_rnc);
            }
        }
        let rbase = (row * params.n_genes + g) * params.n_rnc;
        for (var k = 0u; k < params.n_rnc; k = k + 1u) {
            let v = params.rnc_lo + i32(below(row, g * params.n_rnc + k, STREAM_RNC, params.rnc_span));
            rnc[rbase + k] = f32(v);
        }
    }
    wrapper_id[row] = below(row, 0u, STREAM_WRAPPER, params.n_wrappers);
}
