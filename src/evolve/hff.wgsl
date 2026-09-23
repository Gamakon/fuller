// HFF ON THE DEVICE. `engine.rs::score_rows` did this on the host: for every row
// of the population, for each of the 45 candidates (15 gene-linker combinations
// x 3 wrappers), build the objective vector, scale it, take the TrueNorth angle,
// and keep the best candidate. At population 1,200 that is 54,000 angles and
// costs milliseconds. At 200,000 it is 9,000,000 and cost 238 seconds of a
// 450-second fit, against 1.49 seconds of actual GPU work — the device sat idle
// while the host did arithmetic over three-element arrays.
//
// One thread per ROW. It walks that row's candidates, computes each angle, and
// writes the best: the fitness, which candidate won, and the three 1-R2 values
// the host still needs for the stop bar. The walk is sequential per row and the
// rows are independent, so there is nothing to reduce across threads and no
// atomics — which is as well, since WGSL has no `atomicAdd` on f32.
//
// f32, DELIBERATELY. The device ranks; the host's `confirm` re-scores the winner
// in f64 and that is what may stop a fit or be reported. The same split the
// scoring kernel already lives under.

struct HffParams {
    rows: u32,
    candidates: u32,        // per row: combinations x wrappers
    width: u32,             // the score row's stride (score::WIDTH)
    n_columns: u32,         // how many objectives HFF is over (m)
    wrappers: u32,
    tower: u32,             // 1 = the tower objective joins the vector
    redundancy: u32,        // 1 = the leave-one-gene-out column joins it
    balanced: u32,          // 1 = the selection angle is the balanced pole's
}

@group(0) @binding(0) var<uniform> hp: HffParams;
// The scoring kernel's output: `rows * candidates` rows of `width` f32.
@group(0) @binding(1) var<storage, read> scores: array<f32>;
// Which of the nine objective columns HFF uses, and whether each is log scaled.
@group(0) @binding(2) var<storage, read> columns: array<u32>;
@group(0) @binding(3) var<storage, read> log_scaled: array<u32>;
// The frozen per-column maximum, and the caps var/mad that make an objective.
@group(0) @binding(4) var<storage, read> col_max: array<f32>;
@group(0) @binding(5) var<storage, read> caps: array<f32>;      // var[3] then mad[3]
// The tower depth of each candidate, already reduced over its used genes.
@group(0) @binding(6) var<storage, read> tower: array<u32>;
// Out: the winner's TrueNorth angle, which candidate it was, its three 1-R2,
// and the angle that CHOSE it.
//
// TWO ANGLES, NOT ONE. TrueNorth always judges — it is `Scored::fitness`, what
// the stop bar and the hall of fame read. The balanced pole only decides who
// breeds — it is `Scored::selection`, what the tournament reads. They are the
// same number unless `balanced` is on, and when it is on, a kernel that
// returned only the angle it selected by would silently feed the balanced
// angle to the stop bar.
@group(0) @binding(7) var<storage, read_write> best_fitness: array<f32>;
@group(0) @binding(8) var<storage, read_write> best_candidate: array<u32>;
@group(0) @binding(9) var<storage, read_write> best_omr2: array<f32>;
@group(0) @binding(10) var<storage, read_write> best_selection: array<f32>;

const PI: f32 = 3.14159265358979;
// `HFF_LOG_FLOOR` in engine.rs, and its log10, so the kernel does not call log10
// on a constant every iteration.
const LOG_FLOOR: f32 = 1.0e-12;
const NEG_LOG10_FLOOR: f32 = 12.0;

// `Caps::objectives`: mse and mae capped by the constant model's own error, and
// 1-R2 as mse/var. Column k of nine is (metric, block) = (k / 3, k % 3).
fn objective(base: u32, k: u32) -> f32 {
    let metric = k / 3u;
    let block = k % 3u;
    // score columns: a, b, mse_t, mse_v, max_err_v, mse_e, mae_t, mae_v, mae_e
    var mse = 0.0;
    if (block == 0u) { mse = scores[base + 2u]; }
    else if (block == 1u) { mse = scores[base + 3u]; }
    else { mse = scores[base + 5u]; }
    var mae = 0.0;
    if (block == 0u) { mae = scores[base + 6u]; }
    else if (block == 1u) { mae = scores[base + 7u]; }
    else { mae = scores[base + 8u]; }
    let v = caps[block];
    let m = caps[3u + block];
    if (metric == 0u) { return min(mse, v); }
    if (metric == 1u) {
        if (v > 0.0) { return min(mse / v, 1.0); }
        return 1.0;
    }
    return min(mae, m);
}

// `hff_scaled`: each objective on [0, 1] by its frozen range, optionally on a
// log scale so the small errors separate. A non-finite objective makes the whole
// vector unusable, and the caller then scores the row at PI.
fn scaled(base: u32, i: u32, ok: ptr<function, bool>) -> f32 {
    let k = columns[i];
    let v = objective(base, k);
    // The same bit test as the candidate guard, and for the same reason: `v != v`
    // is foldable to `false` and `abs(v) == 3.4028235e38` tested the largest
    // FINITE f32, not an infinity, so neither line rejected what it named. One
    // test rejects both NaN and +-inf.
    if ((bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u) {
        *ok = false;
        return 0.0;
    }
    let mx = col_max[k];
    var x = 0.0;
    if (mx > 0.0) { x = min(v / mx, 1.0); }
    if (log_scaled[i] == 1u) {
        if (x <= LOG_FLOOR) { x = 0.0; }
        else { x = 1.0 + log(x) / 2.302585 / NEG_LOG10_FLOOR; }
    }
    return x;
}

// The angle from TrueNorth, and from the balanced pole when selection asks for
// it. Both read the same scaled vector, so one walk serves both.
@compute @workgroup_size(64)
fn hff_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= hp.rows) {
        return;
    }
    var win = PI;
    var win_truenorth = PI;
    var win_at = 0u;
    var win_r0 = 1.0;
    var win_r1 = 1.0;
    var win_r2 = 1.0;
    let extra = hp.redundancy + hp.tower;
    let m = f32(hp.n_columns + extra);

    for (var c = 0u; c < hp.candidates; c = c + 1u) {
        let base = (row * hp.candidates + c) * hp.width;
        // An unfitted candidate has a non-finite scale; it is not a choice.
        //
        // NOT `a != a`. A WGSL implementation is permitted to assume operands
        // are not NaN, and Metal's does: the self-comparison folds to `false`
        // and every unfitted candidate walks straight into the scoring loop,
        // where its NaN objectives compare false against every bound and it
        // wins the argmin. The dispatched parity test caught exactly that — the
        // device picked the candidate this line exists to skip.
        //
        // A bit test cannot be folded away. NaN is exponent all-ones with a
        // non-zero mantissa; this also rejects the infinities, which are not
        // fitted scales either.
        let bits = bitcast<u32>(scores[base]);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            continue;
        }
        var ok = true;
        var energy = 0.0;
        // THE BALANCED POLE IS A DOT PRODUCT, NOT A DISTANCE.
        //
        // hff_core builds unit "geometric coordinates" first —
        // `sign(x) * sqrt(x*x / energy)`, i.e. the objective vector normalised —
        // and takes cos_theta as their dot product with (1/sqrt(m), ..).
        // Because every coordinate carries the same 1/sqrt(m), that dot product
        // is `sum(|x|) / sqrt(energy) / sqrt(m)`, so the running sum of absolute
        // values is all this loop needs, and it cannot be formed until the
        // energy is known. This used to accumulate `sum((x - 1/sqrt(m))^2)`,
        // a different function altogether.
        var abs_sum = 0.0;
        for (var i = 0u; i < hp.n_columns; i = i + 1u) {
            let x = scaled(base, i, &ok);
            if (!ok) { break; }
            energy = energy + x * x;
            abs_sum = abs_sum + abs(x);
        }
        if (!ok) {
            continue;
        }
        if (hp.redundancy == 1u) {
            let r = clamp(scores[base + 9u], 0.0, 1.0);
            energy = energy + r * r;
            abs_sum = abs_sum + abs(r);
        }
        if (hp.tower == 1u) {
            // `tower_penalty`: a FREE ZONE up to depth 2, then A QUARTER PER
            // LEVEL, 1 from depth 6. The divisor is 4, not 6 — it was 6 here,
            // which made every tower cheaper on the device than on the host and
            // would have changed which candidate won whenever the tower
            // objective was on.
            let t = tower[row * hp.candidates + c];
            var p = 0.0;
            if (t > 2u) { p = min(f32(t - 2u) / 4.0, 1.0); }
            energy = energy + p * p;
            abs_sum = abs_sum + abs(p);
        }
        let cos_theta = clamp(1.0 - min(energy / m, 1.0), -1.0, 1.0);
        var fitness = 0.0;
        if (cos_theta <= 1.0 - 1.1920929e-7) { fitness = acos(cos_theta); }
        // TrueNorth always JUDGES, AND IT ALSO CHOOSES THE CANDIDATE. The
        // balanced pole only decides who breeds ONCE THE CANDIDATE IS CHOSEN:
        // the host takes its argmin over `fitness` and then records the balanced
        // angle OF THAT CANDIDATE. Selecting by the balanced angle here would
        // pick a different candidate than the host for the same row -- a
        // uniformly mediocre one, which is exactly what the balanced pole is
        // banned as a fitness for rewarding.
        var key = fitness;
        if (hp.balanced == 1u) {
            // hff_core returns 0 outright for a vector with no energy, because
            // the normalisation is undefined there. Its threshold is
            // `f64::EPSILON` against an energy it computed in f64 — about
            // 2.2e-16, NOT an f32 epsilon. Using 1.19e-7 here swallowed every
            // genuinely good candidate: a model fitted to 1e-11 has an energy
            // near 3e-11, which is far below 1.19e-7 and nowhere near zero, and
            // the kernel scored it a perfect 0 and promoted it over the real
            // winner. Denormals are flushed on the device, so an energy this
            // small is still exact enough to normalise by.
            if (energy <= 2.220446049250313e-16) {
                key = 0.0;
            } else {
                let cb = clamp(abs_sum / sqrt(energy) / sqrt(m), -1.0, 1.0);
                key = acos(cb);
            }
        }
        if (fitness < win_truenorth) {
            win = key;
            win_truenorth = fitness;
            win_at = c;
            let v0 = caps[0];
            let v1 = caps[1];
            let v2 = caps[2];
            win_r0 = select(1.0e30, scores[base + 2u] / v0, v0 > 0.0);
            win_r1 = select(1.0e30, scores[base + 3u] / v1, v1 > 0.0);
            win_r2 = select(1.0e30, scores[base + 5u] / v2, v2 > 0.0);
        }
    }
    best_fitness[row] = win_truenorth;
    best_selection[row] = win;
    best_candidate[row] = win_at;
    best_omr2[row * 3u] = win_r0;
    best_omr2[row * 3u + 1u] = win_r1;
    best_omr2[row * 3u + 2u] = win_r2;
}
