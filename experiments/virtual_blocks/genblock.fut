-- Virtual weight blocks: block = f(seed, offset, tables).
-- One block = 64 values. Nothing stored per block but (mbid, seed, offset).
--
-- Steps mirror the numpy reference:
--   1 counter-based hash  -> uniforms            (replaces a stateful PRNG)
--   2 body table lookup   -> body shape
--   3 spike-count pmf     -> how many outliers
--   4 per-count magnitude -> how big they are
--   5 top-k by hash       -> which cells spike   (replaces argsort(argsort))
--   6 body-volume table   -> quiet body when a spike is large
--   7 standardise, apply the master's std/mean
--   8 quantise onto the real block's scale/bias grid

def D : i64 = 64

-- ---- 1. counter-based hash: stateless, parallel, reproducible -------------
def mix64 (x: u64) : u64 =
  let x = x ^ (x >> 33)
  let x = x * 0xff51afd7ed558ccdu64
  let x = x ^ (x >> 33)
  let x = x * 0xc4ceb9fe1a85ec53u64
  in x ^ (x >> 33)

def hash3 (seed: u64) (off: u64) (lane: u64) : u64 =
  mix64 (mix64 (seed * 0x9E3779B97F4A7C15u64 ^ off * 0xBF58476D1CE4E5B9u64) ^ lane)

def u01 (h: u64) : f32 = f32.u64 (h >> 40) / 16777216.0f32   -- 24 bits -> [0,1)

-- ---- 2/4/6. table lookup with linear interpolation ------------------------
def interp [n] (tab: [n]f32) (u: f32) : f32 =
  let x = u * f32.i64 (n - 1)
  let i = i64.min (n - 2) (i64.max 0 (i64.f32 x))
  let t = x - f32.i64 i
  in tab[i] * (1 - t) + tab[i + 1] * t

-- ---- 3. sample the spike count from a small pmf (inverse CDF) -------------
def pick_count [m] (pmf: [m]f32) (u: f32) : i64 =
  -- count how many CDF steps u exceeds: that index IS the sample
  let cdf = scan (+) 0f32 pmf
  in i64.min (m - 1) (i64.sum (map (\c -> i64.bool (u >= c)) cdf))

-- ---- the generator: one block ---------------------------------------------
entry gen_block [nb] [sq] [bq]
                (seed: u64) (off: u64)
                (m_std: f32) (m_mean: f32)
                (scale: f32) (bias: f32)
                (body_q: [bq]f32)            -- body shape quantiles
                (cnt_pmf: [6]f32)            -- spike-count pmf
                (spike_q: [6][sq]f32)        -- magnitude quantiles, per count
                (body_sd: [12][nb]f32)       -- body volume vs largest spike
                (mx_edges: [13]f32)
              : [D]f32 =
  -- 1. one hash per lane, plus three block-level draws
  let h      = map (\l -> hash3 seed off (u64.i64 l)) (iota D)
  let u_body = map u01 h
  let u_mag  = map (\l -> u01 (hash3 seed (off + 1) (u64.i64 l))) (iota D)
  let u_pick = map (\l -> u01 (hash3 seed (off + 2) (u64.i64 l))) (iota D)
  let u_sign = map (\l -> u01 (hash3 seed (off + 3) (u64.i64 l))) (iota D)
  let u_cnt  = u01 (hash3 seed (off + 4) 0xFFFFu64)
  let u_vol  = u01 (hash3 seed (off + 5) 0xFFFEu64)

  -- 2. body
  let body = map (interp body_q) u_body
  -- 3. how many spikes
  let k = pick_count cnt_pmf u_cnt
  -- 5. which cells: the k largest u_pick values. No sort: count how many
  --    lanes beat me, which is a rank in O(D) per lane and fully parallel.
  let rank = map (\i -> i64.sum (map (\j -> i64.bool (u_pick[j] > u_pick[i])) (iota D)))
                 (iota D)
  let is_spike = map (\r -> r < k) rank
  -- 4. magnitudes from the table for THIS count
  let mag = map (interp spike_q[k]) u_mag
  let sgn = map (\u -> if u < 0.5f32 then -1f32 else 1f32) u_sign
  let spiked = map3 (\s m g -> if s then g * m else 0f32) is_spike mag sgn
  -- 6. a big spike implies a quiet body
  let mx = f32.maximum (map f32.abs spiked)
  let mx = if k == 0 then 2.3f32 else mx
  let bin = loop b = 0i64 for j < 12 do if mx >= mx_edges[j + 1] then j + 1 else b
  let bin = i64.min 11 bin
  let bsd = interp body_sd[bin] u_vol
  let x = map3 (\s b sp -> if s then sp else b * bsd) is_spike body spiked

  -- 7. standardise, then wear the master's scale and offset
  let mu = f32.sum x / f32.i64 D
  let c  = map (\v -> v - mu) x
  let sd = f32.sqrt (f32.sum (map (\v -> v * v) c) / f32.i64 D)
  let sd = f32.max sd 1e-12f32
  let g  = map (\v -> v / sd * m_std + m_mean) c

  -- 8. land on the real 4-bit grid: q in [0,15], value = q*scale + bias
  in map (\v -> let q = f32.round ((v - bias) / scale)
                let q = f32.max 0f32 (f32.min 15f32 q)
                in q * scale + bias) g

-- Many blocks at once: this is the shape a real layer wants.
entry gen_layer [n] [nb] [sq] [bq]
                (seed: u64)
                (offs: [n]u64) (m_std: [n]f32) (m_mean: [n]f32)
                (scale: [n]f32) (bias: [n]f32)
                (body_q: [bq]f32) (cnt_pmf: [6]f32) (spike_q: [6][sq]f32)
                (body_sd: [12][nb]f32) (mx_edges: [13]f32)
              : [n][D]f32 =
  map (\i -> gen_block seed offs[i] m_std[i] m_mean[i] scale[i] bias[i]
                       body_q cnt_pmf spike_q body_sd mx_edges)
      (iota n)
