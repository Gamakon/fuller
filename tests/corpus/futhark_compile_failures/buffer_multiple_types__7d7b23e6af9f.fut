-- a batch of 10 programs (fuller::futhark_batch)
def k0_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in tabulate k (\i_k -> (map2 f32.max (hist f32.min f32.highest d (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 d) x))) (map2 f32.max (map2 (-) (scan f32.min f32.highest (opaque (zeros_d))) (hist f32.min f32.highest d (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 d) x))) (scan f32.max f32.lowest (opaque (zeros_k)))) (hist f32.max f32.lowest k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_d)) (zeros_d)))) (zeros_d))) (zeros_d)) (loop (s_a_d: [d]f32) = (map2 (+) ((points)[(i64.max 0 ((let x = (f32.minimum (zeros_k)) in i64.max 0 (i64.min (n - 1) (if f32.isnan x then 0i64 else i64.f32 (f32.max 0f32 (f32.min (f32.i64 (n - 1)) x))))) - 1))]) (zeros_d)) for ti_n < n - 1 do let i_n = ti_n + 1 in (zeros_d))))

entry k0 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k0_main (sample[start:start + n] :> [n][d]f32) centres

def k1_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in tabulate k (\i_k -> (zeros_d))

entry k1 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k1_main (sample[start:start + n] :> [n][d]f32) centres

def k2_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_k)) (map2 (\r s -> map (/ s) r) (hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (hist (+) 0f32 n (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 n) x))) ((centres)[(k - 1)])) (zeros_d))) (points)) (zeros_k))

entry k2 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k2_main (sample[start:start + n] :> [n][d]f32) centres

def k3_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in let s_a_n = (hist (+) 0f32 n (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 n) x))) (zeros_d)) (let s_a_n = (zeros_n) in (zeros_d))) in (hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (loop (s_a_k: [k]f32) = (hist (+) 0f32 k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_k)) (map2 f32.max (map2 f32.min (replicate k ((loop ((s_f32, s2_f32): (f32, f32)) = (((f32.i64 d), (f32.i64 n))) for ti_k < k - 1 do let i_k = ti_k + 1 in ((0f32, 0f32))).0)) (zeros_k)) (map2 (*) (zeros_k) (zeros_k)))) for i_n < n do (map2 (-) (replicate k (f32.i64 d)) (hist f32.max f32.lowest k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_n)) (zeros_n))))) (centres))

entry k3 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k3_main (sample[start:start + n] :> [n][d]f32) centres

def k4_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in map2 (\r s -> map (/ s) r) (hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (hist f32.max f32.lowest k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (hist (+) 0f32 n (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 n) x))) (zeros_d)) (zeros_d))) (map2 (-) (zeros_n) (zeros_n)))) (centres)) (map2 (-) (zeros_k) (zeros_k))

entry k4 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k4_main (sample[start:start + n] :> [n][d]f32) centres

def k5_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in tabulate k (\i_k -> ((hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (map2 (+) (zeros_n) (zeros_n))) (points))[(i_k)]))

entry k5 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k5_main (sample[start:start + n] :> [n][d]f32) centres

def k6_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in tabulate k (\i_k -> (loop (s_a_d: [d]f32) = (zeros_d) for ti_d < d - 1 do let i_d = ti_d + 1 in (loop (s2_a_d: [d]f32) = (scan f32.max f32.lowest (opaque (loop (s2_a_d: [d]f32) = (map2 (-) (hist f32.max f32.lowest d (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 d) x))) (zeros_k)) (zeros_k)) (loop (s2_a_d: [d]f32) = (zeros_d) for ti2_d < d - 1 do let i2_d = ti2_d + 1 in ((centres)[(i64.min (k - 1) ((0i64) + 1))]))) for ti2_k < k - 1 do let i2_k = ti2_k + 1 in ((points)[(n - 1)])))) for ti2_d < d - 1 do let i2_d = ti2_d + 1 in ((centres)[(k - 1)]))))

entry k6 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k6_main (sample[start:start + n] :> [n][d]f32) centres

def k7_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_k)) (centres)

entry k7 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k7_main (sample[start:start + n] :> [n][d]f32) centres

def k8_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in tabulate k (\i_k -> (loop (s_a_d: [d]f32) = (hist f32.max f32.lowest d (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 d) x))) (loop (s_a_k: [k]f32) = (zeros_k) for i2_k < k do (zeros_k))) (hist f32.max f32.lowest k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_k)) (loop (s_a_k: [k]f32) = (zeros_k) for ti_d < d - 1 do let i_d = ti_d + 1 in (loop (s2_a_k: [k]f32) = (hist (+) 0f32 k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (zeros_n)) (hist f32.min f32.highest n (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 n) x))) (tabulate d (\i2_d -> (f32.i64 d)))) (zeros_d))) for i2_k < k do (s2_a_k))))) for i_d < d do (zeros_d)))

entry k8 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k8_main (sample[start:start + n] :> [n][d]f32) centres

def k9_main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in hist (map2 (+)) (replicate d 0f32) k (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 k) x))) (hist (+) 0f32 n (map (\x -> if f32.isnan x then -1i64 else i64.f32 (f32.max (-1f32) (f32.min (f32.i64 n) x))) (loop (s_a_n: [n]f32) = (replicate n (f32.i64 d)) for i_k < k do (map2 f32.max (s_a_n) (s_a_n)))) (tabulate n (\i_n -> (f32.i64 d))))) (tabulate n (\i_n -> (zeros_d)))

entry k9 [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  k9_main (sample[start:start + n] :> [n][d]f32) centres

module mkyard = {
def main [n][d][k] (points: [n][d]f32) (centres: [k][d]f32) : [k][d]f32 =
  let zeros_n = replicate n 0f32
  let zeros_d = replicate d 0f32
  let zeros_k = replicate k 0f32
  in let s_a_n = (tabulate n (\i_n -> ((loop ((s_f32, s2_f32): (f32, f32)) = (((f32.sqrt (loop (s_f32: f32) = (0f32) for i_d < d do ((s_f32) + (((((points)[(i_n)])[(i_d)]) - (((centres)[(0i64)])[(i_d)])) * ((((points)[(i_n)])[(i_d)]) - (((centres)[(0i64)])[(i_d)])))))), (0f32))) for ti_k < k - 1 do let i_k = ti_k + 1 in (if ((f32.sqrt (loop (s3_f32: f32) = (0f32) for i_d < d do ((s3_f32) + (((((points)[(i_n)])[(i_d)]) - (((centres)[(i_k)])[(i_d)])) * ((((points)[(i_n)])[(i_d)]) - (((centres)[(i_k)])[(i_d)])))))) - (s_f32)) < 0f32 then (((f32.sqrt (loop (s3_f32: f32) = (0f32) for i_d < d do ((s3_f32) + (((((points)[(i_n)])[(i_d)]) - (((centres)[(i_k)])[(i_d)])) * ((((points)[(i_n)])[(i_d)]) - (((centres)[(i_k)])[(i_d)])))))), (f32.i64 (i_k)))) else (((s_f32), (s2_f32))))).1))) in (tabulate k (\i_k -> (tabulate d (\i_d -> (if (-((loop ((s_f32, s2_f32, s3_f32): (f32, f32, f32)) = ((0f32, 0f32, 0f32)) for i_n < n do (if ((f32.abs (((s_a_n)[(i_n)]) - (f32.i64 (i_k)))) - (1f32)) < 0f32 then ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))) - (s_f32)) - ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((s3_f32) + (1f32)))) else (((s_f32), (s2_f32), (s3_f32))))).2)) < 0f32 then (((loop ((s_f32, s2_f32, s3_f32): (f32, f32, f32)) = ((0f32, 0f32, 0f32)) for i_n < n do (if ((f32.abs (((s_a_n)[(i_n)]) - (f32.i64 (i_k)))) - (1f32)) < 0f32 then ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))) - (s_f32)) - ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((s3_f32) + (1f32)))) else (((s_f32), (s2_f32), (s3_f32))))).0) / ((loop ((s_f32, s2_f32, s3_f32): (f32, f32, f32)) = ((0f32, 0f32, 0f32)) for i_n < n do (if ((f32.abs (((s_a_n)[(i_n)]) - (f32.i64 (i_k)))) - (1f32)) < 0f32 then ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((((s_f32) + ((((points)[(i_n)])[(i_d)]) - (s2_f32))) - (s_f32)) - ((((points)[(i_n)])[(i_d)]) - (s2_f32))), ((s3_f32) + (1f32)))) else (((s_f32), (s2_f32), (s3_f32))))).2)) else (((centres)[(i_k)])[(i_d)]))))))

}
entry kyard [m_sample][d][k] (sample: [m_sample][d]f32) (start: i64) (n: i64) (centres: [k][d]f32) : [][]f32 =
  mkyard.main (sample[start:start + n] :> [n][d]f32) centres

