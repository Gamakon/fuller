-- Appended by bench.rs to faxl-core's genblock.fut (vendored verbatim beside this file), so the timed pair shares ONE
-- argument list and one upload: the runtime's `p` command calls both entries on the same inputs.
--
-- gen_b   : faxl-core's gen_layer, unchanged; it ignores `packed`.
-- read_b  : the memory-read baseline. It reads each block's stored 4-bit weights (8 u32 words = 32 bytes, MLX's
--           packing: nibble l of a block is bits 4*(l%8).. of word l/8) and its scale and bias, and writes the 64
--           dequantised f32 values q*scale + bias, the same output shape and the same final step as gen_layer.
--           It ignores the generator's own inputs.

entry gen_b [n] [nb] [sq] [bq]
            (seed: u64)
            (offs: [n]u64) (m_std: [n]f32) (m_mean: [n]f32)
            (scale: [n]f32) (bias: [n]f32)
            (body_q: [bq]f32) (cnt_pmf: [6]f32) (spike_q: [6][sq]f32)
            (body_sd: [12][nb]f32) (mx_edges: [13]f32)
            (_packed: [n][8]u32)
          : [n][D]f32 =
  gen_layer seed offs m_std m_mean scale bias body_q cnt_pmf spike_q body_sd mx_edges

entry read_b [n] [nb] [sq] [bq]
             (_seed: u64)
             (_offs: [n]u64) (_m_std: [n]f32) (_m_mean: [n]f32)
             (scale: [n]f32) (bias: [n]f32)
             (_body_q: [bq]f32) (_cnt_pmf: [6]f32) (_spike_q: [6][sq]f32)
             (_body_sd: [12][nb]f32) (_mx_edges: [13]f32)
             (packed: [n][8]u32)
           : [n][D]f32 =
  map3 (\w s b ->
          tabulate D (\l -> let q = (w[l / 8] >> (4 * u32.i64 (l % 8))) & 0xFu32
                            in f32.u32 q * s + b))
       packed scale bias
