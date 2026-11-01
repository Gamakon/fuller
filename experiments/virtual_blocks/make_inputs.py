"""Write the benchmark's inputs in Futhark's binary format: one file a set, the arguments of gen_b / read_b
(bench_entries.fut) in order.

    python3 make_inputs.py OUT_DIR

Sets:
  auc_s{1,2,3}  the 12,000 real blocks of fitness_data.npz: each block's own scale, bias, std and mean (the draft's
                m_std/m_mean = the block it stands in for), its real nibbles packed for read_b; generator seed 1, 2, 3.
  layer         Qwen2.5-0.5B-Instruct-4bit layer 13 MLP, every block (gate, up, down in that order), from the model's
                safetensors: the stored f16 scales/biases, the stored u32 words; m_std/m_mean from the dequantised block.
  n16k, n1m     the first 16,384 layer blocks, and the layer tiled to 2^20 blocks (timing sizes).
Offsets are 8 * block index: the draft hashes streams off .. off+5, so offsets closer than 6 apart share streams
between blocks (see PLAN.md, "a defect in the draft").
"""
import json
import pathlib
import struct
import sys

import numpy as np

FAXL = pathlib.Path("/Users/andrewmorgan/Dev/gamakon/faxl-core/docs/virtual-blocks")
MODEL = pathlib.Path("/Users/andrewmorgan/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
                     "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3/model.safetensors")
TYPES = {np.dtype(np.uint64): b" u64", np.dtype(np.uint32): b" u32", np.dtype(np.float32): b" f32"}
OFF_STRIDE = 8


def put(f, a):
    """One value in Futhark's binary format (version 2): 'b', 2, rank, the type, the dims as i64, the data LE."""
    a = np.asarray(a)  # not ascontiguousarray: it makes a scalar 1-d
    f.write(b"b" + bytes([2, a.ndim]) + TYPES[a.dtype])
    for d in a.shape:
        f.write(struct.pack("<q", d))
    f.write(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())


def tables():
    d = np.load(FAXL / "fitness_data.npz")
    return [d["BODY_Q"].astype(np.float32), d["CNT_P"].astype(np.float32), d["SPIKE_Q"].astype(np.float32),
            d["BODYSD"].astype(np.float32), d["MXED"].astype(np.float32)]


def pack(q):
    """n x 64 nibble levels -> n x 8 u32 words, MLX's packing (nibble l in bits 4*(l%8) of word l/8)."""
    q = q.astype(np.uint32).reshape(len(q), 8, 8)
    return (q << (4 * np.arange(8, dtype=np.uint32))).sum(2, dtype=np.uint32)


def write(path, seed, scale, bias, m_std, m_mean, packed):
    n = len(scale)
    with open(path, "wb") as f:
        put(f, np.array(seed, dtype=np.uint64))
        put(f, (np.arange(n, dtype=np.uint64) * np.uint64(OFF_STRIDE)))
        put(f, m_std.astype(np.float32))
        put(f, m_mean.astype(np.float32))
        put(f, scale.astype(np.float32))
        put(f, bias.astype(np.float32))
        for t in tables():
            put(f, t)
        put(f, packed.astype(np.uint32))
    print(f"{path}: {n} blocks")


def layer13():
    """Every block of layer 13's MLP: (scale, bias, words, dequantised values)."""
    with open(MODEL, "rb") as f:
        hn = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hn))
        base = 8 + hn

        def get(name, dt):
            a, b = head[name]["data_offsets"]
            f.seek(base + a)
            return np.frombuffer(f.read(b - a), dtype=dt).reshape(head[name]["shape"])

        S, B, W = [], [], []
        for p in ("gate_proj", "up_proj", "down_proj"):
            k = f"model.layers.13.mlp.{p}"
            S.append(get(k + ".scales", np.float16).astype(np.float32).ravel())
            B.append(get(k + ".biases", np.float16).astype(np.float32).ravel())
            W.append(get(k + ".weight", np.uint32).reshape(-1, 8))
    s, b, w = np.concatenate(S), np.concatenate(B), np.concatenate(W)
    q = ((w[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(len(w), 64).astype(np.float32)
    return s, b, w, q * s[:, None] + b[:, None]


def main(out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(FAXL / "fitness_data.npz")
    R, s, b = d["real_blocks"], d["real_scale"], d["real_bias"]
    q = np.round((R.astype(np.float64) - b[:, None]) / s[:, None]).astype(np.int64)
    assert q.min() >= 0 and q.max() <= 15, "a real block off its own grid"
    for seed in (1, 2, 3):
        write(out / f"auc_s{seed}.in", seed, s, b, R.std(1), R.mean(1), pack(q))
    ls, lb, lw, lv = layer13()
    print(f"layer 13 MLP: {len(ls)} blocks of 64 (gate 4864x896, up 4864x896, down 896x4864)")
    np.save(out / "layer_real.npy", lv)
    write(out / "layer.in", 1, ls, lb, lv.std(1), lv.mean(1), lw)
    k = 16384
    write(out / "n16k.in", 1, ls[:k], lb[:k], lv[:k].std(1), lv[:k].mean(1), lw[:k])
    reps = -(-(1 << 20) // len(ls))
    t = lambda a: np.concatenate([a] * reps)[: 1 << 20]
    write(out / "n1m.in", 1, t(ls), t(lb), t(lv.std(1)), t(lv.mean(1)), t(lw))


if __name__ == "__main__":
    main(sys.argv[1])
