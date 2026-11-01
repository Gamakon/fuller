"""Per-block (m_mean, m_std, scale, bias) of one Qwen2.5-0.5B-Instruct-4bit MLP layer, in safetensors order
(gate, up, down; block i = scales.ravel()[i]); also the real dequantised blocks.  python3 layer_params.py <layer>"""
import json, struct, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from make_inputs import MODEL
L = int(sys.argv[1])
with open(MODEL, "rb") as f:
    hn = struct.unpack("<Q", f.read(8))[0]; head = json.loads(f.read(hn)); base = 8 + hn
    def get(name, dt):
        a, b = head[name]["data_offsets"]; f.seek(base + a)
        return np.frombuffer(f.read(b - a), dtype=dt).reshape(head[name]["shape"])
    S, B, W = [], [], []
    for p in ("gate_proj", "up_proj", "down_proj"):
        k = f"model.layers.{L}.mlp.{p}"
        S.append(get(k + ".scales", np.float16).astype(np.float32).ravel())
        B.append(get(k + ".biases", np.float16).astype(np.float32).ravel())
        W.append(get(k + ".weight", np.uint32).reshape(-1, 8))
s, b, w = np.concatenate(S), np.concatenate(B), np.concatenate(W)
q = ((w[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(len(w), 64).astype(np.float64)
v = q * s[:, None] + b[:, None]
here = __file__.rsplit('/', 1)[0]
np.column_stack([v.mean(1), v.std(1), s, b]).astype(np.float32).tofile(f"{here}/params_layer{L}.f32")
np.save(f"{here}/real_layer{L}.npy", v.astype(np.float32))
print(f"layer {L}: {len(s)} blocks")
