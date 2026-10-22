#!/usr/bin/env python3
"""Load an ST8b/ST9 population dump (phylu true_wgsl_fit `--dump-every`) into UltraDim.

Read docs/Rules.md (qdrant repo): the UltraDim client is the Python gRPC client, never https.

Input: DUMP_DIR/gNNNNNN.bin, one file per generation, little-endian u32 words
(writer: phylu src/evolve/true_wgsl_fit.rs `dump_bytes`):
  header  magic 'ST8BDMP1' (2), version, generation, rows, n_t, B, I, record_words, C;
          then n_t triples (type code, W_T, h_T+t_T)
  record  generation, row, HFF f32, picture_exact (1/0/0xFFFFFFFF), GPU ms f32 (NaN unknown),
          born, hof, cells_cut, node_count; (a) C u32 symbols padded 0xFFFFFFFF;
          (b) B u32 subtree-hash bucket counts; (c) I f32 HFF inputs (NaN unknown);
          (d, version 2) 4 u32 lineage: first parent, second parent (0xFFFFFFFF none),
          operator (0 one-parent child, 1 one-point, 2 two-point, 3 gene, 4 subtree, 5 copies), mutated

Three point sets, one row per individual:
  st9p3_cells      (a) as a ONE-HOT over (cell position, symbol) pairs seen in the loaded dumps
                   (padding has no one); cosine on these binary rows ranks by shared expressed cells
                   (the Hamming overlap). Vocabulary written to cells_vocab.csv.
  st9p3_subtrees   (b) the 1024 bucket counts as float32.
  st9p3_behaviour  (c) the 29 HFF inputs as float32. Rows with any NaN input are NOT upserted
                   (cosine on NaN is undefined); they stay in the .npy and payload.csv.
Payload per point: uid = generation*1000+row, generation, row, hff, picture_exact (-1 unknown),
gpu_ms (absent when unknown), born, hof, cells_cut, node_count.

UltraDim's UpsertUltradimPoints requires point ids contiguous from 0 (ordinal-addressed store,
ultradimdb/src/v1_ops.rs validate_contiguous_ids), so the point id is the ordinal in load order
(generation, then row) and the requested id generation*1000+row is the payload field `uid`.

Always writes, beside the dump (OUT_DIR, default DUMP_DIR/../ultradim): cells_csr.npz (one-hot,
CSR), cells.npy (the same, dense, when under 1 GiB), cells_vocab.csv, types.csv, subtrees.npy, behaviour.npy, payload.csv. If an UltraDim gRPC server
answers on --addr it creates the three collections (Cosine) and upserts; a collection that already
exists is an error (the store is append-only by ordinal; use --prefix for a fresh set).

  python3.12 scripts/st9_dump_to_ultradim.py /private/tmp/claude-501/st9p3/run/dump
"""

import argparse
import csv
import glob
import math
import os
import struct
import sys
import time

import numpy as np

MAGIC = 0x3150_4D44_4238_5453
PAD = 0xFFFFFFFF
FIXED = 9
HEADER = 10


def stamp(msg):
    print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def read_dump(path):
    w = np.fromfile(path, dtype="<u4")
    if len(w) < HEADER or (int(w[0]) | (int(w[1]) << 32)) != MAGIC:
        raise SystemExit(f"{path}: not an ST8BDMP1 dump")
    version, gen, rows, n_t, b, i, rec, c = (int(x) for x in w[2:10])
    if version not in (1, 2):
        raise SystemExit(f"{path}: dump version {version}, this loader reads 1 and 2")
    lin = 4 if version == 2 else 0
    if rec != FIXED + c + b + i + lin:
        raise SystemExit(f"{path}: record_words {rec} != 9 + C {c} + B {b} + I {i} + lineage {lin}")
    types = w[HEADER:HEADER + 3 * n_t].reshape(n_t, 3)
    if int(types[:, 1].sum()) != c:
        raise SystemExit(f"{path}: per-type widths sum {int(types[:, 1].sum())} != C {c}")
    body = w[HEADER + 3 * n_t:]
    if len(body) != rows * rec:
        raise SystemExit(f"{path}: {len(body)} record words, expected {rows} x {rec}")
    r = body.reshape(rows, rec)
    return dict(gen=gen, rows=rows, b=b, i=i, c=c, types=types, fixed=r[:, :FIXED],
                cells=r[:, FIXED:FIXED + c], bag=r[:, FIXED + c:FIXED + c + b],
                inputs=r[:, FIXED + c + b:FIXED + c + b + i].copy().view("<f4"),
                lineage=r[:, FIXED + c + b + i:])


def f32(u):
    return struct.unpack("<f", struct.pack("<I", int(u)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--addr", default="localhost:6334")
    ap.add_argument("--prefix", default="st9p3")
    ap.add_argument("--stubs", default=os.path.expanduser("~/dev/minkymorgan/qdrant/tmp/pystubs"))
    ap.add_argument("--batch", type=int, default=250)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dump_dir, "g*.bin")))
    if not files:
        raise SystemExit(f"no g*.bin in {a.dump_dir}")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.dump_dir.rstrip("/"))), "ultradim")
    os.makedirs(out, exist_ok=True)
    stamp(f"reading {len(files)} dumps from {a.dump_dir}")

    vocab = {}
    ind_rows, bags, inputs, pay = [], [], [], []
    shape0 = None
    for f in files:
        d = read_dump(f)
        shape = (d["c"], d["b"], d["i"], d["types"].tobytes())
        if shape0 is None:
            shape0 = shape
            np.savetxt(os.path.join(out, "types.csv"), d["types"], fmt="%d", delimiter=",",
                       header="type_code,W_T,h_T_plus_t_T", comments="")
        elif shape != shape0:
            raise SystemExit(f"{f}: layout differs from {files[0]}")
        for k in range(d["rows"]):
            fx = d["fixed"][k]
            cells = d["cells"][k]
            pos = np.nonzero(cells != PAD)[0]
            idx = []
            for p, s in zip(pos.tolist(), cells[pos].tolist()):
                key = (p, s)
                j = vocab.get(key)
                if j is None:
                    j = vocab[key] = len(vocab)
                idx.append(j)
            ind_rows.append(np.array(sorted(idx), dtype=np.int64))
            bags.append(d["bag"][k].astype(np.float32))
            inputs.append(d["inputs"][k])
            gpu = f32(fx[4])
            pay.append(dict(uid=int(fx[0]) * 1000 + int(fx[1]), generation=int(fx[0]), row=int(fx[1]),
                            hff=f32(fx[2]), picture_exact=-1 if int(fx[3]) == PAD else int(fx[3]),
                            gpu_ms=gpu, born=int(fx[5]), hof=int(fx[6]), cells_cut=int(fx[7]),
                            node_count=int(fx[8])))
    n = len(pay)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in ind_rows])
    indices = np.concatenate(ind_rows) if n else np.zeros(0, np.int64)
    subtrees = np.stack(bags)
    behaviour = np.stack(inputs).astype(np.float32)
    np.savez(os.path.join(out, "cells_csr.npz"), indptr=indptr, indices=indices, shape=np.array([n, len(vocab)]))
    with open(os.path.join(out, "cells_vocab.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["column", "cell_position", "symbol"])
        for (p, s), j in sorted(vocab.items(), key=lambda kv: kv[1]):
            wr.writerow([j, p, s])
    if n * len(vocab) * 4 <= 1 << 30:
        dense = np.zeros((n, len(vocab)), dtype=np.float32)
        for k in range(n):
            dense[k, indices[indptr[k]:indptr[k + 1]]] = 1.0
        np.save(os.path.join(out, "cells.npy"), dense)
    else:
        stamp(f"cells.npy not written: {n} x {len(vocab)} dense exceeds 1 GiB; cells_csr.npz holds it")
    np.save(os.path.join(out, "subtrees.npy"), subtrees)
    np.save(os.path.join(out, "behaviour.npy"), behaviour)
    cols = ["uid", "generation", "row", "hff", "picture_exact", "gpu_ms", "born", "hof", "cells_cut", "node_count"]
    with open(os.path.join(out, "payload.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(pay)
    nan_rows = int(np.isnan(behaviour).any(axis=1).sum())
    stamp(f"wrote {out}: {n} individuals, cells one-hot dim {len(vocab)} (nnz {len(indices)}), "
          f"subtrees dim {subtrees.shape[1]}, behaviour dim {behaviour.shape[1]} ({nan_rows} rows with NaN)")

    import grpc
    sys.path.insert(0, a.stubs)
    import ultradim_unified_pb2 as pb
    import ultradim_unified_pb2_grpc as pbg

    ch = grpc.insecure_channel(a.addr, options=[("grpc.max_send_message_length", 512 << 20),
                                                ("grpc.max_receive_message_length", 512 << 20)])
    try:
        grpc.channel_ready_future(ch).result(timeout=5)
    except grpc.FutureTimeoutError:
        stamp(f"NOT LOADED: no UltraDim gRPC server answers on {a.addr}; files are ready in {out}")
        return 2
    stub = pbg.UltraDimDBStub(ch)

    def payload(p):
        fields = {}
        for k, v in p.items():
            if isinstance(v, float):
                if not math.isnan(v):
                    fields[k] = pb.Value(double_value=v)
            else:
                fields[k] = pb.Value(integer_value=v)
        return pb.Payload(fields=fields)

    def load(name, dim, rows_iter, total):
        stub.CreateUltradimCollection(pb.CreateUltradimCollectionRequest(
            collection_name=name, vector_size=dim, distance="Cosine"), timeout=600)
        ordinal = 0
        buf = []

        def flush():
            nonlocal ordinal
            stub.UpsertUltradimPoints(pb.UpsertUltradimPointsRequest(
                collection_name=name,
                vectors=[pb.Vector(data=v.tolist()) for v, _ in buf],
                point_ids=[pb.PointId(num=ordinal + j) for j in range(len(buf))],
                payloads=[payload(p) for _, p in buf]), timeout=600)
            ordinal += len(buf)
            buf.clear()

        for v, p in rows_iter:
            buf.append((v, p))
            if len(buf) >= a.batch:
                flush()
        if buf:
            flush()
        stamp(f"loaded {name}: {ordinal} of {total} points, dim {dim}")

    def cells_rows():
        for k in range(n):
            v = np.zeros(len(vocab), dtype=np.float32)
            v[indices[indptr[k]:indptr[k + 1]]] = 1.0
            yield v, pay[k]

    load(f"{a.prefix}_cells", len(vocab), cells_rows(), n)
    load(f"{a.prefix}_subtrees", subtrees.shape[1], ((subtrees[k], pay[k]) for k in range(n)), n)
    ok = ~np.isnan(behaviour).any(axis=1)
    load(f"{a.prefix}_behaviour", behaviour.shape[1], ((behaviour[k], pay[k]) for k in range(n) if ok[k]), n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
