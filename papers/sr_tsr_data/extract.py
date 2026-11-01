import re, sys, os
L = "/Users/andrewmorgan/Dev/gamakon/phylu/logs"
OUT = "/Users/andrewmorgan/Dev/gamakon/fuller/papers/sr_tsr_data"
# CASCADE2 cumulative
rows = []
fey = stro = 0
for line in open(f"{L}/CASCADE2/sweep.log"):
    m = re.match(r"^(FOUND|FAIL)\s+\[\s*(\d+)/133\]\s+(\S+)\s+(\d+) gens\s+([\d.]+) s\s+\(found so far (\d+)\)", line)
    if m:
        st, i, law, g, s, so = m.groups()
        rows.append((int(i), int(so), law, st, float(s)))
        if st == "FOUND":
            if law.startswith("strogatz"): stro += 1
            else: fey += 1
with open(f"{OUT}/cascade2_cumulative.dat", "w") as f:
    f.write("# source: phylu/logs/CASCADE2/sweep.log ; law_index found_so_far\n")
    f.write("i found\n")
    for i, so, *_ in rows: f.write(f"{i} {so}\n")
print("cascade2 rows", len(rows), "feynman found", fey, "strogatz found", stro, "fit wall s", round(sum(r[4] for r in rows),1))
# KS rescues
ks = [l for l in open(f"{L}/CASCADE2/ks.log") if "KS FOUND" in l]
print("ks found lines", len(ks)); [print("  ", l.split()[0]) for l in ks]
# TF_DEV1 histogram
hist = {}
f_ = s_ = []; fey_v=[]; str_v=[]
sec = False
for line in open(f"{L}/TF_DEV1/summary.txt"):
    if "per problem" in line: sec = True; continue
    p = line.split()
    if sec and len(p) == 2:
        v = float(p[1]); hist[v] = hist.get(v, 0) + 1
        (str_v if p[0].startswith("strogatz") else fey_v).append(v)
with open(f"{OUT}/tf_dev1_hist.dat", "w") as f:
    f.write("# source: phylu/logs/TF_DEV1/summary.txt ; solved_fraction problems\n")
    f.write("frac n\n")
    for k in sorted(hist): f.write(f"{k:.1f} {hist[k]}\n")
print("tfdev1 feynman", len(fey_v), round(100*sum(fey_v)/len(fey_v),2), "strogatz", len(str_v), round(100*sum(str_v)/max(1,len(str_v)),2))
