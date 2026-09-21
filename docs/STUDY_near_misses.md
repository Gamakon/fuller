# STUDY: near misses — which unsolved laws fall with more applied evolution effort

*Read-only study of one race. No engine code changed, no evolution run, nothing committed. Scripts: `scratchpad/nearmiss/` (load.py, cross.py, classify.py, evidence.py, traj.py, core.py, build.py, doc.py); tables generated 2026-09-21.*

## Context page

**Objective.** HFF-SR is to beat the SRBench ground-truth benchmark: exact symbolic recovery of 133 generating laws (119 Feynman, 14 Strogatz), judged by SRBench's own sympy scorer. Best published: AIFeynman, 54.1%.

**Approach.** GEP in Rust + wgpu: a chromosome of 3 genes, each a Karva K-expression with a head of 34; model y = a*WRAPPER(LINKER(g1,g2,g3)) + b with a, b by least squares; tournaments on HFF (TrueNorth pole) over train MSE / 1-R2 / MAE plus t_depth (free up to 2); two islands (intake 1,500 + champion 1,500) with the pump every 5 generations; whole-number constants -5..5; a fit stops when validation 1-R2 <= 1e-10.

**The unique feature.** HFF: selection by the hyperspherical angle to the TrueNorth pole, reported as log10 p. Laws sit at -19.4 or below; the 71 fits where the search did not find the law sit at -10.4 or above, so the angle separates a law from an imitation without knowing the law. Two exceptions in this race, both between those bands: strogatz_bacres2, an imitation at -17.55, and strogatz_barmag2, the law behind a dead arccos(cos(x)) term, at -18.99.

**Status.** The race studied (development seed 7013, 6 s per problem, TrueNorth, population 3,000): **59 of 133 solved** after the 8 re-fit corrections (53 in the race log). One more is found by search and lost in reporting (strogatz_barmag2), one is found and lost in SRBench's simplify (feynman_II_6_15b), one is a fake under the bar (strogatz_bacres2). That leaves **71 laws the search did not find**.

**Why this study.** Andrew's request: "find the ones that we can solve with more applied evolution effort", and his working conclusion to test: "evolving from the data doesnt naturally climb out of basins, and diversity is the issue". The study uses every piece of effort evidence on disk: 33 run folders, 1772 scored fits, from 6 s to 10,000 generations.

**The true laws are used here OFFLINE only**, to understand what the race found and missed. Nothing in this study feeds a law's formula to the search; the no-cheating line is untouched.

**Headline numbers** (all from the scripts above; one seed unless stated):

| | |
|---|---|
| solved at generation 1 (the initial population) | 32 of 59 |
| solved by generation 5 / by 3 seconds | 41 of 59 / 55 of 59 |
| laws never solved here that WERE solved or found in some other run | 10 of 71 (+1 unchecked) — section 4, "yes" |
| laws with no direct evidence either way, in cells where we do solve | 25 — "maybe" |
| laws where the evidence says effort alone does not do it | 36 — "no" |
| the ten hard laws given 1,000-10,000 generations: solved | 1 of 10 (II_11_27: at generation 16 in growhead200_s12, at generation 523 in head34_s12) |
| train 1-R2 gained between generation 150 and 5,000 on the other nine | 0.3 to 3.4 decades, verdict unchanged in all nine |

---

## 4. Solvable with effort (the headline section, placed first)

### 4.1 How each verdict is reached

Three kinds of evidence, in this order of weight:

1. **The same law under other effort.** 1772 scored fits in 33 folders: ten other full 133-law races (seeds 7012 and 7013; population 800 to 20,000; constants -100..100 or -5..5; one on the balanced pole) and the long runs on the ten hard laws (growhead200_s12 5,000 generations; head34, tower, allfixes, hfflog 1,000 generations; big_long 10,000 generations). "SOLVED" = SRBench's scorer said yes. "FOUND" = the fit met the 1e-10 bar and the full string is the law by eye, but the scorer said n (an `Abs`, the asin spelling) — reachability evidence, reporting loss.
2. **Was it still improving?** Section 6 tables: error at generation 150, 1,000, 5,000.
3. **Structural neighbours.** The law's class, head needed, variables against the solve rate of its cell (section 3), and the generation its nearest solved neighbour was found at. A neighbour found at generation 1-5 came out of the initial population: a larger POPULATION is the effort that matters. One found at generation 20-150 came from search: more GENERATIONS matters.

Verdict rule (mechanical, in build.py): **yes** = solved or found in at least one other run. **no** = needs a non-whole-number constant inside the structure; or ran 1,000+ generations unsolved; or belongs to the 1/sqrt(1 - v^2/c^2) family (0 of 9 solved in any run, two members run to 5,000-10,000 generations); or head needed >= 17; or 6+ variables. **maybe** = the rest, ranked by the solve rate of their cells, then by log10 p.

Counts over the 71: **yes 10, maybe 25, no 36.**

A measured caution on "yes": of the 74 laws unsolved here, 68 were solved in **0** of the ten other full races, 5 in one, 1 in three. Of the 59 solved here, 37 were solved in 9 or 10 of the ten. The solved set is largely the same set in every race; the union of everything ever solved by SRBench's verdict in any folder is 66 laws.

### 4.2 Ranked list: most likely to fall with more effort (top 25 of 35)

The settings behind every folder named in the evidence column (head 48 and constants -100..100 for the older races are as given in the brief for this study, not re-read from the engine; "head 12->34" is the growing head). The race studied is seed 7013, population 3,000, head 34, constants -5..5, 6 s. Only the same-seed rows are evidence about EFFORT on the same data; the seed-7012 rows show the law is reachable by this engine under another split and setup.

| run folder | seed | population | head | constants | effort | same seed as the race studied? |
|---|---|---|---|---|---|---|
| ab1_cleanse | 7012 | 800 | 48 | -100..100 | 6 s | no (other data split) |
| growhead200_s12 | 7013 | 6000 | 12->34 | -5..5 | 5000 gens / 700 s | yes |
| head34_s12 | 7013 | 3000 | 34 | -100..100 | 1000 gens / 240 s | yes |
| ab3_rnc5_s11 | 7012 | 800 | 48 | -5..5 | 6 s | no (other data split) |
| ab2_base_s12 | 7013 | 800 | 48 | -100..100 | 6 s | yes |
| balanced_6s_s12_b | 7013 | 3000 | 34 | -5..5 | 6 s, balanced-pole | yes |
| big_pop_redundancy_g35_s11 | 7012 | 20000 | 48 | -100..100 | 35 gens / 9 s | no (other data split) |
| big_pop_redundancy_s11 | 7012 | 20000 | 48 | -100..100 | 4 s (partial) | no (other data split) |
| ab3_restart3_s11 | 7012 | 800 | 48 | -100..100 | 6 s, 3 restarts | no (other data split) |
| ab1_base | 7012 | 800 | 48 | -100..100 | 6 s | no (other data split) |
| ab2_base_s11 | 7012 | 800 | 48 | -100..100 | 6 s | no (other data split) |
| ab2_harvest_s11 | 7012 | 800 | 48 | -100..100 | 6 s | no (other data split) |
| rust_race_7012_6s | 7012 | 800 | 48 | -100..100 | 6 s | no (other data split) |

Generations at about 25 per second at population 3,000 (150 in 6 s), so 500 generations is about 20 s and 1,000 about 40 s per law.

| # | law | more effort? | the law | class; vars; head | log10 p / 1-R2 val (6 s) | nearest solved law, generation found | what differs | evidence | recommended effort |
|---|---|---|---|---|---|---|---|---|---|
| 1 | I_43_43 | yes | `kb*v/(A*(gamma - 1))` | rational (sum in denominator); 4 vars; head 7 | -7.77 / 2.2e-05 | I_34_1 `omega_0/(1 - v/c)` gen 45 | +1 variables; head +1; depth -1 | SOLVED in ab1_cleanse gen 9 | population: it appeared by generation 9 elsewhere; 12,000-20,000, 50 generations |
| 2 | II_11_27 | yes | `Ef*alpha*epsilon*n/(-alpha*n/3 + 1)` | rational (sum in denominator); 4 vars; head 10 | -7.02 / 1.1e-04 | I_34_1 `omega_0/(1 - v/c)` gen 45 | +1 variables; head +4; rational number inside (-1/3) | SOLVED in growhead200_s12 gen 16, head34_s12 gen 523 | population 6,000, growing head from 12 (the growhead200 setting): found at generation 16 |
| 3 | II_21_32 | yes | `0.08*q/(epsilon*r*(1 - v/c))` | rational (sum in denominator); 5 vars; head 10 | -6.47 / 2.7e-04 | I_34_1 `omega_0/(1 - v/c)` gen 45 | +2 variables; head +4; depth +1 | FOUND: ab3_rnc5_s11 gen 158: 0.0796*q*c/(eps*r*Abs(c-v)) met the bar; scorer n (the Abs) | generations: found at 158 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 4 | test_5 | yes | `6.28*d^(3/2)/sqrt(G*(m1 + m2))` | sqrt; 4 vars; head 8 | -5.91 / 8.0e-04 | I_47_23 `sqrt(gamma*pr/rho)` gen 1 | +1 variables; head +5; depth +1; a sum under a root; a sum in a denominator; a sum where the neighbour is a pure product | FOUND: ab2_base_s12 gen 85: 6.2832*sqrt(d^3/(G*(m1+m2))) met the bar; scorer n | generations: found at 85 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 5 | I_26_2 | yes | `arcsin(n*sin(theta2))` | arcsin/arccos/tanh; 2 vars; head 4 | -5.85 / 1.2e-03 | I_30_5 `arcsin(lambd/(d*n))` gen 39 | -1 variables; adds sin/cos | FOUND: balanced_6s_s12_b gen 68: asin(n*sin(theta2)) met the bar (log10 p -21.71); scored n only through the asin/arcsin spelling; but 0 solves in 15 long runs up to 5000 generations (whether arcsin was in the symbol set for each of those was not checked) | balanced-pole tournaments, 6 s: found at generation 68; or TrueNorth with a few seeds |
| 6 | I_44_4 | yes | `T*kb*n*log(V2/V1)` | exp/log; 5 vars; head 6 | -5.57 / 9.3e-04 | II_34_29b `6.28*B*Jz*g_*mom/h` gen 1 | depth -1; adds log | SOLVED in big_pop_redundancy_g35_s11 gen 25, big_pop_redundancy_s11 gen 23 | population: it appeared by generation 23 elsewhere; 12,000-20,000, 50 generations |
| 7 | III_15_12 | yes | `2*U*(1 - cos(d*k))` | sin/cos; 3 vars; head 7 | -3.96 / 1.2e-02 | III_17_37 `beta*(alpha*cos(theta) + 1)` gen 71 | depth +1 | SOLVED in ab3_restart3_s11 gen 199 | generations: found at 199 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 8 | I_12_11 | yes | `q*(B*v*sin(theta) + Ef)` | sin/cos; 5 vars; head 7 | -3.19 / 2.8e-02 | III_17_37 `beta*(alpha*cos(theta) + 1)` gen 71 | +2 variables | SOLVED in ab3_restart3_s11 gen 196 | generations: found at 196 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 9 | I_8_14 | yes | `sqrt((-x1 + x2)^2 + (-y1 + y2)^2)` | sqrt; 4 vars; head 10 | -3.17 / 5.6e-02 | I_47_23 `sqrt(gamma*pr/rho)` gen 1 | +1 variables; head +7; depth +2; a sum under a root; a sum where the neighbour is a pure product | SOLVED in ab1_base gen 152, ab2_base_s11 gen 152, ab2_harvest_s11 gen 164, rust_race_7012_6s gen 152 | generations: found at 152-164 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 10 | s_shearflow1 | yes | `cos(x)*cot(y)` | sin/cos; 2 vars; head 3 | -2.77 / 5.1e-02 | II_15_4 `-B*mom*cos(theta)` gen 1 | -1 variables | SOLVED in ab3_restart3_s11 gen 197; FOUND: ab3_rnc5_s11 gen 163: cos(x)*cos(Abs(y))/sin(y) met the bar; scorer n (the Abs) | generations: found at 163-197 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 11 | I_27_6 | maybe | `1/(n/d2 + 1/d1)` | rational (sum in denominator); 3 vars; head 4 | -5.22 / 1.9e-03 | II_10_9 `sigma_den/(epsilon*(chi + 1))` gen 19 | head -1 | 0 solves in 11 other runs (none longer than 206 generations); class x head cell 1/2, vars x head cell 17/18 | generations (neighbour came at generation 19): population 3,000-6,000, 1,000 generations |
| 12 | I_18_4 | maybe | `(m1*r1 + m2*r2)/(m1 + m2)` | rational (sum in denominator); 4 vars; head 5 | -3.89 / 1.3e-02 | II_10_9 `sigma_den/(epsilon*(chi + 1))` gen 19 | +1 variables | 0 solves in 11 other runs (none longer than 189 generations); class x head cell 3/5, vars x head cell 7/13 | generations (neighbour came at generation 19): population 3,000-6,000, 1,000 generations |
| 13 | I_13_4 | maybe | `m*(u^2 + v^2 + w^2)/2` | polynomial (sum of products); 4 vars; head 7 | -3.54 / 1.8e-02 | I_24_6 `m*x^2*(omega^2 + omega_0^2)/4` gen 27 | depth +1 | 0 solves in 10 other runs (none longer than 200 generations); class x head cell 3/6, vars x head cell 7/13 | generations (neighbour came at generation 27): population 3,000-6,000, 1,000 generations |
| 14 | s_lv1 | maybe | `-x^2 - 2*x*y + 3*x` | polynomial (sum of products); 2 vars; head 8 | -4.46 / 6.8e-02 | s_lv2 `-x*y - y^2 + 2*y` gen 76 | head +1 | 0 solves in 11 other runs (none longer than 392 generations); class x head cell 3/6, vars x head cell 3/6 | generations (neighbour came at generation 76): population 3,000-6,000, 1,000 generations |
| 15 | test_18 | maybe | `0.119*(H_G^2 + c^2*k_f/r^2)/G` | polynomial (sum of products); 5 vars; head 12 | -2.47 / 5.9e-02 | II_2_42 `A*kappa*(-T1 + T2)/d` gen 18 | head +3; depth +1 | 0 solves in 13 other runs (none longer than 453 generations); class x head cell 2/3, vars x head cell 2/6 | generations (neighbour came at generation 18): population 3,000-6,000, 1,000 generations |
| 16 | s_glider1 | maybe | `-0.05*x^2 - sin(y)` | sin/cos; 2 vars; head 6 | -6.46 / 2.1e-03 | s_glider2 `x - cos(y)/x` gen 56 | depth -1; rational number inside (20) | 0 solves in 11 other runs (none longer than 402 generations); class x head cell 3/7, vars x head cell 3/6 | generations (neighbour came at generation 56): population 3,000-6,000, 1,000 generations |
| 17 | III_10_19 | maybe | `mom*sqrt(Bx^2 + By^2 + Bz^2)` | sqrt; 4 vars; head 8 | -4.63 / 4.3e-03 | I_47_23 `sqrt(gamma*pr/rho)` gen 1 | +1 variables; head +5; depth +2; a sum under a root; a sum where the neighbour is a pure product | 0 solves in 11 other runs (none longer than 192 generations); class x head cell 0/2, vars x head cell 7/13 | population (neighbour came at generation 1): 12,000, 150 generations |
| 18 | I_11_19 | maybe | `x1*y1 + x2*y2 + x3*y3` | polynomial (sum of products); 6 vars; head 5 | -3.62 / 1.4e-02 | I_13_12 `G*m1*m2*(1/r2 - 1/r1)` gen 30 | +1 variables; head -2 | FOUND: ab2_base_s12 gen 119: met the bar with x1*y1 + ... plus a Piecewise term; not checked further | generations: found at 119 elsewhere; population 3,000, 500 generations (about 20 s), 3 restarts |
| 19 | III_14_14 | maybe | `I_0*(exp(Volt*q/(T*kb)) - 1)` | exp/log; 5 vars; head 8 | -4.03 / 9.7e-03 | II_2_42 `A*kappa*(-T1 + T2)/d` gen 18 | head -1; depth +1; adds exp | 0 solves in 11 other runs (none longer than 207 generations); class x head cell 0/4, vars x head cell 3/7 | generations (neighbour came at generation 18): population 3,000-6,000, 1,000 generations |
| 20 | II_35_21 | maybe | `mom*n_rho*tanh(B*mom/(T*kb))` | arcsin/arccos/tanh; 5 vars; head 8 | -3.3 / 3.0e-02 | I_30_5 `arcsin(lambd/(d*n))` gen 39 | +2 variables; head +4; depth +1 | 0 solves in 11 other runs (none longer than 227 generations); class x head cell 0/1, vars x head cell 3/7 | generations (neighbour came at generation 39): population 3,000-6,000, 1,000 generations |
| 21 | s_shearflow2 | maybe | `(0.1*sin(y)^2 + cos(y)^2)*sin(x)` | sin/cos; 2 vars; head 11 | -8.27 / 1.1e-05 | s_barmag1 `-sin(x) + 0.5*sin(x - y)` gen 120 | head +1; rational number inside (0.1) | 0 solves in 10 other runs (none longer than 374 generations); class x head cell 1/7, vars x head cell 2/8 | generations (neighbour came at generation 120): population 3,000-6,000, 1,000 generations |
| 22 | II_11_28 | maybe | `alpha*n/(-alpha*n/3 + 1) + 1` | rational (sum in denominator); 2 vars; head 10 | -8.61 / 1.6e-05 | II_38_14 `Y/(2*sigma + 2)` gen 144 | head +5; depth +2; rational number inside (-1/3) | 0 solves in 11 other runs (none longer than 206 generations); class x head cell 0/8, vars x head cell 2/8 | generations (neighbour came at generation 144): population 3,000-6,000, 1,000 generations |
| 23 | s_predprey2 | maybe | `y*(x/(x + 1) - 0.075*y)` | rational (sum in denominator); 2 vars; head 9 | -4.5 / 8.9e-03 | II_38_14 `Y/(2*sigma + 2)` gen 144 | head +4; depth +1; rational number inside (-0.075) | 0 solves in 11 other runs (none longer than 373 generations); class x head cell 0/8, vars x head cell 2/8 | generations (neighbour came at generation 144): population 3,000-6,000, 1,000 generations |
| 24 | s_predprey1 | maybe | `x*(-x - y/(x + 1) + 4)` | rational (sum in denominator); 2 vars; head 9 | -3.36 / 3.0e-02 | II_38_14 `Y/(2*sigma + 2)` gen 144 | head +4; depth +1; integer number inside (4) | 0 solves in 10 other runs (none longer than 386 generations); class x head cell 0/8, vars x head cell 2/8 | generations (neighbour came at generation 144): population 3,000-6,000, 1,000 generations |
| 25 | I_30_3 | maybe | `Int_0*sin(n*theta/2)^2/sin(theta/2)^2` | sin/cos; 3 vars; head 12 | -3.57 / 1.7e-02 | II_6_11 `0.08*p_d*cos(theta)/(epsilon*r^2)` gen 6 | -1 variables; head +5; depth +3; rational number inside (1/2, 1/2) | 0 solves in 12 other runs (none longer than 431 generations); class x head cell 1/7, vars x head cell 0/10 | generations (neighbour came at generation 6): population 3,000-6,000, 1,000 generations |

Remaining "maybe" laws, same columns:

| # | law | more effort? | the law | class; vars; head | log10 p / 1-R2 val (6 s) | nearest solved law, generation found | what differs | evidence | recommended effort |
|---|---|---|---|---|---|---|---|---|---|
| 26 | I_37_4 | maybe | `I1 + I2 + 2*sqrt(I1*I2)*cos(delta)` | sin/cos; 3 vars; head 11 | -3.13 / 3.1e-02 | s_barmag1 `-sin(x) + 0.5*sin(x - y)` gen 120 | +1 variables; head +1; a sum under a root | 0 solves in 11 other runs (none longer than 188 generations); class x head cell 1/7, vars x head cell 0/10 | generations (neighbour came at generation 120): population 3,000-6,000, 1,000 generations |
| 27 | I_50_26 | maybe | `x1*(alpha*cos(omega*t)^2 + cos(omega*t))` | sin/cos; 4 vars; head 12 | -1.96 / 1.5e-01 | III_17_37 `beta*(alpha*cos(theta) + 1)` gen 71 | +1 variables; head +5; depth +2 | 0 solves in 13 other runs (none longer than 222 generations); class x head cell 1/7, vars x head cell 0/5 | generations (neighbour came at generation 71): population 3,000-6,000, 1,000 generations |
| 28 | s_bacres1 | maybe | `-x*y/(0.5*x^2 + 1) - x + 20` | rational (sum in denominator); 2 vars; head 15 | -10.04 / 5.8e-07 | II_38_14 `Y/(2*sigma + 2)` gen 144 | head +10; depth +2; rational number inside (0.5) | 0 solves in 12 other runs (none longer than 388 generations); class x head cell 0/2, vars x head cell 0/1 | generations (neighbour came at generation 144): population 3,000-6,000, 1,000 generations |
| 29 | test_3 | maybe | `d*(1 - alpha^2)/(alpha*cos(theta1 - theta2) + 1)` | sin/cos; 4 vars; head 16 | -5.65 / 1.0e-03 | III_17_37 `beta*(alpha*cos(theta) + 1)` gen 71 | +1 variables; head +9; depth +2; a sum in a denominator | 0 solves in 12 other runs (none longer than 184 generations); class x head cell 0/4, vars x head cell 0/3 | generations (neighbour came at generation 71): population 3,000-6,000, 1,000 generations |
| 30 | I_6_2b | maybe | `0.399*exp(-(theta - theta1)^2/(2*sigma^2))/sigma` | exp/log; 3 vars; head 12 | -4.88 / 6.8e-03 | I_6_2a `0.399*exp(-theta^2/2)` gen 76 | +2 variables; head +8; depth +4; a sum where the neighbour is a pure product | 0 solves in 12 other runs (none longer than 434 generations); class x head cell 0/3, vars x head cell 0/10 | generations (neighbour came at generation 76): population 3,000-6,000, 1,000 generations |
| 31 | I_16_6 | maybe | `(u + v)/(1 + u*v/c^2)` | rational (sum in denominator); 3 vars; head 9 | -3.84 / 1.5e-02 | I_34_1 `omega_0/(1 - v/c)` gen 45 | head +3 | 0 solves in 12 other runs (none longer than 216 generations); class x head cell 0/8, vars x head cell 0/10 | generations (neighbour came at generation 45): population 3,000-6,000, 1,000 generations |
| 32 | test_14 | maybe | `Ef*(d^3*(alpha - 1)/(r^2*(alpha + 2)) - r)*cos(theta)` | sin/cos; 5 vars; head 15 | -3.3 / 2.4e-02 | s_barmag1 `-sin(x) + 0.5*sin(x - y)` gen 120 | +3 variables; head +5; depth +1; a sum in a denominator | 0 solves in 11 other runs (none longer than 217 generations); class x head cell 0/4, vars x head cell 0/4 | generations (neighbour came at generation 120): population 3,000-6,000, 1,000 generations |
| 33 | test_8 | maybe | `E_n/(E_n*(1 - cos(theta))/(c^2*m) + 1)` | sin/cos; 4 vars; head 15 | -3.05 / 3.4e-02 | III_17_37 `beta*(alpha*cos(theta) + 1)` gen 71 | +1 variables; head +8; depth +3; a sum in a denominator | 0 solves in 11 other runs (none longer than 201 generations); class x head cell 0/4, vars x head cell 0/3 | generations (neighbour came at generation 71): population 3,000-6,000, 1,000 generations |
| 34 | II_35_18 | maybe | `n_0/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))` | exp/log; 5 vars; head 16 | -3.03 / 3.7e-02 | II_2_42 `A*kappa*(-T1 + T2)/d` gen 18 | head +7; depth +2; a sum in a denominator; adds exp | 0 solves in 10 other runs (none longer than 184 generations); class x head cell 0/2, vars x head cell 0/4 | generations (neighbour came at generation 18): population 3,000-6,000, 1,000 generations |
| 35 | test_13 | maybe | `0.08*q/(epsilon*sqrt(d^2 - 2*d*r*cos(alpha) + r^2))` | sin/cos; 5 vars; head 15 | -2.67 / 6.1e-02 | s_barmag1 `-sin(x) + 0.5*sin(x - y)` gen 120 | +3 variables; head +5; depth +2; a sum under a root; a sum in a denominator | 0 solves in 10 other runs (none longer than 183 generations); class x head cell 0/4, vars x head cell 0/4 | generations (neighbour came at generation 120): population 3,000-6,000, 1,000 generations |

Reading the "yes" rows: 3 of the 10 appeared by generation 25 in some other run (I_43_43 gen 9, II_11_27 gen 16, I_44_4 gen 23-25 at population 20,000) — population / a different draw, not generations. 7 appeared between generation 68 and 199: two inside the ~150 generations a 6 s fit gets here but under another setting (I_26_2 at 68 on the balanced pole, test_5 at 85 at population 800), five at generation 152-199, which population 800 reaches inside 6 s and population 3,000 does not. Three of those five came in ab3_restart3_s11 (three restarts inside 6 s) — a fresh population each time. None of the ten "yes" laws was reached by running one population for thousands of generations.

Two comparisons on disk change one factor at a time (seed 7012, population 800 unless stated, 6 s):
- **Restarts:** ab2_harvest_s11 (1 search per fit) 47 solved; ab3_restart3_s11 (3 fresh searches inside the same 6 s, all else as logged equal) **54 solved, +7**. Fresh populations bought more than any other setting on disk.
- **Population for generations:** big_pop_redundancy_g35_s11 (population 20,000, capped at 35 generations, 9 s) 42 solved, against 47 for population 800 at ~360 generations. 25x the population with a tenth of the generations solved 5 fewer — so population alone is not the lever either (it did catch I_44_4, which nothing else has).

### 4.3 Effort alone will not do it (36 laws)

| reason | laws |
|---|---|
| 6+ variables and head needed >= 17 | 8 |
| sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | 7 |
| a non-whole-number constant INSIDE the structure | 7 |
| ran 1,000-10,000 generations, never solved | 6 |
| 6+ variables | 4 |
| head needed >= 17 | 4 |

| law | reason | the law | class; vars; head | log10 p (6 s) | evidence | our best model (6 s) |
|---|---|---|---|---|---|---|
| test_17 | 6+ variables | `(m^2*omega^2*x^2*(alpha*x/y + 1) + p^2)/(2*m)` | polynomial (sum of products); 6; 16 | -5.78 | head 16, 6 variables: cell solve rate 0/3; 0 solves in 12 other runs | `0.502*omega*sqrt(tanh(x)/tanh(alpha))*(omega*x^2 + 2*y - 3)*(alpha*...` |
| II_6_15a | 6+ variables | `0.239*p_d*z*sqrt(x^2 + y^2)/(epsilon*r^5)` | sqrt; 6; 14 | -5.74 | head 14, 6 variables: cell solve rate 0/3; 0 solves in 10 other runs | `0.392*sqrt(p_d*y*z*(p_d*z*(x + x^(-2)) - sin(y) + 0.757)/(tanh(y) +...` |
| I_32_17 | 6+ variables | `4.19*Ef^2*c*epsilon*omega^4*r^2/(omega^2 - omega_0^2)^2` | rational (sum in denominator); 6; 16 | -4.86 | head 16, 6 variables: cell solve rate 0/3; 0 solves in 10 other runs | `2.1*Ef^3*c*epsilon*omega^4*r^2*exp(omega)/(omega_0^6*tanh(Ef/(omega...` |
| I_40_1 | 6+ variables | `n_0*exp(-g*m*x/(T*kb))` | exp/log; 6; 10 | -2.35 | head 10, 6 variables: cell solve rate 0/1; 0 solves in 12 other runs | `0.957*sqrt(kb*n_0*(T + tan(n_0^(-3)))/(g*m*x)) - 0.663` |
| test_1 | 6+ variables and head needed >= 17 | `Z_1^2*Z_2^2*alpha^2*c^2*hbar^2/(16*E_n^2*sin(theta/2)^4)` | sin/cos; 7; 20 | -4.26 | head 20, 7 variables: cell solve rate 0/9; 0 solves in 13 other runs | `0.046*Z_1^2*Z_2^2*alpha^2*hbar*(c + theta^2/(E_n^3/c + theta^2)^2)^...` |
| I_9_18 | 6+ variables and head needed >= 17 | `G*m1*m2/((-x1 + x2)^2 + (-y1 + y2)^2 + (-z1 + z2)^2)` | rational (sum in denominator); 9; 20 | -3.87 | head 20, 9 variables: cell solve rate 0/9; 0 solves in 11 other runs | `-0.823*G*m1*m2*sqrt(y2)*(G + z2)/(y1*z1*(-m2 + 1/z1 + 1/y1)) - 0.005` |
| test_6 | 6+ variables and head needed >= 17 | `sqrt(2*E_n*L^2*epsilon^2/(Z_1^2*Z_2^2*m*q^4) + 1)` | sqrt; 7; 21 | -3.71 | head 21, 7 variables: cell solve rate 0/9; 0 solves in 11 other runs | `2.68*L*epsilon*(E_n*sqrt(L) + Z_1 - Z_2 + epsilon)*exp(-Z_1)*exp(-s...` |
| test_20 | 6+ variables and head needed >= 17 | `0.08*alpha^2*h^2*omega_0^2*(omega/omega_0 - sin(beta)^2 +...` | sin/cos; 7; 24 | -3.67 | head 24, 7 variables: cell solve rate 0/9; 0 solves in 12 other runs | `0.064*alpha^2*(h + exp(-arcsin(beta*c)/omega))^2/(c^2*m^2*arcsin(om...` |
| test_19 | 6+ variables and head needed >= 17 | `-0.04*(H_G^2*c^2*(1 - 2*alpha) + c^4*k_f/r^2)/G` | polynomial (sum of products); 6; 21 | -3.62 | head 21, 6 variables: cell solve rate 0/9; 0 solves in 11 other runs | `-0.602 + 0.083*c^2*(H_G - k_f*log(Abs(c - log(c)))/(alpha*r))^2*(al...` |
| test_16 | 6+ variables and head needed >= 17 | `Volt*q + sqrt(c^4*m^2 + c^2*(-A_vec*q + p)^2)` | sqrt; 6; 20 | -3.19 | head 20, 6 variables: cell solve rate 0/9; 0 solves in 14 other runs | `2.07*sqrt((m/log(A_vec*q + 1) + q)*(A_vec + Volt + c^3 + q)*(A_vec ...` |
| II_36_38 | 6+ variables and head needed >= 17 | `H*mom/(T*kb) + M*alpha*mom/(T*c^2*epsilon*kb)` | polynomial (sum of products); 8; 18 | -2.12 | head 18, 8 variables: cell solve rate 0/9; 0 solves in 11 other runs | `3.92 - 2.27*log(Abs(T + kb*tanh(epsilon) - mom + c/H))` |
| test_2 | 6+ variables and head needed >= 17 | `k_G*m*(sqrt(2*E_n*L^2/(k_G^2*m) + 1)*cos(theta1 - theta2)...` | sin/cos; 6; 25 | -1.42 | head 25, 6 variables: cell solve rate 0/9; 0 solves in 11 other runs | `1.24 - 2.29*log(Abs(0.333*L - 0.333*log(k_G*Abs(m - log(L))) + 0.33...` |
| test_12 | a non-whole-number constant INSIDE the structure | `0.08*q*(12.6*Volt*d*epsilon - d*q*y^3/(-d^2 + y^2)^2)/(ep...` | rational (sum in denominator); 5; 29 | -10.02 | needs 12.6 inside the structure; 0 solves in 10 other runs | `1.0*Volt*d*q/y^2 - 0.012` |
| III_4_32 | a non-whole-number constant INSIDE the structure | `1/(exp(0.159*h*omega/(T*kb)) - 1)` | exp/log; 4; 8 | -9.21 | needs 0.159 inside the structure; 0 solves in 17 other runs, incl. 5000 generations | `6.28*T*kb/(h*omega) - 0.472` |
| II_24_17 | a non-whole-number constant INSIDE the structure | `3.14*sqrt(-1/d^2 + 0.101*omega^2/c^2)` | sqrt; 3; 11 | -7.01 | needs 0.101 inside the structure; 0 solves in 24 other runs, incl. 5000 generations | `15.172291005813764*sqrt(0.2*x_0^2/(x_1*(x_0 + x_1)) + 0.2*tanh(x_0*...` |
| III_4_33 | a non-whole-number constant INSIDE the structure | `0.159*h*omega/(exp(0.159*h*omega/(T*kb)) - 1)` | exp/log; 4; 11 | -6.62 | needs 0.159 inside the structure; 0 solves in 11 other runs | `1.0*T*kb - 0.2*h - 0.2*omega + 0.2*sqrt(Abs(h - omega)) + 0.276 + 0...` |
| test_7 | a non-whole-number constant INSIDE the structure | `2.89*sqrt(G*rho - 0.119*alpha*c^2/d^2)` | sqrt; 5; 10 | -6.58 | needs -0.119 inside the structure; 0 solves in 11 other runs | `2.06*sqrt(2)*sqrt(G*rho*tanh((d + sqrt(G*rho/alpha))/c)*tanh((d + s...` |
| I_41_16 | a non-whole-number constant INSIDE the structure | `0.016*h*omega^3/(c^2*(exp(0.159*h*omega/(T*kb)) - 1))` | exp/log; 5; 15 | -6.47 | needs 0.159 inside the structure; 0 solves in 12 other runs | `0.067*T*kb*omega*(omega + 0.2)*sqrt(Abs(2 - h*tanh(T)/(T*kb)))/c^2 ...` |
| III_8_54 | a non-whole-number constant INSIDE the structure | `sin(6.28*E_n*t/h)^2` | sin/cos; 3; 6 | -0.99 | needs 6.28 inside the structure; 0 solves in 11 other runs | `1.11 - 0.891*sqrt(Abs(-0.333*E_n + 0.111*h*log(h + 0.05) - 0.333*h ...` |
| test_4 | head needed >= 17 | `sqrt(2)*sqrt((E_n - L^2/(2*m*r^2) - U)/m)` | sqrt; 5; 18 | -5.12 | head 18, 5 variables: cell solve rate 0/3; 0 solves in 16 other runs | `1.15*sqrt((0.416 + 1/E_n)*(E_n - U + sqrt(Abs(arcsin(-L + m + r))) ...` |
| test_10 | head needed >= 17 | `arccos((cos(theta2) - v/c)/(1 - v*cos(theta2)/c))` | arcsin/arccos/tanh; 3; 17 | -5.09 | head 17, 3 variables: cell solve rate 0/1; 0 solves in 12 other runs | `2.06*log(theta2 + 1/(theta2*(exp(theta2) - sin(theta2*v + v + tanh(...` |
| test_9 | head needed >= 17 | `-32*G^4*m1^2*m2^2*(m1 + m2)/(5*c^5*r^5)` | polynomial (sum of products); 5; 19 | -4.06 | head 19, 5 variables: cell solve rate 0/3; 0 solves in 11 other runs | `2.03e+3 - 1.64e+3*sqrt(-0.04*m1 + (0.2*sqrt(r) + 1)^2 - 0.00201*exp...` |
| test_11 | head needed >= 17 | `4*I_0*sin(alpha/2)^2*sin(delta*n/2)^2/(alpha^2*sin(delta/...` | sin/cos; 4; 17 | -2.65 | head 17, 4 variables: cell solve rate 0/4; 0 solves in 12 other runs | `4.23 - 2.31*log(-I_0 + alpha + n + arcsin(delta - n - sin(delta) + ...` |
| I_48_2 | ran 1,000-10,000 generations, never solved | `c^2*m/sqrt(1 - v^2/c^2)` | sqrt; 3; 12 | -8.91 | 0 solves in 20 other runs incl. 9 long runs up to 10000 generations | `0.429*c*m/(tanh(0.428/c)*tanh(3*log(c)/(v + 1))) + 0.0186` |
| I_15_10 | ran 1,000-10,000 generations, never solved | `m_0*v/sqrt(1 - v^2/c^2)` | sqrt; 3; 11 | -5.62 | 0 solves in 19 other runs incl. 7 long runs up to 5000 generations | `-0.019*m_0*v*(c - v^2 + v + arccos(v + sin(c)) + cos(c) - 60) - 0.026` |
| I_6_2 | ran 1,000-10,000 generations, never solved | `0.399*exp(-theta^2/(2*sigma^2))/sigma` | exp/log; 2; 8 | -4.81 | 0 solves in 18 other runs incl. 7 long runs up to 5000 generations | `0.224 - 0.078*log(2*sigma - 2.42 + (theta + sqrt(1/(sigma*(sigma - ...` |
| II_11_3 | ran 1,000-10,000 generations, never solved | `Ef*q/(m*(-omega^2 + omega_0^2))` | rational (sum in denominator); 5; 11 | -2.93 | 0 solves in 18 other runs incl. 7 long runs up to 5000 generations | `0.17431598841207724 - 0.24518262177115893*log(Abs(0.333333333333333...` |
| I_29_16 | ran 1,000-10,000 generations, never solved | `sqrt(x1^2 - 2*x1*x2*cos(theta1 - theta2) + x2^2)` | sin/cos; 4; 17 | -2.54 | 0 solves in 17 other runs incl. 6 long runs up to 5000 generations | `0.717*sqrt(Abs((theta1 - theta2)*(x1 + x2/x1)*(x1 + x2 + sin(0.333*...` |
| III_9_52 | ran 1,000-10,000 generations, never solved | `25.1*Ef*p_d*sin(t*(omega - omega_0)/2)^2/(h*t*(omega - om...` | sin/cos; 6; 23 | -1.94 | 0 solves in 23 other runs incl. 9 long runs up to 5000 generations | `9.06 - 24.0*log(Abs(-0.333*t/(h + 1.5) + 0.333*exp(cos(p_d)) + 0.33...` |
| II_13_23 | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `rho_c_0/sqrt(1 - v^2/c^2)` | sqrt; 3; 9 | -10.4 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 15+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `0.825*sqrt(rho_c_0*(rho_c_0 + 3/(125*(0.2*tanh(log(1/c)) + 1)^3 - 2...` |
| I_10_7 | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `m_0/sqrt(1 - v^2/c^2)` | sqrt; 3; 9 | -7.42 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 12+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `0.993*m_0*sqrt(tanh(sqrt(m_0))*tanh(5*sqrt(v) - 1.41*sqrt(c/2 - 1))...` |
| II_13_34 | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `rho_c_0*v/sqrt(1 - v^2/c^2)` | sqrt; 3; 11 | -6.94 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 14+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `0.576*v*(rho_c_0 - exp(-v))*(2 + 1/(c - v) - 1/(3 - 3/c))*tanh(sqrt...` |
| I_15_3x | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `(-t*u + x)/sqrt(1 - u^2/c^2)` | sqrt; 4; 14 | -6.49 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 10+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `-1.01*t*(u + 1/(c*x)) + 1.01*u*(c/(-x + log(Abs(log(c) + 3)) + 3) +...` |
| I_15_3t | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `(t - u*x/c^2)/sqrt(1 - u^2/c^2)` | sqrt; 4; 17 | -6.16 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 11+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `0.089*sqrt((125 - 6.31/c)*(t + 5 + t*u/(c*(c + sin(t) - tanh(sin(t ...` |
| test_15 | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `omega*sqrt(1 - v^2/c^2)/(1 + v*cos(theta)/c)` | sin/cos; 4; 18 | -4.98 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 10+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `0.543*omega*(c - v*cos(theta))*tan(tanh(1/c))/cos(tanh(c - v)) - 0.015` |
| I_34_14 | sqrt(1 - v^2/c^2) family (0 of 9 ever solved) | `omega_0*(1 + v/c)/sqrt(1 - v^2/c^2)` | sqrt; 3; 15 | -4.73 | 1/sqrt(1-v^2/c^2) family: 0 of 9 solved in any of 10+ runs; siblings I_48_2 and I_15_10 ran 5,000-10,000 generations unsolved | `(omega_0 + 1/(c^3*arcsin(sin(1/v))^3))*(0.166*v^2 + 2.65)/sqrt(log(...` |

Notes on the reasons.
- **A non-whole-number constant inside** (7 laws; 0 solved in any run, with either constants range): 0.159 = 1/(2 pi) inside an exp (III_4_32, III_4_33, I_41_16), 6.28 inside a sin (III_8_54), 0.101 = 1/pi^2 and 0.119 under a root (II_24_17, test_7), 12.6 = 4 pi between two terms (test_12). A leading constant is absorbed by the scale `a`; these are not. III_4_32 is the clearest measurement: 5,000 generations took train 1-R2 from 1.6e-6 to 1.1e-9 and log10 p from -9.2 to -15.1 — three decades closer, still a series and not the law.
- **The sqrt(1 - v^2/c^2) family** (9 laws; the reason table shows 7 because I_48_2 and I_15_10 are counted under "ran 1,000-10,000 generations"; test_15 carries the root in the numerator, I_34_14 in the denominator with a (1 + v/c) factor: I_10_7, II_13_23, II_13_34, I_15_10, I_48_2, I_15_3x, I_15_3t, I_34_14, test_15): head needed 9 to 18, 3-4 variables, no constants problem — and 0 of 9 solved in any of the folders. I_48_2 had 9 long runs up to 10,000 generations at population 12,000; I_15_10 had 7. The verdict for the five members without a long run of their own rests on these siblings, which is weaker evidence than a direct run.
- **Head needed >= 17 / 6+ variables**: solve rate 0 of 17 and 0 of 14 in this race (section 3), and 0 by SRBench's verdict in every other folder. The 4 laws that are "no" on variable count alone (I_40_1, II_6_15a, test_17, I_32_17; head 10-16) are the weakest "no" here: I_11_19, also 6 variables, met the bar once elsewhere.

---

## 1. When solutions are found (n = 59 solved)

| generation found | solved | cumulative | of the 59 | of the 133 | seconds (min-max) |
|---|---|---|---|---|---|
| 1 | 32 | 32 | 54% | 24% | 0.2-1.2 |
| 2-5 | 9 | 41 | 69% | 31% | 0.3-0.4 |
| 6-20 | 6 | 47 | 80% | 35% | 0.5-1.0 |
| 21-50 | 5 | 52 | 88% | 39% | 1.3-2.6 |
| 51-100 | 5 | 57 | 97% | 43% | 2.0-3.3 |
| 100+ | 2 | 59 | 100% | 44% | 3.2-6.0 |

Cumulative, "solved by generation g":

| by generation | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|
| solved | 32 | 36 | 41 | 44 | 47 | 52 | 57 | 59 |

By wall-clock (`fit_wall`; it includes start-up, so the first fit of the race shows 1.1 s at generation 1):

| race length | 0.5 s | 1 s | 2 s | 3 s | 4 s | 6 s |
|---|---|---|---|---|---|---|
| solved by then (fit_wall) | 40 | 45 | 51 | 55 | 58 | 59 |

**A 1-second race would have caught 45 of the 59; a 3-second race 55.** The last 3 seconds of every fit produced 4 solves across 133 problems.

The 18 solves that came from search (generation 6 or later):

| law | generation | seconds | class | vars | head | the law |
|---|---|---|---|---|---|---|
| II_11_20 | 6 | 0.5 | monomial | 5 | 5 | `Ef*n_rho*p_d^2/(3*T*kb)` |
| II_6_11 | 6 | 0.5 | sin/cos | 4 | 7 | `0.08*p_d*cos(theta)/(epsilon*r^2)` |
| I_32_5 | 7 | 0.5 | monomial | 4 | 7 | `0.053*a^2*q^2/(c^3*epsilon)` |
| III_19_51 | 16 | 0.9 | monomial | 5 | 10 | `-0.125*m*q^4/(epsilon^2*h^2*n^2)` |
| II_2_42 | 18 | 1.0 | polynomial (sum of products) | 5 | 9 | `A*kappa*(-T1 + T2)/d` |
| II_10_9 | 19 | 1.0 | rational (sum in denominator) | 3 | 5 | `sigma_den/(epsilon*(chi + 1))` |
| I_24_6 | 27 | 1.3 | polynomial (sum of products) | 4 | 7 | `m*x^2*(omega^2 + omega_0^2)/4` |
| I_13_12 | 30 | 1.5 | polynomial (sum of products) | 5 | 7 | `G*m1*m2*(1/r2 - 1/r1)` |
| I_39_11 | 32 | 1.5 | rational (sum in denominator) | 3 | 3 | `V*pr/(gamma - 1)` |
| I_30_5 | 39 | 1.8 | arcsin/arccos/tanh | 3 | 4 | `arcsin(lambd/(d*n))` |
| I_34_1 | 45 | 2.6 | rational (sum in denominator) | 3 | 6 | `omega_0/(1 - v/c)` |
| s_glider2 | 56 | 2.0 | sin/cos | 2 | 6 | `x - cos(y)/x` |
| III_17_37 | 71 | 3.1 | sin/cos | 3 | 7 | `beta*(alpha*cos(theta) + 1)` |
| I_6_2a | 76 | 3.3 | exp/log | 1 | 4 | `0.399*exp(-theta^2/2)` |
| s_lv2 | 76 | 2.1 | polynomial (sum of products) | 2 | 7 | `-x*y - y^2 + 2*y` |
| s_vdp1 | 96 | 2.7 | polynomial (sum of products) | 2 | 11 | `-10*x^3/3 + 10*x/3 + 10*y` |
| s_barmag1 | 120 | 3.2 | sin/cos | 2 | 10 | `-sin(x) + 0.5*sin(x - y)` |
| II_38_14 | 144 | 6.0 | rational (sum in denominator) | 2 | 5 | `Y/(2*sigma + 2)` |

The same histogram for the ten other full races on disk (different settings; in the harvest, restart and big-population races the smallest recorded generation of any fit is 2 to 4, so their "1" column is empty — not checked why):

| race | seed | population | effort | solved | 1 | 2-5 | 6-20 | 21-50 | 51-100 | 100+ |
|---|---|---|---|---|---|---|---|---|---|---|
| ab1_base | 7012 | 800 | 6 s | 46 | 29 | 4 | 3 | 6 | 1 | 3 |
| ab1_cleanse | 7012 | 800 | 6 s | 46 | 29 | 4 | 5 | 5 | 3 | 0 |
| ab2_base_s11 | 7012 | 800 | 6 s | 46 | 30 | 4 | 3 | 5 | 1 | 3 |
| ab3_rnc5_s11 | 7012 | 800 | 6 s | 48 | 30 | 5 | 2 | 8 | 3 | 0 |
| ab2_base_s12 | 7013 | 800 | 6 s | 44 | 29 | 5 | 5 | 4 | 1 | 0 |
| balanced_6s_s12_b | 7013 | 3000 | 6 s, balanced-pole | 25 | 11 | 0 | 4 | 2 | 2 | 6 |
| ab2_harvest_s11 | 7012 | 800 | 6 s | 47 | 0 | 31 | 4 | 3 | 3 | 6 |
| ab2_harvest_s12 | 7013 | 800 | 6 s, harvests 4 | 45 | 0 | 28 | 9 | 3 | 3 | 2 |
| ab3_restart3_s11 | 7012 | 800 | 6 s, 3 restarts | 54 | 0 | 31 | 4 | 3 | 6 | 10 |
| big_pop_redundancy_g35_s11 | 7012 | 20000 | 35 gens / 9 s | 42 | 0 | 23 | 14 | 5 | 0 | 0 |

## 2. What kind of solution each is (n = 59)

Classification is programmatic (classify.py, sympy on SRBench's `true_model`, `gamma`/`beta`/`I`/`E_n` passed as Symbols) and then checked by eye; three "number inside" flags were overridden by eye where the number factors out into the scale `a` (II_38_14, test_18, test_19). Head needed / nodes / depth are from `law_heads.json` for 127 laws and computed here for the 6 it lacks (my tree builder agrees with law_heads.json exactly on 52 of 127 laws, mean absolute difference 1.2 — so treat head needed as +/- 1). Each law gets ONE primary class, by the first that applies: arcsin/arccos/tanh, exp/log, sin/cos, sqrt, rational (a sum in a denominator), polynomial (a sum of products), monomial. "Number inside" = a number left in the structure after the leading factor (absorbed by `a`) and an additive constant (absorbed by `b`) are removed: integer (|n| <= 5), rational (1/2, 1/3, 0.05, 0.075, 0.1), or irrational-rounded (0.159, 6.28, 0.101, 0.119, 12.6).

| class | laws | solved | rate | found by search | median generation found | latest |
|---|---|---|---|---|---|---|
| monomial | 38 | 38 | 100% | 38 | 1.0 | 16 |
| polynomial (sum of products) | 14 | 6 | 43% | 6 | 28.5 | 96 |
| rational (sum in denominator) | 19 | 4 | 21% | 4 | 38.5 | 144 |
| sqrt | 18 | 1 | 6% | 1 | 1 | 1 |
| sin/cos | 30 | 8 | 27% | 10 | 3.5 | 120 |
| exp/log | 10 | 1 | 10% | 1 | 76 | 76 |
| arcsin/arccos/tanh | 4 | 1 | 25% | 1 | 39 | 39 |
| ALL | 133 | 59 | 44% | 61 | 1 | 144 |

| law | gen | secs | class | vars | head | nodes | depth | inside | the law |
|---|---|---|---|---|---|---|---|---|---|
| III_12_43 | 1 | 1.2 | monomial | 2 | 1 | 3 | 2 | none | `0.159*h*n` |
| III_21_20 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `-A_vec*q*rho_c_0/m` |
| III_7_38 | 1 | 0.3 | monomial | 3 | 2 | 5 | 3 | none | `12.6*B*mom/h` |
| II_27_16 | 1 | 0.3 | monomial | 3 | 3 | 6 | 3 | none | `Ef^2*c*epsilon` |
| II_27_18 | 1 | 0.3 | monomial | 2 | 3 | 4 | 3 | none | `Ef^2*epsilon` |
| II_34_11 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `B*g_*q/(2*m)` |
| II_34_2 | 1 | 0.3 | monomial | 3 | 2 | 5 | 3 | none | `q*r*v/2` |
| II_34_29a | 1 | 0.3 | monomial | 3 | 2 | 5 | 3 | none | `0.08*h*q/m` |
| II_34_29b | 1 | 0.3 | monomial | 5 | 6 | 9 | 5 | none | `6.28*B*Jz*g_*mom/h` |
| II_34_2a | 1 | 0.3 | monomial | 3 | 2 | 5 | 3 | none | `0.159*q*v/r` |
| II_38_3 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `A*Y*x/d` |
| II_3_24 | 1 | 0.3 | monomial | 2 | 3 | 4 | 3 | none | `0.08*Pwr/r^2` |
| II_8_31 | 1 | 0.3 | monomial | 2 | 3 | 4 | 3 | none | `Ef^2*epsilon/2` |
| I_12_1 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `Nn*mu` |
| I_12_5 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `Ef*q2` |
| I_14_3 | 1 | 1.1 | monomial | 3 | 2 | 5 | 3 | none | `g*m*z` |
| I_14_4 | 1 | 0.3 | monomial | 2 | 3 | 4 | 3 | none | `k_spring*x^2/2` |
| I_25_13 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `q/C` |
| I_29_4 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `omega/c` |
| I_34_27 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `0.159*h*omega` |
| I_34_8 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `B*q*v/p` |
| I_38_12 | 1 | 0.3 | monomial | 4 | 7 | 9 | 4 | none | `0.318*epsilon*h^2/(m*q^2)` |
| I_39_1 | 1 | 0.3 | monomial | 2 | 1 | 3 | 2 | none | `3*V*pr/2` |
| I_39_22 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `T*kb*n/V` |
| I_43_16 | 1 | 0.3 | monomial | 4 | 4 | 7 | 4 | none | `Volt*mu_drift*q/d` |
| I_43_31 | 1 | 0.3 | monomial | 3 | 2 | 5 | 3 | none | `T*kb*mob` |
| s_vdp2 | 1 | 0.2 | monomial | 1 | 0 | 1 | 1 | none | `-x/10` |
| III_13_18 | 2 | 0.3 | monomial | 4 | 5 | 8 | 4 | none | `12.6*E_n*d^2*k/h` |
| III_15_27 | 2 | 0.3 | monomial | 3 | 3 | 5 | 3 | none | `6.28*alpha/(d*n)` |
| II_8_7 | 2 | 0.3 | monomial | 3 | 3 | 6 | 3 | none | `0.048*q^2/(d*epsilon)` |
| III_15_14 | 3 | 0.4 | monomial | 3 | 6 | 7 | 4 | none | `0.013*h^2/(E_n*d^2)` |
| II_13_17 | 3 | 0.4 | monomial | 4 | 8 | 10 | 5 | none | `0.159*I/(c^2*epsilon*r)` |
| II_4_23 | 5 | 0.4 | monomial | 3 | 3 | 5 | 3 | none | `0.08*q/(epsilon*r)` |
| I_12_2 | 5 | 0.4 | monomial | 4 | 7 | 8 | 4 | none | `0.08*q1*q2/(epsilon*r^2)` |
| I_12_4 | 5 | 0.4 | monomial | 3 | 5 | 6 | 4 | none | `0.08*q1/(epsilon*r^2)` |
| II_11_20 | 6 | 0.5 | monomial | 5 | 5 | 10 | 4 | none | `Ef*n_rho*p_d^2/(3*T*kb)` |
| I_32_5 | 7 | 0.5 | monomial | 4 | 7 | 10 | 4 | none | `0.053*a^2*q^2/(c^3*epsilon)` |
| III_19_51 | 16 | 0.9 | monomial | 5 | 10 | 14 | 5 | none | `-0.125*m*q^4/(epsilon^2*h^2*n^2)` |
| II_37_1 | 2 | 0.3 | polynomial (sum of products) | 3 | 3 | 7 | 3 | none | `B*mom*(chi + 1)` |
| II_2_42 | 18 | 1.0 | polynomial (sum of products) | 5 | 9 | 10 | 5 | none | `A*kappa*(-T1 + T2)/d` |
| I_24_6 | 27 | 1.3 | polynomial (sum of products) | 4 | 7 | 10 | 4 | none | `m*x^2*(omega^2 + omega_0^2)/4` |
| I_13_12 | 30 | 1.5 | polynomial (sum of products) | 5 | 7 | 12 | 4 | none | `G*m1*m2*(1/r2 - 1/r1)` |
| s_lv2 | 76 | 2.1 | polynomial (sum of products) | 2 | 7 | 12 | 5 | integer | `-x*y - y^2 + 2*y` |
| s_vdp1 | 96 | 2.7 | polynomial (sum of products) | 2 | 11 | 12 | 5 | integer | `-10*x^3/3 + 10*x/3 + 10*y` |
| II_10_9 | 19 | 1.0 | rational (sum in denominator) | 3 | 5 | 7 | 4 | none | `sigma_den/(epsilon*(chi + 1))` |
| I_39_11 | 32 | 1.5 | rational (sum in denominator) | 3 | 3 | 7 | 3 | none | `V*pr/(gamma - 1)` |
| I_34_1 | 45 | 2.6 | rational (sum in denominator) | 3 | 6 | 8 | 5 | none | `omega_0/(1 - v/c)` |
| II_38_14 | 144 | 6.0 | rational (sum in denominator) | 2 | 5 | 7 | 4 | none | `Y/(2*sigma + 2)` |
| I_47_23 | 1 | 0.3 | sqrt | 3 | 3 | 6 | 4 | none | `sqrt(gamma*pr/rho)` |
| II_15_4 | 1 | 0.3 | sin/cos | 3 | 3 | 6 | 3 | none | `-B*mom*cos(theta)` |
| II_15_5 | 1 | 0.3 | sin/cos | 3 | 3 | 6 | 3 | none | `-Ef*p_d*cos(theta)` |
| I_18_12 | 1 | 0.3 | sin/cos | 3 | 3 | 6 | 3 | none | `F*r*sin(theta)` |
| I_18_14 | 1 | 0.3 | sin/cos | 4 | 4 | 8 | 4 | none | `m*r*v*sin(theta)` |
| II_6_11 | 6 | 0.5 | sin/cos | 4 | 7 | 9 | 4 | none | `0.08*p_d*cos(theta)/(epsilon*r^2)` |
| s_glider2 | 56 | 2.0 | sin/cos | 2 | 6 | 7 | 5 | none | `x - cos(y)/x` |
| III_17_37 | 71 | 3.1 | sin/cos | 3 | 7 | 8 | 5 | none | `beta*(alpha*cos(theta) + 1)` |
| s_barmag1 | 120 | 3.2 | sin/cos | 2 | 10 | 11 | 6 | integer | `-sin(x) + 0.5*sin(x - y)` |
| I_6_2a | 76 | 3.3 | exp/log | 1 | 4 | 5 | 4 | rational | `0.399*exp(-theta^2/2)` |
| I_30_5 | 39 | 1.8 | arcsin/arccos/tanh | 3 | 4 | 6 | 4 | none | `arcsin(lambd/(d*n))` |

## 3. The unsolved laws (n = 74) and the solve rate by class

`*` = not a search loss (II_6_15b, s_barmag2 found; s_bacres2 a fake under the bar). "model size" = sympy nodes in our reported model. test R2 `nan` is as recorded in the JSON.

| law | class | vars | head | nodes | depth | inside | test R2 | log10 p | 1-R2 val | t_depth | gens | model size | the law |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_17 | polynomial (sum of products) | 6 | 16 | 21 | 7 | none | 0.9991 | -5.78 | 7.1e-04 | 2 | 129 | 38 | `(m^2*omega^2*x^2*(alpha*x/y + 1) + p^2)/(2*m)` |
| s_lv1 | polynomial (sum of products) | 2 | 8 | 13 | 5 | integer | nan | -4.46 | 6.8e-02 | 2 | 231 | 45 | `-x^2 - 2*x*y + 3*x` |
| test_9 | polynomial (sum of products) | 5 | 19 | 21 | 7 | none | 0.8515 | -4.06 | 2.0e-02 | 2 | 139 | 86 | `-32*G^4*m1^2*m2^2*(m1 + m2)/(5*c^5*r^5)` |
| I_11_19 | polynomial (sum of products) | 6 | 5 | 11 | 4 | none | nan | -3.62 | 1.4e-02 | 2 | 125 | 39 | `x1*y1 + x2*y2 + x3*y3` |
| test_19 | polynomial (sum of products) | 6 | 21 | 26 | 7 | integer | 0.9672 | -3.62 | 3.8e-02 | 2 | 130 | 37 | `-0.04*(H_G^2*c^2*(1 - 2*alpha) + c^4*k_f/r^2)/G` |
| I_13_4 | polynomial (sum of products) | 4 | 7 | 10 | 5 | none | nan | -3.54 | 1.8e-02 | 2 | 148 | 37 | `m*(u^2 + v^2 + w^2)/2` |
| test_18 | polynomial (sum of products) | 5 | 12 | 16 | 6 | none | -1.0e+16 | -2.47 | 5.9e-02 | 2 | 139 | 38 | `0.119*(H_G^2 + c^2*k_f/r^2)/G` |
| II_36_38 | polynomial (sum of products) | 8 | 18 | 22 | 6 | none | 0.8747 | -2.12 | 1.1e-01 | 1 | 149 | 24 | `H*mom/(T*kb) + M*alpha*mom/(T*c^2*epsilon*kb)` |
| s_bacres2* | rational (sum in denominator) | 2 | 12 | 13 | 6 | rational | 1.0000 | -17.55 | 1.0e-11 | 2 | 40 | 28 | `-x*y/(0.5*x^2 + 1) + 10` |
| s_bacres1 | rational (sum in denominator) | 2 | 15 | 16 | 6 | rational | 0.9976 | -10.04 | 5.8e-07 | 2 | 218 | 65 | `-x*y/(0.5*x^2 + 1) - x + 20` |
| test_12 | rational (sum in denominator) | 5 | 29 | 30 | 9 | irrational-rounded | 1.0000 | -10.02 | 1.0e-06 | 2 | 146 | 10 | `0.08*q*(12.6*Volt*d*epsilon - d*q*y^3/(-d^2 + y^2)^2)/(epsilon*y^2)` |
| II_11_28 | rational (sum in denominator) | 2 | 10 | 13 | 6 | rational | 1.0000 | -8.61 | 1.6e-05 | 2 | 150 | 30 | `alpha*n/(-alpha*n/3 + 1) + 1` |
| I_43_43 | rational (sum in denominator) | 4 | 7 | 9 | 4 | none | 1.0000 | -7.77 | 2.2e-05 | 2 | 129 | 24 | `kb*v/(A*(gamma - 1))` |
| II_11_27 | rational (sum in denominator) | 4 | 10 | 15 | 5 | rational | 0.9999 | -7.02 | 1.1e-04 | 2 | 151 | 17 | `Ef*alpha*epsilon*n/(-alpha*n/3 + 1)` |
| II_21_32 | rational (sum in denominator) | 5 | 10 | 12 | 6 | none | 0.9995 | -6.47 | 2.7e-04 | 2 | 150 | 30 | `0.08*q/(epsilon*r*(1 - v/c))` |
| I_27_6 | rational (sum in denominator) | 3 | 4 | 7 | 4 | none | nan | -5.22 | 1.9e-03 | 2 | 127 | 64 | `1/(n/d2 + 1/d1)` |
| I_32_17 | rational (sum in denominator) | 6 | 16 | 21 | 6 | none | 0.9980 | -4.86 | 1.7e-03 | 2 | 128 | 30 | `4.19*Ef^2*c*epsilon*omega^4*r^2/(omega^2 - omega_0^2)^2` |
| s_predprey2 | rational (sum in denominator) | 2 | 9 | 11 | 5 | rational | nan | -4.5 | 8.9e-03 | 2 | 226 | 46 | `y*(x/(x + 1) - 0.075*y)` |
| I_18_4 | rational (sum in denominator) | 4 | 5 | 11 | 4 | none | 0.9851 | -3.89 | 1.3e-02 | 2 | 134 | 54 | `(m1*r1 + m2*r2)/(m1 + m2)` |
| I_9_18 | rational (sum in denominator) | 9 | 20 | 23 | 7 | none | 0.9873 | -3.87 | 1.4e-02 | 2 | 133 | 31 | `G*m1*m2/((-x1 + x2)^2 + (-y1 + y2)^2 + (-z1 + z2)^2)` |
| I_16_6 | rational (sum in denominator) | 3 | 9 | 12 | 5 | none | 0.9848 | -3.84 | 1.5e-02 | 2 | 149 | 34 | `(u + v)/(1 + u*v/c^2)` |
| s_predprey1 | rational (sum in denominator) | 2 | 9 | 13 | 5 | integer | nan | -3.36 | 3.0e-02 | 2 | 214 | 44 | `x*(-x - y/(x + 1) + 4)` |
| II_11_3 | rational (sum in denominator) | 5 | 11 | 12 | 6 | none | 0.9199 | -2.93 | 3.8e-02 | 2 | 133 | 34 | `Ef*q/(m*(-omega^2 + omega_0^2))` |
| II_13_23 | sqrt | 3 | 9 | 11 | 7 | none | 1.0000 | -10.4 | 3.7e-06 | 2 | 150 | 46 | `rho_c_0/sqrt(1 - v^2/c^2)` |
| I_48_2 | sqrt | 3 | 12 | 14 | 7 | none | 1.0000 | -8.91 | 6.8e-06 | 2 | 145 | 26 | `c^2*m/sqrt(1 - v^2/c^2)` |
| I_10_7 | sqrt | 3 | 9 | 11 | 7 | none | 0.9999 | -7.42 | 1.5e-04 | 2 | 140 | 37 | `m_0/sqrt(1 - v^2/c^2)` |
| II_24_17 | sqrt | 3 | 11 | 13 | 6 | irrational-rounded | nan | -7.01 | 1.2e-04 | 2 | 148 | 53 | `3.14*sqrt(-1/d^2 + 0.101*omega^2/c^2)` |
| II_13_34 | sqrt | 3 | 11 | 13 | 7 | none | 0.9999 | -6.94 | 1.0e-04 | 2 | 142 | 35 | `rho_c_0*v/sqrt(1 - v^2/c^2)` |
| test_7 | sqrt | 5 | 10 | 14 | 6 | irrational-rounded | 0.9997 | -6.58 | 1.9e-04 | 2 | 126 | 41 | `2.89*sqrt(G*rho - 0.119*alpha*c^2/d^2)` |
| I_15_3x | sqrt | 4 | 14 | 16 | 7 | none | 0.9995 | -6.49 | 5.6e-04 | 2 | 144 | 48 | `(-t*u + x)/sqrt(1 - u^2/c^2)` |
| I_15_3t | sqrt | 4 | 17 | 19 | 7 | none | 0.9994 | -6.16 | 4.8e-04 | 2 | 135 | 48 | `(t - u*x/c^2)/sqrt(1 - u^2/c^2)` |
| test_5 | sqrt | 4 | 8 | 10 | 5 | none | 0.9993 | -5.91 | 8.0e-04 | 2 | 139 | 42 | `6.28*d^(3/2)/sqrt(G*(m1 + m2))` |
| II_6_15a | sqrt | 6 | 14 | 16 | 6 | none | 0.9984 | -5.74 | 1.8e-03 | 2 | 130 | 36 | `0.239*p_d*z*sqrt(x^2 + y^2)/(epsilon*r^5)` |
| I_15_10 | sqrt | 3 | 11 | 13 | 7 | none | nan | -5.62 | 1.9e-03 | 2 | 139 | 22 | `m_0*v/sqrt(1 - v^2/c^2)` |
| test_4 | sqrt | 5 | 18 | 20 | 8 | rational | nan | -5.12 | 4.9e-03 | 2 | 145 | 43 | `sqrt(2)*sqrt((E_n - L^2/(2*m*r^2) - U)/m)` |
| I_34_14 | sqrt | 3 | 15 | 17 | 7 | none | 0.9954 | -4.73 | 4.3e-03 | 2 | 142 | 29 | `omega_0*(1 + v/c)/sqrt(1 - v^2/c^2)` |
| III_10_19 | sqrt | 4 | 8 | 11 | 6 | none | nan | -4.63 | 4.3e-03 | 2 | 127 | 52 | `mom*sqrt(Bx^2 + By^2 + Bz^2)` |
| test_6 | sqrt | 7 | 21 | 24 | 8 | integer | 0.9848 | -3.71 | 1.4e-02 | 2 | 127 | 33 | `sqrt(2*E_n*L^2*epsilon^2/(Z_1^2*Z_2^2*m*q^4) + 1)` |
| test_16 | sqrt | 6 | 20 | 22 | 9 | none | 0.9705 | -3.19 | 2.7e-02 | 2 | 129 | 39 | `Volt*q + sqrt(c^4*m^2 + c^2*(-A_vec*q + p)^2)` |
| I_8_14 | sqrt | 4 | 10 | 12 | 6 | none | 0.9513 | -3.17 | 5.6e-02 | 2 | 144 | 42 | `sqrt((-x1 + x2)^2 + (-y1 + y2)^2)` |
| II_6_15b* | sin/cos | 4 | 9 | 12 | 5 | none | 1.0000 | -21.36 | 3.0e-14 | 1 | 41 | 13 | `0.239*p_d*sin(theta)*cos(theta)/(epsilon*r^3)` |
| s_barmag2* | sin/cos | 2 | 10 | 11 | 6 | integer | 1.0000 | -18.99 | 1.4e-13 | 2 | 45 | 22 | `-sin(y) - 0.5*sin(x - y)` |
| s_shearflow2 | sin/cos | 2 | 11 | 12 | 6 | rational | 1.0000 | -8.27 | 1.1e-05 | 2 | 234 | 18 | `(0.1*sin(y)^2 + cos(y)^2)*sin(x)` |
| s_glider1 | sin/cos | 2 | 6 | 8 | 4 | rational | 0.9974 | -6.46 | 2.1e-03 | 2 | 231 | 39 | `-0.05*x^2 - sin(y)` |
| test_3 | sin/cos | 4 | 16 | 17 | 7 | none | 0.9988 | -5.65 | 1.0e-03 | 2 | 151 | 32 | `d*(1 - alpha^2)/(alpha*cos(theta1 - theta2) + 1)` |
| test_15 | sin/cos | 4 | 18 | 20 | 8 | none | 0.9955 | -4.98 | 4.9e-03 | 2 | 138 | 26 | `omega*sqrt(1 - v^2/c^2)/(1 + v*cos(theta)/c)` |
| test_1 | sin/cos | 7 | 20 | 24 | 7 | rational | 0.9917 | -4.26 | 9.2e-03 | 2 | 144 | 47 | `Z_1^2*Z_2^2*alpha^2*c^2*hbar^2/(16*E_n^2*sin(theta/2)^4)` |
| III_15_12 | sin/cos | 3 | 7 | 9 | 6 | none | 0.9889 | -3.96 | 1.2e-02 | 2 | 129 | 24 | `2*U*(1 - cos(d*k))` |
| test_20 | sin/cos | 7 | 24 | 31 | 7 | none | nan | -3.67 | 3.2e-02 | 2 | 139 | 37 | `0.08*alpha^2*h^2*omega_0^2*(omega/omega_0 - sin(beta)^2 + omega_0/o...` |
| I_30_3 | sin/cos | 3 | 12 | 15 | 7 | rational | 0.9842 | -3.57 | 1.7e-02 | 2 | 135 | 25 | `Int_0*sin(n*theta/2)^2/sin(theta/2)^2` |
| test_14 | sin/cos | 5 | 15 | 21 | 7 | integer | nan | -3.3 | 2.4e-02 | 1 | 125 | 56 | `Ef*(d^3*(alpha - 1)/(r^2*(alpha + 2)) - r)*cos(theta)` |
| I_12_11 | sin/cos | 5 | 7 | 10 | 5 | none | 0.9745 | -3.19 | 2.8e-02 | 2 | 147 | 41 | `q*(B*v*sin(theta) + Ef)` |
| I_37_4 | sin/cos | 3 | 11 | 13 | 6 | integer | 0.9704 | -3.13 | 3.1e-02 | 2 | 150 | 50 | `I1 + I2 + 2*sqrt(I1*I2)*cos(delta)` |
| test_8 | sin/cos | 4 | 15 | 16 | 8 | none | 0.9655 | -3.05 | 3.4e-02 | 2 | 136 | 46 | `E_n/(E_n*(1 - cos(theta))/(c^2*m) + 1)` |
| s_shearflow1 | sin/cos | 2 | 3 | 5 | 3 | none | 0.9541 | -2.77 | 5.1e-02 | 2 | 234 | 23 | `cos(x)*cot(y)` |
| test_13 | sin/cos | 5 | 15 | 19 | 8 | integer | 0.8963 | -2.67 | 6.1e-02 | 2 | 134 | 64 | `0.08*q/(epsilon*sqrt(d^2 - 2*d*r*cos(alpha) + r^2))` |
| test_11 | sin/cos | 4 | 17 | 24 | 7 | rational | nan | -2.65 | 5.5e-02 | 2 | 145 | 50 | `4*I_0*sin(alpha/2)^2*sin(delta*n/2)^2/(alpha^2*sin(delta/2)^2)` |
| I_29_16 | sin/cos | 4 | 17 | 18 | 7 | integer | 0.9226 | -2.54 | 8.7e-02 | 2 | 130 | 35 | `sqrt(x1^2 - 2*x1*x2*cos(theta1 - theta2) + x2^2)` |
| I_50_26 | sin/cos | 4 | 12 | 14 | 7 | none | 0.8494 | -1.96 | 1.5e-01 | 2 | 147 | 40 | `x1*(alpha*cos(omega*t)^2 + cos(omega*t))` |
| III_9_52 | sin/cos | 6 | 23 | 24 | 8 | rational | 0.8505 | -1.94 | 1.7e-01 | 2 | 147 | 34 | `25.1*Ef*p_d*sin(t*(omega - omega_0)/2)^2/(h*t*(omega - omega_0)^2)` |
| test_2 | sin/cos | 6 | 25 | 29 | 10 | integer | 0.6947 | -1.42 | 3.1e-01 | 2 | 135 | 31 | `k_G*m*(sqrt(2*E_n*L^2/(k_G^2*m) + 1)*cos(theta1 - theta2) + 1)/L^2` |
| III_8_54 | sin/cos | 3 | 6 | 9 | 6 | irrational-rounded | 0.5061 | -0.99 | 4.7e-01 | 2 | 128 | 47 | `sin(6.28*E_n*t/h)^2` |
| III_4_32 | exp/log | 4 | 8 | 13 | 7 | irrational-rounded | 1.0000 | -9.21 | 3.8e-06 | 2 | 120 | 12 | `1/(exp(0.159*h*omega/(T*kb)) - 1)` |
| III_4_33 | exp/log | 4 | 11 | 16 | 7 | irrational-rounded | 0.9998 | -6.62 | 1.6e-04 | 1 | 126 | 30 | `0.159*h*omega/(exp(0.159*h*omega/(T*kb)) - 1)` |
| I_41_16 | exp/log | 5 | 15 | 20 | 8 | irrational-rounded | 0.9997 | -6.47 | 2.2e-04 | 2 | 125 | 29 | `0.016*h*omega^3/(c^2*(exp(0.159*h*omega/(T*kb)) - 1))` |
| I_44_4 | exp/log | 5 | 6 | 10 | 4 | none | 0.9988 | -5.57 | 9.3e-04 | 2 | 148 | 28 | `T*kb*n*log(V2/V1)` |
| I_6_2b | exp/log | 3 | 12 | 13 | 8 | rational | 0.9933 | -4.88 | 6.8e-03 | 2 | 129 | 34 | `0.399*exp(-(theta - theta1)^2/(2*sigma^2))/sigma` |
| I_6_2 | exp/log | 2 | 8 | 10 | 6 | rational | 0.9965 | -4.81 | 3.6e-03 | 2 | 130 | 45 | `0.399*exp(-theta^2/(2*sigma^2))/sigma` |
| III_14_14 | exp/log | 5 | 8 | 12 | 6 | none | 0.9874 | -4.03 | 9.7e-03 | 2 | 141 | 48 | `I_0*(exp(Volt*q/(T*kb)) - 1)` |
| II_35_18 | exp/log | 5 | 16 | 20 | 7 | none | 0.9597 | -3.03 | 3.7e-02 | 2 | 132 | 42 | `n_0/(exp(B*mom/(T*kb)) + exp(-B*mom/(T*kb)))` |
| I_40_1 | exp/log | 6 | 10 | 13 | 7 | none | 0.9170 | -2.35 | 8.7e-02 | 1 | 120 | 24 | `n_0*exp(-g*m*x/(T*kb))` |
| I_26_2 | arcsin/arccos/tanh | 2 | 4 | 5 | 4 | none | 0.9988 | -5.85 | 1.2e-03 | 2 | 149 | 29 | `arcsin(n*sin(theta2))` |
| test_10 | arcsin/arccos/tanh | 3 | 17 | 18 | 8 | none | 0.9972 | -5.09 | 2.6e-03 | 2 | 140 | 40 | `arccos((cos(theta2) - v/c)/(1 - v*cos(theta2)/c))` |
| II_35_21 | arcsin/arccos/tanh | 5 | 8 | 12 | 5 | none | 0.9706 | -3.3 | 3.0e-02 | 2 | 132 | 32 | `mom*n_rho*tanh(B*mom/(T*kb))` |

69 of the 74 unsolved fits end at t_depth 2 (the free limit); of the 59 solved, 15 do. Median reported model size: 37 nodes unsolved against 13 nodes in their laws; 8 against 7 for the solved.

### Solve rate by feature (a law can carry several)

| feature present (laws can carry several) | laws | solved | rate | found by search | median generation found | latest |
|---|---|---|---|---|---|---|
| pure product-and-power monomial | 38 | 38 | 100% | 38 | 1.0 | 16 |
| has a sum (sum of products) | 76 | 13 | 17% | 14 | 45 | 144 |
| rational: sum or difference in a denominator | 39 | 4 | 10% | 4 | 38.5 | 144 |
| has sqrt | 23 | 1 | 4% | 1 | 1 | 1 |
| has exp | 9 | 1 | 11% | 1 | 76 | 76 |
| has log | 1 | 0 | 0% | 0 | - | - |
| has sin/cos | 32 | 8 | 25% | 10 | 3.5 | 120 |
| has arcsin/arccos/tanh | 4 | 1 | 25% | 1 | 39 | 39 |
| number inside the structure: none | 98 | 55 | 56% | 56 | 1 | 144 |
| number inside the structure: integer | 13 | 3 | 23% | 4 | 96 | 120 |
| number inside the structure: rational | 15 | 1 | 7% | 1 | 76 | 76 |
| number inside the structure: irrational-rounded | 7 | 0 | 0% | 0 | - | - |

### By input variables, and by head needed

| input variables | laws | solved | rate | found by search | median generation found | latest |
|---|---|---|---|---|---|---|
| 1-2 | 30 | 18 | 60% | 19 | 1.0 | 144 |
| 3 | 37 | 22 | 59% | 22 | 1.5 | 71 |
| 4 | 32 | 14 | 44% | 15 | 1.0 | 27 |
| 5 | 20 | 5 | 25% | 5 | 16 | 30 |
| 6+ | 14 | 0 | 0% | 0 | - | - |

| head needed | laws | solved | rate | found by search | median generation found | latest |
|---|---|---|---|---|---|---|
| 0-4 | 40 | 37 | 92% | 37 | 1 | 76 |
| 5-8 | 34 | 18 | 53% | 18 | 6.5 | 144 |
| 9-12 | 30 | 4 | 13% | 6 | 57.0 | 120 |
| 13-16 | 12 | 0 | 0% | 0 | - | - |
| 17+ | 17 | 0 | 0% | 0 | - | - |

### Cells: variables x head needed (solved/laws), and class x head needed

| vars \ head needed | 0-4 | 5-8 | 9-12 | 13-16 | 17+ |
|---|---|---|---|---|---|
| 1-2 | 13/15 | 3/6 | 2/8 | 0/1 | . |
| 3 | 17/18 | 5/7 | 0/10 | 0/1 | 0/1 |
| 4 | 7/7 | 7/13 | 0/5 | 0/3 | 0/4 |
| 5 | . | 3/7 | 2/6 | 0/4 | 0/3 |
| 6+ | . | 0/1 | 0/1 | 0/3 | 0/9 |

| class \ head needed | 0-4 | 5-8 | 9-12 | 13-16 | 17+ |
|---|---|---|---|---|---|
| monomial | 28/28 | 9/9 | 1/1 | . | . |
| polynomial (sum of products) | 1/1 | 3/6 | 2/3 | 0/1 | 0/3 |
| rational (sum in denominator) | 1/2 | 3/5 | 0/8 | 0/2 | 0/2 |
| sqrt | 1/1 | 0/2 | 0/8 | 0/3 | 0/4 |
| sin/cos | 4/5 | 3/7 | 1/7 | 0/4 | 0/7 |
| exp/log | 1/1 | 0/4 | 0/3 | 0/2 | . |
| arcsin/arccos/tanh | 1/2 | 0/1 | . | . | 0/1 |

**At or near 100%:** monomials 38 of 38 (any head, any variable count up to 5); head needed <= 4: 37 of 40. **At 0%:** head needed >= 13: 0 of 29; 6+ variables: 0 of 14; irrational-rounded number inside: 0 of 7; sqrt laws other than the one monomial under a root (I_47_23): 0 of 17. **In between:** head 5-8: 18 of 34; head 9-12: 4 of 30. The rate falls with head needed much faster than with variable count: at head <= 4 the rate is 93% whatever the variable count.

## 5. Near misses proper (n = 19)

Unsolved fits with test R2 >= 0.999 or log10 p <= -8, less the three non-search losses. Grouped by what the model put in place of the piece it lacks (grouping by eye from the full strings).

| group | substitution pattern | near misses |
|---|---|---|
| A | an imitation built from tanh / exp / log / arcsin for a pole or a root (1/sqrt(1 - x^2), 1/(1 - x), sqrt(a - b), 1/sqrt(a + b)) | 10 |
| B | a series: the first one or two terms of the expansion of the missing function (a Taylor-like polynomial) | 5 |
| C | the dominant term alone; the small term is absent | 1 |
| D | the right structure with a wrong number inside, and an extra factor to cover it | 1 |
| E | a patchwork of reciprocals or products for a sum of products | 2 |

| group | law | the law | our model (SRBench-tidied, law variable names) | test R2 | log10 p | model size | law nodes | what it got right | what it put in place of the missing piece |
|---|---|---|---|---|---|---|---|---|---|
| A | II_13_23 | `rho_c_0/sqrt(1 - v^2/c^2)` | `0.825*sqrt(rho_c_0*(rho_c_0 + 3/(125*(0.2*tanh(log(1/c)) + 1)^3 - 2))*cos(tanh(c)))*exp(exp(v^2/c^2)/2) - 0.024` | 0.999997 | -10.4 | 46 | 11 | the factor rho_c_0 and the argument v^2/c^2 | exp(exp(v^2/c^2)/2) times a tanh/log/cos patch, for 1/sqrt(1 - v^2/c^2) |
| A | I_48_2 | `c^2*m/sqrt(1 - v^2/c^2)` | `0.429*c*m/(tanh(0.428/c)*tanh(3*log(c)/(v + 1))) + 0.0186` | 0.999993 | -8.91 | 26 | 14 | the product c*m | 1/(tanh(0.43/c)*tanh(3*log(c)/(v+1))) for c/sqrt(1 - v^2/c^2) |
| A | I_10_7 | `m_0/sqrt(1 - v^2/c^2)` | `0.993*m_0*sqrt(tanh(sqrt(m_0))*tanh(5*sqrt(v) - 1.41*sqrt(c/2 - 1))/tanh(c/(2*v))) + 0.155` | 0.999864 | -7.42 | 37 | 11 | the factor m_0 and the ratio c/v | sqrt of a ratio of tanh terms for 1/sqrt(1 - v^2/c^2) |
| A | II_13_34 | `rho_c_0*v/sqrt(1 - v^2/c^2)` | `0.576*v*(rho_c_0 - exp(-v))*(2 + 1/(c - v) - 1/(3 - 3/c))*tanh(sqrt(c)) + 0.347` | 0.999905 | -6.94 | 35 | 13 | the product v*rho_c_0 (rho_c_0 - exp(-v)) | (2 + 1/(c - v) - ...)*tanh(sqrt(c)): a pole at v = c standing in for the root |
| A | I_15_3t | `(t - u*x/c^2)/sqrt(1 - u^2/c^2)` | `0.089*sqrt((125 - 6.31/c)*(t + 5 + t*u/(c*(c + sin(t) - tanh(sin(t - u)))))*(t + cos(x/c) + 2)) - 3.91` | 0.999415 | -6.16 | 48 | 19 | t as the leading term and the ratios x/c, t*u/c | sqrt of a product of three sums with sin/tanh, for the root in the denominator |
| A | II_11_28 | `alpha*n/(-alpha*n/3 + 1) + 1` | `0.028 - 1.75*log(-0.333*alpha*n + 0.333*tanh(10.0*alpha*n + 0.4*sqrt(n)*log(n) + 4.0) + 0.24)` | 0.999986 | -8.61 | 30 | 13 | the argument -alpha*n/3 exactly, inside the function | log(-alpha*n/3 + tanh(...)/3 + 0.24) for 1/(1 - alpha*n/3) |
| A | II_21_32 | `0.08*q/(epsilon*r*(1 - v/c))` | `0.078*q*exp(v/(c - tanh(v)))/(epsilon*r*tanh(3/sqrt(v)))` | 0.999502 | -6.47 | 30 | 12 | the whole product q/(epsilon*r) and the scale 0.078 ~ 0.0796 | exp(v/(c - tanh(v)))/tanh(3/sqrt(v)) for 1/(1 - v/c) |
| A | I_43_43 | `kb*v/(A*(gamma - 1))` | `0.00299 + 1.0*gamma*kb*v*arcsin(1.56/gamma^2)/(A*sqrt(log(gamma))*tanh(gamma))` | 0.999976 | -7.77 | 24 | 9 | the whole product kb*v/A, scale 1.00 | gamma*arcsin(1.56/gamma^2)/(sqrt(log(gamma))*tanh(gamma)) for 1/(gamma - 1) |
| A | test_7 | `2.89*sqrt(G*rho - 0.119*alpha*c^2/d^2)` | `2.06*sqrt(2)*sqrt(G*rho*tanh((d + sqrt(G*rho/alpha))/c)*tanh((d + sqrt(G*rho/alpha))*tanh(d))) - 0.054` | 0.999726 | -6.58 | 41 | 14 | sqrt(G*rho) as the core under the root | a product of two tanh terms under the root for the subtraction of 0.119*alpha*c^2/d^2 |
| A | test_5 | `6.28*d^(3/2)/sqrt(G*(m1 + m2))` | `0.023 + 2.69*sqrt(-d*(d + exp(-m1^3)*log(Abs(log(4/m2))))*(d*log(0.2*m2) - d)/G)/m1^(1/4)` | 0.999293 | -5.91 | 42 | 10 | sqrt(d*d*d/G) - the right powers of d and G under a root | m1^(-1/4) and log terms in m2 for 1/sqrt(m1 + m2) |
| B | II_11_27 | `Ef*alpha*epsilon*n/(-alpha*n/3 + 1)` | `0.947*Ef*alpha*epsilon*n*(0.54*alpha*n + 1) + 0.005` | 0.999905 | -7.02 | 17 | 15 | the whole product Ef*alpha*epsilon*n | 1 + 0.54*alpha*n: the first two terms of the series of 1/(1 - alpha*n/3) |
| B | III_4_32 | `1/(exp(0.159*h*omega/(T*kb)) - 1)` | `6.28*T*kb/(h*omega) - 0.472` | 0.999997 | -9.21 | 12 | 13 | the argument T*kb/(h*omega) and the scale 6.28 = 1/0.159 | 6.28*T*kb/(h*omega) - 0.47: the series 1/x - 1/2 of 1/(exp(x) - 1) |
| B | III_4_33 | `0.159*h*omega/(exp(0.159*h*omega/(T*kb)) - 1)` | `1.0*T*kb - 0.2*h - 0.2*omega + 0.2*sqrt(Abs(h - omega)) + 0.276 + 0.2/(T*kb)` | 0.999830 | -6.62 | 30 | 16 | the leading term T*kb with scale 1.00 | T*kb - 0.2*h - 0.2*omega + ...: the series kT - x/2 with h*omega replaced by h + omega |
| B | I_41_16 | `0.016*h*omega^3/(c^2*(exp(0.159*h*omega/(T*kb)) - 1))` | `0.067*T*kb*omega*(omega + 0.2)*sqrt(Abs(2 - h*tanh(T)/(T*kb)))/c^2 + 0.00101` | 0.999704 | -6.47 | 29 | 20 | T*kb*omega^2/c^2 - the low-frequency limit of the law | sqrt(Abs(2 - h*tanh(T)/(T*kb))) as the correction for the exp |
| B | I_15_3x | `(-t*u + x)/sqrt(1 - u^2/c^2)` | `-1.01*t*(u + 1/(c*x)) + 1.01*u*(c/(-x + log(Abs(log(c) + 3)) + 3) + x)*exp(-c) + 1.01*x - 0.112 + 1.01*u/c` | 0.999477 | -6.49 | 48 | 16 | x - t*u, the numerator, scale 1.01 | small additive corrections (u/c, exp(-c) terms) for the 1/sqrt(1 - u^2/c^2) factor |
| C | test_12 | `0.08*q*(12.6*Volt*d*epsilon - d*q*y^3/(-d^2 + y^2)^2)/(epsilon*y^2)` | `1.0*Volt*d*q/y^2 - 0.012` | 0.999999 | -10.02 | 10 | 30 | the first term Volt*d*q/y^2 exactly (scale 1.0001) | nothing: the second term d*q*y^3/(y^2 - d^2)^2 is absent, an offset -0.012 covers it |
| D | s_shearflow2 | `(0.1*sin(y)^2 + cos(y)^2)*sin(x)` | `(0.083*cos(y)^2 + 0.01)*(sqrt(Abs(cos(y))) + 9.87)*sin(x)` | 0.999988 | -8.27 | 18 | 12 | the structure (cos(y)^2 + k)*sin(x); law has k = 1/9 = 0.111, ours 0.120 | a factor (sqrt(Abs(cos(y))) + 9.87) to cover the error in k |
| E | s_bacres1 | `-x*y/(0.5*x^2 + 1) - x + 20` | `-0.983*x - 0.492*log(sqrt(x)) + 20.3 + 0.983/(0.333*x*(x + cos(x/(x - 7.39))) - 0.333*y) + 0.983/(0.394*y + 3) - (1.97*y + 5.9/y)/x - 2.46/(x*y)` | 0.997630 | -10.04 | 65 | 16 | the terms -x + 20 (ours -0.98*x + 20.3) | a sum of five reciprocals and a log for -x*y/(0.5*x^2 + 1) |
| E | test_17 | `(m^2*omega^2*x^2*(alpha*x/y + 1) + p^2)/(2*m)` | `0.502*omega*sqrt(tanh(x)/tanh(alpha))*(omega*x^2 + 2*y - 3)*(alpha*m*x + m + y + tanh(m - 3))/y - 5.63` | 0.999083 | -5.78 | 38 | 21 | the factor (alpha*m*x + m ...)/y and omega*x^2 | a product of two sums times sqrt(tanh/tanh), for the sum of two products |

In 17 of the 19 the model has the law's core product or leading term right, with the fitted scale within 1-2% of the law's constant (1.0001, 1.00, 0.993, 0.078 for 0.0796, 6.28 for 1/0.159). What is missing is one sub-expression, and in 10 of 19 that sub-expression is a pole or a root, imitated by stacked tanh/exp/log at t_depth 2. II_11_28 has the exact argument -alpha*n/3 sitting inside a log where the law has a reciprocal.

Not search losses, for the record: II_6_15b `0.2387*sin(theta)*cos(theta)*p_d/(epsilon*r^3)` (the law; scorer n), s_barmag2 (the law plus `x/2 + arccos(cos(x))/2 - pi`, which is 0 on the data's range), s_bacres2 (val 1-R2 1.0e-11, log10 p -17.55 where every law is <= -19.4).

## 6. Does the data support "found early or not at all"?

**The 6 s race.** Solved-rate as a function of generations run cannot be read off this race: a solved fit stops at the generation it was found, and every unsolved fit ran to the clock (120 to 234 generations, median 139). The cumulative curve of section 1 is the usable view: 32 at generation 1, 41 by 5, 47 by 20, 52 by 50, 57 by 100, 59 by 150. The curve is steep then flat, but not dead: 7 solves arrive between generation 51 and 144, over the 81 problems still open at generation 50.

**The 30 s races on disk** (rust_race_7012 and _b, seed 7012, population 800; partial: 39 and 20 laws scored): 15 and 9 solved; the only solve after the first 6 seconds in either is strogatz_barmag1 at generation 1,091 (17-20 s).

**The long runs** (growhead200_s12: population 6,000, head growing 12 to 34 by 1 every 200 generations, 5,000 generations; one progress row per 25 generations; "improvement" = best train 1-R2 falls by 10% or more between rows):

| law | 6 s race: gens / 1-R2 train / log10 p | growhead200 1-R2 train @150 | @1000 | @5000 / log10 p | decades gained 150->5000 | long runs (all settings) | of which solved | last >=10% drop at generation | share of run after it |
|---|---|---|---|---|---|---|---|---|---|
| II_11_27 | 151 / 9.9e-05 / -7.02 | - | - | solved at generation 16 | - | 9 | 2 | - | - |
| II_24_17 | 148 / 1.0e-04 / -7.01 | 1.7e-04 | 4.9e-05 | 1.2e-05 / -8.79 | 1.2 | 10 | 0 | 3200 | 36% |
| II_11_3 | 133 / 5.9e-02 / -2.93 | 3.6e-03 | 1.1e-03 | 2.3e-04 / -6.32 | 1.2 | 7 | 0 | 2650 | 47% |
| III_4_32 | 120 / 2.8e-06 / -9.21 | 1.6e-06 | 4.3e-07 | 1.1e-09 / -15.13 | 3.2 | 6 | 0 | 2275 | 55% |
| I_48_2 | 145 / 6.6e-06 / -8.91 | 6.2e-06 | 1.9e-07 | 9.8e-08 / -11.67 | 1.8 | 9 | 0 | 3875 | 22% |
| I_15_10 | 139 / 1.9e-03 / -5.62 | 2.4e-04 | 4.4e-05 | 9.1e-08 / -11.64 | 3.4 | 7 | 0 | 4100 | 18% |
| I_26_2 | 149 / 1.1e-03 / -5.85 | 6.8e-04 | 3.3e-04 | 1.6e-04 / -7.66 | 0.6 | 15 | 0 | 1525 | 70% |
| I_6_2 | 130 / 3.5e-03 / -4.81 | 6.8e-03 | 3.5e-03 | 1.6e-03 / -5.59 | 0.6 | 7 | 0 | 2275 | 55% |
| I_29_16 | 130 / 7.8e-02 / -2.54 | 7.9e-02 | 4.6e-02 | 3.4e-02 / -3.17 | 0.4 | 6 | 0 | 3650 | 27% |
| III_9_52 | 147 / 1.5e-01 / -1.94 | 1.2e-01 | 9.6e-02 | 6.0e-02 / -2.8 | 0.3 | 9 | 0 | 3575 | 28% |

- 78 improvements across the nine: 19 by generation 150, 22 between 151 and 1,000, 37 after 1,000. Search does not stop improving the fit: the last 10% drop lands between generation 1,525 and 4,100.
- The error does fall — by 0.3 to 3.4 decades between generation 150 and 5,000 — and log10 p falls with it (I_15_10 -6.2 to -11.6, III_4_32 -9.3 to -15.1). None crosses the -19.4 where laws sit, and none is scored solved. The final strings (in the log) are longer imitations, not shorter ones.
- Head steps: 18 of the 55 improvements between generation 200 and 4,425 (the head reaches 34 at 4,400) fall in the row at a head step or the one after it; 2 of every 8 rows are such rows, so about 14 are expected by chance. 18 against 14 is not separable at this n.
- big_long_I_48_2 (population 12,000, 10,000 generations, SMOGD/SMOTE block on): 10 improvements, the last at generation 1500; 85% of the run came after it; train 1-R2 1.9e-7 at generation 150, 2.2e-8 at the end.
- The one solve in the long runs, II_11_27, came at generation 16 in growhead200 (head 12, population 6,000) — and at generation 523 in head34_s12, the single case on disk of a law first reached after generation 200 in a long run.
- I_26_2, a 5-node law, was not solved in 15 long runs up to 5,000 generations, and was found at generation 68 in a 6 s balanced-pole race.

**What the numbers say.** Solves concentrate at the start: 54% at generation 1, 69% by generation 5, and the same shape in all ten other races. Long single-population runs keep lowering the error of an imitation for thousands of generations without turning it into the law (0 of 9). Every law that moved from unsolved to solved/found between runs did so under a different draw (seed, pole, restart, population), at generation 9 to 199 — not deep into a long run, II_11_27 at 523 excepted. That is what "does not climb out of the basin; diversity is the issue" predicts.

**What they cannot say.** One seed for the race studied; the other races differ in several settings at once (population, head, constants range, engine version), so no single factor is isolated. The long runs cover only the ten hardest laws, chosen because they were never solved — they say nothing about whether the 25 "maybe" laws would fall at 1,000 generations; no run on disk tests that. The 6 s race has 18 search-found solves up to generation 144, so generations are not worthless inside the first 150; the data only shows the return falling steeply.

## 7. What to try, ranked by expected solves

Each item is one of Andrew's mechanisms tied to a count above. The counts are upper bounds on what the item could reach, not forecasts.

| # | what | tied to | reach |
|---|---|---|---|
| 1 | **Restarts / more islands fed by the pump, rather than longer runs.** Same 6 s, several fresh intake populations. | All 10 "yes" laws appeared under a different draw, 3 of them in the 3-restart race; restarts alone took 47 to 54 solved on seed 7012 (section 4.2); 0 of 9 long runs solved. | the 10 "yes" laws |
| 2 | **Reporting fixes (not search).** `Abs` of a positive quantity, arccos(cos(x)) on the principal range (a data guided rewrite already covers acos(cos e)), sin*cos vs sin(2x). | s_barmag2, II_6_15b now; II_21_32, s_shearflow1, test_5 met the bar elsewhere and were scored n. | 2 now, 3 more when found |
| 3 | **Population for the head <= 8 cells.** | Head 5-8: 18 of 34 solved; 9 of the 25 "maybe" laws need head <= 8 (I_27_6, I_18_4, I_13_4, s_lv1, s_glider1, III_10_19, I_11_19, III_14_14, II_35_21). I_44_4 was solved at generation 23-25 only at population 20,000. Counterweight: the population-20,000 race solved 42 against 47 at population 800 (section 4.2), so population needs its generations too. | up to 9 "maybe" |
| 4 | **t_depth: the free allowance of 2 is where imitations live.** | 69 of 74 unsolved fits end at t_depth 2; 10 of 19 near misses are tanh/exp/log stacks standing in for a pole or root. Worth an A/B of free-up-to-1 on the 9-law 1/sqrt(1 - v^2/c^2) family, where the law itself has t_depth 1. | up to 9 + several of group A |
| 5 | **Constants inside the structure: snap + data guided rewrites.** | 7 laws need 1/(2 pi), 2 pi, 1/pi^2, 4 pi inside; 0 of 7 ever solved; rational inside 1 of 15. Snap's constant table holds these numbers; today they cannot enter a gene. III_4_32's fitted scale is 6.28 = 1/0.159 — the number is already in the hall of fame as a scale. | up to 7 (+ some of the 14 rational-inside) |
| 6 | **The hall of fame as seed for a pole/root completion.** | 17 of 19 near misses hold the correct core product with the scale within 2%; the missing piece is one sub-expression. None of Andrew's current mechanisms swaps an imitation sub-tree for a simpler exact one; data guided rewrites are the nearest fit (a sub-tree that matches 1/(1-x) or 1/sqrt(1-x^2) on the data IS that form). | up to 15 near misses |
| 7 | **The growing head.** | Improvements land at head steps 18 times against 14 expected by chance — not separable at n = 55 (section 6); head needed >= 13 is 0 of 29. Growing the head does not by itself reach them. Open question: whether it helps at the start (II_11_27 came at generation 16 with head 12). | unknown |

Open questions with no mechanism of Andrew's that fits: head needed >= 17 (17 laws) and 6+ variables (14 laws, 12 of them also head >= 13) — 0 solved anywhere; nothing in the tables points to a lever.

## What was not done

- No fit was run; every "recommended effort" is an inference from other runs, untested.
- Whether arcsin/arccos were in the symbol set for each older run was not checked; older races also differ in engine version, so "0 solves in N other runs" mixes settings.
- "FOUND" rows (met the bar, scorer n) were judged by eye from the string, not re-scored. I_11_19's Piecewise term was not resolved.
- Head needed for the 6 laws missing from law_heads.json comes from my own tree builder, which agrees with law_heads.json on only 52 of 127 laws exactly (mean difference 1.2).
- The near-miss substitution groups are a reading by eye of 19 strings.
