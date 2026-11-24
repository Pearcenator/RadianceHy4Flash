# Hy4-flash serving path: optimisation backlog (ticket-ready)

Analysis and planning only; nothing here is implemented. Status 2026-11-22, `src` @ c655f58c
(branch claude/peaceful-wright-hhqnhi), container `hy4-flash-iqr-v4-u3.rad`, TP2 on 2 x R9700 at
210 W. Every item carries a measured basis from §2 (this note's own runs on the snapshot build) or
from the note it cites. Sources: five read-only code audits (`work/backlog/audit/A-kernels-decode.md`,
`B-kernels-prefill.md`, `C-plugin-graph.md`, `D-host-engine-sched-server.md`,
`E-memory-placement-kv.md`: op lists, kernel fact sheets, thread / wake maps and the longer
derivations live there), the prior notes (`kernels/{perf,levers-general,levers-tp2,experts-streaming,
dfly2,...}.md`, `notes/{g3,tp2t1,tp2t3,t4,g1g5,flashnext-port,xs-0,xs-1,hangfix,perf,u3-perf}.md`,
`work/{levers,tp2,churn,longctx,sweep}`), and the measurements of §2.

**Contents.** §1 result in one page · §2 fresh measured breakdown · §3 items by area (130 item
headings: K kernels 50, P plugin 16, D drafter 11, S scheduler 10, C engine 7, H server 4,
M memory / placement 17, X KV / prefix / checkpoints 6, B bring-up 4, T tooling 5) · §4 the
friend's decisions (21 format / lossy options, not ours) · §5 closed items (27) · §6 global ranking.

## 1. Result in one page

1. **Where the time is now** (§2). One stream without speculation: 13.5-13.8 ms a step (72-74 tok/s),
   1330 dispatches; 7.7 ms are weight bytes at or near the HBM rate, 6.0 ms the other 942 launches
   (junction 1.3, attention core 1.4, router 0.77, quantisers 0.74, small glue). Eight streams:
   25 ms a step (306-317 tok/s), routed GEMMs 12.3 ms. DFly2 coupled, one stream: 21.9-22.4 ms a
   step for 2.7-3.1 tokens (125-139 tok/s): verify 18.1 ms + draft pass 4.1 ms (five heads 1.51,
   IQ8R linears 1.03, ~190 small launches ~1.1). DFly2, eight streams: 446 tok/s; the ~40-row
   verify pass is 56.5 ms and **68 % of it is the two routed GEMMs**. Prefill: 3360-3540 tok/s; a
   2048-row chunk is 565 ms of op time: attention 34 % (sink kernels 28.7 %), routed MoE 32 %,
   trunk GEMMs 14.5 %, two-rank junction 11 %.
2. **New findings of this note (measured):**
   - `checkpoint-interval: 1984` removes the remainder step that every 2048-token interval takes
     while decoders ride (each with its own interleave run): arrival TTFT under load **-27 % at one
     decoder, -14 % at four**, lone prefill unchanged. Config only (S-01).
   - **Small prefill steps are unstaged and cost 0.08-0.34 s**: 73 tokens 78-88 ms, 256 tokens
     223 ms, a 537-row last chunk 338 ms with 214 ms of in-place expert reads. Every resumed
     agentic turn and every prompt tail pays it; the stager arms only above 1024 rows (M-03).
   - In the eight-stream DFly2 verify the routed gate_up (`moe_iqr`) takes 2.27x the routed down's
     time against 1.6x at one row: ~8 ms a pass beyond its bytes; its decode grid splits runs at
     fixed 16-row tiles and launches idle blocks, which `moe_down_dec` no longer does (K-08).
   - The draft pass's small launches cost 5-12 us each (the trunk's overlap: 0.3-0.5 us a removed
     launch), so the launch fusions that were low value in the trunk (G-3) do pay in the drafter
     (D-01..D-04).
   - Several kernels leave their decode-class fast path at verify heights (17-48 rows):
     `mla_sink_rows` 48.8 us a layer, `gemm_nt_f32` 33.3 us, the unchained gather (K-16, D-10).
   - `merge_sink` is 19 us in sliding layers against 6.3 us in sparse ones (K-14); the fused router
     select is 10.7 us, not a floor launch (K-09); the router weight is stored bf16 but served as
     f32 (K-11); the junction is 15.3 us x 84 = 9.3 % of a one-stream pass (K-12).
   - From the code: Hy4's u3 gate_up can never take the packed prefill map (K-44, answers FP-8);
     the "host mirror" test T5 changed a budget the mover does not read (M-01); the in-step tail
     snapshot FP-2 needs a scheduler change besides the plugin flag (X-02); idle conversations hold
     their VRAM KV for an hour because no host tier is configured (X-01).
3. **Ceilings (lossless; §6 end):** one stream no-spec about -2.3..-2.9 ms (73 -> ~90 tok/s);
   DFly2 one stream +18-25 %; DFly2 eight streams -25-30 % of a step; prefill +27-30 % from the
   kernels plus ~170 ms per prompt from M-03; the arrival stall keeps its 0.45-0.58 s until the
   layer-sliced chunk (S-02, L). Beyond that: formats (§4), which are the friend's.
4. **Top 15** (§6 tier 1): S-01, P-16, M-03, X-01, K-08, K-11, K-12, K-14, P-01, K-44, K-16,
   D-01, K-47, K-45, M-05.

## 2. Fresh measured breakdown (2026-11-22 evening)

Build: snapshot of `src/build` @ c655f58c (`work/backlog/snap/`, binary of 20:46), container
`hy4-flash-iqr-v4-u3.rad`, TP2 on 2 x R9700, 210 W (cap read back before every run: 210000000 uW),
Engram in RAM, sampled T 0.9 / top_p 1.0, code prompts, 256 tokens a request, configs
`work/backlog/nospec.yaml` and `coupled03.yaml` (= `work/u3/*` on port 8041). Every run is its own
server under `flock gpu.lock`, 67 s from start to `/health`. Tools: `work/backlog/{run.sh, probe.py,
agg.py, layer.py, profdiff.py}`; raw data `ns-st.*`, `cp-st.*`, `prof.*`, `nsv.*`, `cpv.*`,
`ck1984.*`, `ck2048.*` in the same directory (stamp CSVs 30-60 MB: delete when the tables are no
longer questioned).

Method: decode and DFly2 passes from `RADIANCE_AQL_STAMP=7` (one pass in seven of card 0, every
dispatch with its own completion signal; "exclusive" time of a dispatch = end - max(start, previous
end on the queue); the sum equals the busy union within 1 %). Inside a run of overlapping chained
launches the split between neighbours is fuzzy; sums over a group are reliable. Prefill from
`profile-ops` with `RADIANCE_PROFILE_EVERY=1` and `profdiff.py` (delta between consecutive
cumulative dumps = one step); `profile-ops` adds a synchronisation around every op, so shares are
what to read and the absolute chunk time comes from the unprofiled runs.

### 2.1 Decode, one stream, no speculation: 13.5-13.9 ms a step (72-74 tok/s), 1330 records, busy 13.4-13.7 ms

| category (card 0, a pass) | launches | us | share |
|---|---|---|---|
| trunk linears IQR u5 (`iqr_gemv`) | 251 | 2558 | 18.6 % |
| trunk linears int8 (`gemm_i8_kpar`) | 125 | 1959 | 14.3 % |
| routed gate_up u3 (`moe_iqr`) | 41 | 1779 | 13.0 % |
| two-rank junction (`ar_ihc_junction_2rank`, 84 x 15.3 us) + 5 `ihc_pre` | 89 | 1306 | 9.5 % |
| routed down w4nl64 (`moe_down_dec`) | 41 | 1104 | 8.0 % |
| attention: `mla_sink_rows` 42 x 8.5 + `merge_sink` 42 x 15.2 | 84 | 996 | 7.3 % |
| router: `gemm_nt_f32` 41 x 8.0 + `router_sigmoid_scatter` 41 x 10.7 | 82 | 766 | 5.6 % |
| attention: `kv_expand_gate_quant` 42 x 6.0, `kv_project_wmma` 42 x 3.0, Engram GEMMs (`vit_gemm_wa` 4 x 68.3, `vit_gemm` 4 x 27.3) | 92 | 760 | 5.5 % |
| quantisers: `gated_quant` 83 x 3.5, `quant_act_fp8` 88 x 3.3, `rmsnorm_quant` 42 x 3.8 | 213 | 741 | 5.4 % |
| cache row / rope / copies: `mla_row_front` 42 x 5.9, `rope` 42 x 3.0, `copy16` 41 x 3.0, `copy2d16` 4 x 13.6 | 129 | 550 | 4.0 % |
| small bf16 GEMMs (`kv_a` 42, `index_key` 12; 6.8 us mean) | 54 | 366 | 2.7 % |
| head (`mvm_kernel`, unchained) | 1 | 302 | 2.2 % |
| indexer key side (`idx_pool_gated` 12 x 9.4, `qsa_work`, `qsa_tail_store`) | 36 | 185 | 1.3 % |
| `moe_gather` | 41 | 146 | 1.1 % |
| Engram small (`engram_gate` 4 x 17.7, ids, state, norm, lookup) | 20 | 129 | 0.9 % |
| norms, embed | 19 | 85 | 0.6 % |
| **busy** | **1318** | **13733** | |

Weight-byte kernels (trunk linears, routed GEMMs, head): 7.7 ms = 56 %; the other 942 launches 6.0 ms.
Expert hit 0.984-0.988 (8-11 missed units a step = 0.5-0.7 ms of link inside the routed figures).

One layer in issue order (us, median over 37 passes; `layer.py ns-st.csv ns-st.json c1b 1330 <n>`):

- sparse layer (16 heads a rank, 34 records, **258 us**): `gated_quant 2.9 · moe_down_dec 20.0 · moe_gather 3.5 · copy16 2.9 · quant 2.9 · gemm_i8 5.1 (latent_out) · iqr_gemv 9.0 (shared gate_up) · gated_quant 2.9 · gemm_i8 3.1 (shared down) · junction 11.0 · iqr_gemv 5.6 (q_a) · rmsnorm_quant 3.0 · iqr_gemv 4.7 (q_nope) · iqr_gemv 2.9 (q_rope) · gemm_bf16 6.1 (kv_a) · row_front 3.0 · rope 2.9 · iqr_gemv 18.0 (attn_gate) · gemm_bf16 2.9 (index_key) · rmsnorm 2.9 · qsa_work 2.9 · idx_pool_gated 9.6 · qsa_tail_store 2.9 · kv_project 2.9 · mla_sink_rows 11.3 · merge_sink 6.3 · expand_gate_quant 9.4 · gemm_i8 22.3 (attn_output) · junction 10.0 · iqr_gemv 9.2 (latent_in) · gemm_nt_f32 8.7 · router_sigmoid_scatter 10.8 · quant 3.1 · moe_iqr 33.5`
- sliding layer (32 heads a rank, 29 records, **287 us**): as above without the five indexer records, with `q_nope 8.1 · q_rope 3.9 · kv_a 8.1 · row_front 6.7 · attn_gate 28.0 · mla_sink_rows 7.3 · merge_sink 19.0 · expand_gate_quant 4.4 · attn_output 42.7 · junctions 15.2 / 16.5`.
- per layer over the pass: sliding 285-294 us, sparse 255-258 us, the four Engram layers 348-417 us (`vit_gemm_wa` 68.7 + `vit_gemm` 30.3 + `engram_gate` 17.8 + own `ihc_pre` / norm / quantiser); pass front (embedding to the first routed GEMM, incl. layer 0's dense MLP 43.5 + 38.0 us) 589 us; tail (last down to head) 367 us of which the head 302.
- 14-15 of a layer's records sit at the chained floor (2.9-3.5 us).

### 2.2 Decode, eight streams, no speculation: 25.5 ms a step stamped (306-317 tok/s), busy 24.9 ms

| category | us | share | against one stream |
|---|---|---|---|
| routed gate_up (`moe_iqr` 41 x 188.4) | 7723 | 31.0 % | x4.3 |
| routed down (`moe_down_dec` 41 x 111.3) | 4562 | 18.3 % | x4.1 |
| trunk linears IQR (`iqr_m16` 251 x 11.4) | 2870 | 11.5 % | x1.12 |
| trunk linears int8 (125 x 17.9) | 2241 | 9.0 % | x1.14 |
| junction (84 x 14.8) | 1274 | 5.1 % | x1.0 |
| attention expand / project / Engram GEMMs (`kv_expand_gate_quant` 42 x 16.0) | 1201 | 4.8 % | x1.6 |
| sink rows 42 x 12.3 + merge 42 x 16.2 | 1196 | 4.8 % | x1.2 |
| router (`gemm_nt_f32` 10.2, select 12.3) | 926 | 3.7 % | x1.2 |
| quantisers | 859 | 3.5 % | x1.16 |
| everything else | 2042 | 8.2 % | |
| **busy** | **24894** | | |

4105 routed units a step (2.99 us a unit incl. 34 missed units = 2.2 ms of link); hit 0.992. Four streams: 18.4 ms busy, 208-215 tok/s.

### 2.3 DFly2 coupled (`coupled03.yaml`), one stream: 22.4-22.9 ms a step stamped, 2.8-3.1 tokens a step (125-136 tok/s)

| pass | records | busy | notes |
|---|---|---|---|
| verify (trunk) pass, mean 4.4 rows | 1335 | 18.1-18.2 ms | routed gate_up 4.17 ms (41 x 101.8), routed down 2.53 ms (41 x 61.7), trunk IQR 2.69, int8 2.08, junction 1.24, expand / project 1.06, sink + merge 1.03, router 0.83, quantisers 0.63; 1852-2146 routed units a step, 25-32 missed (1.7-2.1 ms of link) |
| draft pass, one sequence | 303 | 4.0-4.6 ms | head 5 x 301.4 = **1.51 ms (37 %)**; IQ8R linears 46 x 22.4 = **1.03 ms (25 %)**; 46 quantisers x 5.0 = 0.23; 38 `rmsnorm` x 5.5 = 0.21; 21 `copy2d16` x 5.7 = 0.12; 10 standalone reductions x 9.0 = 0.09; sink rows 5 x 15.5, `kv_project` 10 x 7.6, `binary` 16 x 4.6, `propose` 5 x 13.9, `gather_rows` 8 x 7.2, `rope` 10 x 5.5, top-k stages 15 launches 0.08 |

A verify row costs (18.15 - 13.5) / 3.4 = **1.37 ms** at one stream here (the online table of the run: "1 seq: 0.125 tok/ms, row 1.095 ms"). Depth shares [1: 0.07, 2: 0.27, 3: 0.20, 4: 0.15, 5: 0.31]; acceptance at position 0.53-0.64; "host waits 424 (31954.5 us each, all blocked)" (not a throughput lever: Z-06).
Small launches in the draft pass cost 5-7 us each (trunk: 2.9-3.5 us with overlap), so the draft pass's ~190 small launches are 1.0-1.2 ms of its 4.1 ms.

Eight streams (18.96 tokens a step, 108 steps, 443 tok/s summed):

| pass | records | busy | notes |
|---|---|---|---|
| verify, ~40 rows | 1375 | **56.5 ms** | routed gate_up **26.7 ms (47.3 %; 41 x 652 us)**, routed down 11.8 ms (20.8 %; 41 x 287), trunk IQR 3.53, int8 2.47, router 2.36 (`gemm_nt_f32` 41 x 33.3, select 41 x 24.3), junction 2.36 (84 x 27.6), `mla_sink_rows` 2.05 (42 x 48.8: unsplit fallback above 16 / 32 rows), expand 1.49 (42 x 35.6), `gather_kernel` 41 x 7.5 (the unchained form above 32 rows) |
| draft pass, 8 sequences | 298 | 6.3 ms | head 5 x 329 = 1.65 ms; IQ8R linears 46 x 28.4 = 1.31; quantisers 46 x 12.3 = 0.57; reductions 10 x 46.5 = 0.47; `rmsnorm` 38 x 12.0 = 0.46; `copy2d16` 21 x 11.5 = 0.24; `binary` 16 x 11.8; `kv_project` 10 x 16.8; `rope` 10 x 14.1 |

7446 routed units a step, 111 missed (7.3 ms of link), hit 0.985. Routed gate_up / down time ratio: 1.61 (one row), 1.69 (8 rows), **2.27 (40 rows)** against a byte ratio of 1.52: at verify heights `moe_iqr` loses ~8 ms a pass beyond its bytes (K-08). Four streams: verify 35.5 ms, draft 4.8 ms, 331 tok/s summed.

### 2.4 Prefill: one 6681-token prompt = chunks of 2048 + 2048 + 2048 + 537 (`prof`, rank 0, op time per step)

Third chunk (2048 rows at context 4096-6144, every row absorbed, sparse layers selecting): **564.7 ms** of op time (+20.5 ms waits), 51 op kinds:

| op | calls | ms | share | us a call |
|---|---|---|---|---|
| routed `moe_gemm_q` gate_up | 41 | 87.2 | 15.4 % | 2126 |
| `mla_attn_paged_sink` (30 sliding layers) | 30 | 82.2 | 14.6 % | 2740 |
| `mla_attn_sel_sink` (12 sparse layers) | 12 | 79.9 | 14.2 % | 6660 |
| `ar_rmsnorm_ihc_junction` | 84 | 62.3 | 11.0 % | 741 |
| routed `moe_gemm_q` down | 41 | 45.6 | 8.1 % | 1112 |
| trunk GEMMs `gemm_nt_q`: attn_output 22.5, attn_gate 20.5, q_nope 6.4, latent_out 5.7, shared gate_up 5.4, latent_in 5.2, q_a 4.2, shared down 3.6, q_rope 2.8; `kv_a` (bf16) 5.3 | 416 | 81.6 | 14.5 % | |
| `mla_expand_gate` | 42 | 21.2 | 3.7 % | 504 |
| router `gemm_nt_f32` 17.3 + `all_gather` 4.3 + select 1.4 + scatter 1.0 | 164 | 24.0 | 4.3 % | 422 |
| `moe_gather` | 41 | 14.6 | 2.6 % | 356 |
| `swiglu_clamped_had_quant_i8g` | 41 | 10.4 | 1.8 % | 254 |
| `mla_project_kv` (absorb) | 42 | 8.5 | 1.5 % | 203 |
| Engram: `embed_lookup_q` 4 x 1040, key 4 x 612, `engram_norm` 4 x 412, gate 4 x 311, value 4 x 162 | 20 | 10.2 | 1.8 % | |
| `idx_score` 12 x 512 us, `qsa_select` 12 x 123, index GEMMs, pool | 96 | 11.9 | 2.1 % | |
| `had_quant_act_i8g` (130 calls x 46 us), other fused quantisers | 214 | 8.9 | 1.6 % | |
| `cast` 28 x 170 us, `mla_row_norm_rope_store` 42 x 112, `rope` 54 x 66 | 124 | 13.0 | 2.3 % | |
| rest | | 3.2 | 0.6 % | |

Groups: attention 191.8 ms = **34.0 %** (sink 28.7 %), routed MoE incl. gather / swiglu / router 181.8 ms = **32.2 %**, trunk GEMMs 14.5 %, junction 11.0 %.
First two chunks together 1087 ms: the first (non-absorbed) chunk adds `attn_kv_sink` 42 x 764 us, `cast` 158 calls x 205 us = 32 ms (3 %), `mla_kv_split`, the `kv_b` GEMM; `attn_q_a` shows 43.5 ms over 84 calls there against 4.2 ms in a later chunk (first-use cost: one-time).

**Small steps are disproportionately expensive** (same run):

| step | rows | op time | routed gate_up + down | share |
|---|---|---|---|---|
| full chunk | 2048 | 564.7 ms | 132.8 ms (staged: hit 0.864, pooled units copied ahead) | 23.5 % |
| last chunk of the prompt | 537 | **338.3 ms** | **213.6 ms** (3023 + 2187 us a call: unstaged, pooled units read in place) | 63.1 % |
| a 73-token prompt (two passes) | 73 | 96.9 ms each | 64.2 ms each (892 + 674 us a call) | 66.3 % |
| a decode step | 1 | 23.1 ms (profiled) | 3.4 ms | 15 % |

A 537-row step costs 60 % of a 2048-row chunk for 26 % of the rows, and a 73-token prompt's TTFT is 76-84 ms unprofiled (5-6 decode steps): below 1025 rows the stager is not armed, the step still routes almost every expert of every layer and reads the pooled 12-14 % over PCIe synchronously (M-03).

### 2.5 Unprofiled rates and host phases (`nsv`, `cpv`: `server: log: debug`)

| phase | no speculation | DFly2 coupled |
|---|---|---|
| one stream, code | 13.84 ms a step, 72.3 tok/s | 21.9-22.4 ms a step, 2.8-3.1 tok a step, 127.7-138.7 tok/s |
| one stream, thinking text (384 tokens) | | 21.3 ms a step, 2.67 tok a step, 125.1 tok/s |
| eight streams | 315.7 tok/s summed (hit 0.991, 37 missed units a step) | 446.1 tok/s summed (19.1 tok a step) |
| prefill 6680 tokens | TTFT 1.92-1.99 s = 3360-3475 tok/s | |
| prefill 24856 tokens | TTFT 7.03 s = 3536 tok/s | |

Host phases of a decode step (mean of 128 steps): plan + begin + build + pre + save + commit + draft <= 0.05 ms; `ranks` (two rank threads replaying the tape) 0.48-0.50 ms at one stream, 0.66-1.0 ms at eight, 0.50-1.37 ms with DFly2, all overlapped with the card; the rest is `collect` (waiting for the card). The host is not on the critical path of a decode step.

### 2.6 Small prefill steps, the arrival stall, remainder steps (FP-1, measured)

Cost of a lone prefill by size (unprofiled TTFT, cold prefix, `nsv` and `ck*` runs):

| prompt tokens | steps | TTFT | tok/s |
|---|---|---|---|
| 73 | 1 | 78-88 ms | 900 |
| 256 | 1 | **223 ms** | 1150 |
| 717 | 1 | **255 ms** | 2800 |
| 2393 | 2048 + 336 + tail | 707 ms | 3385 |
| 6680 | 3 x 2048 + 528 + 9-token tail step | 1.92-1.99 s | 3360-3475 |

A step of 73 rows costs about 6 decode steps and one of 256 rows about 16: the fixed cost of a small step is the unstaged pooled experts (§2.4: 64 ms of a 97 ms 73-row step, 214 ms of a 338 ms 537-row step under the profile). Every resumed agentic turn (213-450 re-prefilled tokens, `work/churn`) pays this 0.2-0.25 s, as does every prompt's last partial chunk, and the tail step exists on every prompt (`step 100 prefill n_tok=9`).

Arrival of a 5.7-5.9K-token prompt 1.5 s into running decoders (`probe.py stall1 / stall4`; client-side gaps of the background streams):

| run | background | arrival TTFT | lone TTFT, same size | the three largest gaps of a background stream | typical gap |
|---|---|---|---|---|---|
| no spec, interval 2048 (`nsv`) | 1 | 3.08 s | ~1.65 s | 466, 469, 563 ms | 13.2 ms |
| no spec, interval 2048 (`nsv`) | 4 | 2.82 s | ~1.70 s | 459, 516, 565-586 ms | 18.5 ms |
| no spec, interval 2048 (`ck2048`) | 1 / 4 | 3.25 s / 2.83 s | | 453, 460, 573 / 455, 511, 561 ms | |
| **no spec, `checkpoint-interval: 1984` (`ck1984`)** | 1 / 4 | **2.37 s / 2.43 s** | lone 6.7K: 2.00 s against 1.99 s | 446, 493, 548 / 449, 541, 553-578 ms | |
| DFly2 coupled, interval 2048 (`cpv`) | 1 / 4 | 2.79 s / 2.98 s | | 469, 475, 575 / 465, 526, 574 ms | 21.6 / 40.1 ms |

Step log (`RADIANCE_LOG_STEPS=1`), one decoder + the arrival: interval 2048: `MIXED 2033 · 24 decode steps · MIXED 17 · 24 decode · MIXED 2033 · 18 decode · MIXED 17 · 24 decode · MIXED 1601 · 19 decode · MIXED 12 (tail)`; interval 1984: `MIXED 1985 · 15 decode · MIXED 1985 · 17 decode · MIXED 1729 · 21 decode · MIXED 11 (tail)`.
Readings: (1) the stall is the chunk step (0.45-0.58 s), with or without DFly2, whatever the number of decoders; (2) with the default interval every 2048 prompt tokens run as a chunk **and a 16-20-row remainder step, each followed by its own interleave run** of ~24 decode-only steps, which is where the 1.9x TTFT under load comes from; (3) `checkpoint-interval: 1984` removes the remainder steps: **arrival TTFT -27 % at one decoder, -14 % at four, lone prefill unchanged** (single runs of each arm; the 2048 arm was run twice: 3.08 / 3.25 s and 2.82 / 2.83 s); the stall length itself does not change; (4) a remainder or tail step is a 40-65 ms gap for the decoders.
## 3. Items by area

Field key, one line each: **Area** · **Lossless** (BI = byte-identical; ORD = same law, f32 sum order or
row-grouping moves bits inside the op's stated bound; PF = prefill rows only move inside the E4 bound;
PL = placement/scheduling only; DRAFT = draft bits only, output law exact) · **Gain** as
c1 / c8 / DFly2 step / prefill (percent of step or chunk; "0" = none, "?" = unknown until measured) ·
**Conf** (confidence that the gain materialises) · **Effort** S / M / L · **Deps**.
"M:" introduces the measured basis from §2 (this note's runs) or the cited note; "est." is arithmetic.
Paths are relative to `src/`. Line numbers were read at c655f58c by the area audits
(`work/backlog/audit/{A..E}-*.md`, which hold the longer derivations) and spot-checked.

### 3.1 Kernels (libr4d): decode-class rows

#### K-01 Dual-weight IQR launch: `latent_in` + shared `gate_up` (same junction codes)
- Area kernel+plugin · BI · Gain 1.5-2 % / 1 % / 1 % / 0 · Conf medium · Effort M · Deps none (first user of the dual entry; K-02..K-04 reuse it)
- Change: one `iqr_gemv` / `iqr_m16` launch over two weights and two outputs reading the same rotated int8 codes. Both are u5, K 3072, N 1536, `kw` 8 (`libr4d/r4d_gemm_iqr.hip:954` rule `N >= 4096 ? 4 : 8`, `:1031` for m16), so a concatenated grid is byte-identical. Plugin: `arch/hyv4flash/hyv4flash_moe.h:390` (`lat_in.step`) and `:399` (`shared.step` -> GatedMlp `:88`). Removes the shared arm's first GEMM from the ordered tail of the layer (the plugin-level form of X5).
- Why: M: `latent_in` 9.2 us + shared gate_up 9.0 us a layer for 2 x 2.95 MB (byte floor 10.4 us at 570 GB/s): each runs at 2x its floor because a 3-unit-a-wave stream never reaches bandwidth. 41 layers x ~6 us = ~0.25 ms of a 13.7 ms pass.
- Verify: new `r4d_selftest` case dual vs two `gemm_iqr_nt_m64` calls at M 1..16 (bytes); kbench fusion oracle (`rad-kbench -m` container run, as G-3: "154 fusions byte-identical"); `layer.py` before/after; greedy text + logprobs equal with a `RADIANCE_HY4_FUSE_DUAL=0/1` switch.
- Closed, without code (`notes/backlog/closed/K-01.md`): built in full (`36b16cfc`: the `gemm_iqr_dual` row over the existing `iqr_gemv` / `iqr_m16` kernels, the layout hook, the fusion claim, `Lin::declare_dual` behind `RADIANCE_HY4_FUSE_DUAL`, a selftest and an arch-test case) and passed every card-less check and both plan-level oracles — the dual case byte-identical to two `gemm_iqr_nt_m64` at M 1..16, 1204/1204 libref, ISA lint PASS (1028 kernels, 0 errors), the base's card-less failure set — but the two weights are **not** in the same IQR form: `rad-info -v` gives `ffn_latent_in … tcq_rows=256` beside `ffn_gate_up_shexp … tcq_rows=128`, so their form words are `0x02310056` and `0x01f10056`, differing in `lrun` alone (8 vs 7, `run_tiles` 16 vs 8), and the row's `dual_pair` refuses the op at issue. On the item's lane (local TP2, `nospec.yaml`, c1/c1b/c8/greedy) the off arm reproduces the basis (c1 13.779 ms/step, 72.6 tok/s) and the on arm stops at its first decode step on both ranks (`refused op 'gemm_nt_q_dual' at M=1 [band M<=16] … dtype this kernel does not serve`), so the fused op never ran and there is no stamp pair to weigh. The item's "a concatenated grid is byte-identical" needs the pair to be converted with one run; the plugin's guard (`CodeForm::same`, i8/rot/g8) cannot see the run, so it claimed a fusion the row refused and the step aborted rather than falling back. K-02..K-04 must check their pair's `tcq_rows` first; making this one pay means re-converting a weight (lossy, §4) or two forms in the kernel. `36b16cfc` is reverted and stays in the branch's history; nothing merged.

#### K-02 Dual-weight launch `q_nope` + `q_rope` (G-3d, unblocked)
- Area kernel+plugin · BI · Gain 1 % / 0.5 % / 0.7 % / 0 · Conf medium · Effort M · Deps K-01's entry
- Change: G-3d stopped because sliding layers give `q_nope` (N 6144) kw 4 and `q_rope` (N 2048) kw 8 (`notes/g3.md`). Use 8-wave blocks for both parts: a kw-4 part's block takes two 16-row tiles x 4 waves (each tile's LDS partials summed over its own 4 waves in ascending order, exactly the plain kernel), a kw-8 part's block one tile x 8 waves. Sparse layers (N 3072 / 1024) are kw 8 for both already. Plugin `hyv4flash_attn.h:509-510`. Wire it behind the existing dead switch `RADIANCE_HY4_FUSE_QPROJ` (P-14).
- Why: M: `q_rope` 2.9-3.9 us for 0.66-1.3 MB = pure launch/ramp cost; `q_nope` 4.7-8.1 us. 42 x ~3 us = 0.13 ms.
- Verify: as K-01; sliding and sparse layer cases separately (kw mix).

#### K-03 Dual-weight launch `q_a` + `attn_gate` (same attention-norm codes)
- Area kernel+plugin · BI · Gain 1-1.5 % / 0.7 % / 1 % / 0 · Conf medium · Effort M · Deps K-01, K-02 (kw-mix blocks)
- Change: `hyv4flash_attn.h:502` (`q_a`) and `:549` (`gate`, `gate.feed = &q_a` at `:338`): one launch; `q_a` (N 1024, kw 8) rides the gate's 7.9 / 15.7 MB stream. The gate is consumed only by `mla_expand_gate`, so it is off the critical path until then; the engine cannot overlap two dependent chained launches otherwise (`core/device/aql.cpp:1676`: a dependent record targets `chain_issued`).
- Why: M: `q_a` 5.6 us (1.97 MB at 350 GB/s) + `attn_gate` 18.0 (sparse) / 28.0 us (sliding); merged est. 18.5 / 28.5 us: -5 us x 42 = 0.2 ms.
- Verify: as K-01; stamps of the attention front.

#### K-04 Dual-weight kpar launch `latent_out` + shared `down`
- Area kernel+plugin · BI · Gain 0.5-1 % / 0.5 % / 0.5 % / 0 · Conf medium · Effort S-M · Deps K-01 (shared gate_up must be done earlier), P-01
- Change: `libr4d/r4d_gemm_fp8a8.hip:1162-1183` (kpar grid N/16, KW 4 for >= 4 groups): `latent_out` (K 1536, 12 groups) and shared down (K 768, 6 groups), both N 3072, KW 4: concatenated grid `[0,192)` A, `[192,384)` B, two outputs. Plugin `hyv4flash_moe.h:397-402`.
- Why: M: 5.2 us + 3.1 us a layer (c1), 5.8 + 6.4 at c8; est. -2..-3 us x 41 = 0.1 ms.
- Verify: selftest dual-kpar vs pair; fusion oracle.

#### K-05 Chained GEMMs: issue the first static-weight loads before the chain wait
- Area kernel · BI · Gain 1.5-4 % / 1-2 % / 1.5-3 % / 0 · Conf medium-low (mechanism certain, size unmeasured) · Effort S-M · Deps none
- Change: `r4d_gemm_iqr.hip:157` (wait) then `:184` (first `D::fetch`), `:358/:411` (m16); `r4d_gemm_fp8a8.hip:868` then `:790`; `r4d_chain.h:107-128`. Chained packets are launched ahead "as far as the card has slots" (`aql.cpp:1380-1386`), so a GEMM's blocks sit in the poll while the predecessor runs. For VRAM-static trunk weights (not routed experts, not weights a tier copy or LoRA can rewrite: gate on the row) issue the wave's first one or two weight units and the activation-independent scale loads before `r4d_chain_wait*`; registers survive the acquire. Same for `mla_project_kv` / expand's `kv_b` (`r4d_mla_kv_bf16.hip:236, 351` waits; `:169-172, 292-295` loads; the whole quarter fits the registers they use).
- Why: M: 251 IQR + 125 int8 trunk launches a pass at 10.2 / 15.7 us mean; the small ones (q_a, q_nope, latent_in, shared arm, latent_out: ~250 launches) run 1.4-4x above their byte floor, which is one DRAM latency + ramp per launch. Hiding ~1-2 us on ~250 launches = 0.25-0.5 ms; the expand's kv_b under the latency-bound sink + merge (M: 26-31 us a layer) up to 7 us x 30 sliding layers more.
- Verify: `r4d_selftest` chained cases; spec test bit-equal; `tests/chain_scalar_audit.py`; stamps: start gap of GEMMs; `RADIANCE_AQL_STAMP` layer table.

#### K-06 Deeper prefetch for short K loops (3-unit waves)
- Area kernel · BI · Gain 1 % / 0.5 % / 0.7 % / 0 · Conf medium-low · Effort S · Deps after K-05 (measure what is left)
- Change: `r4d_gemm_iqr.hip:184-186` (GEMV), `:411-417` (m16), `r4d_gemm_fp8a8.hip:789-805`: a wave holds the current unit and one in flight; with nkb / KW <= 4 (q_a, latent_in, shared gate_up: 3 units a wave; shared down: 2,2,1,1) issue all units up front (14 VGPR a u5 unit: 3-4 fit).
- Why: those launches are latency-bound (K-01 table); est. ~1 us on 4-5 launches a layer.
- Verify: kbench `gemm_iqr` / `gemm_i8a8` rows at (1, 1024..1536, 3072), VGPR/spill report in the commit message.
- Closed, without code (`notes/backlog/closed/K-06.md`): the three-unit ring was built in all three loops (GEMV and m16 `cur` / `n1` / `n2`, primes a unit and two ahead, refill three ahead after the shift; int8 kpar `ac` / `a1` / `a2` with the activation still one unit ahead) and it is lossless — the four greedy sha256 identical on the served lane (local TP2, `nospec.yaml`, c1/c8/greedy) and `r4d_selftest --case gemm_iqr_nt_m64` 2054 checks with `the chained form is the int8 launch's bytes` at M 1 and 16 — but every targeted kernel is slower: stamps `iqr_gemv_kernel` 9.2 -> 9.9 us a launch (c1), `iqr_m16_kernel` 11.0 -> 12.2 (c8), `gemm_i8_kpar_kernel` 17.2 -> 17.6, and at the item's own geometry on one card (`--perfiqr` / `--perfw8`, N 3072 K 1536) 10.3 -> 10.4 us at M 1, 10.0 -> 11.7 at M 16, `gemm_i8a8_nt_m16` 5.9 -> 6.7. The cost is the resident unit: `res.sh` gives +21 VGPR on average (mean over 2176 GEMV forms, max +62) and a lost wave on 68 % of the GEMV and 53 % of the m16 forms against `R4D_WAVES(6)`'s 256-VGPR cap, and at hy4-flash's trunk shapes (`nkb` 12 at `K` 1536, `kw` 8) a wave walks two units, so the ring is never refilled. (The kbench rows the Verify names do not exist: `data/kernels_hy4.rkb` has no `gemm_nt_q` case — its gemm cases are `gemm_nt_bias` with bf16 operands — and `data/kernels.rkb` reaches `gemm_iqr_tiled` only at prefill geometry.)

#### K-07 `moe_iqr` / `moe_down_dec`: one descriptor load instead of search + pointer-table round trips per block
- Area kernel · BI · Gain 1.5-2.5 % / 1 % / 1.5 % / 0 · Conf medium · Effort M · Deps none
- Change: `r4d_moe_iqr.hip:278-318` (wave 0: 17 `off[]` loads a lane, scan, `lds_barrier`), `:333-334` (`C.w[e >> cshift]` dependent pointer load), then `D::fetch`. At M = 1 the grid is 16 runs x 96 y-blocks of 128 threads and every block repeats the head. Either (a) let a KW 4 block take 2-4 tiles in turn (lift `static_assert(TPW == 1u || KW == 1u)` `:228`), or (b) have `router_sigmoid_topk_scatter` write a run descriptor list (expert, rs, re, W ptr, S ptr) so a block does one load. `moe_down_dec` has the same head (`r4d_moe.hip:1595-1654`).
- Why: M: routed gate_up 43.4 us mean (33.5 warm layer) for 14.8 MB (floor 26 us), routed down 26.9 us (19.8) for 9.7 MB (floor 17): ~17 us a layer above the byte floor at one row = 0.7 ms a pass; est. a third of it recoverable.
- Verify: selftest `moe_iqr` decode-band cases, `--perfmoeiqr`; kbench `moe_gemm_q` rows M = 1..4 of `data/kernels_hy4.rkb`; stamps.
- Closed, without code (`notes/backlog/closed/K-07.md`): (a) built both ways — a wide-K decode block taking 2 or 4 column tiles in turn (the `static_assert` lifted, the fold guarded, one extra `__syncthreads()` a tile), byte-identical at every pin (`r4d_selftest --case moe_gemm_iqr`, 80 checks, the Y of the three pin values `memcmp`ed) — and it is not faster. Served lane (local TP2, `nospec.yaml`, c1/c8/greedy): `moe_iqr` 46.6 -> 46.4 us a launch, routed gate_up 1909 -> 1903 us a pass, c1 15.866 -> 15.795 ms (+0.4 % tok/s), the four greedy sha256 identical. Kernel level (`--perfmoeiqr`, one card, gate_up 768:512 x 2560, us a launch): 16 rows B = 3/4/5 21.7 / 22.2 / 23.1 -> 22.3 / 23.4 / 26.6, 10 rows 15.8 / 15.9 / 16.6 -> 16.0 / 17.0 / 18.4, four a block 17.3 / 17.9 / 18.9. Deleting half (then three quarters) of the 17-load searches removes no time, so the head is not the 17 us a layer above the byte floor and (b)'s descriptor list would shorten the same prologue; the gap is the tile's own work (16 columns a wave, the LDS transpose, the trellis decode). `moe_down_dec` already takes G column tiles a block and its own G rule's measurement rejects more at low rows. The kbench row the item names does not exist: `data/kernels_hy4.rkb` has no `moe_gemm_q` case (they are in `data/kernels_moe.rkb`).

#### K-08 `moe_iqr` decode grid: run-aligned tiles (an expert's run is split at fixed 16-row boundaries and read twice)
- Area kernel · ORD (a row moves between the dot and WMMA paths; rows already depend on the step's other rows today) · Gain 0 / 0.5-1 % / **up to 10-12 % of a c8 DFly2 step**, ~2 % at c4 / 0 · Conf high (code), medium-high (size: measured anomaly) · Effort M · Deps none
- Change: `r4d_moe_iqr.hip:270-322` (`r0 = (blockIdx.x / kRows) * kRows`), `:911` (`g.gx = T`). Port `moe_down_dec`'s numbering (`r4d_moe.hip:1521-1545`: "tiles start at the run's first row"). Also removes the idle x-slots that still run the 17-load search (`:318`).
- Why: M (§2.3): in the c8 DFly2 verify pass (~40 rows, 640 sorted rows) routed gate_up is 41 x 652 us = 26.7 ms = 47 % of the pass, and its time ratio to the routed down is 2.27 against 1.61 at one row and 1.69 at eight (byte ratio 1.52): `moe_iqr` loses ~8 ms a pass at verify heights that `moe_down_dec` (run-aligned since its own fix) does not. Code: ~23 extra expert decodes a layer at T = 768 (+6 % bytes) plus x = T slots x 24-96 y-tiles of blocks that run the 17-load search and return. First step: `--perfmoeiqr` at T = 640 with the two effects separated.
- Verify: selftest vs libref bounds at T = 16, 128, 480, 768; `/stats` unchanged; greedy c1 identical (single-row runs unchanged).
- Done: merged (`99866066`) — the band's grid is the runs' own 16-row tiles, numbered in expert order from each run's first row: every thread holds four experts' offsets and the fifth, counts its own tiles, the threads before it scan them in the wave and the waves before it read the per-wave counts from LDS after one barrier (`moe_down_dec`'s numbering), the thread holding `blockIdx.x` hands over its expert and tile, and the last block zeroes the dropped slots' rows. `g.gx` is the tile bound `(T + 15 min(T, n_expert)) / 16` plus that block, so 768 blocks a layer at 768 rows become 529 where the runs make at most 528. `--perfmoeiqr` B=4, 512 experts, card 1, three interleaved pairs of the old and the new plugin: gate_up 768:512 x 2560 at 640 rows 536.9 -> 530.8 us (-1.0 %), at 768 rows 586.7 -> 574.3 (-2.1 %), down at 768 rows 372.1 -> 366.1 (-1.6 %); 320 rows (356.4 -> 355.8) and 40 (43.5 -> 43.0) inside the harness's scatter, 16 rows 22.0 -> 22.2 (+0.9 %, one block more). Selftest against the dense IQR GEMM over 512 experts at 16, 48, 128, 480 and 768 rows, gate_up and a sorted down at 768: all `ok` inside `|a-b| <= |b| 2^-7 + rms 2^-8`. Lane `local` (TP2, u3, cap 210 W), `coupled03.yaml` greedy,c1,c8, three runs (one before, two after): c1 109.5 -> 131.7/133.7 tok/s (29.44 -> 24.51/24.14 ms a step), c8 363.3 -> 359.8/396.2 tok/s at 20.7 -> 18.1/21.1 tokens a step and 99 -> 113/97 steps — the after arm's own two runs differ by 10 % at c8, more than any arm-to-arm difference, so the lane shows no e2e signal either way. `/stats` keeps its keys and no error counter moves. The verify's "greedy c1 identical" does not hold: the two after runs' greedy texts are identical to each other and differ from the before run's, because a c1 verify step routes six tokens' slots (48 rows over 512 experts), where runs of three and more rows exist and a run of three that crosses a 16-row boundary moves from its dot pieces into its own tile — the ORD class the item declares, not the single-row case its verify reasoned from. Runs of one and two rows are byte-identical (`--perfmoeiqr`'s fnv at 16 and 40 rows, both arms). The ~8 ms a pass the item expected is not reproduced: 1.0-2.1 % of routed gate_up at 640-768 rows is ~0.5 % of a c8 pass, so the 2.27-against-1.61 ratio anomaly has another cause.

#### K-09 Router select: 16 dependent rounds -> pairwise tree / sorted pop (the fused router is 10.7 us, not floor)
- Area kernel · BI (same winners, same order, ties to the lower index) · Gain 2-2.5 % / 1 % / 1.5 % / 0 · Conf medium · Effort M · Deps none
- Change: `r4d_glm.hip:160-199` (16 rounds of 64-bit compare-select + DPP max), `:170-181`. Depth-4 max tree, or a per-lane sorted list popped per round (`r4d_rowtopk_bf16.hip:106-120` `tk_round`).
- Why: M: `router_sigmoid_scatter` 10.7 us x 41 = 0.44 ms (3.2 % of a c1 pass; 12.3 us at c8). A floor launch is 2.9-3.3 us, so ~7 us a layer is the selection chain.
- Verify: `r4d_selftest` router cases incl. ties and correction bias; router picks identical on harness prompts (fusion oracle); `--perff32`.

#### K-10 Router GEMM + select + scatter in one launch (G-3f; exactness worry resolved)
- Area kernel · BI · Gain 0.5-1 % / 0.3 % / 0.5 % / 0 · Conf medium · Effort M · Deps K-09 (same file), K-11
- Change: every `gemm_nt_f32` tile form has the same per-lane k order and butterfly (`r4d_glm.hip:13-77, 116-127`), so the fused form copies it as `router_gemm_topk_scatter_kernel` does (`r4d_moe.hip:3669-3767`). Use the 7 idle waves of the select block for the latent's `had_quant` (12 groups) too (-1 more launch: `rad_block_moe_fp8.h:1925`).
- Why: M: `gemm_nt_f32` 8.0 us + select 10.7 us + quant 3.2 us a layer; fusion removes two chained launches (G-3 price: 0.3-0.5 us each) and the logits round trip; mostly valuable together with K-09.
- Verify: fusion oracle; picks identical.

#### K-11 Router weight read as stored bf16 (container: `glm_router.weight bf16 512x3072 3.00 MiB`), widened in registers
- Area kernel+plugin (memory too) · BI (bf16 -> f32 is exact; same products and order) · Gain 1-1.5 % / 0.7 % / 1 % / +0.3 % (residency) · Conf high · Effort S-M · Deps none
- Change: `arch/common/rad_block_glm_router.h:44-45` declares `RAD_F32`, so the loader widens to 6.29 MB a layer. Declare bf16 and give `gemm_nt_f32` (and the fused router rows) a bf16-weight load path with the same f32 arithmetic. Prefill split form `rad_glm_tp.h:118-124` likewise.
- Why: M: `gemm_nt_f32` 8.0 us x 41 = 0.33 ms for 258 MB a pass a rank; half the bytes -> est. -0.15 ms. Also 129 MiB of VRAM a card = 90 units (+0.4 points residency). Not the closed "router on one rank" and not a format change (the container already stores bf16; `work/levers/radinfo-v.txt`).
- Verify: selftest bf16-weight vs f32-weight bytes equal; greedy texts; startup receipt static weights 3.50 -> 3.37 GiB.

#### K-12 Junction at decode: warm `fn` while thread 0 waits on the peer flag
- Area kernel · BI (loads only) · Gain 1.5-2.5 % / 1 % / 1.5 % / 0 · Conf medium · Effort S · Deps none
- Change: `r4d_ar_ihc_junction.hip:137-200` (`:178` spin on the peer flag while 511 threads sit at `__syncthreads`), arithmetic `r4d_ihc.h:207-260`. Let the idle threads touch `fn` (8 x 12288 f32 = 393 KB), the two gain vectors and `h` (24 KB) during the wait.
- Why: M: junction 15.3 us x 84 = 1.28 ms = 9.3 % of a c1 pass (was 12.6 us pre-fusion-b; it now also writes the next norm's codes). The plain row is 7.3 us at TP1, of which `fn` streaming on one WGP is most; est. -2..-4 us each = 0.17-0.34 ms. Not the closed "faster wire": the wire is untouched.
- Verify: `tests/r4d_emu/test_ihc.cpp`; selftest junction cases; TP2 greedy bit-equal; junction p50 in stamps.
- Closed, without code (`notes/backlog/closed/K-12.md`): built as written — the 511 idle threads read `h`, both gains and `fn` in 16-byte words during thread 0's spin, loads only, `tests/r4d_emu/test_ihc.cpp`'s 22 cases holding the warm to its operands exactly — and byte-identical at TP2 (the greedy phase's four texts). Served lane (iks-A TP2, `nospec.yaml`, c1, `RADIANCE_AQL_STAMP=7`, 3 before runs against 4 after, the arms interleaved in wall time so no drift sits in the gap): junction 2.83 / 2.84 / 2.90 -> 3.48 / 3.48 / 3.49 ms a pass (+22 %), the two `fn` instantiations 52.80 -> 63.76 us and 24.96 -> 29.20 us a launch, c1 13.720 / 13.729 -> 14.287 … 14.307 ms a step (+4.2 %, 72.9 -> 69.9 tok/s). The release flag goes to the peer before the warm, so a launch is `max(spin, warm) + row` and the warm (442,368 B a launch in the 79 POST = 1 launches, the 393 kB `fn` plus 24 KiB `h` and two 12 KiB gains, and 240 KiB in the one POST = 0 tail launch — 80 of a pass's 84, through one block) outlasts the spin by ~11 us. A second build of the same change with four independent read chains — four loads in flight a warp, same reads — recovers all of it and gains nothing: junction 2.865 against 2.859 ms a pass, step 13.697 / 13.726 against 13.720 / 13.729 ms, inside the lane's ±0.03 ms noise. So the reads are the change's floor (its bytes are the row's own, its cost is their latency) and the wait is shorter than them: the item's `spin + 2..4 us > warm` is the wrong way round at this shape. K-13's "K-12 first, then measure" dependency can be dropped.

#### K-13 Junction at decode: split the plain part's piece sums over several blocks (last-arriver finish)
- Area kernel · BI (per-wave order kept, `r4d_ihc.h:73-105`) · Gain +1 % on top of K-12 / 0.5 % / 0.7 % / 0 · Conf medium-low · Effort M · Deps K-12 first, then measure
- Change: `r4d_ar_ihc_junction.hip:287-293` (`nb <= M`): at M <= 8 several blocks each own some of the 16 waves and write `part[wave][k]`; the last arriver finishes.
- Why: one 512-thread block = one WGP for 393 KB + arithmetic; 63 CUs idle.
- Verify: as K-12.

#### K-14 `merge_sink` at 32 heads costs 19 us (sliding) against 6.3 us at 16 heads: find and fix
- Area kernel · BI · Gain 2-3 % / 1 % / 2 % / 0 · Conf medium (measured anomaly, cause not proven) · Effort S (measure) + S-M
- Change: `r4d_mla_sink.hip:443-482`: (a) value loads wait on the denominator load (`:452-455`): load `part[s]` unconditionally (the fold already skips den <= 0, `:461`), load `sinks[head]` early, store 4 bf16 as one 8-byte store (`:478-481`); (b) check the (M, heads) x 128 grid at 32 heads x 8 slots for per-head serial slot folds with dependent L2 round trips and for the stable-split scratch layout (16 workgroups write, 32 read).
- Why: M: `merge_sink` 42 x 15.2 us = 0.64 ms = 4.7 % of a c1 pass; sliding layers 19.0 us, sparse 6.3 us with the same key count (short context). Arithmetic is ~0.5 us. Exclusive attribution inside a chained trio is fuzzy, so first kbench the row alone (heads 32, slots 8, M 1, chained and plain).
- Verify: kbench `mla_attn_paged_sink` M = 1 cases (93 in `kernels_hy4.rkb`); `tests/r4d_emu/test_mla_sink.cpp` bytes; stamps.
- Closed, without code (`notes/backlog/closed/K-14.md`): (a) was built lossless (the slot values loaded with the
  maxima, `sinks[head]` hoisted, four bf16 as one 8-byte store where the out strides leave them aligned, emu bytes and
  the fused form equal) and (b) checks clean, but the premise does not hold — the merge's own body is **1.15 us** at the
  item's geometry (32 heads x 8 slots, `mla_sink_rows` + merge 8.87 us, 7.72 us with the body cut) and **1.9 us** at the
  20-slot law, not 15.2 us. That op is the sink and the merge only (the expand is its own launch), so its 8.83 us is not
  the served trio's total; the 19.0 us bracket stays unexplained, but it is not the slot chain the item names: the
  16-head layers fold 20 slots per thread and are bracketed at 6.3 us, the 32-head 8-slot ones at 19.0. Measured on
  one card (iks, gfx1201, kbench `--timing-only --max-cases 1 --op mla_attn_paged_sink`, plain and `--chain-forms`, three
  builds), the change moves the op 8.87 -> 8.83 us at the item's row (its own +-0.5 %) and 9.66 -> 9.56 us on the
  20-slot rows: not measurable against the 0.64 ms the item was going to win, so the merge bar is not met and the code
  is reverted. K-15 and K-17's launch-count prize inherit an open question (what the brackets carry), not the body's.

#### K-15 Fold `merge_sink` into the expand (G-3h alternative) or `mla_sink_rows` + merge one launch
- Area kernel+plugin · BI · Gain 0.5-1.5 % / 0.5 % / 1 % / 0 · Conf medium · Effort M · Deps K-14 result (if K-14 fixes the 19 us, the rest is launch price only)
- Change: when the sink is split, the expand workgroup (a head, 16 tokens) merges its head's partials into LDS (16 KB) with `merge_sink_body`'s fold, then multiplies (`r4d_mla_sink.hip:443-482, 673-678`; `r4d_mla_kv_bf16.hip:262-299`; `hyv4flash_attn.h:665-676`).
- Why: -1 launch and the `lat` round trip per layer.
- Verify: fusion oracle; emu bytes.

#### K-16 `stable 1` sink: raise the split cap from 16 / 32 rows to 48-64 (verify heights)
- Area kernel · BI (split and fused forms are byte-equal by the stable law) · Gain 0 / 0 / 2-4 % of a c8 verify step / 0 · Conf medium · Effort S · Deps none
- Change: `r4d_mla_sink.hip:76` (`kRowsWindow 16`, `kRowsFull 32`), `:598`, `:267-311`. The cap was justified by a `stable 0` measurement; a c4-c8 DFly2 verify (17-48 rows) falls back to one workgroup a row walking all chunks. Scratch 25 MB at 48 rows x 32 heads x 8 slots.
- Why: M (§2.2): sink rows + merge are 1.2 ms at c8 no-spec (8 rows) and grow super-linearly in the verify (levers-general: 1.75 ms at ~30 rows); est. 35-58 -> 5-9 us a sliding layer at 48 rows.
- Verify: kbench M = 24, 48 cases; verify of 4 streams equals 4 plain runs (16.4a test).

#### K-17 Expand launches sit on the full-card cliff wave count at 32 heads: 10-minute check
- Area kernel · BI · Gain ? (0 if the cliff does not apply to 1024-thread workgroups) · Conf low · Effort S
- Change: `r4d_mla_kv_bf16.hip:530` grid (2, 32, ceil(M/16)) x 1024 threads = 2048 waves; rule `r4d_common.h:349-375` (waves in 2017..2048 of a multiple of 2048 run ~5x slower; measured on 256-thread workgroups); no expand launcher calls `r4d_cliff_rows`. Also on the count: absorb 32 heads M 17..32, `merge_sink` sliding M = 16 / sparse M = 32, sel_sink 16 slots M = 32.
- Why: M: the stamps do **not** show a 5x expand in sliding layers (4.4 us exclusive at c1, 19.5 at c8 against 9.4 / 10.9 in sparse layers), so the cliff probably does not bite at 1024-thread workgroups; the M = 16 / 32 merge cases (c4-c8 verify heights) are unmeasured. A kbench A/B (one padded idle workgroup) settles it.
- Verify: `rad-kbench --timing-only` heads 32, M 1 / 16 / 32, with and without the pad.

#### K-18 `moe_gather`: template TOPK = 16, all slot loads first, 16-byte column loads; declare the `inv` plane
- Area kernel+plugin · BI (same `fmaf` order) · Gain 0.7 % / 0.5 % / 1 % (verify > 32 rows) / +0.8 % (K-45) · Conf medium-high · Effort S-M
- Change: `r4d_moe_gather.h:78-86` (runtime `top_k` loop: LDS slot, 2-byte `ye` load, fmaf serial). Plugin: declare `inv` so the router writes it and `moe_gather_inv` runs (`hyv4flash_moe.h:106-146`; `r4d_moe.hip:3441-3456, 3500`): above 512 slots (M > 32, the c8 verify) gather today adds two unchained prepass launches and loses its chained form.
- Why: M: `moe_gather` 3.6 us x 41 (c1), 5.6 us (c8).
- Verify: `r4d_moe_gather_pin_form` byte compare; stamps at c8 DFly2.

#### K-19 `moe_down_dec` writes the per-token sum (gather fused by a last-arriver fold; G-3g) and/or gather + `lat_out` quantiser
- Area kernel · BI by construction (finisher reproduces gather's order) · Gain 0.5 % / 0.5 % / 0.5 % / +0.5 % and -192 MiB arena at the full plan (M-12) · Conf medium · Effort M-L
- Change: `r4d_moe.hip:1851-1860`: per (token, 64-column tile) an arrival counter; the block bringing the token's last slot applies the gather (`r4d_gemm_iqr.hip:716-745` finisher pattern). Alternative, smaller: gather wave (256 columns = two 128-groups) runs `r4d_fwht_wave<128>` + quant for `lat_out`'s input (`rad_block_moe_fp8.h:2096-2101`), -1 quantiser launch.
- Why: M: gather 3.6 + copy16 3.0 + quant 2.9 us a layer; at prefill `edn` is a 100-192 MB HBM round trip a layer.
- Verify: fusion oracle; prefill profile `moe_gather` line.

#### K-20 Head at M <= 2 has no chained form (`mvm_kernel` keeps the barrier bit)
- Area kernel · BI · Gain 0.1 % / 0 / 0.2 % / 0 · Conf high · Effort S
- Change: `r4d_gemm_fp8a16_nt_m1.hip:806-807, 879-886`: add `r4dchain_mvm` (wait_vec) and pass `chain` through the M <= 2 branch (`lm_wmma` has `r4dchain_lm_wmma`, `:708`).
- Why: M: `mvm_kernel` 302 us unchained at c1 (at the byte wall: 186 MB at 620 GB/s); the barrier gap is 4.4-4.6 us; one a trunk step, five a draft pass at one sequence.
- Verify: selftest logits chained case; stamps `chained` column.

#### K-21 Head: fused top-8 epilogue for the draft rounds (no f32 logits plane, no `cast`, no stage-1 `row_topk`)
- Area kernel+drafter · BI if rounding matches `cast_f32_bf16` · Gain 0 / 0 / 0.3-0.5 % / 0 · Conf medium · Effort M · Deps D-04
- Change: `logits_gemm` variant whose blocks keep a per-block top-8 (value, id) and hand stage 2 the candidates (`r4d_rowtopk_bf16.hip:148-176, 217-258`); skips the 242 KB a row f32 logits write.
- Why: M (§2.3): a draft round's head + cast + row_topk + merge.
- Verify: `row_topk` kbench rows; `dfly.cand_*` taps identical.

#### K-22 Standalone quantiser body: all groups at once, uint2 loads; row staged in LDS for `rmsnorm_had_quant_i8g`
- Area kernel · BI · Gain 0.5 % / 0.3 % / 0.5 % / 0.5 % · Conf medium · Effort S
- Change: `r4d_quant_act_fp8.hip:65, 71-75, 154` (a block of 8 waves a row, 24 groups at K 3072 -> 3 groups serial a wave; four 2-byte loads a lane); `r4d_fused_quant_fp8.hip:211-246, 290-294, 368-371` (norm re-reads the row from global: stage 6 KiB in LDS).
- Why: M: 213 quantiser-class launches = 0.74 ms a c1 pass (`gated_quant` 83 x 3.5, `quant_act_fp8` 88 x 3.3, `rmsnorm_quant` 42 x 3.8): 0.3-0.9 us above the 2.9 us floor each.
- Verify: selftest quantiser rows byte compare; kbench.

#### K-23 Fuse the remaining standalone quantisers into their producing GEMM epilogue (`gemm_nt_q_gated` exists; shared expert does not use it)
- Area plugin+kernel · BI · Gain 0.3 % / 0.2 % / 0.3 % / 0 · Conf high (code) · Effort S
- Change: `r4d_rows.cpp:9366-9382` (`gemm_iqr_gated` = GEMM + `gated_had_quant_i8g`, M <= 16, fusion oracle `r4d_fuse_gemm_nt_q_gated_iqr`); plugin GatedMlp issues gate_up then `gated_had_quant_i8g` (`hyv4flash_moe.h:31-78, 88-90`). Declare the gated row for decode-class rows of the shared expert and layer 0's dense MLP (two instances by row range). Pick this or K-01 for the shared gate_up first (K-01 pays more; the dual launch then needs the gated epilogue per part).
- Why: M: `gated_quant` 83 launches x 3.5 us.
- Verify: fusion oracle (exists).

#### K-24 Indexer key side at decode: early-out when no group closes; fuse `qsa_work` + `idx_pool_gated` + tail store
- Area kernel+plugin · BI · Gain 0.7 % / 0.5 % / 0.5 % / 0 · Conf high (early-out) / medium (fusion) · Effort S / M
- Change: `r4d_idx_pool_gated_f32.hip:72-182` (page check only at `:177-180`): after the wait read the page and return when no group of the workgroup closes and `out` / `q8` are null (3 of 4 decode steps of a lone sequence close nothing). Then `:133-137` (one-thread 128-term f64 sum) overlapped with the trig `:159-163`; then one op for work + pool + tail (`r4d_qsa_work_bf16.hip:57-114`, `r4d_qsa_tail_store_bf16.hip:41-61`, `hyv4flash_indexer.h:354-376`; check the alias proof against the rollback contract in the tail store's header).
- Why: M: `idx_pool_gated` 12 x 9.4 us + `qsa_work` 12 x 3.0 + `qsa_tail_store` 12 x 3.0 + key norm = 0.19 ms a pass (1.3 %).
- Verify: `idx_pool_gated` kbench rows; rollback gate `tools/hy4flash_rollback_gate.py`.

#### K-25 `qsa_select` at long context: multi-workgroup select (154 us a launch at 64K, 305 us at 200K)
- Area kernel · BI (same selection, ties as today) · Gain at >= 32K context: 10-20 % of a c1 decode step / less at c8 / same / 1 % · Conf medium (numbers from `notes/idxint8.md` `--perfidx`, not re-measured in a served run) · Effort M
- Change: `r4d_qsa_select_bf16.hip:417-760`: one workgroup of 256 a row does the radix select over the row's whole score row; at decode (M <= 48) split the histogram passes over ceil(G/4096) workgroups with an arrival counter, or threshold-pass on the sampled pivot in parallel. Also `masked 1` option: no block-table load in threshold passes (`:466, 484-485, 556, 578`; `idx_score` already writes -inf for absent pages).
- Why: 12 launches a selecting decode step: 1.8-3.7 ms a step once a session passes ~32-64K tokens (agentic histories), i.e. the largest context-dependent decode term. M (longctx sweep, `work/longctx`): c1 decode 69.7 tok/s at 2K and 69.4 at 56K, so at one stream the term is not visible in tok/s; re-measure with stamps at 60K before building (T-03).
- Verify: `--perfidx 16384 1`, `qsa_select` kbench; selections identical on a real 60K trace.

#### K-26 Row front, rope, small stores: wider accesses; rope inside the decode sink (G-3e alternative)
- Area kernel · BI · Gain 0.5 % / 0.3 % / 0.4 % / 0 · Conf medium · Effort S (widths) / M (fusion)
- Change: `r4d_model_bf16.hip:817-830`, `kv_store_body` `:770-782` (3 dependent global phases, 2-byte copies): write the cache slot from registers. Rope: apply `rope_body`'s exact formula (`:662-698`) on the query fragment inside the `stable 1` sink (`r4d_mla_sink.hip:186-206`; sel_sink the same), -1 launch a layer.
- Why: M: `mla_row_front` 42 x 5.9 us (above floor by 3 us), `rope` 42 x 3.0 us.
- Verify: emu bytes; fusion oracle.

#### K-27 Engram bf16 GEMMs: chained row for `gemm_nt_bias` (the 4 `vit_gemm_wa` keep barrier bits), `engram_gate` loads
- Area kernel · BI · Gain 0.3-0.5 % / 0.2 % / 0.3 % / 0.2 % · Conf medium · Effort S-M
- Change: `hyv4flash_engram.h:286-293` -> `vit_gemm_wa` (key, N 12288, 37.7 MB bf16) unchained; add a chained bf16 `gemm_nt_bias` row at decode heights. `engram_gate` (`r4d_engram_bf16.hip:230-303`): 4 barriers, f32 weights re-read per row, 2-byte loads.
- Why: M: `vit_gemm_wa` 4 x 68.3 us (at its byte floor: 37.7 MB) + `vit_gemm` 4 x 27.3 us + `engram_gate` 4 x 17.7 us = 0.45 ms a pass (3.3 %); lossless part: 4 barrier gaps (4.4 us each) + ~10 us of gate latency a layer. The rest is bf16 bytes (F-06).
- Verify: stamps `chained`; kbench `engram_gate` (17.4 us in perf.md §2.2).

#### K-28 `copy2d16` embedding -> four streams as one `hc_enter`; `embed_lookup` 16-byte copies
- Area plugin+kernel · BI · Gain 0.3 % / 0.2 % / 0.2 % / 0.3 % · Conf high · Effort S
- Change: `hyv4flash.cpp:964` (decl), `:1147-1148` (4 `cast` launches) -> `hc_enter` (`r4d_rows.cpp:8848`) or lookup straight into stream 0 + one copy. `r4d_model_bf16.hip:82` (`embed_lookup` copies 2-byte elements): uint4.
- Why: M: `copy2d16` 4 x 13.6 us + `embed_lookup` 23.4 us at the pass head = 78 us (0.6 %).
- Verify: taps `t_embed`; greedy equal.

#### K-29 Step metadata read before the chain wait in short kernels
- Area kernel · BI if no kernel of the pass writes the operand (unverified for DFly2's device-written tables) · Gain 0.3-0.5 % / 0.2 % / ? / 0 · Conf low · Effort M
- Change: sink prologue `r4d_mla_sink.hip:162, 313-321`, `qsa_work`, tail store, row front, rope, `idx_score`: host-uploaded positions / tables as vector loads before the wait; `r4d_chain_wait` -> `_vec` where the scalar audit agrees (`r4d_mla_kv_bf16.hip:236, 351`).
- Why: 1-2 dependent round trips on ~250 floor-class launches.
- Verify: `tests/chain_scalar_audit.py`; spec test.

#### K-30 `sample_topk`: one block a row with LDS-atomic contention; `sample_chain_f32` top_p uses float LDS `atomicAdd` (order-dependent)
- Area kernel (sampler) · BI for the counts; the float atomics are a determinism defect to fix first · Gain 0.5-1.5 % / 0.5 % / 0.5 % / 0 · Conf medium-low · Effort M
- Change: `r4d_sample_stages_f32.hip:413-470, 1192`: per-wave histograms or a multi-block stage 1; make the top_p accumulation order-fixed.
- Why: the sampler chain is off the tape; levers-general bounded it at 0.2-0.5 ms a step ("not a lever" as greedy-vs-sampled A/B: CLOSED), but the kernel form was never optimised and the determinism note is new. M (§2.5): see `prof` c1s rows `sample_*`.
- Verify: seeded sampled output reproducible run to run at c1; chi-square unchanged (`mtp_rrs_test`).

### 3.2 Kernels (libr4d): prefill-class rows (M = 256..2048)

#### K-40 Sink attention prefill, step 1: column-major value tile (16.4b part 1)
- Area kernel · BI · Gain 0 / 0 / 0 / +3-5 % (more above 2048 context) · Conf high (mechanism) / medium (share) · Effort M · Deps none
- Change: `r4d_mla_sink.hip:391-396` reads V through `fragT` = 8 `ds_read_u16` a lane a fragment (`r4d_gdn_wmma.h:85-96`): 64 LDS reads a wave a chunk against 18 `ds_read_b64` on the score side. Write a `vt[512][16]` tile at stash time; the value fragment becomes one `ds_read_b128` a lane.
- Why: M (§2.4): `mla_attn_paged_sink` 30 x 2.74 ms + `mla_attn_sel_sink` 12 x 6.66 ms = 162 ms = **28.7 % of a chunk**, now the largest block of prefill; neither is WMMA-bound (est. WMMA floor sel 1.4-1.6 ms, paged 0.7-0.8 ms a call).
- Verify: `tests/r4d_emu/test_mla_sink.cpp` bytes vs today; kbench sel / paged rows with tables from a real `qsa_select` (perf.md §2.2 caveat).

#### K-41 Sink attention prefill, step 2: single LDS key buffer (occupancy) as an axis `kb`
- Area kernel · BI · Gain prefill +0.5-1.5 % · Conf medium · Effort S-M · Deps K-40 (frees the 16 KiB K-40 needs)
- Change: `r4d_mla_sink.hip:152, 330, 397`: the next chunk already sits in registers; one `keys[16 x kPitch]` buffer + a third barrier: LDS 41.6 -> 22.9 KiB (HG 1), 45.7 -> 27.0 (HG 2) = 2x workgroups a WGP.
- Verify: emu bytes; kbench M = 1024 / 2048 at 16 and 32 heads; `rad-tune` axis.

#### K-42 Sink attention prefill, step 3: row tiles (16.4b part 2 / FP-7), union staging for sel
- Area kernel · BI under the lo-anchored law (window not yet full: identical grids) · Gain prefill +3-5 % (K-40..K-42 together: sink -25..-35 % = +7-10 %) · Conf medium · Effort M-L · Deps K-40, K-41
- Change: `rt` rows of one sequence a workgroup: keys / `vt` loaded from LDS once for rt WMMAs; paged window staged once for rt rows (rows share 511/512); sel: union of the rows' selected groups, fallback to single-row when the union is too large (Flash-Next `r4d_attn_usel.h` pattern; do not cherry-pick `afa63933`). Target sel <= 2.8 ms, paged <= 1.0 ms a call at 2048 rows (G-4). `r4d_mla_sink.hip:21-29, 337, 392`.
- Verify: emu sweep over (hg, rt, kb); VGPR <= 245, no spills, LDS <= 64 KiB stated in the commit; profile-ops before/after.

#### K-43 Row-tiled `idx_score` (the term that grows with context)
- Area kernel · BI · Gain prefill +1 % at 8K, +5 % at 47K, more at 128K; decode 0 · Conf high · Effort M
- Change: `r4d_idx_score_bf16.hip:49-99, 172-175`: a (row, 64 keys) workgroup re-reads all L/4 pooled keys per row with no LDS (512 B a (row, group)). A workgroup of 8 waves takes 16 rows of one sequence x 256 keys: key fragments stay in registers, each row's 4 KiB query staged once in LDS; same 8-WMMA chain per (row, 16 keys). Add the causal skip (tiles with b0 >= (pos+1)/4; `qsa_select` never reads them, `r4d_qsa_select_bf16.hip:451-453`) and `:60-111` wave early-out.
- Why: M (§2.4): `idx_score` 12 x 512 us = 6.2 ms a chunk at 6K context (1.1 %), linear in context: ~45 ms at 47K (est. 12.3 GB of L2 reads a sparse layer, ~3.5 ms vs a 0.5 ms compute floor); M: BetterBench prefill falls 3964 -> 2964 tok/s from 1.5K to 47K (chunk 517 -> 691 ms).
- Verify: `idx_score_case` (`r4d_selftest.cpp:28656`) bytes; `--perfidx 49152 2048`; kbench `idx_score`.

#### K-44 Routed gate_up at prefill: lift the 4-bit gate of the packed 128-row map (FP-8 answered from code)
- Area kernel · BI (packed map = run map bytes; selftest compares) · Gain prefill +1.5-3 %, more with pooled experts read in place · Conf high (selection), medium (gain) · Effort S-M
- Change: `r4d_moe_iqr.hip:928` (`g.packed = fm.bits == 4 && fm.L == 16 && ...`) and `:773` (`if constexpr (F::B == 4 && F::L == 16)`); Hy4's gate_up is u3, so it always takes the 64-row run map (`kMapTiles 4`, `:679`): an expert with 65-128 rows (mean 64 rows an expert at 2048 x 16 / 512) is decoded twice and, if not resident and not staged, read twice over PCIe. `R4D_MOEIQR_FORMS` already instantiates `X(3, 16)` (`:723-724`). Check VGPR at NT 8 (237 at 4 bits, `:686`).
- Note: the env switches `RADIANCE_MOE_IQR_MAP_TILES` / `RADIANCE_MOE_DOWN_MT` named in FP-8 do not exist in this tree; use `r4d_selftest --perfmoeiqr`, `R4D_PERF_IQR_PINS`, `r4d_moe_iqr_pin_map_tiles` (`r4d_selftest.cpp:7940, 8297-8303`).
- Why: M (§2.4): routed gate_up 41 x 2.13 ms = 87 ms = 15.4 % of a staged chunk; Flash-Next's 4 -> 8 tiles gave -6..-9 % of the gate_up at 17-41K rows, Hy4 has more experts across the 64-row boundary: -6..-15 % = 5-13 ms. In an unstaged step (537 rows: 3.0 ms a call) the doubled entries are doubled link reads.
- Verify: `--perfmoeiqr` at K 1536, top 16, 32768 routed rows, tiles pinned 4 vs 8 (bytes + time); profile-ops.

#### K-45 `moe_gather` wide kernel for f32 routing weights
- Area kernel · BI (F32 law `fmaf`) · Gain prefill +0.8 % · Conf high · Effort S
- Change: `r4d_moe.hip:3476` (`wide_ok = ew->dtype == RAD_BF16 && ...`); Hy4's `w.ew` is f32 (`hyv4flash_moe.h:126`), so 2048-row chunks run the 2-byte-a-lane `gather_kernel` plus two inverse-table launches (`:3449-3456`). Add `gather_wide_kernel<NW, F32>`.
- Why: M (§2.4): `moe_gather` 41 x 356 us = 14.6 ms = 2.6 % of a chunk; est. -6 ms.
- Verify: `r4d_moe_gather_pin_form` (`r4d_moe.hip:2366-2370`) form 1 vs 2 bytes.

#### K-46 Junction arithmetic at prefill: share the `fn` reads across a block's rows
- Area kernel · BI (per-row sum order kept) · Gain prefill +2-2.5 % · Conf high (mechanism), medium (time) · Effort M
- Change: `r4d_ihc.h:207-262`, `r4d_ar_ihc_junction.hip:107-122`, `r4d_ar_entry.hip:1681` (`R4D_AR_MAX_BLOCKS 512` -> a block owns 4 rows at 2048 and walks them one by one, each re-reading the 384 KiB f32 `fn`: 805 MB of L2 a call). Process a block's rows in groups of 4: norms per row into 4 `ybuf`s, `fn` loaded once a piece into `acc[4][R+1]`; barriers of steps 1 and 3 shared. Also the TP1 plain row and the 4-rank twin.
- Why: M (§2.4): junction 84 x 741 us = 62.3 ms = 11.0 % of a chunk; M (`notes/t4.md`): 0.29 ms of a call is arithmetic (24 ms a chunk), est. -12..-16 ms.
- Verify: `tests/r4d_emu/test_ihc.cpp`; kbench `rmsnorm_ihc_junction` (69 cases); TP2 vs TP1 `work/wp8/cmpdirs.py`.

#### K-47 `cast` (2D copy kernel) is mis-shaped for narrow rows: flatten the grid
- Area core/device · BI · Gain prefill +1-1.5 %; helps every narrow 2D copy (DFly2 splits, taps, checkpoints X-06) · Conf high · Effort S
- Change: `core/device/aql_blit.hip:108-119, 260-275`: grid `(ceil(w16/256), min(height, 65535))`, 256 threads, thread = one 16-byte piece of one row: the indexer's 128-byte rows use 8 of 256 lanes in 32768 one-row blocks. Flatten: `i -> (r = i / w16, c = i % w16)`, grid-stride over `min(ceil(total/256), 2048)` blocks.
- Why: M (§2.4): `cast` 28 x 170 us = 4.8 ms a later chunk (0.8 %), 158 calls x 205 us = 32 ms over the first two chunks (3 %): 4-25 MB a call should take 0.01-0.05 ms.
- Verify: kbench `cast` row / tap parity; profile-ops.

#### K-48 Remove the casts and `mla_kv_split` with pitched operands
- Area plugin+kernel · BI · Gain prefill +0.5-1 % (first chunk and selecting steps), decode selecting steps -24 launches · Conf high (b-d), medium (a) · Effort M · Deps K-47 takes most of the time already
- Change: (a) `q_b` as one GEMM writing `[T, H, 256]`, consumers read heads with stride 256 (`mla_project_kv` `xh`, `r4d_mla_kv_bf16.hip:56-58, 148`; sink `row_tensor_ok` accepts `stride[1] >= width`, `r4d_mla_sink.hip:531-533`; `rope` needs a trailing offset: unverified); (b) indexer query as two operands or one `[T, 16, 128]` linear (`hyv4flash_indexer.h:227-234, 383-390`); (c) `gemm_bf16_nt_tiled` with an A pitch instead of `op_lat_copy` (`hyv4flash_attn.h:427, 684-685`; `r4d_gemm_bf16_nt_tiled.hip:222-235`); (d) `attn_kv_sink` reads `kvb` + rope row directly (`Args` `:71-81` strides; `fetch` `:226-228`), dropping `mla_kv_split` (58.7 MB read + 67 MB written on every first chunk).
- Verify: fusion oracle per op; harness chunked prefill parity.

#### K-49 `rope_rows` instead of `rope` at prefill
- Area plugin · BI (the per-(token, pair) form exists and is byte-identical, opt-in by op name) · Gain prefill +0.2-0.4 % · Conf high · Effort S
- Change: `hyv4flash_common.h:188-200` declares `rope` (one `sincosf` a (token, head, pair)); `r4d_model_bf16.hip:1530-1541`, `r4d_rows.cpp:7384`.
- Verify: kbench; harness parity.

#### K-50 `mla_expand_gate` at prefill: stage the token block once a head; lift the 64-row cap of the had_quant fusion
- Area kernel · BI · Gain prefill +0.5-1 % · Conf medium · Effort M
- Change: `r4d_mla_kv_bf16.hip:330-365, 530`: each of a head's 16 column-tile workgroups re-reads the same 64 x 512 bf16 token block (1.07 GB of L2 a call at 32 heads). `kFuseRows 64` (`hyv4flash_attn.h:133, 466, 648`): give `kv_expand_gate_quant` an MB axis and lift the cap (removes a 33.5 MB round trip + one quantiser a layer-step).
- Why: M (§2.4): `mla_expand_gate` 42 x 504 us = 21.2 ms (3.7 %) + its quantiser 42 x ~46 us.
- Verify: `tests/r4d_emu/test_mla_kv.cpp`; kbench `mla_expand_gate` (48 cases).

#### K-51 Small-N trunk GEMMs at prefill run 4x below `attn_output`'s rate: tile arm / global decode claim
- Area kernel · BI · Gain prefill +1-1.5 % · Conf medium (cause not proven) · Effort S (kbench) + M
- Change: `r4d_rows.cpp:9384-9395` (`gemm_iqr_tiled`, NOTUNE), `r4d_gemm_iqr.hip:782-842` (`IqrPre::begin`: blocks of N tile j claim only j's 8 decode parts and spin). q_a (N 1024): 128 blocks, half wait. Options: a 64-wide arm for N <= 1024; a global claim over all N tiles; a tile axis as the w4a8 rows have (`tGemmPrefill`).
- Why: M (§2.4): the eight small trunk GEMMs (q_a 100 us, q_nope 152, q_rope 66, latent_in 128, latent_out 139, shared 132 / 87, kv_a 126 us a call) are 38.6 ms = 6.8 % of a chunk against `attn_output` + `attn_gate` 43 ms for 5x the MACs.
- Verify: kbench `gemm_nt_q` at (2048, 1024, 3072) vs (2048, 3072, 8192); also resolve which row serves `kv_a` (N 576) and `index_head` (N 16) above 64 rows (unverified: both prefill tiles need N % 128).

#### K-52 Router f32 GEMM at prefill: LDS-staged x, wider tile (byte-identical)
- Area kernel · BI · Gain prefill +1 % · Conf medium · Effort M · Deps K-11 (halves its weight bytes)
- Change: `r4d_glm.hip:17-78, 126` (8 two-byte x loads per k step; 12 loads per 32 FMA): stage x as f32 in LDS once, wave tile 16 x 8, keep each lane's k order and the butterfly.
- Why: M (§2.4): `gemm_nt_f32` 41 x 422 us = 17.3 ms (3.1 %) + `all_gather` 41 x 105 us = 4.3 ms; ~7.5 TFLOP/s.
- Verify: `--perff32 256 3072 2048`, `f32_forms`.

#### K-53 SwiGLU in the gate_up GEMM epilogue (prefill: 100 MB `egu` round trip a layer)
- Area kernel · BI if the epilogue rounds as the stored plane · Gain prefill +1.5-2 %, decode -1 launch a layer · Conf medium · Effort L
- Change: `rad_block_moe_fp8.h:1964-1979` (`moe_gemm_q` -> `w.egu` [T*16, 2F] -> `swiglu_clamped_had_quant_i8g`): a tile owns 128 gate columns and the matching 128 up columns (`parts 2`).
- Why: M (§2.4): `swiglu_clamped_had_quant_i8g` 41 x 254 us = 10.4 ms (1.8 %); 32768 x 1536 x 2 B = 100 MB written by the GEMM and read back per layer.
- Verify: fusion oracle, kbench prefill rows, both class tables.

#### K-54 Routed down at prefill: a block walks 2-3 N tiles over one staged activation tile
- Area kernel · BI · Gain prefill ? (+0.5-2 %) · Conf low-medium · Effort M
- Change: `r4d_moe.hip:2841-2845, 3117-3131`: `dmt 4`, 64-row entries, 256-column tile; the 6 N-blocks of an entry each re-stage the same 48 KiB activation tile. (A block taking several row tiles was tried and was no faster, `:2826-2836`; several N tiles was not.)
- Why: M (§2.4): routed down 41 x 1.11 ms = 45.6 ms = 8.1 % of a chunk.
- Verify: `--perfmoe`, `R4D_PERF_MOE_MT`.

#### K-55 Run map: dispatch non-resident experts' entries first
- Area kernel+core · BI · Gain prefill ? (bounded by the non-overlapped part of the in-place link time; small once M-03/M-04 stage everything) · Conf low-medium · Effort M
- Change: the map builder (`moe_iqr_packmap_kernel` / `moe_blockmap_kernel`) takes a residency bitmask and places miss entries first so their link reads start at t = 0.
- Verify: `/stats` `stream_bytes`, chunk time at long context (M-04's overflow case).

#### K-56 `attn_kv_sink` first chunk: 32-key tiles (3 -> 1.5 barriers per 16 keys); `bq` axis (16.6)
- Area kernel · BI for the 32-key tile if max/den stay per 16-key sub-tile; `bq 64` is PF · Gain prefill +0.3-0.5 % (first chunk only) · Conf medium · Effort S-M
- Change: `r4d_attn_kv_sink_h256.hip:242-284` (3 barriers a 16-key tile).
- Verify: kbench window 512 M = 1024 (-8 % or drop, as 16.6).
- The `bq` part is its own ticket now: K-56b (L7). K-56 keeps the 32-key tiles.

#### K-56b (L7) `attn_kv_sink_h256`: 64-row query tile (`bq` axis) at window 512
- Area kernel · PF (moves prefill bits within the bound, E4 of perf.md; decode rows never take this kernel) · Gain prefill: `attn_kv_sink` -8..-10 % at window 512 (perf.md §3.6: 574 -> ~520 us at 64 heads, M = 1024; ~-1.7 ms per 1024 tokens of a one-shot prefill at TP1), i.e. on today's first, non-absorbed chunk 42 x 764 us = 32 ms -> about -3 ms (~0.3-0.5 % of that chunk); 0 for decode / verify · Conf high (mechanism), medium (gain on gfx1201 at TP2) · Effort S (the cheapest kernel package) · Deps none; can share the commit with K-56's 32-key tiles but needs its own A/B
- Change: `libr4d/r4d_attn_kv_sink_h256.hip`. A workgroup of 128 query rows reads keys from its first row's window start to its last row's own key: 639 keys at window 512, of which each row sees 512, so 20 % of the key tiles are masked work. With 64 rows (8 waves, 256 threads) it is 575 keys (11 %). Add an axis `bq` {128, 64}, default 128 until measured; `RG` becomes a template parameter (`kThreads = bq * 4`). LDS and VGPR per wave unchanged (168 VGPR, LDS under 37 KiB). Key tiles start at the workgroup's lowest visible key, so `bq` moves a row's chunk grid: prefill bits move within the stated bound, which the kernel's contract allows (perf.md §1, E4). Window 0 (full causal) gains nothing; keep 128 there.
- Why: perf.md §3.6 (L7) and the measured first chunk (§2.4 of this note: `attn_kv_sink` 42 x 764 us in the non-absorbed chunk). Window-512 sliding layers are 30 of 42; the masked share drops from 20 % to 11 % of the key tiles.
- Verify: `tests/r4d_emu` attention case at bq 64 vs bq 128 within the documented prefill bound (not byte-equal by design) and vs libref; `r4d_selftest --case attn_kv_sink_h256`; `rad-kbench` window 512 at M = 1024 / 2048, 32 and 64 heads (expect -8..-10 %, else drop); greedy text on u3 with a > 2K prompt stays within the prefill bound (logprob drift reported); `rad-tune` picks `bq` per window.

#### K-57 `had_quant_act_i8g` at prefill: uint2 loads; idle waves at n = 1536
- Area kernel · BI · Gain prefill +0.2-0.4 % · Conf low (HBM-bound: 430 GB/s) · Effort S · Deps same file as K-22
- Change: `r4d_quant_act_fp8.hip:52-120`.
- Why: M (§2.4): 130 standalone calls a chunk x 46 us = 6.0 ms (1.1 %).

#### K-58 Engram at prefill: device-domain planes when the table is not on disk; gather + norm fused; `engram_gate` weights re-read per row
- Area plugin+kernel · BI · Gain prefill +0.3-0.5 %, decode 0.25 % · Conf medium · Effort S-M
- Change: `hyv4flash_engram.h:74-78` (`eg.ids`, `eg.ids_ahead`, `eg.rows` are `RAD_DOMAIN_HOST` at every placement: rows written and read back over PCIe, 3 more ordered launches; `notes/adopt.md` "Open": ~8 us a layer); `r4d_engram_bf16.hip:188-197` (norm reads its row twice), `:250-301`.
- Why: M (§2.4): Engram 10.2 ms a chunk (1.8 %): `embed_lookup_q` 4 x 1040 us (the gather into a host-domain plane), `engram_norm` 4 x 412 us, gate 4 x 311 us.
- Verify: stamps Engram layer (M: 348-417 us against 287 for a plain sliding layer); greedy equal.

#### K-59 rad-tune axes missing on the Hy4 prefill path (all bit-neutral)
- Area kernel/tooling · BI · Gain enables K-41/K-42/K-51 tuning; by itself ? · Conf medium · Effort S each
- Change: add axes: sink `hg` (only the test hook `g_sink_form`, `r4d_mla_sink.hip:93-96`), `rt`, `kb`; `gemm_iqr_tiled` / `gemm_i8a8_tiled` tile arms; `moe_gemm_q` map tiles {4, 8} and down `dmt` {2, 4}; `moe_gather` form. Run `rad-tune` once at Hy4 shapes on gfx1201: `mb` of `mla_project_kv` / `mla_expand_gate` was never tuned at M 2048 x 32/16 heads; two-rank rows only via `RADIANCE_TUNE=` at serve (`r4d_rows.cpp:6300-6331`).
- Verify: tune report; kbench.

### 3.3 Plugin graph (arch/hyv4flash, arch/common MoE)

#### P-01 Shared expert beside the routed GEMMs (X5, concrete)
- Area plugin (+ small MoeFP8 hook) · BI (order only) · Gain 1-2 % / 1-2 % / 1-2 % / 0 · Conf medium · Effort M · Deps none; K-01 / K-04 are the kernel-level alternative for the two GEMMs (do P-01 first: plugin only)
- Change: `hyv4flash_moe.h:178` sets `c.n_ff_shared = 0` (the shared arm is the plugin's `GatedMlp`), issued after `lat_out` at `:399`; MoeFP8's interleave (`rad_block_moe_fp8.h:1855-1861, 1939-1993`, `rad_buf_concurrent` `:1571-1605`) never applies. Give `MoeFP8::pass` a `SideOps` hook (pattern `rad_block_attn_gated_fp8.h:194-199`, `rad_arch.h:1527-1543`), issue shared gate_up / `gated_had_quant_i8g` / down behind the router and the routed down; declare the shared arm before `moe.declare` (`:321` vs `:326`) or mark `sgu` / `sff` / `sout` concurrent.
- Why: M: shared gate_up 9.0 us + `gated_quant` 2.9 us + shared down 3.1 us a layer are ordered launches behind `latent_out` on the critical path; beside the link/byte-bound routed GEMMs (M: 53 us a layer at c1, 283 us at c8) they are hidden: 41 x up to 15 us = 0.2-0.6 ms.
- Verify: stamped pass (`layer.py`); greedy text + logprobs equal; `hyv4flash_arch_test` lint.

#### P-02 Routing report without the `copy16` (declare a routing table)
- Area plugin · BI (counts only) · Gain 0.3-0.9 % / 0.2 % / 0.3 % / 0 (and fixes a latent hazard) · Conf high · Effort S
- Change: `hyv4flash_moe.h:128` (`w.ecnt` a plain transient [n_expert_r] buffer, `ecnt_pitch` 0) -> `rad_block_moe_fp8.h:2203-2212` -> `core/runtime/ctx.cpp:706-707` issues an async `copy16` per MoE layer. Declare `moe.expert_counts` as [n_layer, 512] i32 PERSIST + `rad_route_table(b, buf)` + `ecnt_pitch = 512` as `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:1005-1007, 1137` does (`abi/rad_builder.h:840-851`).
- Why: M: `r4dchain_copy16` 41 x 3.0 us = 122 us a pass (0.9 % exclusive; realised gain of a removed chained launch is 0.3-0.5 us, so 15-40 us is the safe figure). qwen4exp's comment warns that a transient histogram can be aliased by a wide prefill chunk: wrong heat counts.
- Verify: `/stats experts` counts identical on a fixed greedy run; dispatches a pass 1330 -> 1289.

#### P-03 Prefill: last layer computes its post-cache work for all rows of the chunk
- Area plugin · BI for row-local ops; prefill attention rows PF at worst · Gain 0 / 0 / 0 / +2-2.5 % · Conf medium · Effort M · Deps interacts with X-02 (`ckpt_cut`) and the drafter anchors
- Change: `hyv4flash.cpp:1153-1211`: in layer 41 keep the cache writers on all T rows (kv_a, row front, index key/norm/work/pool/tail) and run the rest (q path, attention, o_proj, junctions, MoE, final norm) only on the `n_out` rows plus the rows that can be drafter anchors (each prefill sequence's last row); a chunk that ends no prompt skips that tail.
- Why: M (§2.4): a chunk is 565 ms of op time over 42 layers (13.4 ms a layer); the last layer's post-cache share is ~12 ms = 2.1 %.
- Verify: harness chunked-prefill parity, rollback gate, profile-ops.

#### P-04 `sub_i32` cu rebase once a step, not once a layer
- Area plugin · BI · Gain prefill / mixed steps: -41 host-issued launches (+0.1-0.3 %) · Conf high · Effort S
- Change: `hyv4flash_attn.h:379` (decl per layer), `:570`, `:612` -> one model-level `cu_prefill`.
- Why: M (§2.4): `sub_i32` 30-42 calls x 6.4-9.7 us a prefill step.

#### P-05 Engram `ahead` hashing only for a disk-placed table
- Area plugin · BI · Gain prefill +0.1 % · Conf high · Effort S
- Change: `hyv4flash_engram.h:359-368` (A = n_ahead unless `RADIANCE_HY4_ENGRAM_AHEAD=0`); "a device gather ignores them" (`:54-55`).

#### P-06 Engram: one CONV history and one `engram_ids` for the four layers
- Area plugin+op · BI · Gain 0.1 % decode; 3 fewer CONV state rows per checkpoint / rollback · Conf high · Effort S-M
- Change: `hyv4flash.cpp:969-981`, `hyv4flash_engram.h:140, 352/365` (kernels/engram.md §1 designed one shared history).
- Why: M: `engram_ids` 4 x 4.0 us + `engram_state` 4 x 2.3 us.

#### P-07 Junction form for the Engram layers (they fall back to `ihc_pre` + `rmsnorm` + own quantiser)
- Area kernel fusion+plugin · BI · Gain 0.3-0.5 % / 0.2 % / 0.3 % / 0 · Conf medium · Effort M-L
- Change: `hyv4flash.cpp:772-776` (tail-0 junction in front of layers 1, 11, 21, 31), `:1156-1164`; q_a quantises itself (`hyv4flash_attn.h:219-223`). A junction form applying `engram_gate` between `ihc_post` and `ihc_pre`.
- Why: M: Engram layers 386-417 us against 287; 4 extra floor launches each beside the two GEMMs (K-27).

#### P-08 `gather_rows` before the head when every row is an output row
- Area plugin · BI · Gain 0.05 % · Conf medium · Effort S
- Change: `hyv4flash.cpp:1207-1210`; pass key must carry the choice (`:1258`).

#### P-09 DFly snapshot of the indexer selection: write in place, PERSIST only when served
- Area plugin · BI · Gain decode selecting steps -10 launches; VRAM 12.6-34 MiB a rank (9-23 units) · Conf high · Effort S (a-c), M (d)
- Change: `hyv4flash_dfly.h:202-238`: (a) let layer l's `qsa_select` write straight into `dfly.sel_snap.<l>` for non-streamed groups; (b) one `fill` over a [5, T] plane instead of 5 (M: `fill_kernel` 5 x 2.9 us a verify pass); (c) `final_hidden` PERSIST only when `m.df.served` (`hyv4flash.cpp:616`, 12.6 MB); (d) prefill chunks copy only anchor rows.
- Verify: dfly parity taps; rollback gate; startup receipt.

#### P-10 Indexer score plane [max_tok, max_ctx/4] f32: row tiles
- Area plugin · BI · Gain arena peak -134 MB at 64K context (-268 MB at 128K): more room for the ring / experts during prefill · Conf medium · Effort M · Deps K-43 (same kernels need a row offset)
- Change: `hyv4flash_indexer.h:63, 81, 390, 403-409`: score + select per 256-1024 rows.

#### P-11 Arena: multi-interval liveness (attention scratch and MoE scratch never coexist inside a layer)
- Area core/build · BI · Gain full-plan arena -0.3 GiB a rank (prefill residency or a larger stager ring, M-04) · Conf medium (sizes are arithmetic) · Effort M-L
- Change: `core/build/rad_bufplan.cpp:10-11, 91` (one [first_def, last_use] interval a buffer; wires are reused by every layer so they span the whole step). A buffer whose every use-run starts with a full write is dead between runs.
- Verify: startup arena line; `hyv4flash_arch_test` every level binds; add `peak_live` to the log (rad_bufplan.cpp:99-104).

#### P-12 64- / 128-row arena level
- Area core/engine+build · BI · Gain +60-90 resident units a card at decode (0.3-0.4 points): c8 +0.3 %, DFly2 c8 +0.5 % · Conf medium (tile audit) · Effort M
- Change: `core/engine_bringup.cpp:1735` floors levels at 256 rows; M: "256 tokens use 185.82 MiB". Round every transient's rows up to the largest row tile of any kernel that writes it (`rad_bufplan.cpp:150-215`).

#### P-13 Prefill router on bf16 WMMA with f32 accumulation (prefill rows only)
- Area kernel+plugin · ORD on prefill rows (near-tie picks can flip; decode / verify rows keep today's kernel): needs the user's OK like 16.4c · Gain prefill +1.5-2 % · Conf medium · Effort M · Deps alternative to K-52 (which is BI and smaller)
- Change: `rad_glm_tp.h:119-124`, `rad_block_glm_router.h:49-54`. Re-check the `all_gather` split afterwards (with WMMA the 256-column split + 2 MB gather may no longer pay).

#### P-14 Dead switch `RADIANCE_HY4_FUSE_QPROJ`
- Area plugin/docs · n/a · Effort S
- Change: `hyv4flash_common.h:311` documents it (default on), `tests/hyv4flash_arch_test.cpp:120` sets it, nothing reads it. Wire it to K-02 or delete.

#### P-15 `mtp_couple_propose` passes its operand pointers in a by-value struct: verify replay rebinding
- **Status 2026-11-22: CLOSED, no bug** (notes/p15.md). Tape replay rebinds no operand of any kernel; the pass key mixes every device pointer of the batch, so a changed pointer is a different tape. Seed/base/card live in the draws plane and are staged every step. Card A/B tapes on vs `--profile-ops`: proposals bit-identical for all 48 requests, no status -1 warning.
- Area spechub (correctness risk showing as acceptance) · Effort S to check · Conf low
- Change / check: `plugin/spechub/spechub.hip:583-588` (`RadMtpCoupleProposeCall` holding `idx`, `val`, `card` = `batch->seq_ids` raw, `draw`, `state`) against the verify kernel's own comment (`:976-978`: "a struct on the host would freeze the addresses it was built with"). The draft pass is a recorded tape (key 9). If a raw operand (e.g. `seq_ids` in the staging ring) moves between plays, `card` is stale -> status -1 -> the row keeps the argmax proposal (still exact, lower sampled acceptance).
- Why: M: first-draft acceptance 0.53-0.64; a silent fallback on some plays would cost tokens a step, the largest lever of the DFly2 path.
- Verify: count `status == 1` states in replayed vs first-recorded draft passes; if they differ: separate pointer parameters.

#### P-16 Recommended recipe: `draft-depth-load-from: 1` is not in `work/u3/coupled03.yaml`
- Area config · PL · Gain thinking text +9 % at one client (docs/ARCHITECTURES.md TP2 recipe 1b), code +-0 · Conf high (measured there) · Effort S
- Why: M: this note's `cp-st` server log prints "a step drafting for 5 sequence(s) or more" (the rule is not active below five sequences with that yaml).

### 3.4 DFly2 drafter

Measured basis (§2.3): draft pass 4.0-4.6 ms busy at one sequence (303 dispatches): head 5 x 301 us = 1.51 ms (37 %), IQ8R linears 46 x 22.4 us = 1.03 ms (25 %), 46 quantisers x 5.0 us, 38 norms x 5.5 us, 21 `copy2d16` x 5.7 us, 10 standalone reductions x 9.0 us, 16 `binary` x 4.6, 5 `propose` x 13.9. At eight sequences 6.3 ms (298 dispatches): the same small launches cost 11.5-12.3 us each (46 quantisers 0.57 ms, 38 norms 0.46 ms, 21 copies 0.24 ms, 10 reductions x 46.5 us = 0.47 ms). **In the draft pass a small launch costs 5 us (one sequence) to 12 us (eight), not the trunk's 0.3-0.5 us**: its launches do not overlap their predecessors the way the trunk's do, so launch removal pays here.

#### D-01 Draft layers: use the trunk's fused producers and shared codes (plugin only, every op exists)
- Area drafter · BI · Gain 0 / 0 / 1-1.5 % (c1), 1.5-2 % (c8) / 0 · Conf high (launches), medium (us each) · Effort M
- Change: `hyv4flash_dfly.h:502-548` (`linear`: no `lead` / `feed` / `fed_codes`), `:933-1008`. Per draft layer: kv_a and gate fed by q_a's codes (-2 quantisers); `rmsnorm_had_quant_i8g` for q_a_norm writing q_b's codes (-1); `mla_row_norm_rope_store` for latent norm + key rope + `kv_store` (-2); `mla_expand_gate_had_quant_i8g` for expand + `gate_mul` + o_proj quantiser (-2); `gated_had_quant_i8g` for `silu_mul` + down's quantiser (-1); `add_rmsnorm` twice (-2); attn_norm / ffn_norm writing codes (-2). ~-12 a layer, ~-60 of 303.
- Why: M: 60 launches x 5 us = 0.3 ms of a 4.1 ms draft pass at one sequence, x 12 us = 0.7 ms of 6.3 ms at eight.
- Verify: dfly parity harness (taps vs `ref/dfly_ref.py`), greedy + coupled e2e identical, fusion oracle rows (green for the trunk), `mtp_rrs_test`.

#### D-02 Chain round: 24 dispatches, ~9 avoidable
- Area drafter · BI · Gain 0 / 0 / 1 % / 0 · Conf medium · Effort S-M
- Change: `hyv4flash_dfly.h:1015-1055`: one `iota` [depth, n] before the chain (M: `iota_kernel` 6 x 5.6 us); the 5 `hnorm(z[k+1])` + 5 `gather_rows` as one M = 5n launch each; `correct_up` fed by `correct_gate`'s codes (same `ccat`); `mtp_couple_propose` writes `tokens[:,k]` and `prev` itself (the two `cast`s `:1053-1054`); a two-operand `silu_mul` / `gated_had_quant_i8g` instead of the two `cast`s into `cgu` (`:1027-1028`, `:448-450`); stage A: `embed_lookup` straight into `x`, anchor embeddings gathered once (`:907-920, 1019-1020`). ~-50 a pass.
- Why: M: 50 x 5 us = 0.25 ms (c1); `copy2d16` 21 x 5.7 us, `gather_rows` 8 x 7.2 us, `binary` 16 x 4.6 us measured.
- Verify: taps `dfly.cand_*`; `mtp_rrs_test` chi-square.

#### D-03 Fused reduction twins for the draft layers (T6, concrete)
- Area drafter+kernel · BI (E6) · Gain 0 / 0 / 0.3 % (c1), 0.7 % (c8) / 0 · Conf medium · Effort M · Deps D-01 (`add_rmsnorm`)
- Change: `hyv4flash_dfly.h:993-998, 1006-1008`: `all_reduce exact -> rmsnorm -> add` as `ar_add_rmsnorm` built from the junction's AR device functions; coded / staged write at 48 rows.
- Why: M: `r4d_ar_oneshot_2rank_exact` 10 x 9.0 us (c1), 10 x 46.5 us = 0.47 ms (c8; the trunk's fused junction is 27.6 us at the same height including its norms).
- Verify: emu bytes vs chain; TP2 parity.

#### D-04 `row_topk` reads f32 logits (no `cast`), stage 2 by the last stage-1 block
- Area kernel · BI if the rounding equals `cast_f32_bf16` (RNE / NaN) · Gain DFly2 0.2-0.4 % · Conf medium · Effort S-M
- Change: `hyv4flash_dfly.h:761-764, 1035`; `r4d_rowtopk_bf16.hip:148-176, 217-258`. M: `narrow_bf16` 5 x 3.4 us (19 us at c8), stage1 5 x 9.1, stage2 5 x 3.0, merge 5 x 2.9.

#### D-05 Correction chain: precompute the hidden half for all five rounds; shard gate / up / down
- Area drafter (TP2) · DRAFT (K split / reduction order moves proposal bits; the coupled verify is exact for any Q) · Gain DFly2 0.7-1.3 % · Conf medium · Effort M
- Change: `hyv4flash_dfly.h:731-733` (correct_gate / up / down `RAD_SHARD_NONE`: 47.2 MB a round replicated, 236 MB a pass a rank), `:1025-1031`. (a) `W[:, :E] @ hnorm(z[k+1])` for all rounds in one M = 5n GEMM pair: 160 instead of 236 MB; (b) shard gate / up by rows and down by columns + one exact reduction of [n, 3072] a round.
- Why: M: draft IQ8R linears 1.03 ms a pass, ~37 % of their bytes are the chain.
- Verify: `mtp_rrs_test` chi-square; acceptance by depth unchanged within scatter; stamped draft pass.
- Done (b): merged (`a899bed2`, `83f8b819`) — gate and up sharded by rows, down by columns, one exact `[n, E]` reduction a round before the residual add: 47.2 -> 23.6 MB a round a rank, 236 -> 118 MB a pass. iks-A, TP2, u3, `coupled03.yaml` c1,c8, five runs an arm (stamps on three of the pairs). The chain's int8 GEMM bucket 21.4-21.5 -> 17.4-17.6 us a launch, 985-991 -> 802-810 us for the same 46 launches (-18 %); the exact all_reduce 10 -> 15 launches, 93-95 -> 114-120 us (+21 us a pass for the sums); the c1 decode pass 4159/4178/4145 -> 3971/4064/3994 us; the c8 decode pass 18348/18504/18429 -> 18130/18043/18642 us; the gap between passes 196-198 us in both arms and the 47.3-47.9 ms verify passes unchanged, and the c1 pass's dispatch count 303 -> 308: no pass and no gap slower. Acceptance by depth 0.83-0.85 / 1.85-1.95 / 2.72-2.93 / 3.76-3.91 -> 0.87-0.95 / 1.76-1.96 / 2.55-2.89 / 3.64-3.73, unchanged within scatter, and `mtp_rrs_test`'s chi-square passes. e2e c1 128.8-129.7 -> 120.3-131.5 tok/s (median 129.2 -> 131.0, mean -1.6 %) with 23.69-23.87 -> 23.35-23.74 ms a step; c8 379.6-403.1 -> 348.0-400.5 (median 393.8 -> 377.3, mean -4.3 %), the after arm bimodal at 3.24 or 2.94 tokens a step where the before arm is always 3.20 (the adaptive depth's choice) and two of its runs at 61-65 ms a step against 53-57 elsewhere though their passes and gaps match the best run's; Mann-Whitney over the five pairs, U=15 at c1 and U=13 at c8 (p 0.22, 0.31), so the e2e difference is inside the lane's noise while every measured card component is equal or faster.
- (a) (the chain's gate and up as one GEMM pair of M = 5n) is not expressible in the framework yet: `decl_view` / `decl_w_n` declare at world 1 and `rad_weight_shard_span` only narrows a dimension that is already sharded, so a K sub-span of a stored -- let alone quantised -- weight cannot be declared; it wants the layout/LinearFP8 work first and stays open.

#### D-06 Short block under load (L5c, concrete)
- Area drafter · PL (verify unchanged) · Gain DFly2 c8 +1.5 % · Conf high · Effort S-M
- Change: no `rad_drafter_short_block` (`hyv4flash_dfly.h:853-865`); `query` refuses `M != n * block` (`:886-890`), rounds fixed at `depth` (`:1015`). Pattern `qwen35_fp8.cpp:328, 522`.
- Why: M: at eight sequences 2137 drafts made, 1764 positions reached, 1381 accepted in 108 steps (2.0 verified drafts a sequence-step of 5 drafted); two rounds less = 2 x (head 329 us + chain ~0.3 ms) of a 63 ms step.

#### D-07 Stop the chain on the card (L5b) or draft only the depth the rule will take
- Area drafter+core · PL · Gain DFly2 c1 1-2 % · Conf medium · Effort M · Deps D-06 covers the load case
- Change: the depth rule cannot take draft j + 1 once its statistic fails at j, yet rounds j + 2..5 run their head pass (301 us) and correction. Device flag a round + skip forms of `logits_gemm` and the three correction GEMMs; or (cheaper) a per-step `rounds` operand set from the previous step's depth distribution.
- Why: M: depth shares at one sequence [1: 0.07, 2: 0.27, 3: 0.20, 4: 0.15, 5: 0.31] (mean 3.4 of 5): 1.6 rounds x ~0.5 ms = 0.8 ms of a 22.4 ms step are drafts the rule never verifies. levers-general L5b put this at 1-2 % "after L1": this run confirms the upper end.
- Verify: tokens a step and acceptance unchanged; step time.

#### D-08 Draft pass: the five heads are 37 % of the pass and at the byte wall
- Area drafter · lossless options exhausted (K-20 chained M <= 2 form, K-21 top-8 epilogue); everything else is format (F-05) or the rejected shortlist (CLOSED Z-12) · listed so nobody re-derives it.

#### D-09 Tokens are streamed only after the adaptive-depth walk (sampled DFly2)
- Area core/engine · PL · Gain latency: every token 4-6 ms earlier, TTFT -4 ms; rate 0 · Conf medium · Effort S-M
- Change: `core/engine.cpp:1629` (`issue_ahead` -> `block_depth_decide` `:1265` waits for the drafter's laws) runs before `collect_step` / `commit` (`:1631, :1689`). When the decide will wait, commit and stream first (wait on `tok_end_`), then decide.
- Why: M: "host waits 424 (31954.5 us each, 424 ended blocked)"; throughput side is CLOSED (levers-general §1.5), the delivery order is not.
- Verify: client-side inter-token timing; greedy/sampled tokens identical.

#### D-10 Verify-height costs that are not the row's bytes (c8 DFly2 verify pass, ~40 rows, 56.5 ms)
- Area kernel · see K-08 (routed gate_up 652 us a layer = 47 % of the pass; gate_up / down time ratio 2.27 against 1.6 at one row), K-16 (`mla_sink_rows` 48.8 us a layer = 2.0 ms), K-09 / K-10 and the x-staging of `gemm_nt_f32` at 17-63 rows (33.3 us a layer = 1.4 ms; `r4d_glm.hip:46-47, 125`: 8 two-byte x loads per k step -> stage x transposed in LDS, BI, M), K-18 (`gather_kernel` unchained form above 32 rows), junction 27.6 us. Listed as one pointer item: **the c8 verify is where kernels leave their decode-class fast paths**; a `rad-kbench` sweep of every trunk row at M = 24 / 40 / 48 is the cheapest way to find the rest (T-05).

#### D-11 Link-aware verify price (X5 item, open): the load-aware rule prices a row at its calibrated ms, misses included only on average
- Area core/sched · PL · Gain DFly2 c8 +2-5 % (est.), 0 elsewhere · Conf low-medium · Effort M
- Change: `core/sched/draft_depth.h:306/639`: feed the row price from the last steps' `stream_bytes` per routed row (M: 111 missed units a step at c8 DFly2 = 7.3 ms of link in a ~49 ms step, 15 %).
- Verify: `work/levers/run.sh`-style c4/c8 pairs over three seed sets.

### 3.5 Scheduler (batching, chunking, interleave, the arrival stall)

Measured basis: §2.6 (stall probe of this note) and the BetterBench workload phase (`work/sweep/bb-auto.log`: worst stall 695-812 ms, p99 713-792 ms with 2-4 background streams whatever the arrival's depth; arrival TTFT 16K under load 9.5-10.3 s against 3.36 s cold = 2.8-3.1x; background decode 24-36 tok/s a stream in the window against 48-70 outside).

#### S-01 Remainder steps while decoders ride: every 2048-token interval is two steps, not one
- Area scheduler (config first) · PL · Gain **M (§2.6): arrival TTFT under load -27 % at one decoder (3.25 -> 2.37 s), -14 % at four (2.83 -> 2.43 s), lone prefill unchanged (1.99 -> 2.00 s)**; the 0.45-0.58 s stall itself does not change · Conf high (step log + A/B; single runs: repeat over three arrivals) · Effort S (a) / S-M (b)
- Change: `core/sched/scheduler.cpp:1305-1318` (quantum alignment gives a newcomer's chunk `2048 - ceil16(D)` rows, D = the decoders' row tokens), `:1250` (next step is cut to the interval boundary: a 16-48-row remainder step carrying the checkpoint), `:1474` (`busy-prefill-chunk auto` returns the whole budget), comment `:1490-1495`. (a) config: `checkpoint-interval: 1984` with `max-num-batched-tokens: 2048` (chunks <= 1984 rows end on a boundary with 64 rows of decode headroom); (b) FP-3 `--decode-headroom` (dartree `15434c6b`, 9 conflict files incl. `engine_bringup.cpp` arena sizing vs lending).
- Why: M (step log): with interval 2048 a 5.7K arrival beside one decoder runs `MIXED 2033 · 24 decode · MIXED 17 · 24 decode · MIXED 2033 · 18 · MIXED 17 · 24 · MIXED 1601 · 19 · MIXED 12`; with 1984: `MIXED 1985 · 15 · MIXED 1985 · 17 · MIXED 1729 · 21 · MIXED 11`. The remainder step itself is only a 40-65 ms gap, but it arms its own interleave run of ~24 decode-only steps (S-04), which is the TTFT cost.
- Verify: `RADIANCE_LOG_STEPS=1` with 1 and 4 decoders and a 6-7K arrival: steps per interval 2 -> 1; `probe.py stall1,stall4` (this note's `ck1984` / `ck2048` runs); `sched_test` interval expectations; BetterBench workload phase.
- Done (a): merged, config only (`bbca35dc`, `431c72e8`): iks-A, arrival TTFT 2.628 -> 2.205 s at one decoder, 2.750 -> 2.405 s at four, decode 13.3 / 18.3 ms, lone prefill unchanged (commit `0c9447d7`).
- Closed (b), without code (`notes/backlog/closed/S-01b.md`): the cherry-pick did what the item asks at 2048 rows — beside four decoders `MIXED 2036 | MIXED 20 | MIXED 2036 | MIXED 20 | MIXED 1780` -> `MIXED 2052 | MIXED 2052 | MIXED 1780`, arrival TTFT 2.621 -> 2.229 s at one decoder and 2.756 -> 2.426 s at four on iks-A (TP2, u3, three runs an arm), the chunks and the greedy text byte-identical — but (a) had already taken that step out of the served schedule: at `checkpoint-interval: 1984` the flag lands on the same 1984-row chunks (local lane, TP2, u3, three runs an arm: three `MIXED` steps an arrival and the four greedy sha256 identical in every run of both arms, arrival TTFT 2.424 / 2.552 s with the headroom against 2.539 / 2.569 s without, inside the arms' own 0.19 s spread) and it only grows the arena, 1.39 -> 1.43 GiB a rank (`step_tok` 2112, the bring-up's own +43.34 MiB), with `auto` as the default. An A/B against S-01 has nothing faster in it, so per the merge bar the code is reverted (`a74adc13`, `3c3dd6c5`, `8a82fa0d` stay in the branch's history) and the harness it needed (`f5f7f9fa`, `996510ce`) stays. The interval is not S-01(a)'s looser half as an earlier read said — that compared this lane's 1984 arms with iks-A's 2048 arms — so dropping it is a config-only card with its own same-lane A/B; the one shipped config still cut by its interval is `deploy/compose/tp1.radiance.yaml` (4096/4096, four sequences, three draft tokens: a 16-row remainder step), which wants the same config fix, not a 43.34 MiB arena level.

#### S-02 Layer-sliced prefill chunk: decoders ride a slice of the chunk's layers
- Area engine+plugin+scheduler · PL (each chunk row's per-layer math unchanged; rows of a GEMM are the same set inside the slice) · Gain stall 0.5-0.8 s -> ~0.15-0.2 s at m = 4 with prefill rate and TTFT unchanged (est.) · Conf medium (design only) · Effort L · Deps S-01 (no remainder steps), M-03
- Change: plan a chunk as m sub-steps; sub-step j runs the decode rows through all 42 layers and the chunk rows through layers `[L_j, L_{j+1})`; the chunk's residual / iHC streams (2048 x 4 x 3072 bf16 = 50 MiB a rank) and DFly2 taps persist between sub-steps. Mixed-step composition `scheduler.cpp:2650-2700`, `RadBatch::n_seq_decode`, hyv4flash step `hyv4flash.cpp:1106-1213`, pass key `core/runtime/ctx.cpp:1008`. The stager stages per slice.
- Why: the stall equals the chunk step because decoders ride the whole of it; shrinking the chunk does not help (a 256-row step still reads every expert of every layer: M (`work/sweep`): chunk 256 stalls 444 ms at 7.4x TTFT). Slicing keeps "each layer's experts read once per 2048 rows". This is the only stall lever found that leaves the TTFT/stall frontier; FP-10 (second queue) needs a second all-reduce instance, per-stream counters, a second arena region, stager races and CU masking (`ctx.h:757-799`, `r4d_ar_seq.h`): keep FP-10 as research only.
- Verify: greedy continuation byte-identical against unsliced; `sched_test` shapes; stall probe; BetterBench workload.
- Closed, without code (`notes/backlog/closed/S-02-exec.md`): the lever was built and ran (a sliced prefill chunk
  byte-identical to the unsliced one — the chunk's last read and the 63 decode reads after it, all four dumped buffers —
  `sched_test` shapes, the hook's A/B), and on the lane it halves the bulk of the decoder's wait -- 0.43-0.56 s ->
  0.23-0.28 s at m = 4 -- but it does not reach the item's number: the arrival's TTFT rises 2.070 -> 2.608 s (+26 %),
  a ~0.43-0.49 s gap survives in every arm including unsliced (the arrival's boundary, not the chunk), `auto` plans
  m = 2 rather than the 6-7 its 150 ms target asks for, and slicing doubles the expert copy (123 layers / 12.16 GiB
  unsliced against 246 / 24.31 at m = 4) because the decode rows' trunk pass and the chunk's slices evict each other
  from the ring -- so the stager fallbacks of design §5 come first. The code is reverted (`af25a666`); the harness it
  needed (`scripts/backlog/probe.py`'s stage sampler) stays.

#### S-03 Issue the mixed step ahead (admission and mid-prompt chunks at issue-ahead time)
- Area scheduler/engine · PL · Gain arrival TTFT -8..-30 ms (one decode step), host turnaround of every mixed step overlapped · Conf medium · Effort M
- Change: `scheduler.cpp:2959-3042` (`step_next_locked` refuses while `waiting_` is non-empty `:2962`, held chunk `:2965`, `!produces_token` `:2979`); `engine.cpp:1254-1290`; `plan_next` phase is hard-coded DECODE (`:3022`).
- Verify: `sched_test` issue-ahead cases; `engine_steps_ahead_total`; greedy identical.

#### S-04 Interleave EMA mixes every non-decode step
- Area scheduler · PL · Gain steadier decoder cadence under arrivals (no TTFT gain) · Conf high · Effort S
- Change: `scheduler.cpp:3413` (`step_cost_locked(phase != DECODE, wall)`), `:1542-1547`: feed `ema_chunk_s_` only from steps whose prefill rows reach the stager floor; today remainder, tail and small resumed-delta steps pull it down, so fewer decode-only steps follow the big chunks than `prefill-interleave auto:1` intends.

#### S-05 `--mixed-step-target-ms`: clamp at the stager floor, then put it in the sweep
- Area scheduler · PL · Gain a frontier move (stall vs TTFT), no floor change · Conf high (code) · Effort S
- Change: `scheduler.cpp:1500-1530` (`busy_cap`: the decoder reserve is applied after the stager floor `:1503`, `fair` any multiple of Q `:1525`), `:3453-3470`, `config.cpp:80`: the learned cap can push chunks below 1024 rows, where every routed layer's pooled experts are read synchronously.

#### S-06 Suffix / n-gram index built O(prompt) at the first decode under the scheduler lock
- Area scheduler · BI (same tables) · Gain first-token latency -12 ms at 64K, -20 ms at 100K per turn; `mu_` not held that long · Conf high · Effort S
- Change: `core/sched/suffix.h:159-168` (`index_to`), called from `suffix_propose` in `step()` (`scheduler.cpp:2361-2363`) and `commit()` (`:3294-3296`); index after each committed chunk (`index_to(seq, n_computed)`) while the card runs the next one.
- Verify: `sched_test` suffix cases; `t_on` phase line at 64K.

#### S-07 Issue-ahead refusal reasons as counters
- Area scheduler/diagnostics · Effort S
- Change: `scheduler.cpp:2959-2986`: per-reason counters (waiting, held, mid-prompt, grammar, suffix_hot, ends_every, KV) in `/stats`; today only `suffix_held_ahead`. M: `engine_steps_ahead_total` 89 of 91 (c1), 97 of 108 (c8 DFly2): 10 % of c8 steps are not issued ahead and nobody knows why.

#### S-08 `publish()` rebuilds the request table every busy step under `mu_`; sink wakes inside `mu_`
- Area scheduler · PL · Gain a few us a step · Conf high · Effort S
- Change: `scheduler.cpp:4013-4027` (idle publishes throttled to 20 ms, busy ones not), `:1025-1075`; `:3086, 3172, 3232` -> `core/server/sink.cpp:97` (collect sinks, notify after unlock).
- Done: the item's lane (`9f989727`, `057be7ff`): `local` (TP2, `nospec.yaml`, 210 W read back), three runs an arm and
  four interleaved, `greedy` byte-identical (the same four sha256 in every run), `c8` 301.5 -> 301.7 tok/s over the
  blocks and 309.1 -> 312.1 interleaved (255 steps either way, 25.34 -> 25.32 ms a step), `c1` 71.5 / 75.0 both,
  `stall1` cold arrival 2.261 / 2.262 / 2.236 -> 2.315 / 2.429 / 2.316 s over the blocks and 2.296 / 3.047 ->
  2.195 / 2.160 interleaved, inside the lane's 0.9 s spread. The item's op, the same throwaway `RAD_INFO` in both
  binaries (`commit()`'s `mu_` from before the lock to after the unlock, `publish()`'s around its `mu_` section):
  `commit()` 26.7 -> 6.5 us a step at `c8` and 6.8 -> 1.8 at `c1`, so the step's scheduler critical section 32 -> 11 us
  at `c8` (4-8x shorter; a second pair, under heavier host load, read 143 -> 17 us, so the ratio holds and the absolute
  numbers do not); `publish()` 5.1 -> 5.6 us at `c8`, a wash — at `c8`'s 25 ms steps the 20 ms throttle never fires —
  and 4.33 -> 3.32 at `c1`'s 13.6 ms ones, where it rebuilds the table on 52 % of the steps instead of all of them.
  21 us of a 25 ms step is 0.08 %, which is why the served rates do not move: the item's gain is measured on the op.

#### S-09 Prefix lookup / publish hash the whole context per attempt under `mu_`
- Area scheduler/memory · BI · Gain 0.1-0.5 ms a turn at 64K (est.) · Conf medium · Effort S
- Change: `core/mem/prefix.cpp:392` (`chain_all` per lookup incl. every retry of a waiting `admit_one`), `scheduler.cpp:723`: cache the chain hashes in `SchedReq`.

#### S-10 Recall the whole prompt's KV loan once at admission
- Area core/mem+sched · PL · Gain ~15 stream drains less per 60K prompt (5-50 ms), no lent-slot churn between chunks · Conf medium · Effort S
- Change: `core/mem/kv.cpp:927-957` (`grow_for` recalls `shortfall + buffer` per call), `core/place/mover.cpp:1732-1745` (each recall syncs the mover stream and the compute stream on both ranks and walks 20,992 units). Start the arena recall at `add` too (`engine.cpp:1234` -> `arena_step` `:2470-2505`), so the drain overlaps the in-flight decode step.
- Verify: a `reclaim` counter per prompt; TTFT of a 60K prompt after 2 s of decode.
- Closed, without code (`notes/backlog/closed/S-10.md`): the recalls are real and go away — a 60,526-token arrival beside one decoder paid 13 / 14 / 13 of them per 60K prompt (local lane, `coupled03.yaml`, `greedy,arr60`, three reps an arm) and paid 0 / 0 / 0 with the whole prompt's loan recalled at admission, the four `greedy` texts byte-identical — but avoiding them means refusing the 100 ms arbiter the prompt's own blocks (`spare_tokens`), so the expert slab goes without ~843 MiB (60,502 tokens at 14.25 KiB) for the whole prefill instead of between two recalls: arrival TTFT 19.075 -> 21.039 s (medians; 18.815-19.892 vs 20.947-21.623), a prefill chunk 682 -> 750 ms, the shared decoder's step 23.8 -> 25.8 ms, against 65-650 ms the item's 5-50 ms a drain would have saved. `fec5dede` reverts it; the `arr60` phase (`scripts/backlog/probe.py`) stays, the `reclaim` counter does not (core code, and §2 closes a card without merging any). The arena half of the Change needed nothing: `arena_step`'s 2 s floor already pays it once per prefill wave.

### 3.6 Engine step, tape, device queue

#### C-01 Small mixed / prefill passes are always issued live: measure, then record them by bucket
- Area core/runtime · BI (tapes audited every 1024 plays) · Gain tail / remainder / resumed-delta steps: ? (host-bound if live issue of ~1500 ops a rank approaches the card time) · Conf low until measured · Effort S (measure) / M
- Change: `core/runtime/ctx.cpp:1378` (stager-armed passes live), `:1032` (`ckpt_cut`), `:1424` (non-decode keys recorded only on second sighting; the key carries `n_tok`, the context bucket and staging pointers, so prefill shapes rarely repeat); hyv4flash pass key `hyv4flash.cpp:1249-1259` returns 0 for prefill / mixed. `RADIANCE_DEBUG_ISSUE=1` prints own / launch us per op (`core/runtime/issue.cpp:532`). If small steps are host-bound: key prefill rows by bucket (16 / 64 / 256) and record below the stager floor on first sight.
- Why: M (§2.6): a 73-row step takes 78-88 ms, 256 rows 223 ms, 717 rows 255 ms; most of it is unstaged expert reads (M-03), the live-issue share is unknown: after M-03 the host share is what is left of a resumed turn's step.

#### C-02 First doorbell after 64 packets on a step that starts an idle queue
- Area core/device · PL · Gain 20-50 us earlier card start on steps not issued ahead · Conf high · Effort S
- Change: `core/device/aql.cpp:1571` (`kPlayRing = 64`), `PlayRun::ring` `:1597`: ring after the first 4-8 packets when read index == write index.
- Verify: `RADIANCE_AQL_STAMP` first-dispatch time on tail / mixed steps.

#### C-03 Rank barrier: spin-then-futex, pin the step and rank threads
- Area core/engine · PL · Gain 10-100 us per wake on steps not issued ahead; 8 barrier rounds a DFly2 step · Conf low · Effort S
- Change: `core/runtime/rank_barrier.cpp:6-21` (mutex + condvar); no `setaffinity` anywhere in core; pin to one CCX of the EPYC 7402P.
- Why: M: host `ranks` phase 0.45-0.63 ms (c1), 1.23 ms (c8) overlapped with the card (§2.6 `nsv` / `cpv` phase lines).

#### C-04 One patch launch per pass for residency changes
- Area core/runtime · BI · Gain c8: 0.1-0.25 ms a step (est.; count first) · Conf low-medium · Effort S-M
- Change: `core/runtime/issue.cpp:415-447` (`wtab_patch` per op / table); with `publish_landed` before the pass every change is known at the pass start. Keep the per-op path for staged passes.
- Verify: `RADIANCE_DEBUG_ISSUE` patch counts at c8 (M: 12-16 promotions a step no-spec, 32-37 with DFly2).

#### C-05 Checkpoint save / restore as one copy list instead of ~35 blit kernels with fences
- Area core/mem+device · BI · Gain 0.2-0.5 ms a rank per checkpoint (one a 2048-token chunk, one a resume) on the compute stream · Conf medium · Effort S · Deps K-47 (same blit)
- Change: `core/mem/kv.cpp:1785-1840` (per window page one pitched blit + 2 state groups, each with a system-scope fence `aql.cpp:2974-2995`); use `rad_memcpy_ranges_async` (`aql.cpp:2997-`) or one SDMA list.

#### C-06 Penalty history uploaded at full `max_hist` width
- Area core/sample · PL · Gain ~15 us a step when penalties / DRY are used · Effort S
- Change: `core/sample/sampler.cpp:1241-1247`, staging `:966-1000`.

#### C-07 Empty DFly2 context pass still costs a barrier round
- Area core (additive export) · PL · Gain DFly2: 2 barrier rounds a step (~5-20 us) · Conf medium · Effort S-M
- Change: `hyv4flash.cpp:1113-1119` (`draft_pass < 0` issues nothing); kernels/dfly2.md §3.5 point 1 / §9 item 5: a core export that drops the empty pass (`engine_draft.cpp:1038-1041`).

### 3.7 Server, tokenizer, chat template

Checked and already tuned (not items): SSE build (cached chunk head, SIMD `json_quote_append`), lock-free SPSC sink with wake only while asleep, O(1) incremental detokeniser, per-segment tokenizer LRU, ids-only readback (n_rows x 4 B), no per-token logging. Host cost per token per stream est. 10-30 us of HTTP-thread CPU.

#### H-01 Request prep time is not logged (parse / render / encode / grammar / submit)
- Area server · Effort S · Conf high
- Change: `core/server/server.cpp` `record()` line `req ... done`: add the five phase times. Precondition for H-02 and for any TTFT accounting of agentic turns (64K histories).

#### H-02 Chat template rendered twice per request
- Area text/server · BI if the cached suffix is verified · Gain TTFT: half the render time of a long history (unmeasured: H-01 first) · Conf medium · Effort M
- Change: `vendor/llama/common/chat.cpp:964-979` (`common_chat_template_generation_prompt_impl` does one extra full render), from `core/text/chat.cpp:456`; plus `alias_reasoning` copy (`chat.cpp:214`). Cache the generation-prompt suffix per (template, kwargs, last role, thinking flag), checked on first use by doing both renders.

#### H-03 Per-drain syscalls in the chunked loop
- Area server · PL · Gain 1-3 us per token per stream · Effort S
- Change: `httplib.cpp:4384` (`wait_writable` + `is_peer_alive`), `:7951` (`write` polls again), `core/server/server.cpp:1279` (`peer_closed()` one more probe): drop our own probe; skip the poll right after `wait_writable` (vendored patch; mind `vendor_sums`).

#### H-04 HTTP worker count and pinning
- Area server · PL · Gain ? (measure with H-01) · Effort S
- Change: `core/server/http.cpp:398` (one worker per core, 48 here, none pinned): keep HTTP threads off the CCX that runs the step and rank threads (C-03).

### 3.8 Memory, expert placement, streaming, prefill stager

Unit arithmetic: a u3 unit is 1,499,904 B a rank; **100 MiB = 70 units = 0.33 points of residency, 1 GiB = 716 units = 3.4 points** (20,992 units a rank). A missed unit costs 66 us of link (27.6 GB/s). M (this note, §2.1-2.3): hit 0.984-0.988 at c1 (8-11 missed units a step), 0.992 at c8 no-spec (34), 0.985 at c8 DFly2 (111 a step = 7.3 ms of a ~49 ms step), 0.983 at c1 DFly2 (25-32 = 1.7-2.1 ms of 22 ms).

#### M-01 Mirror pool: size the mover's pinned pool to the host-pool budget (reopens T5)
- Area core/place · PL · Gain c8 +1-3 %, DFly2 c8 +2-4 % by itself (d2h copies and donor moves gone); precondition of M-02 · Conf high (code), medium (gain) · Effort M
- Change: `core/place/mover.cpp:135-152` (pool slots only for the plan's HostPinned units), `:203-205` (reserve `n_pool x stride`), `:1517-1521` (a budgeted promotion gives up its pool slot); `core/mem/pools.cpp:175, 227` allocates the whole `host_pool_total_mib`. So ~8 GiB a rank of pinned host memory is allocated, locked and idle, and `host-pool-mib: 61000` changed nothing (`notes/tp2t1.md` T5 "no-op (pool slots unchanged: 7848)": T5 was never really tested). Give each class `min(all its units, room / stride)` pool slots, write resident units' bytes into their pool slot at load, keep the slot on promotion; a demotion of a mirrored unit is an address change.
- Why: M (longctx exit report): 72 GiB d2h, 17,897 "starved of a pool slot", 92 % readmits; M (levers-tp2 §1.3): d2h 14-19 MB a step a rank at c8, 34-40 MB with DFly2.
- Verify: exit report d2h / starved; `/stats experts`; `ctest place_*`; greedy identity (`work/xs1/xs.py`, `cmp.py`).

#### M-02 Admission for many streams (X2): admit on first miss into copy-free slots, replay first
- Area core/place · PL · Gain bounded by the replay oracle: -18 % (no spec) / -29 % (DFly2) of misses = c8 +1-2 %, DFly2 c8 +2-4 % · Conf medium-low · Effort M · Deps M-01; `bin/rad-heat-replay` variants on the `work/xs1` traces before code
- Change: `core/place/heat.h` (`HeatParams`), `HeatEngine::step` / `observe`, `Mover::balance` / `fill` / `drop`: 2Q probation area of lent slots, move budget following the pass width (64 swaps a step and class today).
- Verify: replay misses per dispatch vs lat 2; then `/stats` in the levers-tp2 §1.3 table.

#### M-03 Stage steps of 129-1024 tokens; fix the 1024 / 1025 off-by-one
- Area core/engine+sched · PL (stager copies are byte-identical) · Gain **TTFT -100..-170 ms per prompt whose last chunk is 129-1024 rows (-5..-8 % on a 6.7K prompt) and per resumed turn (a 256-token step is 223 ms today)**; (a) alone: a held-back 1024-row chunk is staged again · Conf high (mechanism), medium-high (gain: §2.4 / §2.6 size the small steps; the staged cost of such a step is extrapolated) · Effort S (a) / M (b)
- Change: `core/engine_bringup.cpp:5086-5104` (`min_rows = top_level + 1` = 1025), `core/engine.cpp:2495-2503` (ring only at level 0 or the span level), `core/sched/scheduler.cpp:1449, 1484` (busy-chunk floor `stage_rows / Q * Q` = 1024). (a) round the floor up to 1040 or arm at `n_tok >= top_level`; (b) give the 1024 level a ring right above its buffers (the span level's `arena_low_off` mechanism: 706 + 1 + 523 MiB fits) and arm prefill steps >= ~128 rows.
- Why: M (§2.4): the 537-row last chunk of a 6.7K prompt costs 338 ms of op time, 214 ms of it routed GEMMs at 3.0 / 2.2 ms a call (a staged 2048-row chunk: 2.1 / 1.1 ms a call for 4x the rows); M (§2.6): a lone 256-token prompt takes 223 ms, 717 tokens 255 ms, 73 tokens 78-88 ms. At 128 rows x 16 of 512 an expert is routed with p = 98 %, so such a step reads the whole pooled set (3.7-4 GiB a rank; 9 GiB with long contexts live) synchronously in place. Cases: every prompt's last chunk below 1025 rows, every resume turn (M `work/churn`: 213-450 tokens re-prefilled, TTFT 0.69-0.72 s), remainder steps (S-01).
- Verify: `/stats experts stage_layers`; `work/churn/resume.py` TTFT; texts identical stager off / on.

#### M-04 Stager ring sizing and heavy layers (overflow is read in place)
- Area core/mem+place · PL · Gain prefill +5-15 % with several long sessions live, 0 at one short session · Conf medium · Effort S-M
- Change: `core/mem/vram_budget.cpp:262-287` (ring = 2 x ceil(f x layer_max), one global f = 523 MiB here), `core/place/stager.cpp:227` (a layer above ring/2 is not staged ahead; what does not fit is `stage_overflow`, `mover.cpp:636-642`). (1) stage a heavy layer partially ahead; (2) size the ring from the per-layer pooled maximum at the KV floor, bounded ~1 GiB (arena top, lent at decode); (3) hottest-first order so the overflow is the coldest.
- Why: M (`work/longctx/res/nospec.json`): overflow 0 at one session, 49 -> 409 units a chunk at 4 x 32K -> 60K, 819-950 units (1.2-1.4 GiB) a chunk at 8 sessions; in place, u3 gate_up is re-read per 64-row entry (K-44).
- Verify: `/stats experts stage_overflow` per chunk in the longctx sweep; prefill tok/s.

#### M-05 Heat `min_gain` 3 at high residency
- Area core/place · PL · Gain tok/s within noise; promotions -40..-70 %, link h2d -55 % (fewer copies beside junctions and the shared arm; less quarantine) · Conf high (measured, `work/churn`: base 1135-1183 promotions / 1K tokens at c8 vs 407; tok/s 301.4-302.0 vs 297.6 / 306.1; DFly2 1067-1125 vs 572, 374-397 vs 387-392) · Effort S
- Change: `core/place/heat.h:61` (default 1.5, tuned on Flash-Next TP1 at 40 % residency): residency-dependent default (1.5 below 75 % resident, 3 above); env `RADIANCE_HEAT_MIN_GAIN` exists. `min-gain 6` loses 5 % at c1; decay 0.97 does nothing (do not retry).
- Verify: `work/churn/churn.py` A/B three runs; greedy identity.

#### M-06 Paced copies ride behind the layer's second junction, sized in bytes
- Area core/place+runtime · PL · Gain c8 no-spec +1-2 % on top of X1b · Conf medium · Effort M
- Change: `core/place/mover.cpp:1988-2001` (`pace_mark` rides the dispatch after the routed down = the shared arm; xs-1: `gemm_i8_kpar` +0.8 ms, `gated_quant` +0.5 ms, junctions +0.4 ms at c8). Mark behind the post-FFN junction (attention stretch, no PCIe) and cap a share at stretch_us x 27 GB/s. With DFly2 pacing is off (held copies wait a whole step): let draft passes carry marks too.
- Verify: `RADIANCE_AQL_STAMP=24`, `work/xs1/stlay.py`, `kdiff.py`.

#### M-07 Hold the link gate around decode junctions while copies are in flight
- Area core/runtime+device · PL · Gain <= 0.4 ms a step at c8 · Conf medium-low · Effort M · Deps after M-06
- Change: `core/runtime/issue.cpp:1154` (`hold = staging_ && p.collective`: only staged passes hold the gate), `core/device/aql.cpp:3732` (yield only if `ever_held`); `RAD_LINK_ENDED` path `issue.cpp:1253`.

#### M-08 Early publish on the device (X3)
- Area core/runtime · PL · Gain bounded by replay "lat 0": -5..-8 % of misses (c8 +0.5-1 %, DFly2 c8 +1-2 %) · Conf medium-low · Effort M-L · Deps M-01
- Change: `core/runtime/xprefetch.cpp` (`xp_step` generalised): a copy that lands mid-pass serves the layers still ahead.

#### M-09 Queue scratch charged from kernel metadata, not 5 MiB x 32 CU
- Area core/device+bringup · PL · Gain up to 160 MiB = 112 units a card (0.5 points): c8 +0.3-0.5 %, DFly2 c8 +0.5-1 % · Conf medium (whether a Hy4 kernel has a private segment is unverified) · Effort S
- Change: `core/device/dispatch.cpp:943-946` (flat `n_cu x 5 MiB` = 160 MiB); `aql.h:45` / `aql_code.cpp:302` already parse `private_segment_fixed_size`. The settle line (`engine_bringup.cpp:5428`) is read before any dispatch, which explains most of "settled at 338.5 MiB free against 131 asked".
- Verify: sysfs `mem_info_vram_used` after c1 / c8 / prefill / DFly2 warm-up against the settle line.

#### M-10 Soak the remaining settle slack; `--gpu-headroom-mib 64`
- Area core/engine + config · PL · Gain 30-200 MiB (20-140 units) + 45 units · Conf low-medium · Effort S-M / S
- Change: `engine.cpp:564-630` (`xflex_tick` grows the slab only with a fleet broker): after warm-up commit `free - headroom - 32 MiB` once as a revocable loan. Headroom 128 -> 64 with the settle line watched over a long run (`engine_bringup.cpp:5441-5470`).

#### M-11 Lend the sliding-window pages of unused sequence slots
- Area core/mem · PL · Gain 73-129 units at 1-4 live sequences (0.35-0.6 points) · Conf medium · Effort M
- Change: `kv_latent_sliding` 400 blocks = 8 slots x 50 = 211 MiB never lent (`core/mem/kv.h:472-474`).

#### M-12 Arena bytes: `edn` (192 MiB at the full plan), score plane, multi-interval liveness
- pointer: K-19 (drops `edn`), P-10, P-11, P-12.

#### M-13 CWSR off on a dedicated box (operator setting)
- Area ops · numerically lossless · Gain ~68 MiB = 48 units a card · Conf low-medium · Effort S (kernel parameter + reboot; the user's call)
- Change: 2 compute queues x 35.36 MiB measured (context-save area with `cwsr_enable 1`); loses mid-wave preemption.

#### M-14 Code objects: measure what part of "already held 341 MiB" is code, then trim
- Area core/device · PL · Gain ? (tens of MiB) · Conf low · Effort S to measure
- Change: libr4d carries 110.6 MB of gfx1201 code in 121 modules, each loaded whole at its first kernel (`aql.cpp:493-511`); sum `im.size` of loaded modules at the settle line.

#### M-15 KV loan reaches only above the highest held block: measure fragmentation, then compact
- Area core/mem · PL · Gain up to 1.5 GiB (1,100 units) in the measured case · Conf low-medium · Effort S (measure) / M
- Change: `core/mem/kv.h:487-491`; M (longctx 4 x 60K mid-run): 5.43 GiB used of 6.98 GiB not lent. Add `/stats kv held_top`, `buffer_blocks`; relocate the highest cached-only entries downward (d2d + prefix remap) above a 256 MiB gap.

#### M-16 Per-layer residency floor / exit-report fixes
- Area core/place · PL · Effort S
- Change: residents a layer 255-447 (sd 69): replay a per-layer minimum (heavy pooled layers are the ones that overflow the ring, M-04). `mover.cpp:2151-2153` prints `stats_.prefetched` as "staged" (the longctx log says "0 staged" while `/stats` shows 2.0 TB staged): print `staged_units`, `staged_bytes`, `stage_overflow`.

#### M-17 Promote a unit that is in the stager ring VRAM->VRAM instead of h2d again
- Area core/place · PL · Gain prefill-phase h2d 64 MB a step a rank -> d2d · Conf medium-low · Effort M
- Change: `engine.cpp:1041` (no swaps behind a staged pass, on purpose); when `heat.step` would promote a unit currently in the ring, copy it from its ring place before it is lapped.

### 3.9 KV cache, prefix cache, checkpoints, resume

#### X-01 Host KV tier on (`prefix-cache-mib`): idle conversations never give VRAM back to the experts today
- Area config (+docs) · PL · Gain 640 units (3 points) per idle 65K session; removes whole re-prefills when the VRAM prefix pool evicts (M longctx 8 x 60K: one request got cache 0 -> TTFT 32.9 s) · Conf high (mechanism), medium (gain, workload dependent) · Effort S
- Change: without the tier a finished conversation's blocks stay held in VRAM until evicted or `prefix-cache-ttl` 3600 s (`core/rad_core.h:1041`, `core/mem/kvtier.h:1-30`); `spare_tokens` (`kv.cpp:959-972`) counts only blocks above the highest held one. Size it from M-01's idle pinned memory or the host pool's unused tail (replicated groups are stored once: `rad_kv_group_replicated` is declared).
- Why: M (longctx): decode 190 -> 148 tok/s at 4 sessions from 16K to 60K (hit 0.971 -> 0.929, VRAM units 18017 -> 14403); 8 sessions: 391K cached tokens hold 5.65 GiB.
- Verify: `work/longctx/multi.sh`, `work/churn/resume.py`; `/stats` VRAM units after idle.

#### X-02 In-step tail snapshot for Hy4 (FP-2) **plus** the non-span relaxation in the scheduler
- Area plugin+core/sched · BI (snapshot bytes equal the tail step's) · Gain per prompt: no extra <= 15-token tail step (M: `step 100 prefill n_tok=9` on a lone 6.7K prompt, `MIXED 12` under load; a 40-65 ms gap for decoders, and with M-03 open an unstaged small step) and no I/4 gap: M (`work/churn` rsB64): 31 % of resumes re-prefill +227 tokens (mean 290 vs ~218 possible) · Conf high (code) · Effort M
- Change: plugin `ckpt_snap_ok` (`hyv4flash.cpp:1238` sets only `shape_probe_ok`): cut the indexer tail ring and Engram CONV at `ckpt_cut` (pattern `rad_qsa.h:527-539`, `rad_block_ple.h:545-560`). Scheduler: the in-step path needs `geo_.span > 1 && span_now_ && cfg_.prefill_span > 1` (`scheduler.cpp:1276-1277`), so the plugin flag alone changes nothing on the recommended recipe (no span); `tail_checkpoint` needs >= max(Q, I/4) = 512 tokens past an interval (`:1403`). Allow `inside` for any prefill alone in its step. FP-2's 5-13 % TTFT estimate is too high: the tail step is at most 15 rows.
- Verify: `RADIANCE_DEBUG_CKPT_HASH=1` equal with / without; `tools/hy4flash_rollback_gate.py --prefix-cache` TP1 / TP2; `resume.py` cached counts.

#### X-03 Rolling decode snapshot (every 256 output tokens)
- Area core/sched+engine · BI · Gain agentic / code turns: next-turn re-prefill <= 255 + new tokens instead of reply + new (a 1K-token reply = 0.25-0.3 s of prefill today) · Conf medium (the template must re-render the reply token-identically, also with thinking on) · Effort M · Deps C-05
- Change: `scheduler.cpp:1717` ("decode writes no snapshot"). Save into the sequence's running-turn slot when the decode position crosses a multiple of 256 on the quantum (17.4 MiB d2h a rank, ~0.7 ms of link in the idle direction).
- Verify: `resume.py` with long replies; hashes.

#### X-04 Persistent prefix store (FP-4)
- Area core/mem+sched · BI (checksummed) · Gain after a restart: 64K history ~17 s -> ~0.7 s; capacity for many projects · Conf medium-high (Flash-Next precedent) · Effort M (5 conflict files of 26: `config.cpp`, `sched/metrics.h`, `scheduler.cpp/.h`, `GUIDE.md`)
- Change: cherry-pick `-x` dartree `a8400909`, `9e883a39`, `46c77e35`, `e0976a0a`, `5cb4c2ad`, `bab7361b`; 14.6 KiB a token; cap + write budget; the key includes the binary (deployed builds only).
- Verify: gpu-mini TP2: restored hit == in-memory hit (text + top-5 logprobs), across Engram CONV.

#### X-05 KV streaming for the 12 sparse layers (research)
- Area core/mem+plugin · PL · Gain at 4 x 60K ~2.5 GiB = 1,800 units back to the experts, at the price of resolve copies on the same link · Conf low · Effort L · Deps X-01 first
- Change: docs/KV-STREAMING.md; hyv4flash does not set `kv_stream_ok`; `mla_attn_sel_sink` reads a selection (13,824 B a token of sparse latent).

#### X-07 Rotated int8 KV cache for Hy4 (`kv-cache-dtype: i8`, as Qwen3.8-Flash-Next does it)
- Area kernel+plugin+core/mem · LOSSY (the user's decision, 2026-11-22: "int8 with rotation like Flash-Next"; keptn's KL against the bf16 reference is the acceptance gate) · Gain VRAM: the KV a token about halves (bf16 latent 13,824 B a token of sparse latent today; one 1M-token context = 14.46 GiB a card at TP2, which leaves the experts 9.33 GiB and fails to start with the TP2 recipe's host pool — measured, `work/kv1m/start.log`). With int8 roughly 7-8 GiB for 1M tokens: the target is a guaranteed 1M-token context without crippling expert residency; at today's context sizes more VRAM for the experts (longctx: hit 0.887 at 8 x 48K with DFly2). Decode reads fewer KV bytes; prefill and decode pay the quantise / dequantise. · Conf medium (mechanism exists for Flash-Next and GLM's fp8 latent), gain on Hy4 not measured · Effort L · Deps a design first (kernel plan), then kernels; X-01 / X-05 / checkpoints must handle i8 groups
- Today: the plugin refuses it at declare — `hyv4flash: the latent and indexer caches are bf16; run with --kv-cache-dtype bf16` (`work/kv1m/start-i8.log`). The core already sizes and moves i8 KV groups (pool, prefix host tier, streaming); the Hy4 readers and writers do not.
- Change: (1) design: copy Flash-Next's rotated int8 KV (Hadamard/WHT rotation before quantising, per-row or per-group scales; where the inverse rotation is folded: into q / the absorbed W_UK and W_UV so attention reads int8 directly, or per tile) and GLM-5.3's fp8 MLA latent cache (`rad_block_glm_mla.h`) as the MLA precedent; decide per group: sparse latent (RAD_KV_FULL), sliding latent (RAD_KV_WINDOW, window 512), indexer keys (`kv_index_keys`, dense-read — may need bf16 for top-k selection quality), indexer tail, Engram tokens, the rope part (64 of 256 K dims), the DFly2 drafter's caches. (2) store: an int8 + scale form of `mla_row_norm_rope_store` (quantise in the store, rotation fused). (3) readers: int8 tile loads (dequant in the load, or int8 WMMA with scales) in `r4d_mla_sink.hip` (`mla_attn_paged_sink`, `mla_attn_sel_sink`, `mla_sink_rows`, `merge_sink`), `r4d_attn_kv_sink_h256.hip` if it reads the cache, `r4d_mla_kv_bf16.hip` (`mla_project_kv` / `mla_expand_gate` where they read the latent), the indexer kernels `r4d_idx_*.hip`; libref (`ref_hy4.cpp`) and libavx (`avx_hy4.cpp`) forms for every new op. (4) plugin: accept `kv-cache-dtype: i8` in `hyv4flash.cpp` (`latent_group` ~660 and the indexer / Engram groups), declare the i8 forms, keep verify == decode bytes (stable law, kernels/mla_sink_attn.md) and TP (the latent cache is replicated on both ranks). (5) window snapshots / checkpoints, prefix host tier and streaming with i8 groups.
- Why: the user wants a guaranteed 1M-token context in the configs (`max-model-len: 1048576`, `flex: {kv: {min-tokens: 1048576}}`); in bf16 that does not fit beside the experts at TP2 (measured above). Earlier note: kernels/perf.md L9 (FP8 / int8 latent cache, 656 B a token, up to -40 % of selection prefill at 8 heads).
- Verify: emu / `r4d_selftest` / kbench cases for every new op (int8 form vs bf16 within a stated bound, libref vs libr4d); host parity (ref/ harness, mini presets) bf16 vs i8 within a bound; DFly2 chi-square unchanged (the law is the target's for any cache); greedy text drift and logprob stats on u3; the 1M-token config starts and its startup line shows the KV and expert split; decode tok/s at 1/8 clients and prefill vs bf16; **keptn's KL run** on the u3 container (mean / p99 KL, top-1) decides whether it ships as default or opt-in.

#### X-06 Global suffix cache + index take-over (FP-5) · Area sched · Gain +0-2 % sampled · Effort M · only together with X-04 (same commit series). DARTree offline sizing (FP-6) · Effort S · go / no-go for an L package expected to be marginal (0-3 % at c1, <= 0 at c >= 4).

### 3.10 Bring-up and startup time (every measurement and restart pays it)

#### B-01 Pin and fill the four Engram tables in parallel, overlapped with the container pass
- Area core/bringup · BI · Gain start 66 s -> 35-40 s · Conf medium · Effort S-M
- Change: `core/engine_bringup.cpp:380-445`, `core/format/table_copy.cpp:36-62`: 4 x 14.75 GiB one after the other, 10.3-10.5 s each (1.43 GB/s against 4.2 GB/s from disk): prefault with 4 threads (`aql.cpp:4589-4601`), 15 sequential 1 GiB locks (`:4650-4680`), 16-reader fill. M: this note's servers took 67 s from start to `/health` (§2).
- Verify: phase timers (B-03).

#### B-02 `ngram-placement: ram=/dev/shm/...` to skip the 60 GiB read on restart (config) · Effort S · Gain a further 15-25 s · mount the tmpfs with `huge=always`.
#### B-03 Startup phase timestamps in the log (pool pin, each Engram table, container pass, arena levels, settle) · Effort S.
#### B-04 Keep the container in page cache when RAM allows (`core/format/radfile.cpp:144-147` drops every range behind the read; conflicts with X-01 / M-01 for the same RAM: pick one) · Effort S.

### 3.11 Measurement and tooling (needed to verify the above)

#### T-01 Stamps for issued (prefill / mixed) passes
- `RADIANCE_AQL_STAMP` times recorded passes only (`aql.cpp:2558-2569`); prefill needs `profile-ops`, which adds a sync per op (+7-25 % wall) and hides overlap. Give live-issued passes the same per-dispatch completion signal. Effort S-M. Without it every prefill ticket is verified through a distorting profile.
#### T-02 `scripts/` versions of `work/levers/{probe,agg,layer}.py` and this note's stall probe (`work/backlog/probe.py` phases `stall1` / `stall4` / `tailN`), plus a per-layer table in `/stats`. Effort S.
#### T-03 Long-context decode stamps (60K session, c1 and c4): K-25's `qsa_select` and `idx_score` terms have only `--perfidx` numbers. One short GPU session.
#### T-04 `rad-kbench` cases at Hy4 TP2 shapes that are missing: sliding 32 heads M 1 chained (`merge_sink`, expand), M = 24 / 40 / 48 for every trunk row (D-10), sel with real `qsa_select` tables, `gemm_nt_q` (2048, 1024, 3072). Effort S.
#### T-05 A fused-vs-chain kbench oracle run in CI for the container (`rad-kbench -m`): G-3 found the gpu-mini junction fixtures stale (34 cases). Effort S.

## 4. Not ours to decide: lossy or format-touching (the friend's decisions; options with their measured price)

We do not change formats (DECISIONS 2026-11-22). Each row is an option to offer, with the bytes and time it would move at TP2 a rank. Times are this note's stamps (§2) unless marked.

| ID | option | bytes today | time today (M) | estimate | note |
|---|---|---|---|---|---|
| F-01 | Routed experts smaller: a container whose routed experts are 7 % smaller (9 % with DFly2 loaded) is fully resident at TP2 with idle KV | 29.32 GiB a rank | misses: 0.6-0.7 ms a c1 step, 2.2 ms at c8, 7.3 ms a c8 DFly2 step; prefill pooled reads 12-18 % of a chunk | link term to zero at decode; prefill +10-15 % | levers-tp2 §4 |
| F-02 | Routed down w4nl64 (0.58 MiB a unit) -> u3 trellis (0.44) | 2.4 GiB a pass at 8 rows | 4.56 ms at c8, 11.8 ms in a c8 verify | -1.1 ms / -2.8 ms (4.5 %); +10 % units resident | levers-general §3 |
| F-03 | Routed gate_up as a 4-bit L16 form that takes the packed prefill map, or keep u3 and wait for K-44 | | prefill gate_up | K-44 makes this unnecessary | |
| F-04 | `attn_output` int8 -> IQR u5 (the one large int8 trunk weight) | 12.2 / 24.4 MiB a layer a rank (sparse / sliding) | 22 / 42.7 us a layer: ~1.55 ms a pass | -0.5 ms (3.5 % of a c1 pass) | bandwidth-bound: scales with bytes |
| F-05 | Head int8 -> u5 | 186 MB a rank | 0.30 ms a trunk pass, **1.51 ms a draft pass** | -0.1 ms decode, -0.55 ms a DFly2 step (2.5 %) | |
| F-06 | Engram key / value projections bf16 -> int8 | 37.7 + 9.4 MB x 4 layers | 0.38 ms a pass (`vit_gemm_wa` 4 x 68 us + `vit_gemm` 4 x 27 us) | -0.19 ms (1.4 %) | |
| F-07 | `kv_a` bf16 -> int8 / IQR (`gm-i8kva` recipe exists, wp13-9) | 3.4 MiB a layer, replicated | 42 x 6.1-8.1 us = 0.3 ms a pass | -0.12 ms (0.9 %) | the 576 rows are no whole 128-row tile (HY4FLASH-QUANT.md) |
| F-08 | `kv_b` bf16 -> int8 (DECISIONS U2 keeps bf16; WP17 on request) | 7 / 3.5 MiB a layer a rank | absorb + expand part of the attention core (~1.4 ms a pass) | -0.2 ms (1.5 %) | |
| F-09 | Trunk IQR u5 -> u4 | ~0.68 GiB a pass | 2.56 ms (251 launches; ~0.8 ms of it floor) | -0.25 ms (2 %) | |
| F-10 | DFly2 linears IQ8R -> u5 / u4 | ~0.55 GiB a draft pass | 1.03 ms | -0.35 ms a DFly2 step (1.5 %) | |
| F-11 | iHC `fn` f32 -> bf16 | 384 KiB x 84 a pass | junction 1.28 ms a pass, 9-10 % of a prefill chunk | decode -0.1 ms; prefill arithmetic -30 % with K-46 | changes gates |
| F-12 | Latent KV cache FP8 / int8, 656 B a token (perf.md L9) | 13.8 KiB a token over 12 sparse layers | long context: residency (M longctx: -22 % decode at 4 x 60K) and sel staging | -43 % KV: +1,000-1,800 units at long context | lossy, own design note first |
| F-13 | Sliding-window KV in FP8 | 211 MiB fixed | | +75 units | |
| F-14 | 16.4c single-bf16 P for `stable 0` (prefill) rows | | sink attention WMMAs 25 -> 17 a wave a chunk | sel / paged -25..-32 % after K-40..42 | prefill rows move inside the E4 bound; tell him before KL runs |
| F-15 | fp8 / int8 indexer (`RADIANCE_HY4_INDEXER`, built) | | `idx_score` key bytes /2 or /4 | prefill at long context; with K-43 mostly moot | moves near-tie selections |
| F-16 | Sink super-chunks at window 0 / sel (`kShiftFull 7 -> 6`, `r4d_mla_sink.hip:72`) | | sparse sink 9 -> 5-6 us x 12 layers | 0.04 ms | law change (moves bits once) |
| F-17 | f32 instead of f64 in `idx_pool_gated`'s norm / trig | | 12 x 9.4 us | -0.05 ms | within bound, not byte-identical |
| F-18 | Fold `fusion_gate` into the fusion weight; concatenate `correct_gate|up`, the drafter's q_b split, one `index_query` weight | | D-02's casts | removes 2-3 launches a round | container / manifest changes (DECISIONS U4) |
| F-19 | Prefill router on bf16 WMMA (P-13) | | router GEMM at prefill | +1.5-2 % prefill | f32 sum order on prefill rows: the user's / friend's OK |
| F-20 | `RADIANCE_MOE_VERIFY_BUDGET` (fewer experts for verify rows; dartree) | | routed GEMMs are 68 % of a c8 verify pass (M) | large, changes the target law | lossy: excluded, listed only |
| F-21 | `tp-wire: wht6` | | | none at decode | not usable: norms and calibration read the exact sum |

## 5. Closed items (do not re-propose without new evidence)

| ID | item | closed by | reason |
|---|---|---|---|
| Z-01 | Chained launches for every Hy4 row (`chain_pays`), shared / fused quantisers, `router_fuse`, Engram read-ahead (hy4-adopt) | levers-general §1.7, merged 46d17a0a | DONE: +17 % c1, +12 % c4 / c8, prefill +6-9 % |
| Z-02 | G-3 fusions a (swiglu + quant), b (junction writes codes), c (expand + gate + quant) | notes/g3.md, 55a794dc | DONE: -161 launches for +1.2 % c1, +0.8 % c8. **A removed chained trunk launch is worth 0.3-0.5 us, not 3.2 us**: pure launch-count fusions in the trunk are low value (the draft pass is different: §3.4) |
| Z-03 | Arena lending + prefill stager (`shape_probe_ok`) | notes/tp2t1.md T1 | DONE: prefill +15-25 %, c8 +5 % / +12 % |
| Z-04 | `draft-depth-stop: 0.3`, load-aware rule from one sequence, exploration of empty bins | levers-general §1.6, notes/g1g5.md | DONE (recipe) |
| Z-05 | q_temp for the proposal law | notes/g1g5.md | swept {1.0, 0.9, 0.8, 0.6}: no winner, default 1.0 |
| Z-06 | The adaptive depth's host wait as a throughput lever | levers-general §1.5 | fixed 5 vs adaptive stop 0: equal within 0.3 ms (the delivery-order item D-09 is separate) |
| Z-07 | Sampler chain at T 0.9 vs greedy | levers-general §1.5 | 0.2-0.5 ms a step |
| Z-08 | Expert prefetch at TP2 (T3 / X4 router-ahead) | notes/tp2t3.md | NO-GO: precision 23-46 %, machinery alone -4..-6 %, copies -2..-5 % more |
| Z-09 | Two-piece pipelined prefill (T4) | notes/t4.md | hideable transport ~2.5 % < 3 % gate; ~50 issue sites need piece forms |
| Z-10 | `prefill-span` / `prefill-span-pass: step` / 4096 batched (T2) | notes/tp2t1.md, recipe (3) | only >= 16K prompts, decode -2..-5 %, tie-level changes |
| Z-11 | Faster junction wire, fewer reductions a layer, thread pinning for rank skew, re-sharding heads / indexer, whole experts a rank, pipeline parallelism over two cards, router on one rank, Engram rows through the peer | levers-tp2 §3 | each by measurement or arithmetic (junction p90 - p50 = 0.2 us; wire = one P2P round trip) |
| Z-12 | Shortlist head for the drafter | user decision | rejected |
| Z-13 | Host-CPU MoE for cold experts; copy-then-compute inside the layer; arming the stager at decode; second compute stream for a predictor; `expert-pin-top` / `expert-profile` | experts-streaming §6 | arithmetic / friend's measurement |
| Z-14 | X1a publish-before-pass | notes/xs-1.md | built, nearly inert (0-0.05 early publishes a step); kept |
| Z-15 | X1b pacing | notes/xs-1.md | built: c8 no-spec +2-3 %; off with a drafter (follow-ups M-06, M-07) |
| Z-16 | Heat decay / hysteresis / presence credit / moves 128 | notes/xs-0.md replay, `work/churn` | no gain (only `min_gain`: M-05) |
| Z-17 | Host mirror by `host-pool-mib: 61000` | notes/tp2t1.md T5 | measured no-op, **but the test did not exercise a mirror** (the mover sizes pool slots from the plan): reopened as M-01 |
| Z-18 | hc-fold, async checkpoint copy (`RADIANCE_Q4_HCFOLD`, `_CKPT_ASYNC`) | flashnext-port §2 #18 | measured 1 % / 20 % slower on Flash-Next |
| Z-19 | DARTree for DFly2 (implementation), `draft-coupling-temp*`, `draft-calib online`, bridge-files | flashnext-port §3 | defer: L effort for 0-3 % at c1; only FP-6's offline sizing is open (X-06) |
| Z-20 | Replicated KV groups, sync-audit / spec-audit fixes, hangfix (slot wait vs stale chain counter) | 16.0c, 16.0b, dc680ce3 | DONE |
| Z-21 | FP-8 "packed map check with RADIANCE_MOE_IQR_MAP_TILES" | this note | the env switches do not exist; answered from code: u3 never packs (-> K-44) |
| Z-22 | FP-1 "size the arrival stall" | this note §2.6 | measured here (small-step cost, stall, remainder steps) |
| Z-23 | FP-2 as "plugin flag only", -5..-13 % TTFT | this note (X-02) | the tail step is <= 15 rows and the scheduler gate needs span: re-scoped |
| Z-24 | Decode tables to LDS, weight re-streaming at M > 1, 2-byte loads on GEMM paths, in-place PCIe double reads at decode | audit A §2 | checked: computed trellis (no table), every weight kernel reads its weight once for all rows (except K-08), in-place reads run at the link's rate |
| Z-25 | Per-token host path (SSE, detokeniser, sink, tokenizer cache, logging) | audit D §5 | already O(1) and tuned; 10-30 us a token |
| Z-26 | `mla_attn_sel_sink` growing with context | audit B §9 | flat beyond 2048 keys; the growing terms are `idx_score`, `qsa_select`, the score plane (K-43, K-25, P-10) |
| Z-27 | Checkpoint slot count | `work/churn` rs64 vs rs256 | auto 64 = 256 for these shapes; 16 collapses |

## 6. Global ranking (gain x confidence / effort)

Score = expected gain on the user's case (sampled code, DFly2 coupled, one to four clients, agentic turns with prefix hits; then eight clients, no speculation, lone prefill) x confidence / effort. "Gain" columns: c1 no-spec / c8 no-spec / DFly2 step / prefill or TTFT. Config-only and S items with a measured basis come first by construction. Items inside one tier are ordered.

### Tier 1: do first (measured basis, S or S-M)

| # | ID | title | gain | conf | effort |
|---|---|---|---|---|---|
| 1 | S-01 | No remainder steps: `checkpoint-interval: 1984` (then FP-3 if 2048-row chunks matter) | **M: arrival TTFT under load -27 % (one decoder), -14 % (four); lone prefill +-0**; one short decoder stall less per 2048 prompt tokens | high | S (config) |
| 2 | P-16 | `draft-depth-load-from: 1` in the served yaml | thinking text +9 % at c1 (measured in the recipe), code +-0 | high | S (config) |
| 3 | M-03 | Stage 129-1024-token steps (+ the 1024 / 1025 off-by-one) | M: a 256-token prompt takes 223 ms, a 537-row last chunk 338 ms with 214 ms of unstaged routed reads: resumed turns and prompt tails -100..-170 ms, TTFT of a 6.7K prompt -5..-8 % | high / medium-high | S + M |
| 4 | X-01 | Host KV tier on (`prefix-cache-mib`) | 640 units per idle 65K session; long-context decode -22 % (M) largely recovered; no 33 s re-prefills | high / medium | S (config) |
| 5 | K-08 | `moe_iqr` run-aligned decode grid | up to 10-12 % of a c8 DFly2 step, ~2 % at c4 (M: gate_up 47 % of the verify, ratio anomaly) | medium-high | M |
| 6 | K-11 | Router weight read as stored bf16 | 1-1.5 % decode + 90 units a card | high | S-M |
| 7 | K-12 | Junction: warm `fn` during the peer wait | 1.5-2.5 % c1, 1 % c8 | medium | S |
| 8 | K-14 | `merge_sink` at 32 heads (19 us vs 6.3) | 2-3 % c1 | medium | S + S-M |
| 9 | P-01 | Shared expert beside the routed GEMMs | 1-2 % at every decode height | medium | M |
| 10 | K-44 | Packed 128-row map for u3 gate_up | prefill +1-2.3 %, more in unstaged steps | high / medium | S-M |
| 11 | K-16 | `stable 1` sink split cap 48-64 rows | 2-3 % of a c8 DFly2 step (M: 48.8 us a layer) | medium | S |
| 12 | D-01 | Drafter: fused producers and shared codes | DFly2 1-1.5 % (c1), 1.5-2 % (c8) | high / medium | M |
| 13 | K-47 | `cast` blit grid flattened | prefill +1-3 % (first chunk most) | high | S |
| 14 | K-45 | `moe_gather` wide kernel for f32 weights | prefill +1 % | high | S |
| 15 | M-05 | `min_gain` 3 at high residency | link h2d -55 %, tok/s +-0 (measured): enabler for M-06 / less copy interference | high | S |

### Tier 2: clear value, M effort or medium confidence

| # | ID | title | gain | conf | effort |
|---|---|---|---|---|---|
| 16 | K-40 + K-41 + K-42 | Sink attention at prefill (value tile, single key buffer, row tiles) | prefill +7-10 % (sink is 28.7 % of a chunk) | medium-high | M + S-M + M-L |
| 17 | K-46 | Junction arithmetic shares `fn` across a block's rows | prefill +2-2.8 % | high / medium | M |
| 18 | K-09 (+ K-10) | Router select chain; GEMM + select + quant fused | 2-3 % c1, 1-2 % c8 verify | medium | M |
| 19 | K-05 | Static weight loads before the chain wait | 1.5-4 % c1 (wide error bar) | medium-low | S-M |
| 20 | K-01, K-03, K-02, K-04 | Dual-weight launches over shared codes | together ~4 % c1, 2 % c8, 3 % DFly2 | medium | M (one entry) + S-M each |
| 21 | K-07 | Routed GEMM block head (descriptor list) | 1.5-2.5 % c1 | medium | M |
| 22 | M-01 | Mirror pool (reopened T5) | c8 +1-3 %, DFly2 c8 +2-4 %; enables M-02 | high / medium | M |
| 23 | D-07 + D-06 | Draft only the rounds the rule can take; short block under load | DFly2 c1 1-2 %, c8 1.5 % | medium / high | M / S-M |
| 24 | D-02 + D-03 + D-04 | Chain round launches, fused reductions, f32 top-k | DFly2 1.5-2 % (c1), 2-3 % (c8) | medium | S-M each |
| 25 | X-02 | In-step tail snapshot + non-span relaxation | resume turns: -1 step, -31 % of the cases re-prefilling +227 tokens | high | M |
| 26 | K-43 (+ P-10) | Row-tiled `idx_score` | prefill +1 % at 6K, +5 % at 47K | high | M |
| 27 | M-04 | Stager ring sizing / heavy layers | prefill +5-15 % with several long sessions | medium | S-M |
| 28 | M-09, M-10, P-12, M-11 | VRAM receipt: queue scratch, settle slack, 64-row level, window pages | 110 + 20-140 + 60-90 + 73-129 units (up to ~2 points of residency) | medium | S / S-M / M / M |
| 29 | D-10 | Verify-height kernel forms (x-staging of `gemm_nt_f32` at 17-63 rows, gather `inv`: K-18) | 2-3 % of a c8 DFly2 step | medium | M / S |
| 30 | B-01 (+ B-02, B-03) | Startup 66 s -> 35-40 s (-> 25-30 s) | every restart and measurement | medium | S-M |
| 31 | S-06, D-09, S-10 | First-token latency: suffix index during prefill, stream before the depth walk, one KV recall | -12..-20 ms at 64K, -4 ms, -5..-50 ms | high / medium | S each |
| 32 | X-03 | Rolling decode snapshot | agentic turns with long replies: re-prefill <= 255 + new tokens | medium | M |
| 33 | K-50, K-51, K-52, K-53, K-48, K-49, K-58 | Prefill kernels second wave | each +0.3-2 %, together ~+6 % | medium | S-M..L |
| 34 | P-03 | Prefill: last layer only on output rows | prefill +2 % | medium | M |
| 35 | X-04 | Persistent prefix store | restart: 64K history 17 s -> 0.7 s | medium-high | M |

### Tier 3: small, speculative or large

| # | ID | title | note |
|---|---|---|---|
| 36 | S-02 | Layer-sliced prefill chunk | the only lever found that takes the arrival stall off the TTFT / stall frontier (0.5-0.8 s -> ~0.15-0.2 s); L, design note first, after S-01 and M-03 |
| 37 | K-25 | `qsa_select` at long context | 1.8-3.7 ms a decode step past 32-64K by `--perfidx`; measure in a served 60K session first (T-03) |
| 38 | M-02, M-08 | Admission (X2), early publish (X3) | bounded by the replay at -18 % / -29 % and -5..-8 % of misses; after M-01, replay first |
| 39 | M-06, M-07 | Copy pacing behind the second junction; link gate in decode | c8 +1-2 %, <= 0.4 ms |
| 40 | K-06, K-13, K-15, K-18, K-19, K-20, K-21, K-22, K-23, K-24, K-26, K-27, K-28, K-29, K-30 | Decode kernel hygiene | 0.1-1 % each; batch by file (`r4d_gemm_iqr.hip`, `r4d_moe*.hip`, `r4d_mla_sink.hip`, quantisers) |
| 41 | P-02, P-04..P-09, P-11, P-14 | Plugin hygiene | launches, VRAM, a latent hazard (P-02), a dead switch |
| 42 | D-05, D-11, C-07 | Drafter chain bytes, link-aware price, empty context pass | 0.5-1.3 %, 2-5 % at c8 (uncertain), ~10 us |
| 43 | S-03, S-04, S-05, S-07..S-09, C-01..C-06, H-01..H-04 | Scheduler / engine / server hygiene | tens of us to tens of ms per turn; C-01 and H-01 are measurements that may promote items |
| 44 | K-54..K-57, K-59, K-17, M-12..M-17, X-05, X-06, B-04, T-01..T-05 | The rest | checks, research, tooling |
| - | P-15 | Coupled-propose struct operands under replay | a 30-minute check; if it fails it is a tier-1 fix (acceptance) |

### What the measured structure says about the ceiling

- One stream, no speculation (13.5 ms): 7.7 ms are weight bytes at or near the HBM rate (F-rows of §4 only), junction 1.3 ms (K-12 / K-13: -0.3..-0.5), attention core 1.4 ms (K-14, K-05: -0.3..-0.5), router 0.77 ms (K-09..K-11: -0.4), small trunk linears above their floor ~1.0 ms (K-01..K-06: -0.5), routed GEMM head latency 0.7 ms (K-07: -0.25), shared arm 0.6 ms (P-01: -0.3), quantisers / glue 1.5 ms (-0.2). Lossless total about **-2.3..-2.9 ms (17-21 %)**: 73 -> ~90 tok/s; the rest is formats.
- Eight streams, no speculation (25 ms): routed GEMMs 12.3 ms are bytes (2.2 us a unit) + 2.2 ms of misses; lossless: K-07 / K-08 -0.5, misses -1 (M-01, M-09.., X-01), the c1 list -2: about **-3.5 ms (14 %)**.
- DFly2 one stream (22.4 ms, 2.8-3.1 tokens): verify -2.5 (as c1, plus K-08 at 4-6 rows), draft pass 4.1 -> ~3.0 (D-01..D-04, D-07), tokens a step +3 % (P-16 on thinking text): about **+18-25 % tok/s**.
- DFly2 eight streams (~49 ms a step): K-08 -5..-8 ms, K-16 -1.5, D-10 -1.5, misses -3 (of 7.3), draft pass -1.5: about **-12..-15 ms (25-30 %)**: this is where the largest lossless decode gain is.
- Prefill (565 ms of op time a 2048-row chunk): sink attention -45..-55 ms, junction -14, routed (K-44, K-45, K-53, K-54) -25, small GEMMs / router / casts / Engram -25, last layer -12: about **-120..-130 ms (+27-30 %)**; and independent of the kernels, M-03 removes ~170 ms from every prompt whose last chunk is 129-1024 rows.
- Arrival stall (0.5-0.8 s a chunk): S-01 halves the number of stalls, M-03 shortens the small ones, kernels shorten the chunk by a quarter; only S-02 changes its order of magnitude.
