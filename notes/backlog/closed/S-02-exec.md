# S-02 (`--chunk-slices`, layer-sliced prefill chunk): CLOSED — the lever halves the bulk of the stall but keeps a 0.43-0.49 s tail, pays the arrival 26 % of its TTFT, and doubles the expert copy

Item as written (backlog S-02): plan a chunk as m layer-sliced sub-steps so the decoders ride a slice of the chunk's
layers; gain "stall 0.5-0.8 s -> ~0.15-0.2 s at m = 4 with prefill rate and TTFT unchanged"; verify: greedy continuation
byte-identical against unsliced, `sched_test` shapes, the stall probe, BetterBench. Design and its measurement plan:
`notes/backlog/design/S-02.md` (§5 the stager's cost, §10 the A/B this note runs, §12 S-02b's Verify).

Closed **without code**, as S-04, S-01b and S-10 were: the lever of §12 (S-02b) was built, run on the lane and
reverted (`af25a666`), because two of the merge bar's three clauses fail. This note is what stays.

## The number

One lane, arms back to back in one call: `iks-A` (TP2, `nospec.yaml`, host-pool 16000 MiB, port 8042), 2026-11-24,
`probe.py` phases `stall1,stall4,pf6,pf20,greedy`. `stall1` is one decoder and a 5.7K-token arrival, `stall4` four
decoders and the same arrival; a gap is the decoder's wait between two of its own steps (ms), and the arrival TTFT is
the one the lever is trying to protect.

| arm | `stall1` arrival TTFT | gaps, 1 decoder | gaps, 4 decoders | sliced steps / sub-steps | staged, `stall1` window |
|---|---|---|---|---|---|
| `chunk-slices: off` | 2.070 s | 34.7 39.1 40.2 40.7 40.7 **428.7 472.3 538.9** | 431.0-562.5 | 0 / 0 | 123 layers, 12.16 GiB |
| `auto` (planned m = 2) | 2.185 s | 36.3 40.6 41.1 59.4 **341.7 367.1 431.9 476.6** | 330.6-434.2, one 727.3 | 1 / 2 | 164 layers, 16.20 GiB |
| `chunk-slices: 4` | 2.608 s | 41.2 44.1 **241.5 249.8 265.6 271.4 429.8 473.7** | 230.1-282.6, tail 433.9-493.9 | 1 / 4 | 246 layers, 24.31 GiB |

The lever does what it was built to do, and the number is in the middle column: at m = 4 the decoder's bulk wait is a
quarter of the chunk, 0.43-0.56 s -> 0.23-0.28 s. It is not the item's number, for three reasons, and the third is the
one that matters:

1. **`auto` does not plan m = 4.** It plans 2 on both lanes (sub-steps 2 for one chunk). The m law targets 150 ms
   against a ~1 s chunk step, which asks for 6-7; `--chunk-slice-cost-pct 10` holds it at 2, so the default setting of
   the lever lands at 0.34-0.43 s, not 0.15-0.2 s. §10 named this: "If the rate falls and the stall does not, the m
   law's cost cap is wrong".
2. **What is left of the stall is not the chunk.** A ~0.43-0.49 s gap survives in every arm — 428.7 / 472.3 unsliced and
   429.8 / 473.7 at m = 4, the same two gaps to within 5 ms. Slicing the chunk cannot shorten a cost that is not the
   chunk's compute; it sits at the arrival's boundary and is its own item.
3. **Slicing doubles the expert copy** (§10's first check, and §5's premise). Over the same arrival window the unsliced
   arm stages 123 layers / 12.16 GiB and m = 4 stages 246 / 24.31 GiB (m = 2: 164 / 16.20) — two full trunk passes per
   arrival instead of one. The chunk's slices stage their own layers and the decode rows keep traversing the whole
   trunk between slices, so each evicts the other from the ring: "the group's first slice pays the unsliced chunk's
   copy and the later slices pay little" does not hold at 16000 MiB of host pool (the local lane at 39000 reads the
   same ratio, 123 / 12.68 GiB against 246 / 25.11). §10's answer to that failure is §5's fallbacks — stage per slice
   and keep the decode rows' experts resident across the slice boundaries — and "no other number on this page applies"
   until they are built.

The e2e side, same lane: `pf6` 1.838 / 1.851 / 1.845 s and `pf20` 6.834 / 6.893 / 6.865 s (off / auto / 4). A
prefill-only step is never sliced (`only_a_mixed_step_with_one_chunk_is_sliced`), so those rows say the lever costs
nothing when there is no decoder to protect, and they are why the item's "prefill rate unchanged" holds. The arrival
TTFT does not: 2.070 -> 2.608 s at m = 4 (+26 %, and the extra copy above is where it goes) and 2.185 s at `auto`.

Byte identity is two checks, and only the second one can fail. The four `greedy` continuations (256 / 256 / 102 / 64
tokens) are the same four sha256 in all three arms here, and the same four again in the three local arms —
`d6da4fb3…`, `0699709c…`, `e7a23c5d…`, `466e53a2…`. That comparison alone proves nothing about a slice: a `greedy`
request is a prefill-only step, a prefill-only step is never sliced (`only_a_mixed_step_with_one_chunk_is_sliced`), so
those four hashes hold whether or not a sliced row is right. The sliced step is therefore checked on its own, with the
hook (`--chunk-slices-prefill 4` against the same config with it off) and the buffer probe, on one lane (`iks-A`, TP2,
`nospec.yaml`, `greedy` phase, two runs back to back). The bar is exact — the same bytes and not a tolerance, because
nothing a slice changes reorders an accumulation — and the probe's hash is FNV-1a over every byte of the row it reads
(`final_hidden` and `block_in` 3072 bf16 wide, `streams` 4096, the f32 `hc_post` 4; a decode step's read is one row,
the chunk's is its first row):

    greedy  4 prompt(s) identical (text sha256, length, generated count)
    chunk   block_in      n=3072 hash b4ec5d2fd4c3c7cc identical (step 714 vs 744; 715 prefill read(s) unsliced, 745 sliced)
    chunk   final_hidden  n=3072 hash dc50c2993d4ea6b5 identical (step 714 vs 744; 715 prefill read(s) unsliced, 745 sliced)
    chunk   hc_post       n=4    hash 197394f29ecbcc47 identical (step 714 vs 744; 715 prefill read(s) unsliced, 745 sliced)
    chunk   streams       n=4096 hash 3711f5d4a38c03cf identical (step 714 vs 744; 715 prefill read(s) unsliced, 745 sliced)
    decode  block_in         63 read(s) identical (steps 715..777 vs 745..807)
    decode  final_hidden     63 read(s) identical (steps 715..777 vs 745..807)
    decode  hc_post          63 read(s) identical (steps 715..777 vs 745..807)
    decode  streams          63 read(s) identical (steps 715..777 vs 745..807)

The chunk's last prefill read is step 714 unsliced and step 744 through four sub-steps, and the 63 decode reads that
follow it are the arrival's own — one row each, so those are whole reads. So the persistent set the sub-steps hand on
comes out of four slices with the same bytes an unsliced chunk leaves, every layer of it in that row, and the arrival's
64 tokens are the same 64: the four `greedy` sha256 above are this check's end-to-end half, since every one of those
tokens was read off the state the chunk produced. The lever is numerically exact, and what is wrong with it is its
cost.

The check's code goes with the lever — `scripts/backlog/s02_slice.py` and `--chunk-slices-prefill` are deleted in the
revert — so it is replayable from the revert's parent, `64658dc0`, whose lane config still carries `RAD_LOG`:

    RAD_LOG=trace RAD_CHUNK_SLICES_PREFILL=0 RADIANCE_DUMP_BUF=final_hidden,streams,block_in,hc_post \
      scripts/backlog/run.sh s02-whole nospec.yaml greedy
    RAD_LOG=trace RAD_CHUNK_SLICES_PREFILL=4 RADIANCE_DUMP_BUF=final_hidden,streams,block_in,hc_post \
      scripts/backlog/run.sh s02-sliced nospec.yaml greedy
    python3 scripts/backlog/s02_slice.py runs/iks-A/s02-whole runs/iks-A/s02-sliced

The local lane (`local`, TP2, host-pool 39000, port 8041, same phases, back to back) is reported because it was run, not
used for the verdict: the box was at load average 85 with other cards' builds, so `stall1` reads 565.4 / 494.8 / 469.6 ms
worst for off / auto / 4 — the same shape as iks — while its unsliced `stall4` arm queued the arrival for 24.5 s and
showed 3.46-4.54 s decoder gaps, which is the host, not the lever.

## Why it does not carry

The item's gain is one sentence — "0.5-0.8 s -> ~0.15-0.2 s at m = 4 with prefill rate and TTFT unchanged" — and two of
its three clauses fail: the stall lands at 0.23-0.28 s in the bulk with a 0.47 s tail, and the arrival's TTFT rises a
quarter. What the lever does deliver (byte identity, a quarter-length bulk wait, no prefill cost) is real, but it is
paid for with a copy that doubles, and the copy is the part §5 said would be free.

## What stays and what comes next

- The S-02b code is **reverted** (`af25a666`, off this branch in one commit: 105 insertions, 1269 deletions) — the
  sliced pass in `arch/hyv4flash` and the persistent set it resumes from, `core/sched`'s prefill-slices hook and the
  calibration-window guard, the engine bring-up that asks for the declares, the config key, the docs rows, and the
  tests that pin them. CARDS.md §2 and this item's issue close a card that misses its number without code, and an
  approved card merges, so keeping the lever in the branch would land exactly the code the bar keeps out. What a serve
  runs today is therefore unchanged by this item: at the base commit no plugin reports `slice_layers_ok`, so
  `--chunk-slices` plans m = 0 whether or not the flag is given.
- The last round's two review findings were part of the lever and go with it: the prefill-only refusal test needs
  `--chunk-slices-prefill`, which no longer exists, and the calibration-window guard (`--calib-dir`/`--calib-layers`
  keeps a chunk whole at declare and at plan instead of failing to bring the model up) guards declares and plans that
  the base commit never makes.
- The next work is §5's fallbacks, in this order: (a) the stager keeps the decode rows' experts resident across the
  slice boundaries, so the trunk is traversed once per group rather than once per side; (b) the m law's cost cap,
  which plans 2 where the target asks for 6-7; (c) the ~0.43-0.49 s boundary gap, which is not the chunk. The byte
  check above says all three are cost work: nothing has to be re-derived numerically to land them.

The harness this card measured with stays: `scripts/backlog/probe.py`'s `stage` sampler records `stage_layers`,
`staged_bytes` and `stage_copies` from `/stats` across an arrival, and `steps_sliced` / `substeps` from `/metrics` per
phase, so the copy share per slice that this note reports is one run away on any lane.

Commands, both lanes (`local` is `scripts/backlog/lane.sh local`, `iks-A` tars the tree and runs the same in
`~/moebius-gpu/RadianceHy4Flash-run-<run>/src`; the three arms ran in `-run-1954`, the byte check in `-run-1986`):

    RAD_CHUNK_SLICES=off  scripts/backlog/run.sh s02off  nospec.yaml stall1,stall4,pf6,pf20,greedy
    RAD_CHUNK_SLICES=auto scripts/backlog/run.sh s02auto nospec.yaml stall1,stall4,pf6,pf20,greedy
    RAD_CHUNK_SLICES=4    scripts/backlog/run.sh s02m4   nospec.yaml stall1,stall4,pf6,pf20,greedy

## Checks

The revert leaves the tree the base commit's for every file but this note, the backlog row and `probe.py`'s stage
sampler, so its card-less checks are the base's state and not a new claim. Run anyway, on a fresh build (host and
`gfx1201` kernels, `-DRAD_WITH_HIP=ON`):

    cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201 \
      -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++
    cmake --build build-clang -j 40 -- -k 0
    ctest --test-dir build-clang -LE 'gpu|oot' -j 12

89 % of 215 tests pass: 24 fail and 3 skip, which is the count the base commit's notes already carry (K-07).
`hyv4flash_arch_test`, `sched_test`, `config_test`, `fleet_config_test`, the `r4d_emu_*` set and
`tests/chain_scalar_audit.py` are among the passing. `museglimmer_test` does not compile under this box's clang
(`-Wc++11-narrowing` in `tests/museglimmer_test.cpp`, a file this branch does not touch); under gcc 13 the base does
not compile at all (`taking address of a temporary array` in `abi/rad_runtime.h`). The RDNA4 ISA suite has nothing of
this branch to check — no kernel file changes — and pointed at the tree anyway it reports 0 errors over 200 kernels.
