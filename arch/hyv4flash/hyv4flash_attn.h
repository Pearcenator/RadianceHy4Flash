/* hyv4flash_attn.h -- Hy4-flash's gated MLA with a learned sink, both layer kinds.
 *
 *   q   = q_b(rmsnorm(q_a(x)))            served as two linears over q_b's rows, the heads' nope
 *                                         rows and their rope rows (as bailing_hybrid does)
 *   row = [rmsnorm(kv_a(x)[:L]) | rope(kv_a(x)[L:])]   the cache row, one KV head of (L + R) / 2
 *   g   = attn_gate(x)                    heads * v wide, acting after the expansion
 *   o   = rmsnorm(o_proj(ao * sigmoid(g)))
 *
 * The attention itself takes one of two paths (PLAN §1.2):
 *   absorbed      q_nope through kv_b's key rows into the latent (mla_project_kv absorb), the
 *                 latent attention over the paged cache (mla_attn_paged_sink, or mla_attn_sel_sink
 *                 over the indexer's table), the latent expanded through kv_b's value rows
 *   non-absorbed  the step's latent rows through kv_b (a GEMM), split into per-head keys and
 *                 values (mla_kv_split), dense attention with the sink (attn_kv_sink)
 * Decode and verify rows are absorbed, through an instance of the latent attention declared with
 * `stable 1` (a row's bytes independent of the step's other rows and streams: verify == decode at
 * any stream count, WP16 perf.md 3.4.1); the absorbed prefill rows use a second, `stable 0`. A step whose prefill sequences all start at position 0
 * (nothing of them cached) runs those rows non-absorbed, unless the sparse layers select (context
 * past the dense limit), which is absorbed for every row. RADIANCE_HY4_PREFILL=absorbed sends every
 * row down the absorbed path. A prefill row with a cached prefix is absorbed too: gathering the
 * cached rows back into a contiguous key block is not built (no op serves the paged read).
 *
 * The sparse layers' indexer (hyv4flash_indexer.h) runs between the cache store and the
 * attention.
 *
 * Streamed KV (docs/KV-STREAMING.md, --kv-resident): the sparse latent group may stream, as GLM's
 * MLA does (rad_block_glm_mla.h). Its store writes the host mirror always and the ring slot where
 * the block is resident (kv_store_streamed), and on a pass bound to the ring (a decode or verify
 * pass) the selection is resolved between qsa_select and the attention (kv_resolve: misses copied
 * in from the mirror, `sel` rewritten from block ids to ring slots). Such a pass always attends
 * through the selection, also at or below the dense limit: there every closed group is selected
 * (kernels/indexer.md §1), which is the dense result, and the dense kernel would read block ids as
 * ring slots. A pass on the stage (prefill, mixed) reads block ids as the resident path does. The
 * sliding group is a window and never streams; the indexer's key cache is read whole every step
 * and stays resident (rad_kv_group_dense_read).
 *
 * The row form (`tp-attn-split: rows`, off by default; Shape::attn_rows). At more than one rank a
 * sparse layer's selecting prefill step of GlmTp::kSplitRows rows or more, every row a prefill
 * query (no decode rows, no compact pass, no streamed group, no three-component rotary table), a
 * whole number of rows a rank and more than kRowsNarrow of them,
 * splits by rows instead of heads: rank r runs rows [r * T / world, (r + 1) * T / world) at every
 * head of the model, so one launch of the latent attention reads a row's selected keys once for
 * all heads instead of once a rank. The step's other heights keep the head form.
 *
 * The bytes are the head form's. Every per-head op (the q_b rows, the absorb, the attention, the
 * expansion and the gate) is the head form's op at the model's head count over views of the same
 * stored weights, and each writes a (row, head) from that row and head alone. o_proj is where the
 * head form rounds: each rank's partial is a bf16 GEMM over its heads' columns, and the exact
 * all-reduce sums the partials in f32 from +0 in rank order and rounds once. So the row form runs
 * the same `world` GEMMs, one over each rank's columns of the gated plane against that rank's
 * column share of attn_output (the head form's shapes but for M), widens each bf16 partial to f32
 * and adds them in rank order onto a +0 fill, and narrows once (round to nearest even, the
 * all-reduce's). An all_gather of whole rows then replaces the layer's all-reduce, and the
 * junction behind it runs without its reduction (Junction::step's `reduced`).
 *
 * Its cost in weights: whole views of q_nope, q_rope, kv_b, the sinks and attn_gate and every
 * rank's column share of attn_output, a sparse layer (rows_weight_bytes, noted at startup). */
#ifndef HYV4FLASH_ATTN_H
#define HYV4FLASH_ATTN_H

#include "hyv4flash_common.h"
#include <arch/rad_glm_tp.h>

namespace hyv4flash {

/* What a step's attention does, decided once a step on the host (every layer and rank agrees). */
struct StepPlan {
    int64_t T = 0, S = 0, D = 0, DT = 0, P = 0;
    bool nonabs = false;   /* the prefill rows [DT, T) run non-absorbed */
    bool select = false;   /* the sparse layers attend through the indexer's selection */
    int64_t live = 0;      /* the indexer's live block columns */
    /* The row form (Attn::rows_take), set a layer: this rank's rows [r0, r0 + rrows) of the step.
     * rrows 0: the head form. */
    int64_t r0 = 0, rrows = 0;
    bool rlane = false;   /* its gather rides the second lane, where a piped step's collectives do */
};

/* Consecutive original rows can move in one launch. Query projections read the original
 * norm/codes and write compact Q. The cache projections still read the whole chunk. */
struct QueryRows {
    const int32_t* ids = nullptr;
    int64_t n = 0;
    /* Device: the retained prefill queries' bounds rebased to 0 (the prefill_tail derivation's
     * prefill_cu, located by TailLayout in hyv4flash.cpp). Null on a full pass, whose query
     * bounds are the key bounds, the trunk's once-a-step rebase (w.cu_p). */
    const int32_t* pf_cu = nullptr;
    template<class F> void spans(F f) const {
        if (!ids) { if (n) f(0, 0, n); return; }
        for (int64_t i = 0; i < n;) {
            int64_t end = i + 1;
            while (end < n && ids[end] == ids[i] + end - i) ++end;
            f(ids[i], i, end - i);
            i = end;
        }
    }
};

/* The planes of one attention geometry, shared by its layers. */
struct AttnWire {
    rad_buf qa = 0;       /* [T, q_lora] q_a's output, normed in place (HF's q_resid) */
    rad_buf qa_raw = 0;   /* [T, q_lora] q_a's output in front of a fused norm (Attn::qfuse); declared
                             by the first layer that fuses */
    rad_buf q_nope = 0;   /* [T, H*nope] */
    rad_buf q_rope = 0;   /* [T, H*R] rotated in place */
    rad_buf row = 0;      /* [T, L + R] the cache row */
    rad_buf q_lat = 0;    /* [T, H*L] the absorbed query */
    rad_buf lat = 0;      /* [T, H*L] the latent attention's output */
    rad_buf q_full = 0;   /* [T, H, nope + R] the non-absorbed query */
    rad_buf latc = 0;     /* [T, L] the normed latent, contiguous (a GEMM reads whole rows) */
    rad_buf kvb = 0;      /* [T, H*(nope + v)] kv_b over the step's latent rows */
    rad_buf k = 0;        /* [T, H*(nope + R)] */
    rad_buf v = 0;        /* [T, H*v] */
    rad_buf ao = 0;       /* [T, H*v] the attention's output */
    rad_buf gate = 0;     /* [T, H*v] attn_gate's logits */
    rad_buf ag = 0;       /* [T, H*v] gated */
    rad_buf o = 0;        /* [T, n] o_proj's output */
    /* The row form's planes: one rank's run of a step, `run` rows at the model's whole head count HW
     * (0 where the row form is off). */
    int64_t run = 0;
    rad_buf r_q_nope = 0, r_q_rope = 0;   /* [R, HW*nope], [R, HW*R] */
    rad_buf r_q_lat = 0, r_lat = 0;       /* [R, HW*L] */
    rad_buf r_gate = 0, r_ag = 0;         /* [R, HW*v] */
    rad_buf r_agj = 0;                    /* [R, H*v] one rank's columns of r_ag, contiguous */
    rad_buf r_part = 0;                   /* [R, n] one rank's bf16 partial of o_proj */
    rad_buf r_tmp = 0, r_acc = 0;         /* [R, n] f32: a partial widened, the sum */
    rad_buf r_o = 0;                      /* [R, n] the sum rounded, gathered into `o` */
    rad_buf cu_p = 0;     /* [max_seqs + 1] i32: the prefill sequences' bounds rebased to 0; the
                             model's one buffer (P-04), written once a step by Model::op_rebase,
                             read by every layer of both geometries */
    /* `whole` and `rows`: the model's geometry and a rank's run, for the row form (0: none). */
    int declare(RadBuilder* b, Names& nm, const Shape& s, const AttnGeom& a, int64_t T,
                rad_buf cu_prefill, const char* tag, const AttnGeom* whole = nullptr,
                int64_t rows = 0) {
        auto bf = [&](const char* n, std::initializer_list<int64_t> shape) {
            return decl_b(b, nm.f("%s.%s", tag, n), RAD_BF16, shape);
        };
        const int64_t H = a.heads, L = s.kv_lora, R = a.qk_rope;
        qa = bf("q_a", { T, s.q_lora });
        q_nope = bf("q_nope", { T, H * a.qk_nope });
        q_rope = bf("q_rope", { T, H * R });
        row = bf("kv_row", { T, L + R });
        q_lat = bf("q_latent", { T, H * L });
        lat = bf("attn_latent", { T, H * L });
        q_full = bf("q_full", { T, H, a.qk() });
        latc = bf("latent", { T, L });
        kvb = bf("kv_b", { T, a.kv_width() });
        k = bf("k", { T, H * a.qk() });
        v = bf("v", { T, H * a.v });
        ao = bf("attn_value", { T, a.v_width() });
        gate = bf("attn_gate", { T, a.v_width() });
        ag = bf("attn_gated", { T, a.v_width() });
        o = bf("o_proj", { T, s.n_embd });
        cu_p = cu_prefill;
        if (whole && rows > 0) {
            const int64_t HW = whole->heads;
            run = rows;
            r_q_nope = bf("rows.q_nope", { run, HW * a.qk_nope });
            r_q_rope = bf("rows.q_rope", { run, HW * R });
            r_q_lat = bf("rows.q_latent", { run, HW * L });
            r_lat = bf("rows.attn_latent", { run, HW * L });
            r_gate = bf("rows.attn_gate", { run, whole->v_width() });
            r_ag = bf("rows.attn_gated", { run, whole->v_width() });
            r_agj = bf("rows.attn_gated_part", { run, a.v_width() });
            r_part = bf("rows.o_part", { run, s.n_embd });
            r_tmp = decl_b(b, nm.f("%s.rows.o_part_f32", tag), RAD_F32, { run, s.n_embd });
            r_acc = decl_b(b, nm.f("%s.rows.o_sum_f32", tag), RAD_F32, { run, s.n_embd });
            r_o = bf("rows.o_proj", { run, s.n_embd });
            if (!r_q_nope || !r_q_rope || !r_q_lat || !r_lat || !r_gate || !r_ag || !r_agj ||
                !r_part || !r_tmp || !r_acc || !r_o)
                return RAD_E_INVAL;
        }
        return qa && q_nope && q_rope && row && q_lat && lat && q_full && latc && kvb && k && v && ao &&
                       gate && ag && o && cu_p
                   ? RAD_OK
                   : RAD_E_INVAL;
    }
};

struct Indexer;   /* hyv4flash_indexer.h */

struct Attn {
    Shape s{};
    AttnGeom a{};
    AttnWire w{};
    Geom g{};
    int layer = 0;
    bool sliding = false;
    int64_t bs = 0, window = 0;
    rad_buf xa = 0, ya = 0;
    rad_kvgroup kv = 0;
    Lin q_a{}, q_nope{}, q_rope{}, kv_a{}, gate{}, out{};
    rad_weight qa_gain = 0, kv_gain = 0, post_gain = 0, kv_b = 0, sinks = 0;
    rad_op op_qa_norm = 0, op_kv_norm = 0, op_q_rope = 0, op_k_rope = 0, op_store = 0;
    /* The q residual's norm writes q_nope's and q_rope's rotated int8 codes in its pass
     * (rmsnorm_had_quant_i8g, fused_quant_on): q_a then writes `qa_raw`, the norm `qa` and the
     * codes `qcodes`. */
    bool qfuse = false;
    ActFP8 qcodes{};
    rad_buf qa_out = 0;   /* q_a's output: w.qa (normed in place) or, behind the fused norm, qa_raw */
    rad_op op_store_stream = 0, op_resolve = 0;   /* a streamed sparse group (--kv-resident) */
    rad_op op_row_front = 0;   /* the cache row's norm, rotary and store in one launch (WP16.5) */
    /* The latent attention twice (WP16 perf.md 3.4.1): decode-class rows (decode and verify,
     * [0, DT)) with `stable 1`, whose bytes do not depend on the step's other rows or streams, and
     * prefill rows ([DT, T)) with `stable 0`, the fast form. */
    rad_op op_absorb = 0, op_paged = 0, op_paged_pf = 0, op_sel = 0, op_sel_pf = 0, op_expand = 0;
    rad_op op_lat_copy = 0, op_kvb = 0, op_split = 0, op_qn_copy = 0, op_qr_copy = 0, op_dense = 0;
    rad_op op_gate = 0, op_post_norm = 0, op_ar = 0;
    rad_op op_expand_gate = 0;   /* the expansion with the gate folded in, where a library has it */
    /* G-3c: the expansion, the gate and the rotated int8 codes attn_output reads, in one launch, for
     * decode-class steps (at most kFuseRows rows, every row absorbed); `out` then issues no quantiser. */
    rad_op op_expand_quant = 0;
    static constexpr int64_t kFuseRows = 64;
    /* The row form (the file's head comment): the head form's linears and per-head weights as
     * views at the model's width, every rank's column share of attn_output, and the fold. */
    Lin rq_nope{}, rq_rope{}, rgate{};
    std::vector<Lin> rout;   /* [world] rank j's columns of attn_output */
    rad_weight rkv_b = 0, rsinks = 0;
    rad_op op_r_rope = 0, op_r_absorb = 0, op_r_sel = 0, op_r_expand_gate = 0, op_r_expand = 0;
    rad_op op_r_gate = 0, op_r_cut = 0, op_r_fill = 0, op_r_widen = 0, op_r_add = 0;
    rad_op op_r_narrow = 0, op_r_gather = 0;
    static constexpr int64_t kRowsNarrow = 64;
    int64_t rows_weight_bytes = 0;   /* the views' bytes on this rank, about (one a code) */
    Tap t_in_norm{}, t_q_resid{}, t_kv_latent{}, t_kv_rope{}, t_gate{}, t_o_proj{}, t_out{};

    /* G-3b: maps the three linears that read the attention norm's output (q_a, kv_a, the gate) and
     * returns the code form of q_a's weight, so the junction in front can decide whether to write the
     * codes they read (the form comes from the mapped weight). Declare then skips the mapping. */
    CodeForm premap_in(RadBuilder* bld, Names& nm, const Shape& shape, int l, const char* ap) {
        q_a.mapped = kv_a.mapped = gate.mapped = false;
        if (map_linear(bld, nm.f("blk.%d.attn_q_a.weight", l), { nm.ckpt("%s.q_a_proj.weight", ap) }) !=
                RAD_OK ||
            map_linear(bld, nm.f("blk.%d.attn_kv_a.weight", l),
                       { nm.ckpt("%s.kv_a_proj_with_mqa.weight", ap) }) != RAD_OK ||
            map_linear(bld, nm.f("blk.%d.attn_gate.weight", l), { nm.ckpt("%s.linear_gate.weight", ap) }) !=
                RAD_OK)
            return CodeForm{};
        q_a.mapped = kv_a.mapped = gate.mapped = true;
        (void)shape;
        return code_form(bld, nm.f("blk.%d.attn_q_a.weight", l));
    }

    /* `ap` is the checkpoint's `model.layers.<l>.self_attn`. `group` is the layer's latent KV
     * group, declared by the caller right before this (its block size is this layer's ops').
     * `in_codes` (G-3b): the rotated int8 codes of `in` the junction in front writes. */
    int declare(RadBuilder* b, Names& nm, const Shape& shape, const Geom& geom, int l,
                bool is_sliding, AttnWire& wire, rad_buf in, rad_buf outb, rad_kvgroup group,
                int64_t block, const char* ap, bool first_of_group,
                const ActFP8* in_codes = nullptr) {
        s = shape;
        g = geom;
        layer = l;
        sliding = is_sliding;
        a = s.local(sliding);   /* this rank's heads */
        const AttnGeom& whole = sliding ? s.sliding : s.sparse;
        w = wire;
        xa = in;
        ya = outb;
        kv = group;
        bs = block;
        window = sliding ? s.window : 0;
        const int64_t T = g.max_tok, H = a.heads, L = s.kv_lora, R = a.qk_rope, N = a.qk_nope;
        const double theta = sliding ? s.theta_sliding : s.theta_sparse;
        const auto grp = grp_layer(l);
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv));

        /* q_b's rows are head-major, a head's nope rows then its rope rows: one row map a part,
         * a piece a head, eight pieces a map (successive ROWS maps extend one name). The maps
         * name every head; a rank reads its heads' rows of the result (RAD_SHARD_ROW). Mapped
         * first: the q residual's norm takes its form from them. */
        const char* qsrc = nm.ckpt("%s.q_b_proj.weight", ap);
        auto part = [&](const char* name, int64_t off, int64_t width) -> int {
            for (int64_t h0 = 0; h0 < whole.heads; h0 += 8) {
                RadNameMap m = map_base(name, RAD_MAP_ROWS);
                for (int64_t h = h0; h < std::min<int64_t>(h0 + 8, whole.heads); ++h) {
                    m.src[m.n_src] = qsrc;
                    m.src_index[m.n_src] = (int)(h * whole.qk() + off);
                    m.src_rows[m.n_src] = width;
                    ++m.n_src;
                }
                RAD_ARCH_TRY(rad_decl_name_map(b, &m));
            }
            return RAD_OK;
        };
        RAD_ARCH_TRY(part(nm.f("blk.%d.attn_q_nope.weight", l), 0, N));
        RAD_ARCH_TRY(part(nm.f("blk.%d.attn_q_rope.weight", l), N, R));
        q_nope.mapped = q_rope.mapped = true;

        /* The int8 inputs (Lin::feed): q_a, kv_a and the gate read the attention norm's output,
         * q_nope and q_rope the normed q residual; each group quantises once. Where both q_b
         * parts are rotated int8 of one form and libr4d serves it, the residual's norm writes
         * their codes itself (rmsnorm_had_quant_i8g), byte-identical to the norm followed by
         * had_quant_act_i8g. */
        const CodeForm fn = code_form(b, nm.f("blk.%d.attn_q_nope.weight", l));
        qfuse = fused_quant_on() && fn.rot && fn.same(code_form(b, nm.f("blk.%d.attn_q_rope.weight", l))) &&
                s.q_lora % fn.g8 == 0 && rad_op_declared(b, "rmsnorm_had_quant_i8g");
        if (qfuse && !wire.qa_raw) {
            wire.qa_raw = decl_b(b, nm.f("%s.q_a_raw", sliding ? "sliding" : "sparse"), RAD_BF16,
                                 { T, s.q_lora });
            if (!wire.qa_raw) return RAD_E_INVAL;
        }
        w.qa_raw = qfuse ? wire.qa_raw : 0;
        qa_out = qfuse ? w.qa_raw : w.qa;
        q_a.lead = "attn_in";
        q_a.fed_codes = kv_a.fed_codes = gate.fed_codes = in_codes;   /* each takes them by its form */
        RAD_ARCH_TRY(q_a.declare(b, nm, g, nm.f("blk.%d.attn_q_a", l), s.q_lora, s.n_embd,
                                 RAD_SHARD_NONE, grp, { nm.ckpt("%s.q_a_proj.weight", ap) },
                                 act(xa), qa_out));
        qa_gain = decl_f32(b, nm.f("blk.%d.attn_q_a_norm.weight", l),
                           nm.ckpt("%s.q_a_layernorm.weight", ap), { s.q_lora }, grp);
        if (!qa_gain) return RAD_E_INVAL;
        if (qfuse) {
            Codes* cs = codes_of(b);
            const Codes::Pair* pr = cs ? cs->get(s.q_lora, fn.g8, "q_res") : nullptr;
            if (!pr) return RAD_E_INVAL;
            qcodes.q8 = pr->q8;
            qcodes.s8 = pr->s8;
            qcodes.g8 = fn.g8;
            qcodes.q8_fed = qcodes.q8_rot = true;
            op_qa_norm = g.chain_q8
                ? mk(b, "rmsnorm_had_quant_i8g",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", s.q_lora), RAD_F64("eps", s.eps),
                       RAD_INT("group", fn.g8), RAD_STR("dtype", "bf16"), RAD_INT("chained", 1) },
                     { qa_gain }, { w.qa_raw }, { qcodes.q8, qcodes.s8, w.qa })
                : mk(b, "rmsnorm_had_quant_i8g",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", s.q_lora), RAD_F64("eps", s.eps),
                       RAD_INT("group", fn.g8), RAD_STR("dtype", "bf16") },
                     { qa_gain }, { w.qa_raw }, { qcodes.q8, qcodes.s8, w.qa });
            /* Unserved after all: the plain norm out of place, and q_nope's own quantiser. */
            if (!op_qa_norm) qfuse = false;
        }
        if (!qfuse) op_qa_norm = decl_rmsnorm(b, T, s.q_lora, s.eps, qa_gain, qa_out, w.qa);
        HY4_NEED(op_qa_norm, "rmsnorm");
        if (qfuse) q_nope.fed_codes = q_rope.fed_codes = &qcodes;
        q_nope.lead = "q_res";
        q_rope.feed = &q_nope;
        RAD_ARCH_TRY(q_nope.declare(b, nm, g, nm.f("blk.%d.attn_q_nope", l), H * N, s.q_lora,
                                    RAD_SHARD_ROW, grp, { qsrc }, act(w.qa), w.q_nope));
        RAD_ARCH_TRY(q_rope.declare(b, nm, g, nm.f("blk.%d.attn_q_rope", l), H * R, s.q_lora,
                                    RAD_SHARD_ROW, grp, { qsrc }, act(w.qa), w.q_rope));
        /* K-02 (P-14's switch, RADIANCE_HY4_FUSE_QPROJ=0 keeps the two launches): q_nope and
         * q_rope over the one set of codes q_a's quantiser writes, in one launch at a decode step's
         * GEMV, where each of the two is launch and ramp cost. Taken only where the two are one
         * form and one K (declare_dual refuses the rest) and only at M == 1 (kDualMaxM); a wider
         * pass keeps both Lins' own launches. */
        if (dual_fuse_on()) RAD_ARCH_TRY(q_nope.declare_dual(b, q_rope, w.qa,
                                                             w.q_nope, w.q_rope, T));
        q_nope.fed_codes = q_rope.fed_codes = nullptr;   /* `qcodes` moves with the layer */

        /* kv_a's rows are the latent then the shared rotary key (`kv_a_proj_with_mqa`). */
        kv_a.feed = &q_a;
        RAD_ARCH_TRY(kv_a.declare(b, nm, g, nm.f("blk.%d.attn_kv_a", l), L + R, s.n_embd,
                                  RAD_SHARD_NONE, grp,
                                  { nm.ckpt("%s.kv_a_proj_with_mqa.weight", ap) }, act(xa), w.row));
        kv_gain = decl_f32(b, nm.f("blk.%d.attn_kv_a_norm.weight", l),
                           nm.ckpt("%s.kv_a_layernorm.weight", ap), { L }, grp);
        if (!kv_gain) return RAD_E_INVAL;
        op_kv_norm = decl_rmsnorm(b, T, L, s.eps, kv_gain, w.row, w.row);
        const char* sec = !s.mrope ? nullptr : sliding ? s.mrope_sliding.c_str()
                                                         : s.mrope_sparse.c_str();
        /* K-49: the query rope is the wide one -- H heads of pairs, so the row form drops
         * `H` times the `sincosf` over a prefill chunk -- and the key rope is one head a row,
         * where the launch would not take the row form at all (r4d_model_bf16.hip:1538). */
        op_q_rope = decl_rope_gptj(b, T, H, R, theta, w.q_rope, sec, "rope_rows");
        op_k_rope = decl_rope_gptj(b, T, 1, R, theta, w.row, sec);
        HY4_NEED(op_kv_norm && op_q_rope && op_k_rope, "rmsnorm / rope (gptj)");
        /* The row form's q_b: both parts whole, over the codes the head form's read. */
        if (w.run && !sliding) {
            const int64_t HW = whole.heads;
            rq_nope.stored = nm.f("blk.%d.attn_q_nope.weight", l);
            rq_rope.stored = nm.f("blk.%d.attn_q_rope.weight", l);
            rq_nope.fed_codes = rq_rope.fed_codes = qfuse ? &qcodes : nullptr;
            rq_nope.feed = rq_rope.feed = &q_nope;
            RAD_ARCH_TRY(rq_nope.declare(b, nm, g, nm.f("blk.%d.attn_q_nope_rows", l), HW * N,
                                         s.q_lora, RAD_SHARD_NONE, grp, { qsrc }, act(w.qa),
                                         w.r_q_nope, w.run));
            RAD_ARCH_TRY(rq_rope.declare(b, nm, g, nm.f("blk.%d.attn_q_rope_rows", l), HW * R,
                                         s.q_lora, RAD_SHARD_NONE, grp, { qsrc }, act(w.qa),
                                         w.r_q_rope, w.run));
            rq_nope.fed_codes = rq_rope.fed_codes = nullptr;
            op_r_rope = decl_rope_gptj(b, w.run, HW, R, theta, w.r_q_rope, sec, "rope_rows");
            HY4_NEED(op_r_rope, "rope (gptj), the row form's");
            rows_weight_bytes += HW * (N + R) * s.q_lora;
        }
        /* The cache row as one KV head of (L + R) / 2, its first half in the K slot and the rest in
         * the V slot (bailing_hybrid's arrangement). The first op after the group to carry a
         * block size claims it. */
        op_store = mk(b, "kv_store",
                      { RAD_RANGE("M", 1, T), RAD_INT("head_dim", (L + R) / 2),
                        RAD_INT("n_head_kv", 1), RAD_INT("block_size", bs),
                        RAD_STR("kv_dtype", "bf16") },
                      {}, { w.row }, {});
        HY4_NEED(op_store, "kv_store");
        /* The row's norm, the key part's rotary and the store as one op, where a library has it and
         * nothing reads between them (the taps); declared after kv_store, whose block size the group
         * takes. The streamed store and the staged pass keep the three. */
        if (cache_row_on() && !taps_on() && rad_op_declared(b, "mla_row_norm_rope_store")) {
            op_row_front =
                sec && *sec
                    ? mk(b, "mla_row_norm_rope_store",
                         { RAD_RANGE("M", 1, T), RAD_INT("latent", L), RAD_INT("rope_dim", R),
                           RAD_F64("eps", s.eps), RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                           RAD_STR("mode", "imrope_gptj"), RAD_STR("sections", sec),
                           RAD_INT("block_size", bs), RAD_STR("dtype", "bf16") },
                         { kv_gain }, { w.row }, { w.row })
                    : mk(b, "mla_row_norm_rope_store",
                         { RAD_RANGE("M", 1, T), RAD_INT("latent", L), RAD_INT("rope_dim", R),
                           RAD_F64("eps", s.eps), RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                           RAD_STR("mode", "gptj"), RAD_INT("block_size", bs),
                           RAD_STR("dtype", "bf16") },
                         { kv_gain }, { w.row }, { w.row });
            if (op_row_front && !rad_op_resolved(b, op_row_front)) op_row_front = 0;
        }
        /* The streamed store: kv_store's operands plus the residency map and the layer's mirror
         * row, added at issue. Declared only when the architecture can stream and --kv-resident
         * asks for a ring (rad_op_declared); the sliding window never streams. */
        if (!sliding && rad_op_declared(b, "kv_store_streamed"))
            op_store_stream = mk(b, "kv_store_streamed",
                                 { RAD_RANGE("M", 1, T), RAD_INT("head_dim", (L + R) / 2),
                                   RAD_INT("n_head_kv", 1), RAD_INT("block_size", bs),
                                   RAD_STR("kv_dtype", "bf16") },
                                 {}, { w.row }, {});
        if (first_of_group && rad_kv_block_size(b, kv) != bs) {
            fprintf(stderr, "radiance: hyv4flash: layer %d's latent cache took a block of %lld "
                            "tokens; its ops are declared at %lld\n",
                    l, (long long)rad_kv_block_size(b, kv), (long long)bs);
            return RAD_E_INVAL;
        }

        /* kv_b stays plain bf16 (DECISIONS U2): the absorbed ops read its per-head blocks. */
        const char* bn = nm.f("blk.%d.attn_kv_b.weight", l);
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.kv_b_proj.weight", ap)));
        if (!is_plain_bf16(b, bn)) {
            RadEncoding e{};
            char en[256] = "";
            if (weight_enc(b, bn, &e)) rad_enc_format(&e, en, sizeof en);
            fprintf(stderr, "radiance: hyv4flash: '%s' is %s; kv_b must stay plain bf16, because "
                            "the absorbed attention reads its per-head key and value blocks "
                            "(DECISIONS U2). Convert it without a recipe rule.\n", bn, en);
            return RAD_E_UNSUPPORTED;
        }
        /* A head's key rows then its value rows, head after head: a rank's heads are a
         * contiguous run of rows, as are its sinks. */
        kv_b = decl_w(b, bn, RAD_BF16, { a.kv_width(), L }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                      grp);
        sinks = decl_f32(b, nm.f("blk.%d.attn_sinks.weight", l),
                         nm.ckpt("%s.learnable_sink_param", ap), { H }, grp, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_ROW);
        if (!kv_b || !sinks) return RAD_E_INVAL;
        if (w.run && !sliding) {
            rkv_b = decl_view(b, nm.f("blk.%d.attn_kv_b_rows.weight", l), bn, nullptr, RAD_BF16,
                              { whole.kv_width(), L }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
            rsinks = decl_view(b, nm.f("blk.%d.attn_sinks_rows.weight", l),
                               nm.f("blk.%d.attn_sinks.weight", l), nullptr, RAD_F32,
                               { whole.heads }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
            if (!rkv_b || !rsinks) return RAD_E_INVAL;
            rows_weight_bytes += whole.kv_width() * L * 2;
        }

        gate.feed = &q_a;
        RAD_ARCH_TRY(gate.declare(b, nm, g, nm.f("blk.%d.attn_gate", l), a.v_width(), s.n_embd,
                                  RAD_SHARD_ROW, grp, { nm.ckpt("%s.linear_gate.weight", ap) },
                                  act(xa), w.gate));
        /* K-03: the gate reads q_a's codes and both are GEMVs at a decode step, so where the two
         * encodings agree one launch carries the pair -- q_a's part is the eight-wave one -- and the
         * gate's own launch stays the wider passes'. Its product is written at q_a's place, which is
         * ahead of `mla_expand_gate`, its only reader. */
        if (qa_gate_fuse_on())
            RAD_ARCH_TRY(q_a.declare_dual(b, gate, xa, qa_out, w.gate, T));

        if (w.run && !sliding) {
            rgate.stored = nm.f("blk.%d.attn_gate.weight", l);
            rgate.feed = &q_a;
            rgate.fed_codes = in_codes;
            RAD_ARCH_TRY(rgate.declare(b, nm, g, nm.f("blk.%d.attn_gate_rows", l), whole.v_width(),
                                       s.n_embd, RAD_SHARD_NONE, grp,
                                       { nm.ckpt("%s.linear_gate.weight", ap) }, act(xa), w.r_gate,
                                       w.run));
            rgate.fed_codes = nullptr;
            rows_weight_bytes += whole.v_width() * s.n_embd;
        }
        q_a.fed_codes = kv_a.fed_codes = gate.fed_codes = nullptr;
        RAD_ARCH_TRY(t_in_norm.declare(b, nm, T, "attn_in_norm", l, RAD_BF16, s.n_embd, xa));
        RAD_ARCH_TRY(t_q_resid.declare(b, nm, T, "attn_q_resid", l, RAD_BF16, s.q_lora, w.qa));
        RAD_ARCH_TRY(t_kv_latent.declare(b, nm, T, "attn_kv_latent", l, RAD_BF16, L, w.row));
        /* The cache row's rotated key part (what a DFly2 draft layer reads back, WP13). */
        RAD_ARCH_TRY(t_kv_rope.declare(b, nm, T, "attn_kv_rope", l, RAD_BF16, R, w.row));
        /* The gate's logits are the rank's heads only: tapped at one rank. */
        if (s.world == 1)
            RAD_ARCH_TRY(t_gate.declare(b, nm, T, "attn_gate_logits", l, RAD_BF16, a.v_width(),
                                        w.gate));
        return RAD_OK;
    }

    /* The attention proper and the output, declared after the sparse layer's indexer: `sel` and
     * `len` are its table (0 on a sliding layer). Declaration order is liveness. */
    int declare_core(RadBuilder* b, Names& nm, const char* ap, rad_buf sel, rad_buf len,
                     rad_buf nsel) {
        const int64_t T = g.max_tok, H = a.heads, L = s.kv_lora, R = a.qk_rope, N = a.qk_nope;
        const double scale = 1.0 / std::sqrt((double)a.qk());
        const int l = layer;
        const auto grp = grp_layer(l);
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        auto proj = [&](const char* mode, rad_buf in_, rad_buf out_) {
            return mk(b, "mla_project_kv",
                      { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("k_dim", N),
                        RAD_INT("v_dim", a.v), RAD_INT("latent_dim", L), RAD_STR("mode", mode),
                        RAD_STR("dtype", "bf16") },
                      { kv_b }, { in_ }, { out_ });
        };
        op_absorb = proj("absorb", w.q_nope, w.q_lat);
        /* The rotary the sinks fold (their optional `rope_pos` operand): the stable instance's, so
         * a decode step needs no separate `rope` launch (`front` then issues none, and the raw
         * projection columns are what the operand holds). The fast instance keeps the rotated
         * buffer, since a mixed step issues one `rope` over every row. */
        const char* rsec = !s.mrope ? nullptr
                          : sliding ? s.mrope_sliding.c_str() : s.mrope_sparse.c_str();
        const double rtheta = sliding ? s.theta_sliding : s.theta_sparse;
        /* Decode-class rows: one a stream and its speculation rows. */
        const int64_t DTmax = std::max<int64_t>(1, std::min<int64_t>(T, g.max_logit_rows));
        auto sink_params = [&](int64_t rows, int stable, bool windowed, int64_t blocks,
                               int64_t win, int64_t heads = 0) {
            std::vector<RadParam> ps = { RAD_RANGE("M", 1, rows), RAD_INT("heads", heads ? heads : H),
                                         RAD_INT("latent_dim", L), RAD_INT("rope_dim", R) };
            if (windowed) {
                ps.push_back(RAD_INT("block_size", bs));
                ps.push_back(RAD_INT("window", win));
            } else {
                ps.push_back(RAD_INT("blocks", blocks));
                ps.push_back(RAD_INT("block_size", bs));
            }
            ps.push_back(RAD_F64("scale", scale));
            ps.push_back(RAD_STR("dtype", "bf16"));
            ps.push_back(RAD_INT("stable", stable));
            if (stable) {
                ps.push_back(RAD_F64("rope_theta", rtheta));
                ps.push_back(RAD_F64("rope_scale", 1.0));
                if (rsec && *rsec) {
                    ps.push_back(RAD_STR("rope_mode", "imrope_gptj"));
                    ps.push_back(RAD_STR("rope_sections", rsec));
                } else {
                    ps.push_back(RAD_STR("rope_mode", "gptj"));
                }
            }
            return ps;
        };
        auto paged_op = [&](int64_t rows, int stable, std::initializer_list<rad_buf> in) {
            return mk_list(b, "mla_attn_paged_sink", sink_params(rows, stable, true, 0, window),
                           { sinks }, in, { w.lat });
        };
        op_paged = paged_op(DTmax, 1, { w.q_lat, w.q_rope });
        op_paged_pf = paged_op(T, 0, { w.q_lat, w.q_rope, w.cu_p });
        op_expand = proj("expand", w.lat, w.ao);
        HY4_NEED(op_absorb && op_expand, "mla_project_kv");
        /* mla_expand_gate is the expansion followed by gate_mul, byte for byte (libr4d in one
         * launch, the host libraries as the pair): the absorbed rows take it where it resolves,
         * the non-absorbed ones keep gate_mul.
         * RADIANCE_HY4_EXPAND_GATE=0 keeps the pair everywhere (an A/B switch). Declared beside
         * the expansion, where it is issued: declaration order is liveness. */
        const char* eg = std::getenv("RADIANCE_HY4_EXPAND_GATE");
        if (!(eg && !std::strcmp(eg, "0")) && rad_op_declared(b, "mla_expand_gate")) {
            op_expand_gate = mk(b, "mla_expand_gate",
                                { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("k_dim", N),
                                  RAD_INT("v_dim", a.v), RAD_INT("latent_dim", L),
                                  RAD_STR("dtype", "bf16") },
                                { kv_b }, { w.lat, w.gate }, { w.ag });
            if (op_expand_gate && !rad_op_resolved(b, op_expand_gate)) op_expand_gate = 0;
        }
        HY4_NEED(op_paged && op_paged_pf, "mla_attn_paged_sink");
        if (!sliding) {
            auto sel_op = [&](int64_t rows, int stable) {
                return mk_list(b, "mla_attn_sel_sink",
                               sink_params(rows, stable, false, s.ix_group_topk, 0), { sinks },
                               { w.q_lat, w.q_rope, sel, len }, { w.lat });
            };
            op_sel = sel_op(DTmax, 1);
            op_sel_pf = sel_op(T, 0);
            HY4_NEED(op_sel && op_sel_pf, "mla_attn_sel_sink");
            /* Issued after the layer's selection and before mla_attn_sel_sink reads `sel`;
             * `nsel` is qsa_select's entry count, the table's pitch the attention's `blocks`. */
            op_resolve = kv_resolve_declare(b, g, kv, sel, nsel, s.ix_group_topk);
        }

        /* The non-absorbed prefill. */
        op_lat_copy = decl_copy(b, T, L, RAD_BF16, w.row, w.latc);
        op_kvb = mk(b, "gemm_nt",
                    { RAD_RANGE("M", 1, T), RAD_INT("N", a.kv_width()), RAD_INT("K", L),
                      RAD_STR("dtype", "bf16") },
                    { kv_b }, { w.latc }, { w.kvb });
        op_split = mk(b, "mla_kv_split",
                      { RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("k_dim", N),
                        RAD_INT("v_dim", a.v), RAD_INT("rope_dim", R), RAD_STR("dtype", "bf16") },
                      {}, { w.kvb, w.row }, { w.k, w.v });
        op_qn_copy = decl_copy(b, T, H * N, RAD_BF16, w.q_nope, w.q_full);
        op_qr_copy = decl_copy(b, T, H * R, RAD_BF16, w.q_rope, w.q_full);
        op_dense = mk(b, "attn_kv_sink",
                      { RAD_RANGE("M", 1, T), RAD_INT("N", T), RAD_INT("heads", H),
                        RAD_INT("head_dim", a.qk()), RAD_INT("window", window),
                        RAD_F64("scale", scale), RAD_STR("dtype", "bf16") },
                      { sinks }, { w.q_full, w.k, w.v, w.cu_p }, { w.ao });
        HY4_NEED(op_lat_copy && op_kvb, "cast / gemm_nt");
        HY4_NEED(op_split, "mla_kv_split");
        HY4_NEED(op_qn_copy && op_qr_copy, "cast");
        HY4_NEED(op_dense, "attn_kv_sink");

        op_gate = mk(b, "gate_mul",
                     { RAD_RANGE("M", 1, T), RAD_INT("n", a.v_width()), RAD_STR("dtype", "bf16") },
                     {}, { w.ao, w.gate }, { w.ag });
        HY4_NEED(op_gate, "gate_mul");
        /* o_proj reads the rank's heads (its columns) into a partial sum, reduced before the
         * sandwich norm. */
        RAD_ARCH_TRY(out.declare(b, nm, g, nm.f("blk.%d.attn_output", l), s.n_embd, a.v_width(),
                                 RAD_SHARD_COL, grp, { nm.ckpt("%s.o_proj.weight", ap) },
                                 act(w.ag), w.o));
        /* G-3c (RADIANCE_HY4_FUSE_EXPAND_QUANT=0 keeps the expansion and the quantiser apart). Only
         * where o_proj reads rotated int8 codes of the 128 group its own quantiser writes, and no tap
         * or calibration reads the gated plane between. */
        if (fuse_on("RADIANCE_HY4_FUSE_EXPAND_QUANT") && op_expand_gate && out.i8 && out.i8_rot &&
            !out.fed && out.op_q && out.g8 == RAD_FP8_BLOCK && out.q8 && out.s8 &&
            calib_dir() == nullptr && rad_op_declared(b, "mla_expand_gate_had_quant_i8g")) {
            op_expand_quant =
                mk(b, "mla_expand_gate_had_quant_i8g",
                   { RAD_RANGE("M", 1, std::min<int64_t>(T, kFuseRows)), RAD_INT("heads", H),
                     RAD_INT("k_dim", N), RAD_INT("v_dim", a.v), RAD_INT("latent_dim", L),
                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16") },
                   { kv_b }, { w.lat, w.gate }, { out.q8, out.s8 });
            if (op_expand_quant && !rad_op_resolved(b, op_expand_quant)) op_expand_quant = 0;
        }
        /* The row form, in the order it is issued (declaration order is liveness). */
        if (w.run && !sliding) {
            const AttnGeom& whole = s.sparse;
            const int64_t HW = whole.heads, RR = w.run, E = s.n_embd, W = a.v_width();
            auto rproj = [&](const char* mode, rad_buf in_, rad_buf out_) {
                return mk(b, "mla_project_kv",
                          { RAD_RANGE("M", 1, RR), RAD_INT("heads", HW), RAD_INT("k_dim", N),
                            RAD_INT("v_dim", a.v), RAD_INT("latent_dim", L), RAD_STR("mode", mode),
                            RAD_STR("dtype", "bf16") },
                          { rkv_b }, { in_ }, { out_ });
            };
            op_r_absorb = rproj("absorb", w.r_q_nope, w.r_q_lat);
            op_r_sel = mk_list(b, "mla_attn_sel_sink",
                               sink_params(RR, 0, false, s.ix_group_topk, 0, HW), { rsinks },
                               { w.r_q_lat, w.r_q_rope, sel, len }, { w.r_lat });
            HY4_NEED(op_r_absorb && op_r_sel, "mla_project_kv / mla_attn_sel_sink, the row form's");
            /* The expansion and the gate as the head form takes them. */
            if (op_expand_gate) {
                op_r_expand_gate = mk(b, "mla_expand_gate",
                                      { RAD_RANGE("M", 1, RR), RAD_INT("heads", HW),
                                        RAD_INT("k_dim", N), RAD_INT("v_dim", a.v),
                                        RAD_INT("latent_dim", L), RAD_STR("dtype", "bf16") },
                                      { rkv_b }, { w.r_lat, w.r_gate }, { w.r_ag });
                HY4_NEED(op_r_expand_gate, "mla_expand_gate, the row form's");
            } else {
                op_r_expand = rproj("expand", w.r_lat, w.r_ag);
                op_r_gate = mk(b, "gate_mul",
                               { RAD_RANGE("M", 1, RR), RAD_INT("n", whole.v_width()),
                                 RAD_STR("dtype", "bf16") },
                               {}, { w.r_ag, w.r_gate }, { w.r_ag });
                HY4_NEED(op_r_expand && op_r_gate, "mla_project_kv / gate_mul, the row form's");
            }
            op_r_fill = mk(b, "fill",
                           { RAD_RANGE("M", 1, RR), RAD_INT("n", E), RAD_STR("dtype", "f32"),
                             RAD_F64("value", 0.0) },
                           {}, {}, { w.r_acc });
            op_r_cut = decl_copy(b, RR, W, RAD_BF16, w.r_ag, w.r_agj);
            HY4_NEED(op_r_fill && op_r_cut, "fill / cast, the row form's");
            rout.assign((size_t)s.world, Lin{});
            for (int j = 0; j < s.world; ++j) {
                Lin& o = rout[(size_t)j];
                o.stored = nm.f("blk.%d.attn_output.weight", l);
                RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output_rows%d", l, j), E, W,
                                       RAD_SHARD_COL, grp, { nm.ckpt("%s.o_proj.weight", ap) },
                                       act(w.r_agj), w.r_part, RR, {}, j * W, (j + 1) * W));
            }
            rows_weight_bytes += (int64_t)s.world * E * W;
            op_r_widen = mk(b, "cast",
                            { RAD_RANGE("M", 1, RR), RAD_INT("n", E), RAD_STR("from", "bf16"),
                              RAD_STR("to", "f32") },
                            {}, { w.r_part }, { w.r_tmp });
            op_r_add = mk(b, "add",
                          { RAD_RANGE("M", 1, RR), RAD_INT("n", E), RAD_STR("dtype", "f32") }, {},
                          { w.r_acc, w.r_tmp }, { w.r_acc });
            op_r_narrow = mk(b, "cast",
                             { RAD_RANGE("M", 1, RR), RAD_INT("n", E), RAD_STR("from", "f32"),
                               RAD_STR("to", "bf16") },
                             {}, { w.r_acc }, { w.r_o });
            op_r_gather = mk(b, "all_gather",
                             { RAD_INT("world_size", s.world), RAD_RANGE("numel", E, RR * E),
                               RAD_STR("dtype", "bf16") },
                             {}, { w.r_o }, { w.o });
            HY4_NEED(op_r_widen && op_r_add && op_r_narrow, "cast / add (f32), the row form's");
            HY4_NEED(op_r_gather, "all_gather, the row form's");
        }
        int rc = RAD_OK;
        op_ar = decl_all_reduce(b, g, s.n_embd, w.o, &rc);
        RAD_ARCH_TRY(rc);
        post_gain = decl_f32(b, nm.f("blk.%d.attn_post_norm.weight", l),
                             nm.ckpt("%s.post_proj_norm.weight", ap), { s.n_embd }, grp);
        if (!post_gain) return RAD_E_INVAL;
        op_post_norm = decl_rmsnorm(b, T, s.n_embd, s.eps, post_gain, w.o, ya);
        HY4_NEED(op_post_norm, "rmsnorm");

        RAD_ARCH_TRY(t_o_proj.declare(b, nm, T, "attn_o_proj", l, RAD_BF16, s.n_embd, w.o));
        RAD_ARCH_TRY(t_out.declare(b, nm, T, "attn_out", l, RAD_BF16, s.n_embd, ya));
        return RAD_OK;
    }

    /* The decode rows' query rotary folded into the sinks' stable instances (their optional
     * `rope_pos`), so a step with no prefill rows needs no `rope` launch. A step with prefill rows
     * does not fold: one `rope` rotates every row there, and the fast instances and the
     * non-absorbed path read the rotated buffer. Off with RADIANCE_HY4_FOLD_QROPE=0, which issues
     * the `rope` launch back. */
    bool fold_qrope(const StepPlan& p) const {
        return p.T == p.DT && fuse_on("RADIANCE_HY4_FOLD_QROPE");
    }

    /* Whether this layer's pass takes the row form (the file's head comment): the same answer on
     * every rank, from the step's shape alone. `p` is the layer's query plan, `rows` its rows. */
    bool rows_take(const RadBatch* batch, const StepPlan& p, const QueryRows& rows) const {
        if (!op_r_gather || !p.select || p.DT != 0 || rows.ids || rows.n != p.T) return false;
        if (!GlmTp::splits(p.T, s.world) || p.T % s.world || p.T / s.world > w.run) return false;
        /* A run of kRowsNarrow rows or fewer would route its int8 GEMMs to the narrow tile
         * (libr4d's gemm_i8a8_nt_m16, M <= 64) where the head form's whole step takes the tiled
         * one: two kernels, no promise of one rounding. Such a step keeps heads. */
        if (p.T / s.world <= kRowsNarrow) return false;
        /* A run of the three-component rotary table ([3, T], component-major) is a column
         * slice at the table's pitch, which a raw operand cannot say: such a step keeps heads. */
        if (s.mrope && batch->rope_pos) return false;
        const auto* kb = kv_batch(batch, kv);
        return kb && !kb->stream;
    }

    /* The front: projections, norms, rotary and the cache store. */
    bool front(RadCtx* c, const RadBatch* batch, const RadBatch* queries, QueryRows rows,
               const StepPlan& p) const {
        const auto* kb = kv_batch(batch, kv);
        if (!kb || !kb->slot_mapping || !kb->block_table || !kb->seqused) {
            rad_step_fail(c, "hyv4flash: the latent cache's slot mapping, block table and "
                             "sequence lengths are missing");
            return false;
        }
        const int64_t T = batch->n_tok, L = s.kv_lora, R = a.qk_rope;
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        t_in_norm.step(c, brows(xa, T), T);
        const int64_t M = rows.n;
        /* kv_a can reuse the codes q_a normally makes. Produce their full plane even
         * when no query survives this chunk. A fused preceding junction already did so
         * when q_a itself is fed. */
        const bool full_codes = M < T && kv_a.fed && q_a.op_q && kv_a.q8 == q_a.q8;
        if (full_codes)
            RAD_ISSUE_N(c, q_a.op_q, T, brows(xa, T), brows(q_a.q8, T), brows(q_a.s8, T));
        rows.spans([&](int64_t src, int64_t dst, int64_t count) {
            q_a.step(c, act(xa), qa_out, T, src, count, dst, full_codes);
        });
        if (M) {
            if (qfuse)
                RAD_ISSUE_N(c, op_qa_norm, M, brows(qa_out, M), RAD_NONE, RAD_W(qa_gain),
                            brows(qcodes.q8, M), brows(qcodes.s8, M), brows(w.qa, M));
            else
                RAD_ISSUE_N(c, op_qa_norm, M, brows(qa_out, M), RAD_W(qa_gain), brows(w.qa, M));
            t_q_resid.step(c, brows(w.qa, M), M);
            if (p.rrows) {
                /* The codes the head form's q_nope would have written, over this rank's rows. */
                if (rq_nope.fed && !qfuse)
                    q_nope.quantise(c, q_nope.with_codes(act(w.qa)), M, p.r0, p.rrows);
                rq_nope.step(c, act(w.qa), w.r_q_nope, M, p.r0, p.rrows, 0);
                rq_rope.step(c, act(w.qa), w.r_q_rope, M, p.r0, p.rrows, 0);
            } else {
                q_nope.step(c, act(w.qa), w.q_nope, M);
                q_rope.step(c, act(w.qa), w.q_rope, M);
            }
        }
        kv_a.step(c, act(xa), w.row, T);
        const RadOperand rpos = rope_pos_of(s, batch, T);
        const bool qrope_folded = fold_qrope(p);
        const int64_t half = (L + R) / 2;
        /* The row's front in one launch (byte-identical to the norm, the rotary and the store). */
        const bool fused = op_row_front && !kb->stream;
        if (fused) {
            RAD_ISSUE_N(c, op_row_front, T, brows(w.row, T), RAD_W(kv_gain), rpos,
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
        } else {
            RAD_ISSUE_N(c, op_kv_norm, T, bcol(w.row, 0, L, T), RAD_W(kv_gain), bcol(w.row, 0, L, T));
            t_kv_latent.step(c, bcol(w.row, 0, L, T), T);
        }
        if (p.rrows)
            RAD_ISSUE_N(c, op_r_rope, p.rrows, brows(w.r_q_rope, p.rrows),
                        praw(queries->positions + p.r0, RAD_I32, p.rrows));
        else if (M && !qrope_folded)
            RAD_ISSUE_N(c, op_q_rope, M, brows(w.q_rope, M), rope_pos_of(s, queries, M));
        if (!fused) {
            RAD_ISSUE_N(c, op_k_rope, T, bcol(w.row, L, R, T), rpos);
            t_kv_rope.step(c, bcol(w.row, L, R, T), T);
        }
        if (!fused && !kb->stream) {
            RAD_ISSUE_N(c, op_store, T, bcol(w.row, 0, half, T), bcol(w.row, half, half, T),
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
        } else if (!fused) {
            /* The mirror always, the ring slot where the block is resident; the stage wait goes
             * right before the store (core/kvstream/stage.h), and a staged pass also stores at
             * block-id identity, which is how the stage is read. */
            const bool staged = kv_stage_wait(c, kv, kb, layer);
            const KVStreamLayer S = kv_stream_layer(c, kv, layer, kb);
            if (!op_store_stream || !S.host || S.blocks <= 0) {
                rad_step_fail(c, "hyv4flash: the sparse latent cache streams but the build has no "
                                 "kv_store_streamed row to write its host mirror with");
                return false;
            }
            RAD_ISSUE_N(c, op_store_stream, T, bcol(w.row, 0, half, T),
                        bcol(w.row, half, half, T), praw(kb->slot_mapping, RAD_I32, T),
                        kv_cache(kv, layer), praw(S.res, RAD_I32, S.blocks),
                        praw2(S.host, RAD_U8, S.blocks, S.bbytes));
            if (staged)
                RAD_ISSUE_N(c, op_store, T, bcol(w.row, 0, half, T), bcol(w.row, half, half, T),
                            praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
        }
        if (p.rrows) {
            rgate.step(c, act(xa), w.r_gate, T, p.r0, p.rrows, 0);
            return true;
        }
        rows.spans([&](int64_t src, int64_t dst, int64_t count) {
            gate.step(c, act(xa), w.gate, T, src, count, dst);
        });
        t_gate.step(c, brows(w.gate, M), M);
        return true;
    }


    /* The row form's attention and output over this rank's rows, gathered into w.o: the full
     * sum on every rank, as the all-reduce leaves it. */
    void core_rows(RadCtx* c, const StepPlan& p, rad_buf sel, rad_buf len) const {
        const int64_t n = p.rrows, r0 = p.r0, K = s.ix_group_topk, W = a.v_width(), E = s.n_embd;
        RAD_ISSUE_N(c, op_r_absorb, n, brows(w.r_q_nope, n), RAD_W(rkv_b), brows(w.r_q_lat, n));
        RAD_ISSUE_N(c, op_r_sel, n, brows(w.r_q_lat, n), brows(w.r_q_rope, n), kv_cache(kv, layer),
                    brow_slice(sel, r0, n, K), brow_slice(len, r0, n, 1), RAD_W(rsinks),
                    brows(w.r_lat, n));
        if (op_r_expand_gate) {
            RAD_ISSUE_N(c, op_r_expand_gate, n, brows(w.r_lat, n), RAD_W(rkv_b),
                        brows(w.r_gate, n), brows(w.r_ag, n));
        } else {
            RAD_ISSUE_N(c, op_r_expand, n, brows(w.r_lat, n), RAD_W(rkv_b), brows(w.r_ag, n));
            RAD_ISSUE_N(c, op_r_gate, n, brows(w.r_ag, n), brows(w.r_gate, n), brows(w.r_ag, n));
        }
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        /* The exact all-reduce's fold: +0, then each rank's bf16 partial in rank order, in f32. */
        RAD_ISSUE_N(c, op_r_fill, n, brows(w.r_acc, n));
        for (int j = 0; j < s.world; ++j) {
            RAD_ISSUE_N(c, op_r_cut, n, bcol(w.r_ag, j * W, W, n), brows(w.r_agj, n));
            rout[(size_t)j].step(c, act(w.r_agj), w.r_part, n);
            RAD_ISSUE_N(c, op_r_widen, n, brows(w.r_part, n), brows(w.r_tmp, n));
            RAD_ISSUE_N(c, op_r_add, n, brows(w.r_acc, n), brows(w.r_tmp, n), brows(w.r_acc, n));
        }
        RAD_ISSUE_N(c, op_r_narrow, n, brows(w.r_acc, n), brows(w.r_o, n));
        /* Every collective of a step rides one lane (abi/rad_runtime.h); a piped step's is lane 1. */
        if (p.rlane) {
            rad_lane(c, 1);
            rad_lane_join(c, 0, 1);
        }
        RAD_ISSUE_N(c, op_r_gather, n * E, brows(w.r_o, n), brows(w.o, s.world * n));
        if (p.rlane) {
            rad_lane(c, 0);
            rad_lane_join(c, 1, 0);
        }
    }

    /* The absorbed attention over the paged cache: the decode sequences [0, D) (rows [0, DT))
     * with the stable instance, then, when `prefill`, the prefill sequences [D, S) (rows
     * [DT, T)) with the fast one over their bounds rebased to 0 (`w.cu_p`, rebased once a step
     * by the trunk: Model::op_rebase; on the last layer's compact pass, the retained queries'
     * bounds: query_cu). */
    void paged(RadCtx* c, const RadBatch* batch, const StepPlan& p, bool prefill,
               const QueryRows& rows) const {
        const auto* kb = kv_batch(batch, kv);
        const int64_t pitch = kb->block_table_pitch, D = p.D, DT = p.DT, L = s.kv_lora,
                      R = a.qk_rope, H = a.heads;
        /* The rotary the decode rows fold (the stable instance's `rope_pos`): front() then issues
         * no `rope` launch, and w.q_rope holds the projection's columns. Absent (RAD_NONE) when
         * front() did not fold. */
        const RadOperand rpos = fold_qrope(p) ? rope_pos_of(s, batch, DT) : RAD_NONE;
        if (D > 0 && DT > 0)
            RAD_ISSUE_N(c, op_paged, DT, brows(w.q_lat, DT), brows(w.q_rope, DT),
                        kv_cache(kv, layer), praw2(kb->block_table, RAD_I32, D, pitch),
                        praw(kb->seqused, RAD_I32, D), praw(batch->cu_seqlens, RAD_I32, D + 1),
                        RAD_W(sinks), brows(w.lat, DT), rpos);
        const int64_t M = p.T - DT;
        if (!prefill || p.P <= 0 || M <= 0) return;
        RAD_ISSUE_N(c, op_paged_pf, M, brow_slice(w.q_lat, DT, M, H * L),
                    brow_slice(w.q_rope, DT, M, H * R), kv_cache(kv, layer),
                    praw2(kb->block_table + D * pitch, RAD_I32, p.P, pitch),
                    praw(kb->seqused + D, RAD_I32, p.P), query_cu(rows, p), RAD_W(sinks),
                    brow_slice(w.lat, DT, M, H * L));
    }

    /* The selection form: decode-class rows [0, DT) stable, prefill rows [DT, T) fast. */
    void selected(RadCtx* c, const RadBatch* batch, const StepPlan& p, rad_buf sel,
                  rad_buf len) const {
        const int64_t DT = p.DT, M = p.T - DT, L = s.kv_lora, R = a.qk_rope, H = a.heads,
                      K = s.ix_group_topk;
        /* As paged(): the decode rows' rotary is folded into the stable instance. */
        const RadOperand rpos = fold_qrope(p) ? rope_pos_of(s, batch, DT) : RAD_NONE;
        if (DT > 0)
            RAD_ISSUE_N(c, op_sel, DT, brows(w.q_lat, DT), brows(w.q_rope, DT),
                        kv_cache(kv, layer), brows(sel, DT), brows(len, DT), RAD_W(sinks),
                        brows(w.lat, DT), rpos);
        if (M > 0)
            RAD_ISSUE_N(c, op_sel_pf, M, brow_slice(w.q_lat, DT, M, H * L),
                        brow_slice(w.q_rope, DT, M, H * R), kv_cache(kv, layer),
                        brow_slice(sel, DT, M, K), brow_slice(len, DT, M, 1), RAD_W(sinks),
                        brow_slice(w.lat, DT, M, H * L));
    }

    /* The prefill queries' bounds rebased to 0: the compact pass's own (QueryRows::pf_cu), else
     * the step's (w.cu_p, which the trunk rebases once a step from the full batch). */
    RadOperand query_cu(const QueryRows& rows, const StepPlan& p) const {
        return rows.pf_cu ? praw(rows.pf_cu, RAD_I32, p.P + 1) : brows(w.cu_p, p.P + 1);
    }

    /* The non-absorbed attention over the prefill query rows [DT, T) (`p` the query plan), every
     * key this step's (`batch`'s prefill rows, bounded by w.cu_p). */
    void dense(RadCtx* c, const RadBatch* batch, const QueryRows& rows, const StepPlan& p) const {
        const int64_t T = p.T, DT = p.DT, M = T - DT, L = s.kv_lora, R = a.qk_rope,
                      H = a.heads, N = a.qk_nope, Q = a.qk();
        int64_t kd, k0;
        batch_split(batch, &kd, &k0);
        const int64_t nk = batch->n_tok - k0;
        RAD_ISSUE_N(c, op_lat_copy, nk, bcol_at(w.row, k0, L + R, 0, L, nk),
                    brow_slice(w.latc, k0, nk, L));
        RAD_ISSUE_N(c, op_kvb, nk, brow_slice(w.latc, k0, nk, L), RAD_W(kv_b),
                    brow_slice(w.kvb, k0, nk, a.kv_width()));
        RAD_ISSUE_N(c, op_split, nk, brow_slice(w.kvb, k0, nk, a.kv_width()),
                    bcol_at(w.row, k0, L + R, L, R, nk), brow_slice(w.k, k0, nk, H * Q),
                    brow_slice(w.v, k0, nk, H * a.v));
        RadOperand qn = brow_slice(w.q_full, DT, M, H * Q), qr = qn;
        qn.cols = N;
        qr.offset += N;
        qr.cols = R;
        RAD_ISSUE_N(c, op_qn_copy, M, brow_slice(w.q_nope, DT, M, H * N), qn);
        RAD_ISSUE_N(c, op_qr_copy, M, brow_slice(w.q_rope, DT, M, H * R), qr);
        RAD_ISSUE_N(c, op_dense, M, brow_slice(w.q_full, DT, M, H * Q),
                    brow_slice(w.k, k0, nk, H * Q), brow_slice(w.v, k0, nk, H * a.v),
                    query_cu(rows, p), brows(w.cu_p, p.P + 1), RAD_W(sinks),
                    brow_slice(w.ao, DT, M, a.v_width()));
    }

    /* Whether this layer's pass addresses a streamed group's ring: a ring smaller than the id
     * space, and the pass not on the stage. Such a pass must select (the file's head comment).
     * Called after front(), whose store already staged the layer if the pass stages. */
    bool ring_bound(RadCtx* c, const RadBatch* batch) const {
        if (sliding) return false;
        const auto* kb = kv_batch(batch, kv);
        const RadKVStream* st = kb ? kb->stream : nullptr;
        if (!st || st->block_bytes <= 0 || st->slots <= 0) return false;
        const int64_t blocks = st->host_layer_stride / st->block_bytes;
        return st->slots < blocks && !kv_stage_layer(c, kv, kb, layer);
    }

    /* The attention proper and everything after it. `sel`/`len`: the indexer's table, issued
     * already (sparse layers on a selecting step); `nsel` its entry count. `fused_tail`: stop
     * behind o_proj; the layer's Junction runs the reduction and the sandwich norm (and
     * tail_taps() the two taps behind them). */
    void core(RadCtx* c, const RadBatch* batch, const StepPlan& p, rad_buf sel, rad_buf len,
              rad_buf nsel, bool fused_tail, const RadBatch* queries,
              const QueryRows& rows) const {
        const int64_t T = p.T;
        const auto* kb = kv_batch(batch, kv);
        auto act = [](rad_buf x) {
            ActFP8 v{};
            v.x = x;
            return v;
        };
        if (p.rrows) {
            core_rows(c, p, sel, len);
            if (fused_tail) return;
            t_o_proj.step(c, brows(w.o, T), T);
            RAD_ISSUE_N(c, op_post_norm, T, brows(w.o, T), RAD_W(post_gain), brows(ya, T));
            t_out.step(c, brows(ya, T), T);
            return;
        }
        const bool selecting = !sliding && p.select;
        const int64_t abs_rows = selecting ? T : p.nonabs ? p.DT : T;
        /* G-3c: every row of a small step is absorbed, so one launch gives o_proj its codes. */
        const bool fused_q = op_expand_quant && abs_rows == T && T > 0 && T <= kFuseRows;
        if (abs_rows > 0) {
            RAD_ISSUE_N(c, op_absorb, abs_rows, brows(w.q_nope, abs_rows), RAD_W(kv_b),
                        brows(w.q_lat, abs_rows));
            if (selecting && kb && kb->stream && !kv_stage_layer(c, kv, kb, layer)) {
                /* After this layer's selection, before the attention reads `sel`. A pass on the
                 * stage reads block ids and never resolves. */
                const KVStreamLayer S = kv_stream_layer(c, kv, layer, kb);
                if (!op_resolve || !nsel || !S.host || S.blocks <= 0 || S.slots <= 0) {
                    rad_step_fail(c, "hyv4flash: the sparse latent cache streams but the build "
                                     "has no kv_resolve for its selection");
                    return;
                }
                kv_resolve_issue(c, op_resolve, S, brows(sel, T), brows(nsel, T), kv, layer, T);
            }
            if (selecting)
                selected(c, queries, p, sel, len);
            else
                paged(c, queries, p, !p.nonabs, rows);
            if (fused_q)
                RAD_ISSUE_N(c, op_expand_quant, T, brows(w.lat, T), RAD_W(kv_b), brows(w.gate, T),
                            brows(out.q8, T), brows(out.s8, T));
            else if (op_expand_gate)
                RAD_ISSUE_N(c, op_expand_gate, abs_rows, brows(w.lat, abs_rows), RAD_W(kv_b),
                            brows(w.gate, abs_rows), brows(w.ag, abs_rows));
            else
                RAD_ISSUE_N(c, op_expand, abs_rows, brows(w.lat, abs_rows), RAD_W(kv_b),
                            brows(w.ao, abs_rows));
        }
        if (p.nonabs && !selecting && p.T > p.DT) dense(c, batch, rows, p);
        /* the rows the fused expansion has not gated already */
        const int64_t g0 = op_expand_gate ? abs_rows : 0;
        if (g0 < T) {
            const int64_t vw = a.v_width();
            RAD_ISSUE_N(c, op_gate, T - g0, brow_slice(w.ao, g0, T - g0, vw),
                        brow_slice(w.gate, g0, T - g0, vw), brow_slice(w.ag, g0, T - g0, vw));
        }
        out.step(c, act(w.ag), w.o, T, 0, 0, -1, fused_q);
        if (fused_tail) return;
        issue_all_reduce(c, op_ar, w.o, T, s.n_embd);
        t_o_proj.step(c, brows(w.o, T), T);
        RAD_ISSUE_N(c, op_post_norm, T, brows(w.o, T), RAD_W(post_gain), brows(ya, T));
        t_out.step(c, brows(ya, T), T);
        /* After every attention launch of this layer (core/kvstream/stage.h). */
        if (kb && kb->stream) kv_stage_next(c, kv, layer);
    }
    void tail_taps(RadCtx* c, int64_t T) const {
        t_o_proj.step(c, brows(w.o, T), T);
        t_out.step(c, brows(ya, T), T);
    }
};

}  // namespace hyv4flash
#endif
