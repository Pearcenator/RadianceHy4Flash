/* tests/r4d_emu: Hy4-flash's small ops (r4d_ihc_bf16.hip, r4d_engram_bf16.hip, r4d_mla_kv_bf16.hip,
 * r4d_dfly.hip)
 * on the host SIMT emulator, each held to libref's op (libref/ref_hy4.cpp) on the same bytes.
 * ihc_post, mla_kv_split and Engram's ids, mask and window compare byte for byte; the others within
 * the bound their kernels state (one bf16 ulp of a differently ordered or fused sum). Row padding of
 * a pitched operand must come back untouched. idx_score's per-row and row-tiled forms
 * (r4d_idx_score_bf16.hip) compare with each other byte for byte. Needs libref (RADIANCE_HOME), as
 * test_kda. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "../hy4_dfly_cases.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>

static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) return nullptr;
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i)
        if (at(i)->op && !std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: return 4;
        case RAD_BF16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

struct Opd {
    uint32_t dt = RAD_BF16;
    std::vector<int64_t> shape;
    int64_t pitch = 0;
    std::vector<uint8_t> bytes;
    bool absent = false, out = false;
    double rtol = 0, atol = 0;
    int64_t elements() const {
        if (shape.size() == 2 && pitch) return shape[0] * pitch;
        int64_t n = 1;
        for (int64_t e : shape) n *= e;
        return n;
    }
    RadTensor tensor(void* data) const {
        RadTensor t{};
        t.data = absent ? nullptr : data;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i];
            t.stride[i] = st;
            st *= shape[(size_t)i];
        }
        if (shape.size() == 2 && pitch) t.stride[0] = pitch;
        return t;
    }
};
static Opd opd(uint32_t dt, std::vector<int64_t> shape, int64_t pitch = 0) {
    Opd o;
    o.dt = dt;
    o.shape = std::move(shape);
    o.pitch = pitch;
    o.bytes.assign((size_t)o.elements() * esz(dt), 0x7e);
    return o;
}
static Opd absent() { Opd o; o.absent = true; return o; }

static std::vector<int64_t> offsets(const RadTensor& t) {
    std::vector<int64_t> o;
    const int64_t d0 = t.shape[0], d1 = t.rank > 1 ? t.shape[1] : 1, d2 = t.rank > 2 ? t.shape[2] : 1;
    for (int64_t i = 0; i < d0; ++i)
        for (int64_t j = 0; j < d1; ++j)
            for (int64_t k = 0; k < d2; ++k)
                o.push_back(i * t.stride[0] + (t.rank > 1 ? j * t.stride[1] : 0) +
                            (t.rank > 2 ? k * t.stride[2] : 0));
    return o;
}
static void fill(Opd& o, const std::function<double(int64_t)>& f) {
    const RadTensor t = o.tensor(o.bytes.data());
    int64_t idx = 0;
    for (int64_t off : offsets(t)) {
        const double v = f(idx++);
        switch (o.dt) {
            case RAD_BF16: ((uint16_t*)o.bytes.data())[off] = f2b((float)v); break;
            case RAD_F32: ((float*)o.bytes.data())[off] = (float)v; break;
            case RAD_I32: ((int32_t*)o.bytes.data())[off] = (int32_t)v; break;
            case RAD_I64: ((int64_t*)o.bytes.data())[off] = (int64_t)v; break;
            default: break;
        }
    }
}
static double get(uint32_t dt, const uint8_t* p, int64_t off) {
    switch (dt) {
        case RAD_BF16: return b2f(((const uint16_t*)p)[off]);
        case RAD_F32: return ((const float*)p)[off];
        case RAD_I32: return ((const int32_t*)p)[off];
        case RAD_I64: return (double)((const int64_t*)p)[off];
        default: return ((const uint8_t*)p)[off];
    }
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float u() { return (float)(next() >> 8) / 8388608.0f - 1.0f; }
    float n() { return (u() + u() + u() + u()) * 0.866f; }
    uint32_t r(uint32_t m) { return next() % m; }
};

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static int g_fails = 0, g_cases = 0;
/* g_quiet: no line for a passing case. g_ref_shape: libref's RAD_E_SHAPE still compares (dfly_sel's
 * failed rows, which a debug libref also reports; the outputs are written either way). */
static bool g_quiet = false, g_ref_shape = false;

static void check(const char* op, RadLaunchFn kernel, const std::string& what, std::vector<RadParam> p,
                  std::vector<Opd>& ops) {
    ++g_cases;
    const RadKernelInfo* ref = libref_op(op);
    if (!ref) {
        std::printf("FAIL %s: libref's %s not loadable (set RADIANCE_HOME)\n", what.c_str(), op);
        ++g_fails;
        return;
    }
    const size_t n = ops.size();
    std::vector<std::vector<uint8_t>> a(n), b(n);
    std::vector<RadTensor> ta(n), tb(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = b[i] = ops[i].bytes;
        ta[i] = ops[i].tensor(a[i].data());
        tb[i] = ops[i].tensor(b[i].data());
    }
    RadArgs ar{}, ak{};
    ar.t = ta.data(); ar.n_t = (int)n; ar.p = p.data(); ar.n_p = (int)p.size(); ar.world_size = 1;
    ak.t = tb.data(); ak.n_t = (int)n; ak.p = p.data(); ak.n_p = (int)p.size(); ak.world_size = 1;
    const int rr = ref->launch(&ar, nullptr), rk = kernel(&ak, nullptr);
    if ((rr != RAD_OK && !(g_ref_shape && rr == RAD_E_SHAPE)) || rk != RAD_OK) {
        std::printf("FAIL %s: libref rc=%d kernel rc=%d\n", what.c_str(), rr, rk);
        ++g_fails;
        return;
    }
    int64_t bad = 0, total = 0;
    bool padding = true;
    for (size_t i = 0; i < n; ++i) {
        if (ops[i].absent) continue;
        std::vector<char> live(a[i].size() / esz(ops[i].dt), 0);
        for (int64_t off : offsets(ta[i])) {
            live[(size_t)off] = 1;
            if (!ops[i].out) continue;
            const double e = get(ops[i].dt, a[i].data(), off), g = get(ops[i].dt, b[i].data(), off);
            ++total;
            if (!std::isfinite(g) || std::fabs(g - e) > ops[i].atol + ops[i].rtol * std::fabs(e)) {
                if (!bad) std::printf("  first: operand %zu element %lld got %.9g want %.9g\n", i, (long long)off, g, e);
                ++bad;
            }
        }
        const size_t es = esz(ops[i].dt);
        for (size_t off = 0; off < live.size(); ++off)
            if (!live[off] && std::memcmp(&b[i][off * es], &ops[i].bytes[off * es], es)) padding = false;
        if (!ops[i].out && std::memcmp(b[i].data(), ops[i].bytes.data(), b[i].size())) padding = false;
    }
    if (bad || !padding || !total) {
        std::printf("FAIL %s: %lld of %lld out of bound%s\n", what.c_str(), (long long)bad,
                    (long long)total, padding ? "" : ", padding or an input written");
        ++g_fails;
    } else {
        if (!g_quiet) std::printf("ok   %s (%lld values)\n", what.c_str(), (long long)total);
    }
}

static const double kUlp = 0.0079;   /* one bf16 ulp, relative */

extern "C" void r4d_idx_score_pin_form(int form);
/* idx_score, each form pinned: the per-row form (1) and the row-tiled form (2) give the same bytes
 * (the same WMMA chain over the same fragments), and each is within 1e-4 of the row's largest
 * |score| of libref's (the products sum in another order) with -inf exactly where libref has it.
 * The fixtures cross the tiled form's 16-row and 128-key tiles: sequences that share a tile, a
 * one-row sequence, one with no live group, M not a multiple of 16, columns past the table, pages
 * absent or outside the cache, signed and zero gates, the paged and the flat key layout. */
static void idx_score_forms() {
    struct G { std::vector<int> qn, nc; int BT, SC, pages; bool paged; };
    const int64_t H = 16, n = 128;
    for (const G& g : { G{ { 5, 1, 70 }, { 250, 13, 300 }, 300, 290, 400, true },
                        G{ { 1, 37, 1, 3 }, { 9, 40, 0, 50 }, 48, 70, 120, false } }) {
        ++g_cases;
        Rng rng((uint32_t)(g.BT + g.SC));
        const int64_t S = (int64_t)g.qn.size();
        std::vector<int32_t> cu = { 0 };
        for (int q : g.qn) cu.push_back(cu.back() + q);
        const int64_t M = cu.back();
        std::vector<uint16_t> q((size_t)(M * H * n)), gate((size_t)(M * H)), key((size_t)(g.pages * n));
        for (auto& v : q) v = f2b(rng.n() * 0.5f);
        for (auto& v : gate) v = f2b(rng.r(10) == 0 ? 0.0f : rng.n());
        for (auto& v : key) v = f2b(rng.n() * 0.5f);
        std::vector<int32_t> bt((size_t)(S * g.BT)), nc(g.nc.begin(), g.nc.end());
        for (int64_t i = 0; i < (int64_t)bt.size(); ++i)
            bt[(size_t)i] = i % 23 == 7 ? -1 : i % 41 == 5 ? g.pages + 2 : (int32_t)((i * 37) % g.pages);
        auto tens = [](void* p, uint32_t dt, std::vector<int64_t> sh) {
            RadTensor t{};
            t.data = p; t.dtype = dt; t.rank = (uint32_t)sh.size();
            int64_t st = 1;
            for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[(size_t)i]; t.stride[i] = st; st *= sh[(size_t)i]; }
            return t;
        };
        std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("heads", H), pi("blocks", g.BT),
                                    pi("seqs", S), ps("dtype", "bf16") };
        auto launch = [&](RadLaunchFn fn, std::vector<float>& out) {
            out.assign((size_t)(M * g.SC), 7.0f);
            std::vector<RadTensor> t = {
                tens(q.data(), RAD_BF16, { M, H, n }), tens(gate.data(), RAD_BF16, { M, H }),
                g.paged ? tens(key.data(), RAD_BF16, { g.pages, 1, 4, n / 4 })
                        : tens(key.data(), RAD_BF16, { g.pages, n }),
                tens(bt.data(), RAD_I32, { S, g.BT }), tens(cu.data(), RAD_I32, { S + 1 }),
                tens(nc.data(), RAD_I32, { S }), tens(out.data(), RAD_F32, { M, g.SC }) };
            RadArgs a{};
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            return fn(&a, nullptr);
        };
        char w[120];
        std::snprintf(w, sizeof w, "idx_score M %lld, %d columns, %s keys: per-row = row-tiled, both ~ libref",
                      (long long)M, g.SC, g.paged ? "paged" : "flat");
        const RadKernelInfo* ref = libref_op("idx_score");
        if (!ref) { std::printf("FAIL %s: libref's idx_score not loadable (set RADIANCE_HOME)\n", w); ++g_fails; continue; }
        std::vector<float> want, row, tile;
        const int rr = launch(ref->launch, want);
        r4d_idx_score_pin_form(1);
        const int r1 = launch(r4d_rad_idx_score, row);
        r4d_idx_score_pin_form(2);
        const int r2 = launch(r4d_rad_idx_score, tile);
        r4d_idx_score_pin_form(0);
        if (rr != RAD_OK || r1 != RAD_OK || r2 != RAD_OK) {
            std::printf("FAIL %s: libref rc=%d per-row rc=%d tiled rc=%d\n", w, rr, r1, r2);
            ++g_fails;
            continue;
        }
        int64_t bad = 0, fin = 0, inf = 0;
        for (int64_t m = 0; m < M; ++m) {
            double mx = 0;
            for (int64_t b = 0; b < g.SC; ++b)
                if (std::isfinite(want[(size_t)(m * g.SC + b)])) mx = std::max(mx, (double)std::fabs(want[(size_t)(m * g.SC + b)]));
            for (int64_t b = 0; b < g.SC; ++b) {
                const float x = row[(size_t)(m * g.SC + b)], y = want[(size_t)(m * g.SC + b)];
                if (std::isinf(x) || std::isinf(y)) { bad += x != y; inf += std::isinf(y); continue; }
                ++fin;
                bad += std::fabs((double)x - y) > 1e-4 * (mx > 0 ? mx : 1.0);
            }
        }
        const bool same = std::memcmp(row.data(), tile.data(), row.size() * 4) == 0;
        if (bad || !same || !fin || !inf) {
            std::printf("FAIL %s: %lld of %lld out of bound, forms %s (%lld -inf)\n", w, (long long)bad,
                        (long long)(fin + inf), same ? "equal" : "DIFFER", (long long)inf);
            ++g_fails;
        } else {
            std::printf("ok   %s (%lld finite, %lld -inf)\n", w, (long long)fin, (long long)inf);
        }
    }
}

static void ihc_pre() {
    struct G { int64_t hc, n, M, pad; bool post; };
    for (const G& g : { G{ 4, 96, 3, 0, true }, G{ 4, 40, 2, 5, true }, G{ 2, 17, 3, 0, true },
                        G{ 3, 64, 2, 0, false }, G{ 8, 24, 2, 3, true } }) {
        Rng rng((uint32_t)(g.hc * 100 + g.n));
        const int64_t W = g.hc * g.n, rows = g.post ? 2 * g.hc : g.hc;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 2; });
        ops.push_back(opd(RAD_F32, { rows, W }));
        fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
        ops.push_back(opd(RAD_F32, { g.post ? 2 : 1 }));
        fill(ops[2], [&](int64_t i) { return i ? 2.1 : 3.0; });
        ops.push_back(opd(RAD_F32, { rows }));
        fill(ops[3], [&](int64_t) { return rng.n() * 0.7; });
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[4].out = true; ops[4].rtol = kUlp; ops[4].atol = 1e-5;
        if (g.post) {
            ops.push_back(opd(RAD_F32, { g.M, g.hc }, g.pad ? g.hc + 2 : 0));
            ops[5].out = true; ops[5].rtol = 2e-4; ops[5].atol = 1e-6;
        } else {
            ops.push_back(absent());
        }
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pi("rows", rows),
                                    pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
        if (g.post) p.push_back(pf("post_mag", 2.0));
        char w[120];
        std::snprintf(w, sizeof w, "ihc_pre hc %lld n %lld M %lld%s%s", (long long)g.hc, (long long)g.n,
                      (long long)g.M, g.post ? "" : " head", g.pad ? " pitched" : "");
        check("ihc_pre", r4d_rad_ihc_pre, w, p, ops);
    }
}

static void ihc_post() {
    struct G { int64_t hc, n, M, pad; };
    for (const G& g : { G{ 4, 64, 3, 0 }, G{ 4, 64, 2, 8 }, G{ 2, 17, 3, 0 }, G{ 3, 24, 2, 5 } }) {
        Rng rng((uint32_t)(g.hc * 10 + g.n));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_F32, { g.M, g.hc }));
        fill(ops[1], [&](int64_t) { return 0.5 + rng.u(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.hc * g.n }, g.pad ? g.hc * g.n + g.pad : 0));
        fill(ops[2], [&](int64_t) { return rng.n() * 2; });
        ops[2].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "ihc_post hc %lld n %lld M %lld%s, bit for bit", (long long)g.hc,
                      (long long)g.n, (long long)g.M, g.pad ? " pitched" : "");
        check("ihc_post", r4d_rad_ihc_post, w, p, ops);
    }
}

static void gate_mul() {
    struct G { int64_t M, n, pad; };
    for (const G& g : { G{ 3, 64, 0 }, G{ 2, 80, 8 }, G{ 3, 13, 0 }, G{ 2, 37, 3 } }) {
        Rng rng((uint32_t)(g.n * 3 + g.M));
        std::vector<Opd> ops;
        for (int i = 0; i < 2; ++i) {
            ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
            fill(ops.back(), [&](int64_t) { return rng.n() * (i ? 4 : 2); });
        }
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "gate_mul M %lld n %lld%s", (long long)g.M, (long long)g.n, g.pad ? " pitched" : "");
        check("gate_mul", r4d_rad_gate_mul, w, p, ops);
    }
}

static void mla_kv() {
    for (const char* mode : { "absorb", "expand" })
        for (int64_t M : { 1, 3, 9 }) {
            const bool ab = !std::strcmp(mode, "absorb");
            const int64_t H = 2, kd = 192, vd = 256, L = 512, xw = ab ? kd : L, yw = ab ? L : vd;
            Rng rng((uint32_t)(M + (ab ? 7 : 70)));
            std::vector<Opd> ops;
            ops.push_back(opd(RAD_BF16, { M, H, xw }));
            fill(ops[0], [&](int64_t) { return rng.n(); });
            ops.push_back(opd(RAD_BF16, { H * (kd + vd), L }));
            fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
            ops.push_back(opd(RAD_BF16, { M, H, yw }));
            ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 2e-5;
            std::vector<RadParam> p = { pi("M", M), pi("heads", H), pi("k_dim", kd), pi("v_dim", vd),
                                        pi("latent_dim", L), ps("mode", mode), ps("dtype", "bf16") };
            char w[100];
            std::snprintf(w, sizeof w, "mla_project_kv %s heads %lld M %lld", mode, (long long)H, (long long)M);
            check("mla_project_kv", r4d_rad_mla_project_kv, w, p, ops);
        }
    struct G { int64_t M, H, kd, vd, rd, pad; };
    for (const G& g : { G{ 3, 2, 16, 24, 8, 0 }, G{ 2, 3, 16, 8, 8, 8 }, G{ 3, 2, 13, 10, 6, 0 }, G{ 2, 2, 12, 9, 5, 3 } }) {
        Rng rng((uint32_t)(g.M * 7 + g.kd));
        const int64_t bw = g.H * (g.kd + g.vd), kw = g.H * (g.kd + g.rd), vw = g.H * g.vd;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, bw }, g.pad ? bw + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.rd }, g.pad ? g.rd + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, kw }, g.pad ? kw + g.pad : 0));
        ops.push_back(opd(RAD_BF16, { g.M, vw }, g.pad ? vw + g.pad : 0));
        ops[2].out = ops[3].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.H), pi("k_dim", g.kd), pi("v_dim", g.vd),
                                    pi("rope_dim", g.rd), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "mla_kv_split %lld/%lld/%lld heads %lld%s, bit for bit", (long long)g.kd,
                      (long long)g.vd, (long long)g.rd, (long long)g.H, g.pad ? " pitched" : "");
        check("mla_kv_split", r4d_rad_mla_kv_split, w, p, ops);
    }
}

static void engram_ids() {
    const int64_t heads = 4, ngram = 3, pad = 90, msize = 100, image = 95, video = 96;
    /* Form 4: a prefill whose last sequence hashes three tokens of its following prompt into
     * `ahead_ids` (the read-ahead), `tok` running on past the step's rows. */
    for (int form = 0; form < 5; ++form) {
        const bool decode = form & 1, wide = form & 2, ahd = form == 4;
        const int64_t A = ahd ? 3 : 0;
        const int64_t S = 3, W = decode ? 4 : 2;
        const std::vector<int64_t> Ts = decode ? std::vector<int64_t>{ 3, 1, 2 } : std::vector<int64_t>{ 4, 1, 3 };
        int64_t M = 0;
        for (int64_t t : Ts) M += t;
        Rng rng((uint32_t)(form + 3));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_I32, { M + A }));
        fill(ops[0], [&](int64_t) {
            const uint32_t r = rng.r(12);
            return r == 0 ? (double)pad : r == 1 ? (double)image : r == 2 ? (double)video : r == 3 ? 150.0 : (double)rng.r(90);
        });
        ops.push_back(opd(RAD_I32, { 4, W }));
        fill(ops[1], [&](int64_t) { return (double)rng.r(90); });
        ops[1].out = true;
        std::vector<int> cu = { 0 };
        for (int64_t t : Ts) cu.push_back(cu.back() + (int)t);
        ops.push_back(opd(RAD_I32, { S + 1 }));
        fill(ops[2], [&](int64_t i) { return cu[(size_t)i]; });
        ops.push_back(opd(RAD_I32, { M }));
        {
            int64_t i = 0;
            for (int64_t s = 0; s < S; ++s)
                for (int64_t t = 0; t < Ts[(size_t)s]; ++t, ++i)
                    ((int32_t*)ops[3].bytes.data())[i] = (int32_t)((s == 0 ? 0 : 500) + t);
        }
        ops.push_back(opd(RAD_I32, { msize }));
        fill(ops[4], [&](int64_t) { return (double)rng.r(60); });
        ops.push_back(opd(RAD_I64, { ngram }));
        { const int64_t m[3] = { 1000003, 7919000117ll, 0x5851F42D4C957F2Dll };
          for (int i = 0; i < 3; ++i) ((int64_t*)ops[5].bytes.data())[i] = m[i]; }
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[6], [&](int64_t i) { return (double)(1009 + 13 * i); });
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[7], [&](int64_t i) { return (double)(i * 2003); });
        if (decode) {
            ops.push_back(opd(RAD_I32, { S }));
            fill(ops[8], [&](int64_t i) { return (double)(1 + i % 3); });
        } else {
            ops.push_back(absent());
        }
        ops.push_back(opd(RAD_I32, { S }));
        { const int32_t slots[3] = { 3, -1, 0 };
          fill(ops[9], [&](int64_t i) { return (double)slots[i]; }); }
        ops.push_back(opd(wide ? RAD_I64 : RAD_I32, { M, heads }));
        ops.push_back(opd(RAD_I32, { M }));
        ops[10].out = ops[11].out = true;
        if (ahd) {
            ops.push_back(opd(RAD_I32, { A, heads }));
            ops[12].out = true;
        }
        std::vector<RadParam> p = { pi("M", M), pi("heads", heads), pi("ngram", ngram), pi("pad", pad),
                                    pi("map_size", msize), pi("image", image), pi("video", video) };
        char w[100];
        std::snprintf(w, sizeof w, "engram_ids %s%s, ids i%d, bit for bit", decode ? "decode" : "prefill",
                      ahd ? " with ahead_ids" : "", wide ? 64 : 32);
        check("engram_ids", r4d_rad_engram_ids, w, p, ops);
    }
}

static void engram_norm_gate() {
    struct N { int64_t M, heads, n, pad; double eps; };
    for (const N& g : { N{ 3, 4, 24, 0, 0.0078125 }, N{ 2, 3, 37, 5, 1.19e-7 } }) {
        Rng rng((uint32_t)(g.n));
        const int64_t W = g.heads * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_BF16, { W }));
        fill(ops[1], [&](int64_t) { return 1 + rng.n() * 0.3; });
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.heads), pi("n", g.n), pf("eps", g.eps),
                                    ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_norm heads %lld n %lld eps %g", (long long)g.heads, (long long)g.n, g.eps);
        check("engram_norm", r4d_rad_engram_norm, w, p, ops);
    }
    struct G { int64_t M, hc, n, pad; double eps; };
    for (const G& g : { G{ 4, 2, 48, 0, 0.0078125 }, G{ 3, 4, 20, 6, 1.19e-7 }, G{ 3, 1, 33, 0, 1e-6 } }) {
        Rng rng((uint32_t)(g.n * 5));
        const int64_t W = g.hc * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t i) { return i / W == 1 ? 0.0 : rng.n() * 2; });   /* row 1: d == 0 */
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_I32, { g.M }));
        fill(ops[2], [&](int64_t i) { return i % 3 == 2 ? 0.0 : 1.0; });
        for (int k = 0; k < 4; ++k) {
            ops.push_back(opd(RAD_BF16, { k == 2 ? g.n : W }));
            fill(ops.back(), [&](int64_t) { return 1 + rng.n() * 0.2; });
        }
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[7], [&](int64_t) { return rng.n() * 2; });
        ops[7].out = true; ops[7].rtol = 2 * kUlp; ops[7].atol = 0.016;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pf("eps", g.eps), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_gate hc %lld n %lld eps %g%s", (long long)g.hc, (long long)g.n, g.eps,
                      g.pad ? " pitched" : "");
        check("engram_gate", r4d_rad_engram_gate, w, p, ops);
    }
}

/* dfly_sel (r4d_dfly.hip): every golden case alone and all of one G in one call, then 1000 seeded
 * random geometries (sources -1 and past the snapshot, unheld rows, empty sequences, wide tables);
 * then 100 with tables past `blocks` (the rows fail to -1, len 0). Byte for byte, with
 * and without `count`; a pitched `out` keeps its padding and its columns past `blocks`. */
static void dfly_sel_case(const hy4dfly::Case& c, bool with_count, int64_t out_pad, const char* what) {
    std::vector<Opd> ops;
    auto put = [&](std::vector<int64_t> shape, const std::vector<int64_t>& v) {
        ops.push_back(opd(RAD_I32, std::move(shape)));
        fill(ops.back(), [&](int64_t i) { return (double)v[(size_t)i]; });
    };
    put({ c.R, c.G }, c.sel);
    put({ c.R }, c.nsel);
    put({ c.S }, c.src);
    put({ c.S, c.pitch }, c.bt);
    put({ c.M }, c.pos);
    put({ c.S + 1 }, c.cu);
    ops.push_back(opd(RAD_I32, { c.M, c.blocks }, out_pad ? c.blocks + out_pad : 0));
    ops.push_back(opd(RAD_I32, { c.M }));
    ops.push_back(with_count ? opd(RAD_I32, { c.M }) : absent());
    ops[6].out = ops[7].out = ops[8].out = true;
    std::vector<RadParam> p = { pi("M", c.M), pi("blocks_in", c.G), pi("blocks", c.blocks),
                                pi("ratio", c.ratio), pi("seqs", c.S) };
    check("dfly_sel", r4d_rad_dfly_sel, what, p, ops);
}

static void dfly_sel() {
    const int before = g_fails, cases = g_cases;
    g_quiet = true;
    for (int i = 0; i < kDflyCases; ++i)
        dfly_sel_case(hy4dfly::golden({ i }, 1000 + (uint64_t)i), i & 1, (i % 3) == 0 ? 3 : 0,
                      "dfly_sel golden case, one sequence");
    for (const int64_t G : { 4, 512 })
        dfly_sel_case(hy4dfly::golden(hy4dfly::golden_of_g(G), 77), true, 0,
                      G == 4 ? "dfly_sel every G 4 golden case in one call"
                             : "dfly_sel every G 512 golden case in one call");
    for (uint64_t seed = 1; seed <= 1000; ++seed)
        dfly_sel_case(hy4dfly::random(seed), seed & 1, (seed % 4) == 0 ? 2 : 0, "dfly_sel random geometry");
    g_ref_shape = true;
    for (uint64_t seed = 1; seed <= 100; ++seed)
        dfly_sel_case(hy4dfly::random(5000 + seed, true), true, 0, "dfly_sel random geometry, tables past blocks");
    g_ref_shape = false;
    g_quiet = false;
    std::printf("dfly_sel: %d case(s), %d failure(s)\n", g_cases - cases, g_fails - before);
}

/* ---- ihc_pre's order (r4d_ihc.h), restated on the host: 512 threads own the row's 8-value pieces
 * t, t + 512, ...; per thread the pieces and their values ascending; a wave's lanes by the xor
 * butterfly 16, 8, 4, 2, 1; the 16 waves from 0.0f in order. The kernel must give these bytes, on
 * the 16-byte form and on the scalar-load form alike. */
static float host_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static void host_ihc_pre_row(const uint16_t* h, const float* fn, int64_t fp, const float* scale,
                             const float* base, int hc, int n, bool post, float eps, float heps,
                             float pmag, uint16_t* x, float* prow) {
    const int R = post ? 2 * hc : hc, W = hc * n, T = 512;
    std::vector<float> acc((size_t)T * (R + 1), 0.0f);
    for (int p = 0; 8 * p < W; ++p) {
        const int t = p % T, c0 = 8 * p, cnt = W - c0 < 8 ? W - c0 : 8;
        float f[8];
        for (int e = 0; e < cnt; ++e) f[e] = b2f(h[c0 + e]);
        for (int j = 0; j < R; ++j)
            for (int e = 0; e < cnt; ++e) acc[(size_t)t * (R + 1) + j] += fn[(int64_t)j * fp + c0 + e] * f[e];
        for (int e = 0; e < cnt; ++e) acc[(size_t)t * (R + 1) + R] += f[e] * f[e];
    }
    std::vector<float> s((size_t)R + 1, 0.0f);
    for (int k = 0; k <= R; ++k) {
        float tot = 0.0f;
        for (int w = 0; w < T / 32; ++w) {
            float l[32];
            for (int i = 0; i < 32; ++i) l[i] = acc[(size_t)(w * 32 + i) * (R + 1) + k];
            for (int d = 16; d >= 1; d >>= 1) {
                float nl[32];
                for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
                std::memcpy(l, nl, sizeof l);
            }
            tot += l[0];
        }
        s[(size_t)k] = tot;
    }
    const float inv = 1.0f / sqrtf(s[(size_t)R] / (float)W + eps);
    float gate[16];
    for (int j = 0; j < hc; ++j) gate[j] = host_sigmoid(s[(size_t)j] * inv * scale[0] + base[j]) + heps;
    if (post)
        for (int j = 0; j < hc; ++j)
            prow[j] = pmag * host_sigmoid(s[(size_t)(hc + j)] * inv * scale[1] + base[hc + j]) + heps;
    for (int i = 0; i < n; ++i) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += gate[j] * b2f(h[j * n + i]);
        x[i] = f2b(v);
    }
}

static RadTensor t2(void* p, uint32_t dt, int64_t M, int64_t n, int64_t ld) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = M; t.shape[1] = n; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
template <class T>
static std::vector<T> aligned_like(size_t n) { return std::vector<T>(n + 64); }

static void ihc_pre_order() {
    struct G { int64_t hc, n, M, pad; bool post; };
    for (const G& g : { G{ 4, 3072, 2, 0, true }, G{ 4, 3072, 1, 3, true }, G{ 4, 512, 3, 0, true },
                        G{ 2, 128, 2, 0, true }, G{ 4, 512, 2, 0, false }, G{ 3, 37, 2, 1, true } }) {
        Rng rng((uint32_t)(g.hc * 7 + g.n + g.pad));
        const int64_t W = g.hc * g.n, rows = g.post ? 2 * g.hc : g.hc, hp = W + g.pad, xp = g.n + g.pad;
        std::vector<uint16_t> h((size_t)(g.M * hp)), xk((size_t)(g.M * xp), 0x7e7e), xh((size_t)(g.M * g.n));
        for (auto& v : h) v = f2b(rng.n() * 2);
        std::vector<float> fn((size_t)(rows * W)), base((size_t)rows), scale = { 3.0f, 2.1f };
        for (auto& v : fn) v = rng.n() * 0.05f;
        for (auto& v : base) v = rng.n() * 0.7f;
        std::vector<float> pk((size_t)(g.M * g.hc), 0.0f), ph((size_t)(g.M * g.hc), 0.0f);
        RadTensor t[6] = { t2(h.data(), RAD_BF16, g.M, W, hp), t2(fn.data(), RAD_F32, rows, W, W),
                           t1(scale.data(), RAD_F32, g.post ? 2 : 1), t1(base.data(), RAD_F32, rows),
                           t2(xk.data(), RAD_BF16, g.M, g.n, xp),
                           g.post ? t2(pk.data(), RAD_F32, g.M, g.hc, g.hc) : RadTensor{} };
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pi("rows", rows),
                                    pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
        if (g.post) p.push_back(pf("post_mag", 2.0));
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
        ++g_cases;
        char w[120];
        std::snprintf(w, sizeof w, "ihc_pre order hc %lld n %lld M %lld%s%s, bit for bit", (long long)g.hc,
                      (long long)g.n, (long long)g.M, g.post ? "" : " head", g.pad ? " pitched (scalar loads)" : "");
        if (r4d_rad_ihc_pre(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: launch\n", w); ++g_fails; continue; }
        long bad = 0;
        for (int64_t r = 0; r < g.M; ++r) {
            host_ihc_pre_row(h.data() + r * hp, fn.data(), W, scale.data(), base.data(), (int)g.hc, (int)g.n,
                             g.post, 1e-5f, 1e-6f, 2.0f, xh.data() + r * g.n, ph.data() + r * g.hc);
            for (int64_t i = 0; i < g.n; ++i) bad += xk[(size_t)(r * xp + i)] != xh[(size_t)(r * g.n + i)];
            if (g.post) bad += std::memcmp(pk.data() + r * g.hc, ph.data() + r * g.hc, (size_t)g.hc * 4) != 0;
        }
        if (bad) { std::printf("FAIL %s: %ld differ\n", w, bad); ++g_fails; }
        else std::printf("ok   %s\n", w);
    }
}

/* ---- rmsnorm_ihc_junction against its chain, byte for byte: the add (host, add_bf16's
 * arithmetic), rmsnorm in rmsnorm_bf16's order (host: kR4dSumLanes' partition, the wave butterfly,
 * the waves in order, rsqrtf as the emulator has it), the emulated ihc_post and ihc_pre kernels,
 * rmsnorm again. Every form: with and without the add, `y` / `x` stored or not, tail 0, the head. */
static void host_rmsnorm_row(const uint16_t* x, const float* g, int n, float eps, uint16_t* y) {
    float part[256];
    for (int t = 0; t < 256; ++t) {
        float ss = 0.0f;
        for (int i = t; i < n; i += 256) { const float v = b2f(x[i]); ss += v * v; }
        part[t] = ss;
    }
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) {
        float l[32];
        for (int i = 0; i < 32; ++i) l[i] = part[w * 32 + i];
        for (int d = 16; d >= 1; d >>= 1) {
            float nl[32];
            for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
            std::memcpy(l, nl, sizeof l);
        }
        total += l[0];
    }
    const float s = 1.0f / std::sqrt(total / (float)n + eps);
    const float wadd = 0.0f, mult = 1.0f;
    for (int i = 0; i < n; ++i) y[i] = f2b(b2f(x[i]) * s * (g[i] + wadd) * mult);
}

static void ihc_junction() {
    struct G { int64_t hc, n, M; bool add, ys, xs, tail, head; };
    const G cases[] = {
        { 4, 3072, 1, false, false, false, true, false }, { 4, 3072, 6, true, true, true, true, false },
        { 4, 512, 17, true, false, true, true, false },   { 4, 512, 6, false, true, false, true, true },
        { 2, 128, 6, true, true, true, false, false },    { 2, 128, 64, false, false, false, true, false },
        { 4, 512, 3, true, false, false, true, true },    { 8, 64, 2, false, true, true, true, false },
        { 2, 128, 1024, true, true, false, true, false },
    };
    for (const G& g : cases) {
        Rng rng((uint32_t)(g.hc * 13 + g.n + g.M));
        const int64_t M = g.M, n = g.n, hc = g.hc, W = hc * n, rows = g.head ? hc : 2 * hc;
        const bool post = !g.head;
        std::vector<uint16_t> A((size_t)(M * n)), B((size_t)(M * n)), H((size_t)(M * W));
        for (auto& v : A) v = f2b(rng.n() * 1.5f);
        for (auto& v : B) v = f2b(rng.n() * 1.5f);
        for (auto& v : H) v = f2b(rng.n() * 2);
        std::vector<float> wp((size_t)n), wn((size_t)n), P((size_t)(M * hc)), fn((size_t)(rows * W)),
            base((size_t)rows), scale = { 3.0f, 2.1f };
        for (auto& v : wp) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : wn) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : P) v = 0.5f + rng.u();
        for (auto& v : fn) v = rng.n() * 0.05f;
        for (auto& v : base) v = rng.n() * 0.7f;
        /* the fused kernel */
        auto Ak = A, Hk = H;
        auto Pk = P;
        std::vector<uint16_t> XNk((size_t)(M * n), 0x7e7e), Yk((size_t)(M * n), 0x7e7e), Xk((size_t)(M * n), 0x7e7e);
        RadTensor t[12] = {
            t2(Ak.data(), RAD_BF16, M, n, n), g.add ? t2(B.data(), RAD_BF16, M, n, n) : RadTensor{},
            t1(wp.data(), RAD_F32, n), t2(Pk.data(), RAD_F32, M, hc, hc), t2(Hk.data(), RAD_BF16, M, W, W),
            g.tail ? t2(fn.data(), RAD_F32, rows, W, W) : RadTensor{},
            g.tail ? t1(scale.data(), RAD_F32, post ? 2 : 1) : RadTensor{},
            g.tail ? t1(base.data(), RAD_F32, rows) : RadTensor{},
            g.tail ? t1(wn.data(), RAD_F32, n) : RadTensor{},
            g.tail ? t2(XNk.data(), RAD_BF16, M, n, n) : RadTensor{},
            g.ys ? t2(Yk.data(), RAD_BF16, M, n, n) : RadTensor{},
            g.xs && g.tail ? t2(Xk.data(), RAD_BF16, M, n, n) : RadTensor{} };
        std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("hc", hc), pi("rows", rows), pf("eps", 1e-6),
                                    pf("ihc_eps", 1e-5), pf("hc_eps", 1e-6), pi("tail", g.tail ? 1 : 0),
                                    ps("dtype", "bf16") };
        if (post) p.push_back(pf("post_mag", 2.0));
        RadArgs a{}; a.t = t; a.n_t = 12; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
        char w[160];
        std::snprintf(w, sizeof w, "rmsnorm_ihc_junction hc %lld n %lld M %lld%s%s%s%s vs the chain, bit for bit",
                      (long long)hc, (long long)n, (long long)M, g.add ? " add" : "", g.tail ? "" : " tail 0",
                      g.head ? " head" : "", g.ys || g.xs ? " y/x stored" : "");
        ++g_cases;
        if (r4d_rad_rmsnorm_ihc_junction(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: launch\n", w); ++g_fails; continue; }
        /* the chain */
        auto Ac = A, Hc = H;
        auto Pc = P;
        std::vector<uint16_t> Yc((size_t)(M * n)), Xc((size_t)(M * n)), XNc((size_t)(M * n));
        if (g.add)
            for (size_t i = 0; i < Ac.size(); ++i) Ac[i] = f2b(b2f(Ac[i]) + b2f(B[i]));
        for (int64_t r = 0; r < M; ++r) host_rmsnorm_row(Ac.data() + r * n, wp.data(), (int)n, 1e-6f, Yc.data() + r * n);
        {
            RadTensor tp[3] = { t2(Yc.data(), RAD_BF16, M, n, n), t2(Pc.data(), RAD_F32, M, hc, hc),
                                t2(Hc.data(), RAD_BF16, M, W, W) };
            std::vector<RadParam> q = { pi("M", M), pi("n", n), pi("hc", hc), ps("dtype", "bf16") };
            RadArgs b{}; b.t = tp; b.n_t = 3; b.p = q.data(); b.n_p = (int)q.size(); b.world_size = 1;
            if (r4d_rad_ihc_post(&b, nullptr) != RAD_OK) { std::printf("FAIL %s: chain ihc_post\n", w); ++g_fails; continue; }
        }
        if (g.tail) {
            RadTensor tp[6] = { t2(Hc.data(), RAD_BF16, M, W, W), t2(fn.data(), RAD_F32, rows, W, W),
                                t1(scale.data(), RAD_F32, post ? 2 : 1), t1(base.data(), RAD_F32, rows),
                                t2(Xc.data(), RAD_BF16, M, n, n),
                                post ? t2(Pc.data(), RAD_F32, M, hc, hc) : RadTensor{} };
            std::vector<RadParam> q = { pi("M", M), pi("n", n), pi("hc", hc), pi("rows", rows),
                                        pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
            if (post) q.push_back(pf("post_mag", 2.0));
            RadArgs b{}; b.t = tp; b.n_t = 6; b.p = q.data(); b.n_p = (int)q.size(); b.world_size = 1;
            if (r4d_rad_ihc_pre(&b, nullptr) != RAD_OK) { std::printf("FAIL %s: chain ihc_pre\n", w); ++g_fails; continue; }
            for (int64_t r = 0; r < M; ++r) host_rmsnorm_row(Xc.data() + r * n, wn.data(), (int)n, 1e-6f, XNc.data() + r * n);
        }
        long bad = 0;
        auto cmpv = [&](const void* x, const void* y, size_t bytes, const char* what) {
            if (std::memcmp(x, y, bytes)) { if (!bad) std::printf("  first difference in %s\n", what); ++bad; }
        };
        cmpv(Ak.data(), Ac.data(), Ac.size() * 2, "a");
        cmpv(Hk.data(), Hc.data(), Hc.size() * 2, "h");
        cmpv(Pk.data(), Pc.data(), Pc.size() * 4, "post");
        if (g.tail) cmpv(XNk.data(), XNc.data(), XNc.size() * 2, "xn");
        if (g.ys) cmpv(Yk.data(), Yc.data(), Yc.size() * 2, "y");
        else for (auto v : Yk) if (v != 0x7e7e) { cmpv("a", "b", 1, "y written though absent"); break; }
        if (g.xs && g.tail) cmpv(Xk.data(), Xc.data(), Xc.size() * 2, "x");
        if (bad) { std::printf("FAIL %s\n", w); ++g_fails; }
        else std::printf("ok   %s\n", w);
    }
}

/* The indexer's row split (hyv4flash_indexer.h, GLM's DsaCache split): rank r of W scores rows
 * [r R, r R + R) with idx_score row_part r and selects the rows it holds (GlmTp::run_of: a rank
 * past a small step selects its dead rows from 0) over the rebased bounds; the runs laid end to
 * end must be the replicated idx_score + qsa_select's bytes over the step's M rows: the table, the
 * counts, the lengths and the scores. The fixtures cross the tiled form's 256-row rule (replicated
 * tiled, parts per-row), leave a rank past the step, and select from more groups than topk. libref
 * is held to the same split when it serves row_part. */
static void idx_split() {
    struct G { std::vector<int> qn, ctx; int W; };
    const int64_t H = 16, n = 128, ratio = 4, GT = 16, BT = 130, pages = 600;
    const int32_t kSent = 0x5a5a5a5a;
    for (const G& g : { G{ { 70, 1, 200, 30 }, { 400, 120, 480, 64 }, 4 },
                        G{ { 5, 4 }, { 100, 60 }, 4 },
                        G{ { 3, 250, 40 }, { 260, 500, 90 }, 2 } }) {
        ++g_cases;
        const int64_t S = (int64_t)g.qn.size(), W = g.W;
        std::vector<int32_t> cu = { 0 }, pos, nc;
        for (int64_t s = 0; s < S; ++s) {
            cu.push_back(cu.back() + g.qn[(size_t)s]);
            for (int i = 0; i < g.qn[(size_t)s]; ++i) pos.push_back(g.ctx[(size_t)s] - g.qn[(size_t)s] + i);
            nc.push_back(g.ctx[(size_t)s] / (int)ratio);
        }
        const int64_t M = cu.back(), R = 4 * ((M + 4 * W - 1) / (4 * W)), Q = W * R;
        pos.resize((size_t)Q, 0);
        Rng rng((uint32_t)(M * 31 + W));
        std::vector<uint16_t> q((size_t)(Q * H * n)), gate((size_t)(Q * H)), key((size_t)(pages * n));
        for (auto& v : q) v = f2b(rng.n() * 0.5f);
        for (auto& v : gate) v = f2b(rng.r(10) == 0 ? 0.0f : rng.n());
        for (auto& v : key) v = f2b(rng.n() * 0.5f);
        std::vector<int32_t> bt((size_t)(S * BT));
        for (auto& v : bt) v = rng.r(29) == 3 ? -1 : (int32_t)rng.r((uint32_t)pages);
        auto tens = [](const void* p, uint32_t dt, std::vector<int64_t> sh) {
            RadTensor t{};
            t.data = const_cast<void*>(p); t.dtype = dt; t.rank = (uint32_t)sh.size();
            int64_t st = 1;
            for (int i = (int)sh.size() - 1; i >= 0; --i) { t.shape[i] = sh[(size_t)i]; t.stride[i] = st; st *= sh[(size_t)i]; }
            return t;
        };
        /* idx_score over the step (part < 0) or part `part`'s R rows. */
        auto score = [&](RadLaunchFn fn, int part, std::vector<float>& out, std::vector<int32_t>& cup) {
            const int64_t rows = part < 0 ? M : R, qr = part < 0 ? M : Q;
            out.assign((size_t)(rows * BT), 7.0f);
            cup.assign((size_t)(S + 1), kSent);
            std::vector<RadParam> p = { pi("M", rows), pi("n", n), pi("heads", H), pi("blocks", BT),
                                        pi("seqs", S), ps("dtype", "bf16") };
            if (part >= 0) p.push_back(pi("row_part", part));
            std::vector<RadTensor> t = {
                tens(q.data(), RAD_BF16, { qr, H, n }), tens(gate.data(), RAD_BF16, { qr, H }),
                tens(key.data(), RAD_BF16, { pages, 1, 4, n / 4 }), tens(bt.data(), RAD_I32, { S, BT }),
                tens(cu.data(), RAD_I32, { S + 1 }), tens(nc.data(), RAD_I32, { S }),
                tens(out.data(), RAD_F32, { rows, BT }) };
            if (part >= 0) t.push_back(tens(cup.data(), RAD_I32, { S + 1 }));
            RadArgs a{};
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            return fn(&a, nullptr);
        };
        /* qsa_select (signed, own) over `rows` score rows, the scratch the engine gives. */
        auto select = [&](const float* sc, int64_t rows, const int32_t* pp, const int32_t* cp,
                          int32_t* se, int32_t* ns, int32_t* su) {
            std::vector<RadParam> p = { pi("M", rows), pi("blocks", BT), pi("topk", GT - 1),
                                        pi("ratio", ratio), pi("seqs", S), pi("signed", 1), pi("own", 1) };
            std::vector<RadTensor> t = {
                tens(sc, RAD_F32, { rows, BT }), tens(bt.data(), RAD_I32, { S, BT }),
                tens(nc.data(), RAD_I32, { S }), tens(pp, RAD_I32, { rows }),
                tens(cp, RAD_I32, { S + 1 }), tens(se, RAD_I32, { rows, GT }),
                tens(ns, RAD_I32, { rows }), tens(su, RAD_I32, { rows }) };
            RadArgs a{};
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            const int64_t need = r4d_rad_qsa_select_scratch(&a);
            std::vector<uint32_t> scratch((size_t)(need / 4 + 1), 0xA5A5A5A5u);
            a.scratch = scratch.data(); a.scratch_bytes = need;
            return r4d_rad_qsa_select(&a, nullptr);
        };
        char w[140];
        std::snprintf(w, sizeof w, "indexer row split M %lld over %lld ranks (runs of %lld): gathered = replicated",
                      (long long)M, (long long)W, (long long)R);
        std::vector<float> sc_rep, sc_part, sc_gath((size_t)(Q * BT), 7.0f);
        std::vector<int32_t> cu_unused, cup, se_rep((size_t)(M * GT), kSent), ns_rep((size_t)M, kSent),
            su_rep((size_t)M, kSent), se_g((size_t)(Q * GT), kSent), ns_g((size_t)Q, kSent),
            su_g((size_t)Q, kSent);
        int rc = score(r4d_rad_idx_score, -1, sc_rep, cu_unused);
        if (rc == RAD_OK) rc = select(sc_rep.data(), M, pos.data(), cu.data(), se_rep.data(), ns_rep.data(), su_rep.data());
        bool cu_ok = true;
        for (int64_t r = 0; r < W && rc == RAD_OK; ++r) {
            int64_t first = r * R, rows = std::min(R, M - first);
            if (rows <= 0) { first = 0; rows = R; }   /* GlmTp::run_of */
            rc = score(r4d_rad_idx_score, (int)r, sc_part, cup);
            for (int64_t i = 0; i <= S; ++i) cu_ok &= cup[(size_t)i] == cu[(size_t)i] - (int32_t)(r * R);
            std::vector<int32_t> se((size_t)(R * GT), kSent), ns((size_t)R, kSent), su((size_t)R, kSent);
            if (rc == RAD_OK) rc = select(sc_part.data(), rows, pos.data() + first, cup.data(), se.data(), ns.data(), su.data());
            std::copy(sc_part.begin(), sc_part.end(), sc_gath.begin() + r * R * BT);
            std::copy(se.begin(), se.end(), se_g.begin() + r * R * GT);
            std::copy(ns.begin(), ns.end(), ns_g.begin() + r * R);
            std::copy(su.begin(), su.end(), su_g.begin() + r * R);
        }
        if (rc != RAD_OK) { std::printf("FAIL %s: launch rc=%d\n", w, rc); ++g_fails; continue; }
        int64_t full = 0, dead = 0;
        for (int64_t m = 0; m < M; ++m) full += ns_rep[(size_t)m] >= GT - 1 && pos[(size_t)m] / ratio > GT;
        for (int64_t m = M * BT; m < Q * BT; ++m) dead += !std::isinf(sc_gath[(size_t)m]);
        const bool same = !std::memcmp(se_g.data(), se_rep.data(), se_rep.size() * 4) &&
                          !std::memcmp(ns_g.data(), ns_rep.data(), ns_rep.size() * 4) &&
                          !std::memcmp(su_g.data(), su_rep.data(), su_rep.size() * 4);
        const bool sc_same = !std::memcmp(sc_gath.data(), sc_rep.data(), sc_rep.size() * 4);
        /* libref's row_part over the same runs, where it has one. */
        const char* ref_note = "libref not loaded";
        bool ref_ok = true;
        if (const RadKernelInfo* ref = libref_op("idx_score")) {
            std::vector<float> want, part_out;
            std::vector<int32_t> rcup;
            if (score(ref->launch, -1, want, cu_unused) != RAD_OK) { ref_ok = false; ref_note = "libref failed"; }
            else if (score(ref->launch, 0, part_out, rcup) != RAD_OK) ref_note = "libref without row_part";
            else {
                ref_note = "libref's parts = its step";
                for (int64_t r = 0; r < W && ref_ok; ++r) {
                    ref_ok = score(ref->launch, (int)r, part_out, rcup) == RAD_OK;
                    for (int64_t i = 0; i < R && ref_ok; ++i) {
                        const int64_t m = r * R + i;
                        for (int64_t b = 0; b < BT; ++b) {
                            const float x = part_out[(size_t)(i * BT + b)];
                            ref_ok &= m < M ? !std::memcmp(&x, &want[(size_t)(m * BT + b)], 4) : std::isinf(x) && x < 0;
                        }
                    }
                    for (int64_t i = 0; i <= S; ++i) ref_ok &= rcup[(size_t)i] == cu[(size_t)i] - (int32_t)(r * R);
                }
            }
        }
        if (!same || !sc_same || !cu_ok || dead || !full || !ref_ok) {
            std::printf("FAIL %s: tables %s, scores %s, cu_part %s, %lld live past the step, %lld full rows, %s%s\n",
                        w, same ? "equal" : "DIFFER", sc_same ? "equal" : "DIFFER", cu_ok ? "ok" : "WRONG",
                        (long long)dead, (long long)full, ref_note, ref_ok ? "" : " DIFFER");
            ++g_fails;
        } else {
            std::printf("ok   %s (%lld rows select %lld of more; %s)\n", w, (long long)full, (long long)GT, ref_note);
        }
    }
}

int main() {
    ihc_pre();
    ihc_post();
    ihc_pre_order();
    ihc_junction();
    gate_mul();
    mla_kv();
    engram_ids();
    engram_norm_gate();
    dfly_sel();
    idx_score_forms();
    idx_split();
    std::printf("%d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
