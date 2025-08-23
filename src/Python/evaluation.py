import os, time, math, statistics
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import ndf_py as _ndf
import model_G
import re
import matplotlib.pyplot as plt
# Make single-thread CPU runs stable/reproducible
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ----------------------------- Utilities -----------------------------

def disable_inplace(m: nn.Module) -> nn.Module:
    """Turn off inplace for all modules that have an 'inplace' flag (e.g., ReLU, SiLU, Dropout)."""
    for mod in m.modules():
        if hasattr(mod, "inplace") and getattr(mod, "inplace", False):
            mod.inplace = False
    return m

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _stats(arr):
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.median(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "n":       len(arr),
    }

def extract_head_tag_from_filename(fn: str) -> str | None:
    """
    Accept filenames like:
      HeadZ64_128x64_relu_do0.00_s3_best.pth
      HeadZ64_128x64_relu_do0_s0_best.pth
      HeadZ64_256x128x64_silu_do0.05_s4_best.pth
    Return the tag part (without seed/suffix) or None if not matched.
    """
    m = re.match(
        r'^(HeadZ64_[^_]+_(?:relu|silu)_do[0-9]+(?:\.[0-9]+)?)_s[0-9]+_best\.pth$',
        fn
    )
    return m.group(1) if m else None


def parse_head_tag(tag: str):
    """
    Parse 'HeadZ64_<h1>x<h2>x..._<act>_do<dropout>'
    Examples:
      HeadZ64_128x64_relu_do0
      HeadZ64_128x64_relu_do0.00
      HeadZ64_256x128x64_silu_do0.05
    """
    m = re.match(
        r'^HeadZ64_(?P<h>\d+(?:x\d+)*)_(?P<act>relu|silu)_do(?P<do>[0-9]+(?:\.[0-9]+)?)$',
        tag
    )
    if not m:
        raise ValueError(
            f"Invalid head tag: '{tag}'. Expected "
            "'HeadZ64_<h1>x<h2>..._<act>_do<dropout>' "
            "(e.g., HeadZ64_128x64_relu_do0.00)"
        )
    hidden = tuple(int(x) for x in m.group("h").split("x"))
    act = m.group("act")
    dropout = float(m.group("do"))
    return hidden, act, dropout

# ---------- Traditional G1 (C++ via pybind) ----------

def g1_traditional_eval(X_np):
    """
    X_np: (B, 260) float32 = [cosT, sinT, cosP, sinP, Dx(128), Dy(128)]
    Returns: (B,) float32
    """
    
    if hasattr(_ndf, "G1_batch"):
        y = _ndf.G1_batch(X_np.astype(np.float32))
        return np.asarray(y, dtype=np.float32).reshape(-1)
    else:
        # Fallback: per-sample loop (good for BS=1 latency; slower for large BS)
        B, C = X_np.shape
        nc = (C - 4) // 2
        y = np.empty((B,), dtype=np.float32)
        wh = _ndf.Vec3f(0.0, 0.0, 1.0)
        for i in range(B):
            cos_t, sin_t, cos_p, sin_p = X_np[i, :4]
            Dx = X_np[i, 4:4 + nc].tolist()
            Dy = X_np[i, 4 + nc:4 + 2 * nc].tolist()
            w = _ndf.Vec3f(float(sin_t * cos_p), float(sin_t * sin_p), float(cos_t))
            ndf = _ndf.PiecewiseLinearNDF(Dx, Dy)
            y[i] = float(ndf.G1(w, wh))
        return y

def benchmark_traditional_g1(sample_source, batch_sizes=(1, 2048), repeats=200, warmup=50):
    """
    CPU-only benchmark for traditional C++ G1 via pybind.
    Uses the same batch grid and statistics as NN benchmarks.
    Returns: {bs: {"mean_ms","p50_ms","p95_ms","throughput"}}
    """
    import numpy as _np, time
    results = {}
    # warmup
    for bs in batch_sizes:
        for _ in range(warmup):
            X = sample_source(bs).numpy()  # (bs,260) on CPU
            _ = g1_traditional_eval(X)
    # measure
    for bs in batch_sizes:
        times = []
        for _ in range(repeats):
            X = sample_source(bs).numpy()
            t0 = time.perf_counter()
            _ = g1_traditional_eval(X)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        t = _np.array(times, dtype=_np.float64)
        results[bs] = {
            "mean_ms": float(_np.mean(t)),
            "p50_ms":  float(_np.median(t)),
            "p95_ms":  float(_np.percentile(t, 95)),
            "throughput": float(bs / (_np.mean(t) / 1000.0)),
        }
    return results

def pretty_print_trad(title, results):
    print("=" * 80)
    print(f"{title}")
    for bs in sorted(results.keys()):
        r = results[bs]
        print(f"  batch={bs:4d}  mean={r['mean_ms']:8.3f} ms  "
              f"p50={r['p50_ms']:8.3f}  p95={r['p95_ms']:8.3f}  "
              f"throughput={r['throughput']:8.1f}/s")

def sanity_check_traditional_accuracy(x_te: np.ndarray, y_te: np.ndarray, n=20000):
    """Quick check: traditional G1 vs dataset y on a random subset."""
    import numpy as _np, torch
    N = min(n, x_te.shape[0])
    idx = torch.randperm(x_te.shape[0])[:N].numpy()
    X = x_te[idx].astype(_np.float32)
    Y = y_te[idx].astype(_np.float32).reshape(-1)
    pred = g1_traditional_eval(X)
    ae = _np.abs(pred - Y)
    mae = float(ae.mean())
    p95 = float(_np.percentile(ae, 95))
    print(f"[SANITY][Traditional G1 vs dataset y] N={N}  MAE={mae:.6g}  P95={p95:.6g}")

# ------------------------- Generic benchmarking -------------------------

def eval_metrics_on_loader(model: nn.Module,
                           loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           high_theta_threshold: float = 1.2):
    """Compute MAE / RMSE / P95 / HighThetaMAE for a full model over a DataLoader."""
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n_total = 0
    abs_err_chunks = []

    hi_mae_sum = 0.0
    hi_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float().view(-1, 1)

            pred = model(xb)
            err  = (pred - yb).view(-1)

            # accumulate MAE / RMSE
            ae = err.abs()
            mae_sum += ae.sum().item()
            mse_sum += (err * err).sum().item()
            n_total += ae.numel()
            abs_err_chunks.append(ae.detach().cpu())

            # High-theta mask: theta from cos/sin in the first two features
            cos_t = xb[:, 0].clamp(-1, 1).detach().cpu()
            sin_t = xb[:, 1].clamp(-1, 1).detach().cpu()
            theta = torch.atan2(sin_t, cos_t)
            mask  = theta >= high_theta_threshold
            if mask.any():
                hi_mae_sum += ae.detach().cpu()[mask].sum().item()
                hi_count   += int(mask.sum().item())

    mae  = mae_sum / max(1, n_total)
    rmse = float(math.sqrt(mse_sum / max(1, n_total)))

    abs_all = torch.cat(abs_err_chunks) if abs_err_chunks else torch.tensor([])
    p95  = torch.quantile(abs_all, 0.95).item() if abs_all.numel() else float("nan")
    hi_mae = (hi_mae_sum / hi_count) if hi_count > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "P95": p95, "HighThetaMAE": hi_mae, "N": n_total}

def benchmark_once(model, inputs, device, measure="device_only", use_amp=False):
    """One timing sample. No gradients. Optional AMP on CUDA."""
    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()

        if measure == "end_to_end":
            t0 = time.perf_counter()
            x = inputs.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (t0 := t0, time.perf_counter())[1] - t0

        if measure == "split":
            if device.type == "cuda":
                e_h0 = torch.cuda.Event(True); e_h1 = torch.cuda.Event(True)
                e_f0 = torch.cuda.Event(True); e_f1 = torch.cuda.Event(True)
                t0 = time.perf_counter()
                e_h0.record()
                x = inputs.to(device, non_blocking=True)
                e_h1.record()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        e_f0.record(); _ = model(x); e_f1.record()
                else:
                    e_f0.record(); _ = model(x); e_f1.record()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                return {
                    "h2d_ms":  e_h0.elapsed_time(e_h1),
                    "fwd_ms":  e_f0.elapsed_time(e_f1),
                    "total_ms": (t1 - t0) * 1000.0,
                }
            else:
                t0 = time.perf_counter()
                x = inputs.to(device)
                t_mid = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                return {
                    "h2d_ms":  (t_mid - t0) * 1000.0,
                    "fwd_ms":  (t1 - t_mid) * 1000.0,
                    "total_ms": (t1 - t0) * 1000.0,
                }

        # device_only
        x = inputs.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    e0.record(); _ = model(x); e1.record()
            else:
                e0.record(); _ = model(x); e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1)
        else:
            t0 = time.perf_counter(); _ = model(x); t1 = time.perf_counter()
            return (t1 - t0) * 1000.0

def run_benchmark(model, device, *, batch_sizes=(1, 2048), repeats=200, warmup=50,
                  measure="device_only", use_amp=False, sample_source=None):
    """Time a model across batch sizes. Returns per-batch statistics dict."""
    torch.set_grad_enabled(False)
    model.eval()

    def sample(bs):
        if sample_source is not None:
            return sample_source(bs)
        return torch.randn(bs, 260, dtype=torch.float32, device="cpu")

    results = {}
    for bs in batch_sizes:
        # warmup
        for _ in range(warmup):
            _ = benchmark_once(model, sample(bs), device, measure, use_amp)

        # measure
        if measure != "split":
            times = [benchmark_once(model, sample(bs), device, measure, use_amp) for _ in range(repeats)]
            results[bs] = _stats(times)
        else:
            h2d, fwd, total = [], [], []
            for _ in range(repeats):
                d = benchmark_once(model, sample(bs), device, measure, use_amp)
                h2d.append(d["h2d_ms"]); fwd.append(d["fwd_ms"]); total.append(d["total_ms"])
            results[bs] = {
                "h2d":   _stats(h2d),
                "fwd":   _stats(fwd),
                "total": _stats(total),
            }
    return results

def aggregate_bench_results(per_seed_results):
    """Aggregate multiple per-seed benchmark dicts into mean/std per batch."""
    def _agg_scalar_list(vals):
        if len(vals) == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "n": len(vals)}

    agg = {}
    all_bs = set()
    for R in per_seed_results:
        all_bs.update(R.keys())
    for bs in sorted(all_bs):
        any_stats = next((R[bs] for R in per_seed_results if bs in R), None)
        if isinstance(any_stats, dict) and "mean_ms" in any_stats:
            fields = {"mean_ms": [], "p50_ms": [], "p95_ms": []}
            for R in per_seed_results:
                if bs in R:
                    st = R[bs]
                    for k in fields.keys():
                        fields[k].append(st[k])
            agg[bs] = {k: _agg_scalar_list(v) for k, v in fields.items()}
            tput_list = [bs / (ms / 1000.0) for ms in fields["mean_ms"]]
            agg[bs]["throughput"] = _agg_scalar_list(tput_list)
        else:
            agg[bs] = {}
            for part in ("h2d", "fwd", "total"):
                fields = {"mean_ms": [], "p50_ms": [], "p95_ms": []}
                for R in per_seed_results:
                    if bs in R:
                        st = R[bs][part]
                        for k in fields.keys():
                            fields[k].append(st[k])
                agg[bs][part] = {k: _agg_scalar_list(v) for k, v in fields.items()}
            tput_list = [bs / (ms / 1000.0) for ms in [x["mean_ms"] for x in [R[bs]["total"] for R in per_seed_results if bs in R]]]
            agg[bs]["throughput"] = _agg_scalar_list(tput_list)
    return agg

def pretty_print_agg(title, device, params, agg):
    print("=" * 80)
    print(f"{title} | device={device.type} | params={params/1e6:.3f} M")
    for bs, stats in agg.items():
        if "mean_ms" in stats:
            m = stats["mean_ms"]; p50 = stats["p50_ms"]; p95 = stats["p95_ms"]; tp = stats["throughput"]
            print(f"  batch={bs:4d}  mean={m['mean']:8.3f}+/-{m['std']:6.3f} ms "
                  f" p50={p50['mean']:8.3f}+/-{p50['std']:6.3f}  p95={p95['mean']:8.3f}+/-{p95['std']:6.3f} "
                  f" throughput={tp['mean']:8.1f}+/-{tp['std']:6.1f} /s  (n={m['n']})")
        else:
            tot = stats["total"]; h2d = stats["h2d"]; fwd = stats["fwd"]; tp = stats["throughput"]
            print(f"  batch={bs:4d}  TOTAL  mean={tot['mean_ms']['mean']:8.3f}+/-{tot['mean_ms']['std']:6.3f} ms"
                  f"  p50={tot['p50_ms']['mean']:8.3f}+/-{tot['p50_ms']['std']:6.3f}"
                  f"  p95={tot['p95_ms']['mean']:8.3f}+/-{tot['p95_ms']['std']:6.3f}  throughput={tp['mean']:8.1f}+/-{tp['std']:6.1f}/s")
            print(f"             H2D    mean={h2d['mean_ms']['mean']:8.3f}+/-{h2d['mean_ms']['std']:6.3f}  "
                  f"p50={h2d['p50_ms']['mean']:8.3f}+/-{h2d['p50_ms']['std']:6.3f}  "
                  f"p95={h2d['p95_ms']['mean']:8.3f}+/-{h2d['p95_ms']['std']:6.3f}")
            print(f"             FWD    mean={fwd['mean_ms']['mean']:8.3f}+/-{fwd['mean_ms']['std']:6.3f}  "
                  f"p50={fwd['p50_ms']['mean']:8.3f}+/-{fwd['p50_ms']['std']:6.3f}  "
                  f"p95={fwd['p95_ms']['mean']:8.3f}+/-{fwd['p95_ms']['std']:6.3f}")


# --------------------------- Dataset helpers ---------------------------

def make_loaders_from_npz(npz_path, batch=2048, seed=42):
    npz = np.load(npz_path)
    x, y = npz["x"], npz["y"]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=seed)
    te_loader = torch.utils.data.DataLoader(model_G.G1Dataset(x_te, y_te), batch_size=batch, shuffle=False, pin_memory=True)
    return te_loader, (x_te, y_te)

def build_sample_source_from_test(x_te):
    N = x_te.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(0)
    X_cpu = torch.from_numpy(x_te).float()
    if torch.cuda.is_available():
        X_cpu = X_cpu.pin_memory()
    def sample(bs: int) -> torch.Tensor:
        idx = torch.randint(low=0, high=N, size=(bs,), generator=rng)
        return X_cpu[idx]
    return sample


# ----------------------- Head-only evaluation kit -----------------------

class _HeadOnly(nn.Module):
    """Wrapper that exposes only the trained head. Expects input (B, 4+z_dim)."""
    def __init__(self, head: nn.Module, angle_dim=4):
        super().__init__()
        self.head = head
        self.angle_dim = angle_dim
    def forward(self, az):
        angle = az[:, :self.angle_dim].contiguous()
        z     = az[:, self.angle_dim:].contiguous()
        return self.head(z, angle)

def build_head_only_from_ckpt(ckpt_path: str, z_dim: int,
                              angle_dim=4, head_hidden=(128,64),
                              act="relu", dropout=0.0):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = model_G.build_model(z_dim=z_dim, head_hidden=head_hidden,
                               act=act, dropout=dropout)
    full.load_state_dict(torch.load(ckpt_path, map_location=dev), strict=True)
    head = full.head.to(dev).eval()
    disable_inplace(head)
    return _HeadOnly(head, angle_dim=angle_dim).to(dev).eval()

def precompute_z_cache(ckpt_path: str, z_dim: int, x_te: np.ndarray, y_te: np.ndarray,
                       out_path: str, angle_dim: int = 4, batch: int = 4096):
    """Compute z on the test split once and save (angles, z, y) to NPZ."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model_G.build_model(z_dim=z_dim, head_hidden=(128,64), act="relu", dropout=0.0).to(dev).eval()
    m.load_state_dict(torch.load(ckpt_path, map_location=dev), strict=True)
    disable_inplace(m)

    X_cpu = torch.from_numpy(x_te.astype(np.float32))
    Y_cpu = torch.from_numpy(y_te.astype(np.float32)).view(-1, 1)
    if torch.cuda.is_available():
        X_cpu = X_cpu.pin_memory(); Y_cpu = Y_cpu.pin_memory()

    Z_list, ANG_list = [], []
    with torch.no_grad():
        for i in range(0, X_cpu.size(0), batch):
            xb = X_cpu[i:i+batch].to(dev, non_blocking=True)
            angle = xb[:, :angle_dim].detach().cpu().numpy()
            seq   = xb[:, angle_dim:].view(-1, 2, 128)
            z     = m.enc(seq).detach().cpu().numpy()
            Z_list.append(z); ANG_list.append(angle)
    Z   = np.concatenate(Z_list, 0).astype(np.float32)
    ANG = np.concatenate(ANG_list, 0).astype(np.float32)
    np.savez_compressed(out_path, z=Z, angles=ANG, y=y_te.astype(np.float32))
    print(f"[z-cache] saved -> {out_path}  z.shape={Z.shape}  angles.shape={ANG.shape}  y.shape={y_te.shape}")

def make_head_sample_source(z_cache_path: str):
    """Return a callable(bs)->Tensor that samples concatenated (angles,z) rows from the z-cache."""
    data = np.load(z_cache_path)
    ANG = torch.from_numpy(data["angles"].astype(np.float32))
    Z   = torch.from_numpy(data["z"].astype(np.float32))
    N   = ANG.size(0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    if torch.cuda.is_available():
        ANG = ANG.pin_memory(); Z = Z.pin_memory()
    def sample(bs: int) -> torch.Tensor:
        idx = torch.randint(0, N, (bs,), generator=gen)
        return torch.cat([ANG[idx], Z[idx]], dim=1)
    return sample

def eval_headonly_metrics(head_model: nn.Module, z_cache_path: str, high_theta_threshold=1.2):
    """Compute MAE/RMSE/P95/High-Theta MAE for head-only using the z-cache."""
    data = np.load(z_cache_path)
    ANG = torch.from_numpy(data["angles"].astype(np.float32))
    Z   = torch.from_numpy(data["z"].astype(np.float32))
    Y   = torch.from_numpy(data["y"].astype(np.float32)).view(-1, 1)
    X   = torch.cat([ANG, Z], dim=1)

    dev = next(head_model.parameters()).device
    bs  = 4096
    mae_sum = mse_sum = 0.0; n = 0
    abs_errs = []; hi_mae_sum = 0.0; hi_cnt = 0

    with torch.no_grad():
        for i in range(0, X.size(0), bs):
            xb = X[i:i+bs].to(dev, non_blocking=True)
            yb = Y[i:i+bs].to(dev, non_blocking=True)
            pred = head_model(xb)
            err  = (pred - yb).view(-1)
            ae   = err.abs()
            mae_sum += ae.sum().item()
            mse_sum += (err * err).sum().item()
            n += ae.numel()
            abs_errs.append(ae.detach().cpu())

            cos_t = xb[:, 0].clamp(-1, 1).detach().cpu()
            sin_t = xb[:, 1].clamp(-1, 1).detach().cpu()
            theta = torch.atan2(sin_t, cos_t)
            mask  = theta >= high_theta_threshold
            if mask.any():
                hi_mae_sum += ae.detach().cpu()[mask].sum().item()
                hi_cnt     += int(mask.sum().item())

    mae  = mae_sum / max(1, n)
    rmse = float(math.sqrt(mse_sum / max(1, n)))
    abs_all = torch.cat(abs_errs) if len(abs_errs) else torch.tensor([])
    p95  = torch.quantile(abs_all, 0.95).item() if abs_all.numel() else float("nan")
    hi_mae = (hi_mae_sum / hi_cnt) if hi_cnt > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "P95": p95, "HighThetaMAE": hi_mae, "N": n}

@torch.inference_mode()
def render_g1_maps(ndf_dx, ndf_dy, nn_model, device, W=256, H=128, save_prefix="vis",
                   vmin=0.0, vmax=1.0, cmap_val="inferno", cmap_err="magma", err_vmax=None):
    """
    Render G1(theta,phi) heatmaps for a given NDF using a FULL model that takes 260-D input,
    and compare against the traditional C++ implementation. Saves a side-by-side PNG.

    nn_model: expects input (B,260) = [cosT,sinT,cosP,sinP,Dx(128),Dy(128)]
    """
    theta = np.linspace(0, np.pi/2, H, endpoint=True, dtype=np.float32)   # 0..90 deg
    phi   = np.linspace(0, 2*np.pi,  W, endpoint=False, dtype=np.float32) # 0..360 deg
    TT, PP = np.meshgrid(theta, phi, indexing="ij")  # (H,W)

    cos_t = np.cos(TT).reshape(-1,1); sin_t = np.sin(TT).reshape(-1,1)
    cos_p = np.cos(PP).reshape(-1,1); sin_p = np.sin(PP).reshape(-1,1)
    Dx = np.broadcast_to(ndf_dx.reshape(1,-1), (H*W, 128))
    Dy = np.broadcast_to(ndf_dy.reshape(1,-1), (H*W, 128))
    X  = np.concatenate([cos_t, sin_t, cos_p, sin_p, Dx, Dy], axis=1).astype(np.float32)

    xb = torch.from_numpy(X).to(device)
    nn_model.eval()
    y_nn = nn_model(xb).detach().cpu().numpy().reshape(H, W)
    y_tr = g1_traditional_eval(X).reshape(H, W)

    err = np.abs(y_nn - y_tr)
    mae = float(err.mean()); p95 = float(np.percentile(err, 95))

    # draw
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    extent = [0, 360, 0, 90]  # phi deg, theta deg
    im0 = axs[0].imshow(y_nn, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap_val, extent=extent, aspect="auto"); axs[0].set_title("NN G1")
    im1 = axs[1].imshow(y_tr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap_val, extent=extent, aspect="auto"); axs[1].set_title("Traditional G1")
    im2 = axs[2].imshow(err, origin="lower",
                        vmin=0.0 if err_vmax is not None else None,
                        vmax=err_vmax,
                        cmap=cmap_err, extent=extent, aspect="auto")
    axs[2].set_title(f"|err|  MAE={mae:.3f}, P95={p95:.3f}")
    for ax in axs:
        ax.set_xlabel("phi(deg)")
    axs[0].set_ylabel("theta(deg)")

    cbar0 = plt.colorbar(im0, ax=axs[:2], fraction=0.046, pad=0.02)
    cbar0.set_label("G1 value")
    cbar1 = plt.colorbar(im2, ax=axs[2],  fraction=0.046, pad=0.02)
    cbar1.set_label("|err|")

    out = f"{save_prefix}.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"[VIS] saved -> {out}")

@torch.inference_mode()
def render_g1_maps_headonly(ndf_dx, ndf_dy, enc_ckpt_path: str, head_ckpt_path: str,
                            z_dim=64, head_hidden=(256,128), act="relu", dropout=0.0,
                            device=None, W=256, H=128, save_prefix="vis_head",
                            vmin=0.0, vmax=1.0, cmap_val="inferno", cmap_err="magma",err_vmax=None):
    """
    Render G1(theta,phi) using HEAD-ONLY path (angle + z), compare with traditional.
    - enc_ckpt_path: Encoder_64_sX_best.pth
    - head_ckpt_path: HeadZ64_<...>_sX_best.pth (must match seed of encoder)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) build encoder to get z from (Dx,Dy)
    full_for_enc = model_G.build_model(z_dim=z_dim, head_hidden=(128,64), act="relu", dropout=0.0).to(device).eval()
    full_for_enc.load_state_dict(torch.load(enc_ckpt_path, map_location=device), strict=True)

    seq = torch.from_numpy(np.stack([ndf_dx, ndf_dy], axis=0).astype(np.float32)).unsqueeze(0).to(device)  # (1,2,128)
    z = full_for_enc.enc(seq).detach().cpu().numpy().reshape(1, -1)  # (1, z_dim)

    # 2) build head-only model and load head weights
    head_full = model_G.build_model(z_dim=z_dim, head_hidden=head_hidden, act=act, dropout=dropout)
    head_full.load_state_dict(torch.load(head_ckpt_path, map_location="cpu"), strict=True)
    head_only = _HeadOnly(head_full.head, angle_dim=4).to(device).eval()
    disable_inplace(head_only)

    # 3) build angle grid and concatenate z
    theta = np.linspace(0, np.pi/2, H, endpoint=True, dtype=np.float32)
    phi   = np.linspace(0, 2*np.pi,  W, endpoint=False, dtype=np.float32)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    ANG = np.stack([np.cos(TT), np.sin(TT), np.cos(PP), np.sin(PP)], axis=-1).reshape(-1, 4).astype(np.float32)
    Z   = np.repeat(z.astype(np.float32), repeats=ANG.shape[0], axis=0)  # (H*W, z_dim)
    AZ  = np.concatenate([ANG, Z], axis=1)

    # NN (head-only) prediction
    y_nn = head_only(torch.from_numpy(AZ).to(device)).detach().cpu().numpy().reshape(H, W)

    # Traditional reference (use Dx,Dy + angles)
    Dx = np.broadcast_to(ndf_dx.reshape(1,-1), (H*W, 128))
    Dy = np.broadcast_to(ndf_dy.reshape(1,-1), (H*W, 128))
    X  = np.concatenate([ANG, Dx, Dy], axis=1).astype(np.float32)
    y_tr = g1_traditional_eval(X).reshape(H, W)

    # Error map
    err = np.abs(y_nn - y_tr)
    mae = float(err.mean()); p95 = float(np.percentile(err, 95))

    # draw
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    extent = [0, 360, 0, 90]
    im0 = axs[0].imshow(y_nn, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap_val, extent=extent, aspect="auto"); axs[0].set_title("NN G1 (head-only)")
    im1 = axs[1].imshow(y_tr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap_val, extent=extent, aspect="auto"); axs[1].set_title("Traditional G1")
    im2 = axs[2].imshow(err, origin="lower",
                        vmin=0.0 if err_vmax is not None else None,
                        vmax=err_vmax,
                        cmap=cmap_err, extent=extent, aspect="auto")
    axs[2].set_title(f"|err|  MAE={mae:.3f}, P95={p95:.3f}")
    for ax in axs:
        ax.set_xlabel("phi(deg)")
    axs[0].set_ylabel("theta(deg)")
    plt.colorbar(im0, ax=axs[:2], fraction=0.046, pad=0.02).set_label("G1 value")
    plt.colorbar(im2, ax=axs[2],  fraction=0.046, pad=0.02).set_label("|err|")

    out = f"{save_prefix}.png"
    plt.savefig(out, dpi=200); plt.close()
    print(f"[VIS] saved -> {out}")

def _compute_error_max_for_sample(Dx, Dy, full_ckpts, head_ckpts, device, W=256, H=128, pctl=99.0):
    import numpy as np, torch
    # angle grid
    theta = np.linspace(0, np.pi/2, H, endpoint=True, dtype=np.float32)
    phi   = np.linspace(0, 2*np.pi,  W, endpoint=False, dtype=np.float32)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    ANG = np.stack([np.cos(TT), np.sin(TT), np.cos(PP), np.sin(PP)], axis=-1).reshape(-1, 4).astype(np.float32)

    # traditional reference
    Dx_t = np.broadcast_to(Dx.reshape(1,-1), (ANG.shape[0], 128))
    Dy_t = np.broadcast_to(Dy.reshape(1,-1), (ANG.shape[0], 128))
    X260 = np.concatenate([ANG, Dx_t, Dy_t], axis=1).astype(np.float32)
    y_tr = g1_traditional_eval(X260).reshape(H, W)

    errs = []

    # full models
    for name, seed, ckpt in full_ckpts:
        m = _build_full_from_name(name).to(device).eval()
        m.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        with torch.inference_mode():
            y_nn = m(torch.from_numpy(X260).to(device)).cpu().numpy().reshape(H, W)
        errs.append(np.abs(y_nn - y_tr).ravel())

    # head-only
    for tag, seed, head_ckpt, enc_ckpt, hidden, act, dropout in head_ckpts:
        # get z from encoder
        full_for_enc = model_G.build_model(z_dim=64, head_hidden=(128,64), act="relu", dropout=0.0).to(device).eval()
        full_for_enc.load_state_dict(torch.load(enc_ckpt, map_location=device), strict=True)
        seq = torch.from_numpy(np.stack([Dx, Dy], axis=0).astype(np.float32)).unsqueeze(0).to(device)
        z = full_for_enc.enc(seq).detach().cpu().numpy().reshape(1, -1)
        AZ = np.concatenate([ANG, np.repeat(z, ANG.shape[0], axis=0)], axis=1).astype(np.float32)
        # head prediction
        head_full = model_G.build_model(z_dim=64, head_hidden=hidden, act=act, dropout=dropout)
        head_full.load_state_dict(torch.load(head_ckpt, map_location="cpu"), strict=True)
        head_only = _HeadOnly(head_full.head, angle_dim=4).to(device).eval()
        disable_inplace(head_only)
        with torch.inference_mode():
            y_nn = head_only(torch.from_numpy(AZ).to(device)).cpu().numpy().reshape(H, W)
        errs.append(np.abs(y_nn - y_tr).ravel())

    all_err = np.concatenate(errs, axis=0)

    return float(np.percentile(all_err, pctl))


def main():

    torch.backends.cudnn.benchmark = True
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    npz_path = "data/dataset/dataset_G_100k.npz"   
    te_loader, (x_te, y_te) = make_loaders_from_npz(npz_path, batch=2048, seed=42)
    sample_source = build_sample_source_from_test(x_te)
    
    SEEDS = [0,1,2,3,4]

    MODEL_SPECS = [
        ("FCNN",        lambda: model_G.G1FCNN(260)),              
        ("CNN",         lambda: model_G.G1CNN(260)),               
        ("CNN_GAP",     lambda: model_G.G1CNN_GAP(260)),           
        ("CNN_GAP_S",   lambda: model_G.G1CNN_GAP_S(260)),
        ("CNN_GAP_DW",  lambda: model_G.G1CNN_GAP_DW(260, width_mult=1.0)),
        ("CNN_GAP_DW_Wide", lambda: model_G.G1CNN_GAP_DW(260, width_mult=2.0)),
    ]

    MODEL_SPECS += [
        (f"Encoder_{z}", lambda z=z: model_G.build_model(
            z_dim=z, head_hidden=(128,64), act="relu", dropout=0.0))
        for z in [128,64,32,16,8,4,2,1,0]
    ]
    
    # ====== full model evaluation ======
    results_table = []  
    for name, ctor in MODEL_SPECS:

        ckpts = []
        for s in SEEDS:
            p = os.path.join("data", "model", f"{name}_s{s}_best.pth")
            if os.path.exists(p):
                ckpts.append((s, p))
        if not ckpts:
            print(f"[SKIP] {name}: no ckpt found (expect data/model/{name}_sX_best.pth)")
            continue

        # -- accuracy --
        maes, rmses, p95s, hi_maes = [], [], [], []
        params_ref = None
        for s, p in ckpts:
            m = ctor().to(device_cpu).eval()
            sd = torch.load(p, map_location="cpu")
            m.load_state_dict(sd, strict=True)
            if params_ref is None:
                params_ref = count_params(m)

            metrics = eval_metrics_on_loader(m, te_loader, device_cpu)  # MAE/RMSE/P95/HighThetaMAE
            maes.append(metrics["MAE"]); rmses.append(metrics["RMSE"])
            p95s.append(metrics["P95"]);  hi_maes.append(metrics["HighThetaMAE"])

            print(f"[TEST][{name}][seed={s}] "
                  f"MAE={metrics['MAE']:.8f}  RMSE={metrics['RMSE']:.8f}  "
                  f"P95={metrics['P95']:.8f}  HighThetaMAE={metrics['HighThetaMAE']:.8f}")

        # mean and std
        import statistics
        def mstd(x):  # return (mean, std)
            return (float(statistics.mean(x)), float(statistics.pstdev(x)) if len(x)>1 else 0.0)

        mae_m, mae_s   = mstd(maes)
        rmse_m, rmse_s = mstd(rmses)
        p95_m, p95_s   = mstd(p95s)
        htm_m, htm_s   = mstd(hi_maes)

        print(f"[AGG][{name}] seeds={len(ckpts)}  "
              f"MAE={mae_m:.8f}+/-{mae_s:.8f}  RMSE={rmse_m:.8f}+/-{rmse_s:.8f}  "
              f"P95={p95_m:.8f}+/-{p95_s:.8f}  HighTheta={htm_m:.8f}+/-{htm_s:.8f}  "
              f"Params={params_ref/1e6:.3f}M")

        results_table.append((name, len(ckpts), mae_m, mae_s, rmse_m, rmse_s, p95_m, p95_s, htm_m, htm_s, params_ref))

        # -- speed --
        # CPU
        cpu_runs = []
        params_ref = None
        for s, p in ckpts:
            m = ctor().to(device_cpu).eval()
            sd = torch.load(p, map_location="cpu")
            m.load_state_dict(sd, strict=True)
            if params_ref is None: params_ref = count_params(m)
            r = run_benchmark(m, device_cpu,
                              batch_sizes=(1, 2048), repeats=200, warmup=50,
                              measure="device_only", use_amp=False,
                              sample_source=sample_source)
            cpu_runs.append(r)
        cpu_agg = aggregate_bench_results(cpu_runs)
        pretty_print_agg(f"{name} | CPU device-only fp32 (5 seeds)", device_cpu, params_ref, cpu_agg)

        # CPU split
        cpu_split_runs = []
        for s, p in ckpts:
            m = ctor().to(device_cpu).eval()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            r = run_benchmark(m, device_cpu,
                              batch_sizes=(1,), repeats=200, warmup=50,
                              measure="split", use_amp=False,
                              sample_source=sample_source)
            cpu_split_runs.append(r)
        cpu_split_agg = aggregate_bench_results(cpu_split_runs)
        pretty_print_agg(f"{name} | CPU split fp32 (5 seeds)", device_cpu, params_ref, cpu_split_agg)

        # GPU
        if device_gpu is not None:
            gpu_runs = []
            for s, p in ckpts:
                mg = ctor().to(device_gpu).eval()
                mg.load_state_dict(torch.load(p, map_location=device_gpu))
                r = run_benchmark(mg, device_gpu,
                                  batch_sizes=(1, 2048), repeats=300, warmup=80,
                                  measure="device_only", use_amp=False,
                                  sample_source=sample_source)
                gpu_runs.append(r)
            gpu_agg = aggregate_bench_results(gpu_runs)
            pretty_print_agg(f"{name} | GPU device-only fp32 (5 seeds)", device_gpu, params_ref, gpu_agg)

            # AMP(fp16)
            gpu_amp_runs = []
            for s, p in ckpts:
                mg = ctor().to(device_gpu).eval()
                mg.load_state_dict(torch.load(p, map_location=device_gpu))
                r = run_benchmark(mg, device_gpu,
                                  batch_sizes=(1, 2048), repeats=300, warmup=80,
                                  measure="device_only", use_amp=True,
                                  sample_source=sample_source)
                gpu_amp_runs.append(r)
            gpu_amp_agg = aggregate_bench_results(gpu_amp_runs)
            pretty_print_agg(f"{name} | GPU device-only AMP(fp16) (5 seeds)", device_gpu, params_ref, gpu_amp_agg)

            # GPU split
            gpu_split_runs = []
            for s, p in ckpts:
                mg = ctor().to(device_gpu).eval()
                mg.load_state_dict(torch.load(p, map_location=device_gpu))
                r = run_benchmark(mg, device_gpu,
                                  batch_sizes=(1,), repeats=400, warmup=100,
                                  measure="split", use_amp=False,
                                  sample_source=sample_source)
                gpu_split_runs.append(r)
            gpu_split_agg = aggregate_bench_results(gpu_split_runs)
            pretty_print_agg(f"{name} | GPU split fp32 (5 seeds)", device_gpu, params_ref, gpu_split_agg)


    print("\n" + "="*80)
    print("Name, Nseeds, MAE_mean, MAE_std, RMSE_mean, RMSE_std, P95_mean, P95_std, HighThetaMAE_mean, HighThetaMAE_std, Params")
    for row in results_table:
        print(", ".join([row[0], str(row[1])] + [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row[2:]]))
    

    #====== head-only evaluation ======
    print("\n" + "=" * 80)
    print(">>> HEAD-ONLY evaluation (encoder offline; online path runs head only) <<<")

    ENC_Z_LIST = [128,64, 32, 16, 8, 4, 2, 1, 0]
    SEEDS = [0, 1, 2, 3, 4]
    BATCH_GRID = [1, 2048]   # same batches for CPU and GPU

    for z in ENC_Z_LIST:
        name = f"Encoder_{z}"

        # Collect checkpoints
        ckpts = []
        for s in SEEDS:
            p = os.path.join("data", "model", f"{name}_s{s}_best.pth")
            if os.path.exists(p):
                ckpts.append((s, p))
        if not ckpts:
            print(f"[HEAD-ONLY][SKIP] {name}: no ckpt")
            continue

        # Precompute z-cache per seed
        caches = []
        for s, p in ckpts:
            cache_path = os.path.join("data", "model", f"zcache_{name}_s{s}.npz")
            if not os.path.exists(cache_path):
                precompute_z_cache(p, z, x_te, y_te, cache_path, angle_dim=4, batch=4096)
            caches.append((s, p, cache_path))

        # Accuracy using head-only
        maes, rmses, p95s, hi_maes = [], [], [], []
        params_ref = None
        for s, p, cache in caches:
            head_only = build_head_only_from_ckpt(p, z_dim=z).to(device_cpu).eval()
            if params_ref is None:
                params_ref = count_params(head_only)
            M = eval_headonly_metrics(head_only, cache)
            maes.append(M["MAE"]); rmses.append(M["RMSE"]); p95s.append(M["P95"]); hi_maes.append(M["HighThetaMAE"])
            print(f"[HEAD-ONLY][{name}][seed={s}] MAE={M['MAE']:.8f}  RMSE={M['RMSE']:.8f}  P95={M['P95']:.8f}  HighThetaMAE={M['HighThetaMAE']:.8f}")

        def mstd(x):
            return (float(statistics.mean(x)), float(statistics.pstdev(x)) if len(x) > 1 else 0.0)

        mae_m, mae_s   = mstd(maes)
        rmse_m, rmse_s = mstd(rmses)
        p95_m, p95_s   = mstd(p95s)
        htm_m, htm_s   = mstd(hi_maes)
        print(f"[HEAD-ONLY][AGG][{name}] seeds={len(caches)}  "
              f"MAE={mae_m:.8f}+/-{mae_s:.8f}  RMSE={rmse_m:.8f}+/-{rmse_s:.8f}  "
              f"P95={p95_m:.8f}+/-{p95_s:.8f}  HighTheta={htm_m:.8f}+/-{htm_s:.8f}  "
              f"Params(head)={params_ref/1e6:.3f}M")

        # Speed: head-only, unified batch grid
        # CPU device-only
        cpu_runs = []
        for s, p, cache in caches:
            sample_src = make_head_sample_source(cache)
            m_cpu = build_head_only_from_ckpt(p, z_dim=z).to(device_cpu).eval()
            r = run_benchmark(m_cpu, device_cpu,
                              batch_sizes=tuple(BATCH_GRID), repeats=300, warmup=80,
                              measure="device_only", use_amp=False,
                              sample_source=sample_src)
            cpu_runs.append(r)
        cpu_agg = aggregate_bench_results(cpu_runs)
        pretty_print_agg(f"[HEAD-ONLY]{name} | CPU device-only fp32 (seeds={len(caches)})", device_cpu, params_ref, cpu_agg)

        # CPU split (BS=1 end-to-end)
        cpu_split_runs = []
        for s, p, cache in caches:
            sample_src = make_head_sample_source(cache)
            m_cpu = build_head_only_from_ckpt(p, z_dim=z).to(device_cpu).eval()
            r = run_benchmark(m_cpu, device_cpu,
                              batch_sizes=(1,), repeats=400, warmup=100,
                              measure="split", use_amp=False,
                              sample_source=sample_src)
            cpu_split_runs.append(r)
        cpu_split_agg = aggregate_bench_results(cpu_split_runs)
        pretty_print_agg(f"[HEAD-ONLY]{name} | CPU split fp32 (BS=1, seeds={len(caches)})", device_cpu, params_ref, cpu_split_agg)

        # GPU (optional)
        if device_gpu is not None:
            gpu_runs = []
            for s, p, cache in caches:
                sample_src = make_head_sample_source(cache)
                m_gpu = build_head_only_from_ckpt(p, z_dim=z).to(device_gpu).eval()
                r = run_benchmark(m_gpu, device_gpu,
                                  batch_sizes=tuple(BATCH_GRID), repeats=300, warmup=80,
                                  measure="device_only", use_amp=False,
                                  sample_source=sample_src)
                gpu_runs.append(r)
            gpu_agg = aggregate_bench_results(gpu_runs)
            pretty_print_agg(f"[HEAD-ONLY]{name} | GPU device-only fp32 (seeds={len(caches)})", device_gpu, params_ref, gpu_agg)

            # AMP (fp16)
            gpu_amp_runs = []
            for s, p, cache in caches:
                sample_src = make_head_sample_source(cache)
                m_gpu = build_head_only_from_ckpt(p, z_dim=z).to(device_gpu).eval()
                r = run_benchmark(m_gpu, device_gpu,
                                  batch_sizes=tuple(BATCH_GRID), repeats=300, warmup=80,
                                  measure="device_only", use_amp=True,
                                  sample_source=sample_src)
                gpu_amp_runs.append(r)
            gpu_amp_agg = aggregate_bench_results(gpu_amp_runs)
            pretty_print_agg(f"[HEAD-ONLY]{name} | GPU device-only AMP(fp16) (seeds={len(caches)})", device_gpu, params_ref, gpu_amp_agg)

            # GPU split (BS=1)
            gpu_split_runs = []
            for s, p, cache in caches:
                sample_src = make_head_sample_source(cache)
                m_gpu = build_head_only_from_ckpt(p, z_dim=z).to(device_gpu).eval()
                r = run_benchmark(m_gpu, device_gpu,
                                  batch_sizes=(1,), repeats=400, warmup=100,
                                  measure="split", use_amp=False,
                                  sample_source=sample_src)
                gpu_split_runs.append(r)
            gpu_split_agg = aggregate_bench_results(gpu_split_runs)
            pretty_print_agg(f"[HEAD-ONLY]{name} | GPU split fp32 (BS=1, seeds={len(caches)})", device_gpu, params_ref, gpu_split_agg)

 # ----- Traditional G1 CPU benchmark (fair CPU-vs-CPU) -----
    print("\n" + "="*80)
    print(">>> Traditional G1 (C++ via pybind) vs head-only (CPU) <<<")

    # Reuse the same batch grid you used for NN head-only
    BATCH_GRID = [1,2048]

    # Build a sample_source that returns (bs,260) from the test split
    sample_source_full = build_sample_source_from_test(x_te)

    # Optional: sanity-check accuracy
    sanity_check_traditional_accuracy(x_te, y_te, n=20000)

    # Benchmark traditional C++ G1 on CPU
    trad_res = benchmark_traditional_g1(sample_source_full,
                                        batch_sizes=tuple(BATCH_GRID),
                                        repeats=200, warmup=50)
    pretty_print_trad("Traditional G1 | CPU (aligned batches)", trad_res)

    # (Optional) Print a quick side-by-side with your chosen head-only model (e.g., Encoder_64)
    chosen_z = 64
    chosen_ckpts = [p for s in [0,1,2,3,4]
                    for p in [os.path.join("data", "model", f"Encoder_{chosen_z}_s{s}_best.pth")]
                    if os.path.exists(p)]
    if len(chosen_ckpts) > 0:
        # Aggregate CPU head-only device-only results for the chosen z
        cpu_runs = []
        for p in chosen_ckpts:
            m_cpu = build_head_only_from_ckpt(p, z_dim=chosen_z).to(torch.device("cpu")).eval()
            r = run_benchmark(m_cpu, torch.device("cpu"),
                              batch_sizes=tuple(BATCH_GRID),
                              repeats=200, warmup=50,
                              measure="device_only", use_amp=False,
                              sample_source=make_head_sample_source(
                                  os.path.join("data", "model", f"zcache_Encoder_{chosen_z}_s{p.split('_s')[-1].split('_')[0]}.npz")
                              ))
            cpu_runs.append(r)
        cpu_agg = aggregate_bench_results(cpu_runs)
        pretty_print_agg(f"[HEAD-ONLY] Encoder_{chosen_z} | CPU device-only fp32 (seeds={len(chosen_ckpts)})",
                         torch.device("cpu"), count_params(build_head_only_from_ckpt(chosen_ckpts[0], chosen_z)), cpu_agg)


  # ====== head-search evaluation (z=64) ======
    print("\n" + "="*80)
    print(">>> HEAD-SEARCH (z=64) evaluation on test split <<<")

    # collect all head ckpts of the naming pattern
    model_dir = os.path.join("data", "model")
    all_files = [fn for fn in os.listdir(model_dir)
             if fn.startswith("HeadZ64_") and fn.endswith("_best.pth")]

    # collect valid tags
    head_tags = []
    seen = set()
    for fn in all_files:
        tag = extract_head_tag_from_filename(fn)
        if tag is None:
            print(f"[WARN] skip (bad name): {fn}")
            continue
        try:
            parse_head_tag(tag)  # validate
            if tag not in seen:
                head_tags.append(tag)
                seen.add(tag)
        except ValueError as e:
            print(f"[WARN] skip (bad tag): {fn} -> {e}")

    # ensure z-cache for Encoder_64
    caches = {}
    for s in [0,1,2,3,4]:
        enc_ckpt = os.path.join("data","model", f"Encoder_64_s{s}_best.pth")
        cache = os.path.join("data","model", f"zcache_Encoder_64_s{s}.npz")
        if os.path.exists(enc_ckpt) and not os.path.exists(cache):
            precompute_z_cache(enc_ckpt, 64, x_te, y_te, cache, angle_dim=4, batch=4096)
        if os.path.exists(cache):
            caches[s] = cache

    for tag in head_tags:
        hidden, act, dropout = parse_head_tag(tag)
        # gather seeds
        ckpts = []
        for s in [0,1,2,3,4]:
            p = os.path.join("data","model", f"{tag}_s{s}_best.pth")
            if os.path.exists(p) and s in caches:
                ckpts.append((s, p, caches[s]))
        if not ckpts:
            print(f"[HEAD-SEARCH][SKIP] {tag}: no ckpt or no z-cache")
            continue

        # accuracy
        maes, rmses, p95s, hi_maes = [], [], [], []
        for s, p, cache in ckpts:
            head_only = build_head_only_from_ckpt(p, z_dim=64,
                                head_hidden=hidden, act=act, dropout=dropout).to(device_cpu).eval()
            M = eval_headonly_metrics(head_only, cache)
            maes.append(M["MAE"]); rmses.append(M["RMSE"])
            p95s.append(M["P95"]); hi_maes.append(M["HighThetaMAE"])
            print(f"[HEAD-SEARCH][{tag}][seed={s}] MAE={M['MAE']:.8f}  RMSE={M['RMSE']:.8f}  "
                  f"P95={M['P95']:.8f}  HighThetaMAE={M['HighThetaMAE']:.8f}")

        m = lambda a:(float(statistics.mean(a)), float(statistics.pstdev(a)) if len(a)>1 else 0.0)
        mae_m, mae_s = m(maes); rmse_m, rmse_s = m(rmses); p95_m, p95_s = m(p95s); htm_m, htm_s = m(hi_maes)
        print(f"[HEAD-SEARCH][AGG][{tag}] seeds={len(ckpts)}  "
              f"MAE={mae_m:.8f}+/-{mae_s:.8f}  RMSE={rmse_m:.8f}+/-{rmse_s:.8f}  "
              f"P95={p95_m:.8f}+/-{p95_s:.8f}  HighTheta={htm_m:.8f}+/-{htm_s:.8f}")

        # speed (CPU device-only; batch=1 & 2048)
        cpu_runs = []
        for s, p, cache in ckpts:
            head_only = build_head_only_from_ckpt(p, z_dim=64,
                            head_hidden=hidden, act=act, dropout=dropout).to(device_cpu).eval()
            sample_src = make_head_sample_source(cache)
            r = run_benchmark(head_only, device_cpu,
                              batch_sizes=(1,2048), repeats=300, warmup=80,
                              measure="device_only", use_amp=False,
                              sample_source=sample_src)
            cpu_runs.append(r)
        cpu_agg = aggregate_bench_results(cpu_runs)
        pretty_print_agg(f"[HEAD-SEARCH]{tag} | CPU device-only fp32", device_cpu,
                         count_params(head_only), cpu_agg)

def render_case_full(model_name="Encoder_64", seed=0, sample_idx=0, out="vis_full"):
    npz = np.load("data/dataset/dataset_G_100k.npz")
    x, y = npz["x"], npz["y"]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42)
    Dx = x_te[sample_idx][4:4+128]; Dy = x_te[sample_idx][4+128:4+256]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name.startswith("Encoder_"):
        z = int(model_name.split("_")[1])
        ctor = lambda: model_G.build_model(z_dim=z, head_hidden=(128,64), act="relu", dropout=0.0)
    else:
        ctor_map = {
            "FCNN": lambda: model_G.G1FCNN(260),
            "CNN": lambda: model_G.G1CNN(260),
            "CNN_GAP": lambda: model_G.G1CNN_GAP(260),
            "CNN_GAP_S": lambda: model_G.G1CNN_GAP_S(260),
        }
        if hasattr(model_G, "G1CNN_GAP_DW"):
            ctor_map["CNN_GAP_DW"] = lambda: model_G.G1CNN_GAP_DW(260, width_mult=1.0)
        ctor = ctor_map[model_name]
    m = ctor().to(device).eval()
    ckpt = f"data/model/{model_name}_s{seed}_best.pth"
    m.load_state_dict(torch.load(ckpt, map_location=device))
    render_g1_maps(Dx, Dy, m, device, W=256, H=128, save_prefix=out)

def render_case_head(tag="HeadZ64_256x128_relu_do0.00", seed=0, enc_seed=None, sample_idx=0, out="vis_head"):
    """
    tag must match your head ckpt name prefix (e.g., HeadZ64_256x128_relu_do0.00)
    seed chooses which _s{seed}_best.pth to load.
    enc_seed defaults to seed if not provided.
    """
    if enc_seed is None:
        enc_seed = seed
    npz = np.load("data/dataset/dataset_G_100k.npz")
    x, y = npz["x"], npz["y"]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42)
    Dx = x_te[sample_idx][4:4+128]; Dy = x_te[sample_idx][4+128:4+256]

    # parse z & head spec from tag
    hidden, act, dropout = parse_head_tag(tag)
    z_dim = 64  # tag is HeadZ64_...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_ckpt  = os.path.join("data","model", f"Encoder_{z_dim}_s{enc_seed}_best.pth")
    head_ckpt = os.path.join("data","model", f"{tag}_s{seed}_best.pth")
    render_g1_maps_headonly(Dx, Dy, enc_ckpt, head_ckpt, z_dim=z_dim,
                            head_hidden=hidden, act=act, dropout=dropout,
                            device=device, W=256, H=128, save_prefix=out)

def _list_full_model_ckpts(model_dir="data/model"):
    """
    Find full-model checkpoints by naming convention: <ModelName>_s<seed>_best.pth
    ModelName in {FCNN, CNN, CNN_GAP, CNN_GAP_S, CNN_GAP_DW?, Encoder_<z>}
    Returns: list of (model_name, seed:int, ckpt_path)
    """
    out = []
    for fn in os.listdir(model_dir):
        m = re.match(r'^(FCNN|CNN|CNN_GAP|CNN_GAP_S|CNN_GAP_DW|Encoder_\d+)_s(\d+)_best\.pth$', fn)
        if not m: 
            continue
        name, s = m.group(1), int(m.group(2))
        out.append((name, s, os.path.join(model_dir, fn)))
    return sorted(out)

def _list_head_ckpts_with_encoder(model_dir="data/model"):
    """
    Find head-only checkpoints and match encoder ckpt with the same seed.
    Head ckpt: HeadZ64_<...>_s<seed>_best.pth  (z=64 assumed)
    Encoder ckpt to match: Encoder_64_s<seed>_best.pth
    Returns: list of (tag, seed:int, head_ckpt, enc_ckpt, hidden, act, dropout)
    """
    heads = []
    for fn in os.listdir(model_dir):
        m = re.match(r'^(HeadZ64_[^_]+_(?:relu|silu)_do\d+\.\d+)_s(\d+)_best\.pth$', fn)
        if not m: 
            continue
        tag, s = m.group(1), int(m.group(2))
        head_ckpt = os.path.join(model_dir, fn)
        enc_ckpt  = os.path.join(model_dir, f"Encoder_64_s{s}_best.pth")
        if not os.path.exists(enc_ckpt):
            print(f"[VIS][WARN] missing encoder for {fn} -> {enc_ckpt}, skip.")
            continue
        hidden, act, dropout = parse_head_tag(tag)
        heads.append((tag, s, head_ckpt, enc_ckpt, hidden, act, dropout))
    return sorted(heads)

def _build_full_from_name(name: str):
    """
    Construct full model by name (for 260-D input).
    """
    if name.startswith("Encoder_"):
        z = int(name.split("_")[1])
        return model_G.build_model(z_dim=z, head_hidden=(128,64), act="relu", dropout=0.0)
    ctor_map = {
        "FCNN":       lambda: model_G.G1FCNN(260),
        "CNN":        lambda: model_G.G1CNN(260),
        "CNN_GAP":    lambda: model_G.G1CNN_GAP(260),
        "CNN_GAP_S":  lambda: model_G.G1CNN_GAP_S(260),
    }
    if hasattr(model_G, "G1CNN_GAP_DW"):
        ctor_map["CNN_GAP_DW"] = lambda: model_G.G1CNN_GAP_DW(260, width_mult=1.0)
    if name not in ctor_map:
        raise ValueError(f"Unknown model name: {name}")
    return ctor_map[name]()

def _load_test_sample(npz_path="data/dataset/dataset_G_100k.npz", sample_idx=0):
    """
    Use the same split (seed=42) as elsewhere; return one test sample's Dx,Dy.
    """
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42)
    if sample_idx < 0 or sample_idx >= x_te.shape[0]:
        raise IndexError(f"sample_idx out of range: {sample_idx} / {x_te.shape[0]}")
    Dx = x_te[sample_idx][4:4+128]; Dy = x_te[sample_idx][4+128:4+256]
    return Dx.astype(np.float32), Dy.astype(np.float32)

def visualize_all_models(sample_indices=(0,), outdir="vis_all", W=256, H=128,
                         vmin=0.0, vmax=1.0, cmap_val="inferno", cmap_err="magma",
                         npz_path="data/dataset/dataset_G_100k.npz", device=None):
    """
    Render G1(theta,phi) maps for ALL available models (full and head-only) on the same test samples.
    Saves 3-panel PNGs to outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather models
    full_ckpts = _list_full_model_ckpts()
    head_ckpts = _list_head_ckpts_with_encoder()

    if not full_ckpts and not head_ckpts:
        print("[VIS] No checkpoints found. Check 'data/model' directory.")
        return

    print(f"[VIS] Found full models: {len(full_ckpts)}  | head-only: {len(head_ckpts)}")
    for sidx in sample_indices:
        Dx, Dy = _load_test_sample(npz_path=npz_path, sample_idx=sidx)
        err_vmax = _compute_error_max_for_sample(Dx, Dy, full_ckpts, head_ckpts, device, W, H, pctl=99.0)

        # Full models
        for name, seed, ckpt in full_ckpts:
            try:
                m = _build_full_from_name(name).to(device).eval()
                m.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
                prefix = os.path.join(outdir, f"FULL_{name}_s{seed}_i{sidx}")
                render_g1_maps(Dx, Dy, m, device, W=W, H=H, save_prefix=prefix,
                               vmin=vmin, vmax=vmax, cmap_val=cmap_val, cmap_err=cmap_err,err_vmax=err_vmax)
            except Exception as e:
                print(f"[VIS][ERR][FULL] {name}_s{seed}: {e}")

        # Head-only models (z=64)
        for tag, seed, head_ckpt, enc_ckpt, hidden, act, dropout in head_ckpts:
            try:
                prefix = os.path.join(outdir, f"HEAD_{tag}_s{seed}_i{sidx}")
                render_g1_maps_headonly(Dx, Dy, enc_ckpt, head_ckpt, z_dim=64,
                                        head_hidden=hidden, act=act, dropout=dropout,
                                        device=device, W=W, H=H, save_prefix=prefix,
                                        vmin=vmin, vmax=vmax, cmap_val=cmap_val, cmap_err=cmap_err,err_vmax=err_vmax)
            except Exception as e:
                print(f"[VIS][ERR][HEAD] {tag}_s{seed}: {e}")

if __name__ == "__main__":
    #main()
    visualize_all_models(sample_indices=(7185,), outdir="vis_all", W=512, H=256)