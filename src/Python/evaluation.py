import os, time, math, statistics
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import model_G

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

def build_head_only_from_ckpt(ckpt_path: str, z_dim: int, angle_dim=4):
    """Load a full checkpoint, extract the head, disable inplace, return a head-only module."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = model_G.build_model(z_dim=z_dim, head_hidden=(128,64), act="relu", dropout=0.0)
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



if __name__ == "__main__":
    main()