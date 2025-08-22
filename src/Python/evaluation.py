import numpy as np
import torch
import torch.nn as nn
import model_G
import os, time, statistics
import math
from sklearn.model_selection import train_test_split

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _stats(arr):
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.median(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "n":       len(arr),
    }

@torch.inference_mode()
def benchmark_once(model, inputs, device, measure="device_only", use_amp=False):
    model.eval()
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
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

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
            with torch.amp.autocast('cuda',dtype=torch.float16):
                e0.record(); _ = model(x); e1.record()
        else:
            e0.record(); _ = model(x); e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1)
    else:
        t0 = time.perf_counter(); _ = model(x); t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

def run_benchmark(model, device, batch_sizes=(1, 256), repeats=200, warmup=50,
                  measure="device_only", use_amp=False, sample_source=None):

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

@torch.inference_mode()
def eval_metrics_on_loader(model, loader, device, high_theta_threshold=1.2):
    model.eval()
    mae_sum = 0.0; mse_sum = 0.0; n = 0
    abs_errs = []
    hi_mae_sum = 0.0; hi_cnt = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        err = (pred - yb).view(-1)
        ae  = err.abs()
        mae_sum += ae.sum().item()
        mse_sum += (err * err).sum().item()
        n += ae.numel()
        abs_errs.append(ae.cpu())
        # high ¦È£º¦È = atan2(sin¦È, cos¦È)
        cos_t = xb[:, 0].clamp(-1, 1); sin_t = xb[:, 1].clamp(-1, 1)
        theta = torch.atan2(sin_t, cos_t)
        mask = theta >= high_theta_threshold
        if mask.any():
            hi_mae_sum += ae[mask].sum().item()
            hi_cnt += int(mask.sum().item())
    mae  = mae_sum / max(1, n)
    rmse = math.sqrt(mse_sum / max(1, n))
    abs_all = torch.cat(abs_errs) if len(abs_errs) else torch.tensor([])
    p95  = torch.quantile(abs_all, 0.95).item() if abs_all.numel() else float("nan")
    hi_mae = (hi_mae_sum / hi_cnt) if hi_cnt > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "P95": p95, "HighThetaMAE": hi_mae, "N": n}

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
    if torch.cuda.is_available(): X_cpu = X_cpu.pin_memory()
    def sample(bs: int) -> torch.Tensor:
        idx = torch.randint(low=0, high=N, size=(bs,), generator=rng)
        return X_cpu[idx]
    return sample

def pretty_print(title, device, params, results):
    print("="*80)
    print(f"{title} | device={device.type} | params={params/1e6:.3f} M")
    for bs, stats in results.items():
        mean_ms = stats["mean_ms"]; p50 = stats["p50_ms"]; p95 = stats["p95_ms"]
        tput = bs / (mean_ms / 1000.0)
        print(f"  batch={bs:4d}  mean={mean_ms:8.3f} ms   p50={p50:8.3f} ms   p95={p95:8.3f} ms   throughput={tput:8.1f} samples/s")

def pretty_print_split(title, device, params, results):
    print("="*80)
    print(f"{title} | device={device.type} | params={params/1e6:.3f} M")
    for bs, parts in results.items():
        tot = parts["total"]; h2d = parts["h2d"]; fwd = parts["fwd"]
        tput = bs / (tot["mean_ms"] / 1000.0)
        print(f"  batch={bs:4d}  TOTAL  mean={tot['mean_ms']:8.3f} ms  p50={tot['p50_ms']:8.3f} ms  p95={tot['p95_ms']:8.3f} ms  throughput={tput:8.1f}/s")
        print(f"             H2D    mean={h2d['mean_ms']:8.3f} ms  p50={h2d['p50_ms']:8.3f} ms  p95={h2d['p95_ms']:8.3f} ms")
        print(f"             FWD    mean={fwd['mean_ms']:8.3f} ms  p50={fwd['p50_ms']:8.3f} ms  p95={fwd['p95_ms']:8.3f} ms")

def main():

    torch.backends.cudnn.benchmark = True
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    npz_path = "data/dataset/dataset_G_100k.npz"   
    te_loader, (x_te, y_te) = make_loaders_from_npz(npz_path, batch=2048, seed=42)
    sample_source = build_sample_source_from_test(x_te)


    MODELS = [
        ("FCNN",        lambda: model_G.G1FCNN(260),               "data/model/model_G1_FCNN_100K.pth"),
        ("Encoder_4",   lambda: model_G.build_model(z_dim=4, head_hidden=(128,64), act="relu", dropout=0.0),"data/model/model_G1_z4_100K.pth"),
    ]

    for name, ctor, ckpt in MODELS:
        # ---------------- accuracy (TEST)----------------
        m = ctor().to(device_cpu).eval()
        if os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        params = count_params(m)
        metrics = eval_metrics_on_loader(m.to(device_cpu), te_loader, device_cpu)
        print(f"[TEST][{name}] MAE={metrics['MAE']:.8f}  RMSE={metrics['RMSE']:.8f}  "
              f"P95={metrics['P95']:.8f}  HighThetaMAE={metrics['HighThetaMAE']:.8f}  "
              f"Params={params/1e6:.4f}M")

        # ---------------- speed (CPU) ----------------
        r_cpu = run_benchmark(m.to(device_cpu), device_cpu,
                              batch_sizes=(1, 2048), repeats=200, warmup=50,
                              measure="device_only", use_amp=False, sample_source=sample_source)
        pretty_print(f"{name} | CPU device-only fp32", device_cpu, params, r_cpu)

        # ---------------- speed (GPU) ----------------
        if device_gpu is not None:
            mg = ctor().to(device_gpu).eval()
            if os.path.exists(ckpt):
                mg.load_state_dict(torch.load(ckpt, map_location=device_gpu))
            params_g = count_params(mg)
            r_gpu = run_benchmark(mg, device_gpu,
                                  batch_sizes=(1, 1024), repeats=300, warmup=80,
                                  measure="device_only", use_amp=False, sample_source=sample_source)
            pretty_print(f"{name} | GPU device-only fp32", device_gpu, params_g, r_gpu)

            r_gpu_amp = run_benchmark(mg, device_gpu,
                                      batch_sizes=(1, 1024), repeats=300, warmup=80,
                                      measure="device_only", use_amp=True, sample_source=sample_source)
            pretty_print(f"{name} | GPU device-only AMP(fp16)", device_gpu, params_g, r_gpu_amp)

            r_split = run_benchmark(mg, device_gpu,
                                    batch_sizes=(1,), repeats=400, warmup=100,
                                    measure="split", use_amp=False, sample_source=sample_source)
            pretty_print_split(f"{name} | GPU split(fp32, bs=1)", device_gpu, params_g, r_split)



if __name__ == "__main__":
    main()