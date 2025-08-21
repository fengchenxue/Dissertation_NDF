import os, time, statistics, math
import numpy as np
import torch
import torch.nn as nn

import model_G



import os, time, statistics
import numpy as np
import torch
import torch.nn as nn
import model_G

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
    # import os
    # torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"

    torch.backends.cudnn.benchmark = True
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else None

    input_dim = 260
    fcnn_ckpt = "data/model/model_G1_FCNN_100K.pth"
    cnn_ckpt  = "data/model/model_G1_CNN_GAP_DW.pth"

    npz = np.load("data/dataset/dataset_G_100k.npz")
    X = torch.from_numpy(npz["x"]).float()
    if torch.cuda.is_available():
        X = X.pin_memory()

    rng = torch.Generator(device="cpu").manual_seed(0)
    N = X.shape[0]
    def sample_source(bs: int) -> torch.Tensor:
        idx = torch.randint(low=0, high=N, size=(bs,), generator=rng)
        return X[idx]

    fcnn = model_G.G1FCNN(input_dim)
    cnn  = model_G.G1CNN_GAP_DW(input_dim)

    if os.path.exists(fcnn_ckpt):
        fcnn.load_state_dict(torch.load(fcnn_ckpt, map_location="cpu"))
    if os.path.exists(cnn_ckpt):
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location="cpu"))

    # ---------------- CPU ----------------
    for name, model in [("FCNN", fcnn), ("CNN", cnn)]:
        m = model.to(device_cpu).eval()
        params = count_params(m)
        r1 = run_benchmark(m, device_cpu, batch_sizes=(1, 256), repeats=200, warmup=50,
                           measure="device_only", use_amp=False, sample_source=sample_source)
        pretty_print(f"{name} | device-only | fp32", device_cpu, params, r1)

        r2 = run_benchmark(m, device_cpu, batch_sizes=(1, 256), repeats=200, warmup=50,
                           measure="end_to_end", use_amp=False, sample_source=sample_source)
        pretty_print(f"{name} | end-to-end | fp32", device_cpu, params, r2)

        rsplit = run_benchmark(m, device_cpu, batch_sizes=(1,), repeats=200, warmup=50,
                               measure="split", use_amp=False, sample_source=sample_source)
        pretty_print_split(f"{name} | split (CPU)", device_cpu, params, rsplit)

    # ---------------- GPU ----------------
    if device_gpu is not None:
        for name, model in [("FCNN", fcnn), ("CNN", cnn)]:
            m = model.to(device_gpu).eval()
            params = count_params(m)

            r1 = run_benchmark(m, device_gpu, batch_sizes=(1, 1024), repeats=300, warmup=80,
                               measure="device_only", use_amp=False, sample_source=sample_source)
            pretty_print(f"{name} | device-only | fp32", device_gpu, params, r1)

            r2 = run_benchmark(m, device_gpu, batch_sizes=(1, 1024), repeats=300, warmup=80,
                               measure="end_to_end", use_amp=False, sample_source=sample_source)
            pretty_print(f"{name} | end-to-end | fp32", device_gpu, params, r2)

            r3 = run_benchmark(m, device_gpu, batch_sizes=(1, 1024), repeats=300, warmup=80,
                               measure="device_only", use_amp=True, sample_source=sample_source)
            pretty_print(f"{name} | device-only | AMP(fp16)", device_gpu, params, r3)

            rsplit = run_benchmark(m, device_gpu, batch_sizes=(1,), repeats=400, warmup=100,
                                   measure="split", use_amp=False, sample_source=sample_source)
            pretty_print_split(f"{name} | split (GPU fp32, batch=1)", device_gpu, params, rsplit)

if __name__ == "__main__":
    main()