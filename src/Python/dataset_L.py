# --- put near top of dataset_L.py ---
import os, sys, time, shutil, subprocess, traceback
from pathlib import Path
import numpy as np
import traceback


def _early_repo_and_logs():
    here = Path(__file__).resolve()
    p = here
    for _ in range(12):
        if (p / "CMakeLists.txt").exists() and (p / "src").exists():
            break
        p = p.parent
    repo = p
    logs = repo / "build" / "auto_logs"
    logs.mkdir(parents=True, exist_ok=True)
    # drop a run marker so we know this file actually executed
    with open(logs / "run_marker.txt", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start  FILE={__file__}  CWD={os.getcwd()}\n")
    return repo, logs

REPO_ROOT, AUTO_LOGS = _early_repo_and_logs()
def _repo_root_from_here() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "CMakeLists.txt").exists() and (p / "src").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[1]

def _cmake_exe() -> str:
    exe = shutil.which("cmake")
    if not exe:
        raise RuntimeError("cmake not found in PATH")
    return exe

def _ensure_vg_tools(repo: Path, config: str = "Release") -> dict:
    build = repo / "build"

    # primary expected locations (as per your logs)
    exe_dir = build / "external" / "virtualgonio" / "src" / config
    d2s   = exe_dir / ("distrib_to_surface.exe" if os.name == "nt" else "distrib_to_surface")
    gonio = exe_dir / ("virtual_gonio.exe"       if os.name == "nt" else "virtual_gonio")
    plot  = exe_dir / ("plot_scattering.exe"     if os.name == "nt" else "plot_scattering")

    def _found():
        return d2s.exists() and gonio.exists() and plot.exists()

    if not _found():
        # build the three targets
        cfg_log = AUTO_LOGS / "cmake_configure.log"
        bld_log = AUTO_LOGS / "cmake_build_tools.log"
        print("[auto-build] building virtualgonio tools...")
        with open(cfg_log, "w", encoding="utf-8", errors="ignore") as f:
            p = subprocess.run([_cmake_exe(), "-S", str(repo), "-B", str(build)],
                               stdout=f, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"CMake configure failed. See log: {cfg_log}")

        with open(bld_log, "w", encoding="utf-8", errors="ignore") as f:
            p = subprocess.run([_cmake_exe(), "--build", str(build), "--config", config,
                                "--target", "distrib_to_surface", "virtual_gonio", "plot_scattering"],
                               stdout=f, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"CMake build failed. See log: {bld_log}")

    # if still not at primary path, do a recursive search and pick the best match
    if not _found():
        hits = { "d2s": [], "gonio": [], "plot": [] }
        for name, key in [("distrib_to_surface", "d2s"),
                          ("virtual_gonio", "gonio"),
                          ("plot_scattering", "plot")]:
            pat = name + (".exe" if os.name == "nt" else "")
            hits[key] = list((build).rglob(pat))
        def _pick(lst):
            # prefer .../src/<config>/..., then shortest path
            lst = sorted(lst, key=lambda p: ("\\src\\" + config + "\\" not in str(p) and "/src/" + config + "/" not in str(p),
                                             len(str(p))))
            return lst[0] if lst else None
        d2s_alt, gonio_alt, plot_alt = _pick(hits["d2s"]), _pick(hits["gonio"]), _pick(hits["plot"])
        if d2s_alt and gonio_alt and plot_alt:
            d2s, gonio, plot = d2s_alt, gonio_alt, plot_alt
        else:
            raise RuntimeError("Tools built but not found. Looked under:\n  "
                               f"{exe_dir}\n"
                               "Also searched recursively under build/.\n"
                               f"Found:\n  d2s={d2s_alt}\n  gonio={gonio_alt}\n  plot={plot_alt}")

    return {"d2s": str(d2s), "gonio": str(gonio), "plot": str(plot)}

def _seed_dll_paths_for_tools(repo: Path) -> None:
    """Add likely DLL folders (Embree/TBB/etc.) to PATH and DLL search path."""
    build = repo / "build"
    dirs = set()

    # typical release folders
    globs = [
        "Release", "RelWithDebInfo", "Debug",
        "external/**/Release", "external/**/RelWithDebInfo", "external/**/Debug",
        "external/virtualgonio/src/Release",
        "external/virtualgonio/src/modules/*/Release",
        "external/virtualgonio/**/bin",
        "external/virtualgonio/**/lib",
    ]
    for pat in globs:
        for d in build.glob(pat):
            if d.is_dir():
                dirs.add(d)

    # pick folders that actually contain key dlls
    keys = ("embree", "tbb", "omp", "openmp")
    for dll in build.rglob("*.dll"):
        lname = dll.name.lower()
        if any(k in lname for k in keys):
            dirs.add(dll.parent)

    # apply
    for d in dirs:
        try:
            if os.name == "nt":
                os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass


def _check_cli_tools(exe_d2s: str, exe_gonio: str, exe_plot: str) -> None:
    """
    Some upstream tools don't support --help and return non-zero even when healthy.
    We consider a tool 'runnable' if it starts and prints usage/options. We also
    screen out classic DLL-missing return codes on Windows.
    """
    def _probe(path: str) -> tuple[bool, str, int]:
        try:
            # run with NO args so it prints usage; capture both streams
            p = subprocess.run([path], capture_output=True, text=True, timeout=20)
            msg = (p.stdout or "") + (p.stderr or "")
            code = int(p.returncode)
            # common Windows fatal codes for missing DLL etc.
            fatal_codes = {-1073741515, -1073741819, -1073740791}  # 0xC0000135, 0xC0000005, 0xC0000409
            if code in fatal_codes:
                return False, f"fatal code {code}", code
            # treat as OK if it shows option lines (usage text)
            ok_tokens = ["--filename", "--output", "--theta-i", "--phi-i", "required", "optional"]
            looks_like_usage = any(t in msg for t in ok_tokens)
            return looks_like_usage, msg.strip(), code
        except Exception as e:
            return False, f"{e}", -9999

    ok1, m1, c1 = _probe(exe_d2s)
    ok2, m2, c2 = _probe(exe_gonio)
    ok3, m3, c3 = _probe(exe_plot)

    if not (ok1 and ok2 and ok3):
        detail = []
        if not ok1: detail.append(f"d2s: {m1} (code {c1})")
        if not ok2: detail.append(f"gonio: {m2} (code {c2})")
        if not ok3: detail.append(f"plot: {m3} (code {c3})")
        raise RuntimeError("virtualgonio tools not runnable. Details:\n" + "\n".join(detail))

def _run_here_defaults() -> None:
    import ndf_py as vg

    repo = REPO_ROOT
    g_npz = repo / "data" / "dataset" / "dataset_G_100k.npz"
    if not g_npz.exists():
        raise FileNotFoundError(f"Missing G dataset: {g_npz}")

    tools = _ensure_vg_tools(repo, config="Release")
    exe_d2s, exe_gonio, exe_plot = tools["d2s"], tools["gonio"], tools["plot"]

    # ensure DLL search path before any subprocess
    _seed_dll_paths_for_tools(repo)

    vg.set_exe_paths(exe_d2s, exe_gonio, exe_plot)
    _check_cli_tools(exe_d2s, exe_gonio, exe_plot)

    out_dir = repo / "data" / "l2_from_g"
    take    = 20
    H, W    = 128, 64
    N_theta = 64
    N_phi   = 32

    print("[repo]", repo)
    print("[g_npz]", g_npz)
    print("[exe ]", exe_d2s, "|", exe_gonio, "|", exe_plot)
    print("[grid]", f"HxW={H}x{W}  Ni={N_theta*N_phi}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # grids
    def build_o_grid(H, W):
        th = (np.arange(H)+0.5) * (0.5*np.pi / H)
        ph = (np.arange(W)+0.5) * (2.0*np.pi / W)
        TH, PH = np.meshgrid(th, ph, indexing='ij')
        dirs_ow = np.stack([np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)], -1).astype(np.float32)
        dtheta = 0.5*np.pi / H; dphi = 2.0*np.pi / W
        weight = (np.cos(TH) * np.sin(TH) * dtheta * dphi).astype(np.float32)
        return dirs_ow, weight

    def build_i_grid(N_theta, N_phi):
        th = (np.arange(N_theta)+0.5) * (0.5*np.pi / N_theta)
        ph = (np.arange(N_phi)+0.5)   * (2.0*np.pi / N_phi)
        TH, PH = np.meshgrid(th, ph, indexing='ij')
        cos_ti, sin_ti = np.cos(TH).reshape(-1), np.sin(TH).reshape(-1)
        cos_pi, sin_pi = np.cos(PH).reshape(-1), np.sin(PH).reshape(-1)
        return np.stack([cos_ti, sin_ti, cos_pi, sin_pi], -1).astype(np.float32)

    def load_unique_materials_from_G(g_npz_path, max_count=None):
        import hashlib
        d = np.load(g_npz_path)
        X = d["x"]
        K = (X.shape[1] - 4) // 2
        Dx_all = X[:, 4:4+K].astype(np.float32)
        Dy_all = X[:, 4+K:4+2*K].astype(np.float32)
        def pair_hash(dx, dy):
            q = np.round(np.concatenate([dx, dy], 0), 7).tobytes()
            return hashlib.sha1(q).hexdigest()
        seen, mats = set(), []
        for i in range(X.shape[0]):
            h = pair_hash(Dx_all[i], Dy_all[i])
            if h in seen:
                continue
            seen.add(h)
            mats.append((Dx_all[i].copy(), Dy_all[i].copy()))
            if max_count is not None and len(mats) >= max_count:
                break
        return mats, K

    dirs_ow, weight_o = build_o_grid(H, W)
    i_angles = build_i_grid(N_theta, N_phi)
    mats, K = load_unique_materials_from_G(g_npz, max_count=take)
    print(f"[materials] unique={len(mats)} K={K}")

    Ni = i_angles.shape[0]
    for i, (Dx, Dy) in enumerate(mats):
        tag = f"fromG_{i:04d}"
        Linf = np.zeros((Ni, H, W), np.float32)
        L1   = np.zeros((Ni, H, W), np.float32)
        for k in range(Ni):
            if (k % max(1, Ni // 10)) == 0:
                print(f"  [{tag}] incidence {k+1}/{Ni}")
            cti, sti, cpi, spi = map(float, i_angles[k])
            L1[k]   = vg.eval_microfacet_L1_img(Dx.tolist(), Dy.tolist(), cti, sti, cpi, spi, dirs_ow).astype(np.float32)
            Linf[k] = vg.virtual_goniometer_sample(Dx.tolist(), Dy.tolist(), cti, sti, cpi, spi, int(H), int(W)).astype(np.float32)
        L2p = np.maximum(0.0, Linf - L1).astype(np.float32)
        z = np.zeros((0,), np.float32)
        np.savez(out_dir / f"{tag}.npz",
                 z=z, Dx=Dx.astype(np.float32), Dy=Dy.astype(np.float32),
                 i_angles=i_angles, Linf=Linf, L1=L1, L2p=L2p,
                 dirs_ow=dirs_ow, weight_o=weight_o)
        print("  wrote", out_dir / f"{tag}.npz")


if __name__ == "__main__":
    try:
        _run_here_defaults()
        print("\n[done] Baking finished.")
    except Exception:

        _p = Path(__file__).resolve()
        for _ in range(12):
            if (_p / "CMakeLists.txt").exists() and (_p / "src").exists():
                break
            _p = _p.parent
        _logs = _p / "build" / "auto_logs"
        _logs.mkdir(parents=True, exist_ok=True)
        err = _logs / "last_error.txt"
        with open(err, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("\n[ERROR] full traceback saved to:", err)
    finally:
        try:
            input("\nPress Enter to close")
        except Exception:
            pass
