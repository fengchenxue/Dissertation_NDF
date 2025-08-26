import os, numpy as np, torch
from model_L import L2LowRankHead

@torch.no_grad()
def _wmean(x, w):  # x:(B,H,W) w:(H,W)
    return (x*w).sum(dim=(-1,-2)) / (w.sum() + 1e-8)

@torch.no_grad()
def _hi_theta_mask(dirs_ow):     # (H,W,3) xyz
    z = dirs_ow[...,2]
    th = np.arccos(np.clip(z,0.0,1.0))
    return (th > np.deg2rad(70)).astype(np.float32)

@torch.no_grad()
def eval_l2lowrank(npz_path, ckpt_path, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    d = np.load(npz_path)
    z  = torch.from_numpy(d['z'][None]).float().to(device)     # (1,z_dim) or (1,0)
    I  = torch.from_numpy(d['i_angles']).float().to(device)     # (N_i,4)
    L  = torch.from_numpy(d['L2p']).float().to(device)          # (N_i,H,W)
    w  = torch.from_numpy(d['weight_o']).float().to(device)     # (H,W)
    do = d['dirs_ow']                                           # (H,W,3)
    hi = torch.from_numpy(_hi_theta_mask(do)).float().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = L2LowRankHead(**ckpt['cfg']).to(device)
    model.load_state_dict(ckpt['model']); model.eval()

    B=256; maes=[]; rmses=[]; p95s=[]; hmae=[]; epr=[]; egt=[]
    for s in range(0, I.shape[0], B):
        zB = z.expand(min(B, I.shape[0]-s), -1)
        Lp,_ = model.forward_reconstruct(I[s:s+B], z=zB)       # (B,H,W)
        D = torch.abs(Lp - L[s:s+B])
        maes.append(_wmean(D, w).cpu())
        rmses.append(torch.sqrt(_wmean(D**2, w)).cpu())
        p95s.append(torch.quantile((D*w).flatten(1), 0.95, dim=1).cpu())
        hmae.append(_wmean(D, w*hi).cpu())
        epr.append((Lp*w).sum(dim=(-1,-2)).cpu())
        egt.append((L[s:s+B]*w).sum(dim=(-1,-2)).cpu())

    mae  = torch.cat(maes).mean().item()
    rmse = torch.cat(rmses).mean().item()
    p95  = torch.cat(p95s).mean().item()
    h70  = torch.cat(hmae).mean().item()
    epr  = torch.cat(epr); egt = torch.cat(egt)
    erel = torch.mean(torch.abs(epr-egt) / (egt+1e-8)).item()
    print(f"[L2LowRank][{os.path.basename(npz_path)}] "
          f"wMAE={mae:.6f} wRMSE={rmse:.6f} P95={p95:.6f} HighThetaMAE={h70:.6f} EnergyRelErr={erel:.6f}")
