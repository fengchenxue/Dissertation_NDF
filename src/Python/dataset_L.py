import numpy as np
import os
import ndf_py as vg

def build_o_grid(H=64, W=32):
    dtheta = 0.5*np.pi / H
    dphi   = 2.0*np.pi / W
    theta  = (np.arange(H) + 0.5) * dtheta
    phi    = (np.arange(W) + 0.5) * dphi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')          # (H,W)
    dirs_ow = np.stack([np.sin(TH)*np.cos(PH),
                        np.sin(TH)*np.sin(PH),
                        np.cos(TH)], axis=-1).astype(np.float32)     # (H,W,3)
    delta_omega = (np.sin(TH) * dtheta * dphi).astype(np.float32)    # (H,W)
    weight_o = (np.cos(TH) * delta_omega).astype(np.float32)         # (H,W)
    return dirs_ow, weight_o

def build_i_grid(N_theta=64, N_phi=32):
    dti = 0.5*np.pi / N_theta
    dpi = 2.0*np.pi / N_phi
    ti  = (np.arange(N_theta) + 0.5) * dti
    pi  = (np.arange(N_phi) + 0.5) * dpi
    THi, PHi = np.meshgrid(ti, pi, indexing='ij')
    Ii = np.stack([np.cos(THi), np.sin(THi), np.cos(PHi), np.sin(PHi)],
                  axis=-1).reshape(-1,4).astype(np.float32)
    return Ii  

def bake_one_material_npz(out_path, Dx, Dy, encoder=None, H=64, W=32, N_theta=64, N_phi=32):
    # optional z feature
    z = None
    if encoder is not None:
        import torch
        enc = encoder.eval().cpu()
        with torch.no_grad():
            x = torch.from_numpy(np.stack([Dx,Dy],0)[None]).float()  # (1,2,128)
            z = enc(x).squeeze(0).cpu().numpy().astype(np.float32)

    i_angles = build_i_grid(N_theta, N_phi)              # (N_i,4)
    dirs_ow, weight_o = build_o_grid(H, W)               # (H,W,3), (H,W)

    N_i = i_angles.shape[0]
    Linf = np.zeros((N_i, H, W), np.float32)
    L1   = np.zeros((N_i, H, W), np.float32)

    def _split(i_feat):
        cos_ti, sin_ti, cos_pi, sin_pi = map(float, i_feat.tolist())
        return cos_ti, sin_ti, cos_pi, sin_pi

    def eval_microfacet_L1(Dx, Dy, i_feat, dirs_ow):
        cos_ti, sin_ti, cos_pi, sin_pi = _split(i_feat)
        return vg.eval_microfacet_L1_img(
            Dx.astype(np.float32).tolist(),
            Dy.astype(np.float32).tolist(),
            cos_ti, sin_ti, cos_pi, sin_pi,
            dirs_ow.astype(np.float32)
        ).astype(np.float32)

    def virtual_goniometer_sample_safe(Dx, Dy, i_feat, H, W):
        """Call C++ Linf; if not implemented yet, fall back to L1."""
        cos_ti, sin_ti, cos_pi, sin_pi = _split(i_feat)
        if hasattr(vg, "virtual_goniometer_sample"):
            try:
                return vg.virtual_goniometer_sample(
                    Dx.astype(np.float32).tolist(),
                    Dy.astype(np.float32).tolist(),
                    cos_ti, sin_ti, cos_pi, sin_pi,
                    int(H), int(W)
                ).astype(np.float32)
            except Exception:
                pass  # will fall back to L1 below
        # safe fallback (lets the pipeline run before Linf is ready)
        return eval_microfacet_L1(Dx, Dy, i_feat, dirs_ow)

    for k in range(N_i):
        i_feat = i_angles[k]  
        L1[k]   = eval_microfacet_L1(Dx, Dy, i_feat, dirs_ow)
        Linf[k] = virtual_goniometer_sample_safe(Dx, Dy, i_feat, H, W)

    L2p = np.maximum(0.0, Linf - L1).astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             z=(z if z is not None else np.zeros((0,), np.float32)),
             Dx=Dx.astype(np.float32),
             Dy=Dy.astype(np.float32),
             i_angles=i_angles,
             Linf=Linf, L1=L1, L2p=L2p,
             dirs_ow=dirs_ow, weight_o=weight_o)
    print("wrote", out_path)
