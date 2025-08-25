import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model_L import L2LowRankHead 

# ===== L2+ Neural Low-Rank Head =====
class L2LowRankHead(nn.Module):
    """
    Neural low-rank: L2+(i,o) ¡Ö ¦²_r a_r(z,i,ndf_feat) * b_r(o).
    - b_r(o): learnable outgoing bases over hemisphere grid (R,H,W), shared by all materials.
    - a_r(...): small MLP mapping from (z and/or NDF features, i-feat) to R non-negative coefficients.
    """
    def __init__(self, z_dim=64, R=8, H=64, W=32,
                 use_z=True,
                 use_ndf_pca=False, ndf_pca_dim=16,     # Dx_pca + Dy_pca = 2*ndf_pca_dim
                 use_ndf_raw=False, ndf_raw_len=128, ndf_embed_dim=32,
                 widths=(128,128), act='relu'):
        super().__init__()
        self.R, self.H, self.W = R, H, W
        act_fn = {'relu': nn.ReLU, 'silu': nn.SiLU}.get(act, nn.ReLU)

        in_dim = 4  # [cos¦Èi, sin¦Èi, cos¦Õi, sin¦Õi]
        self.use_z = use_z
        if use_z: in_dim += z_dim

        self.use_ndf_pca = use_ndf_pca
        self.use_ndf_raw = use_ndf_raw
        if use_ndf_pca:
            in_dim += 2 * ndf_pca_dim
        if use_ndf_raw:
            self.ndf_proj = nn.Sequential(
                nn.Linear(2 * ndf_raw_len, 128), act_fn(),
                nn.Linear(128, 2 * ndf_embed_dim), act_fn(),
            )
            in_dim += 2 * ndf_embed_dim

        layers = []
        last = in_dim
        for w in widths:
            layers += [nn.Linear(last, w), act_fn()]
            last = w
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(last, R)

        # Learnable bases b_r(o) ¡Ý 0 (R,H,W)
        self.basis = nn.Parameter(torch.full((R, H, W), 1e-3))
        self.energy_scale = nn.Parameter(torch.tensor(1.0))

    def _pack_input(self, i_feat, z=None, ndf_pca=None, ndf_raw=None):
        feats = [i_feat]
        if self.use_z and z is not None:
            feats.append(z)
        if self.use_ndf_pca and ndf_pca is not None:
            feats.append(ndf_pca)
        if self.use_ndf_raw and ndf_raw is not None:
            flat = ndf_raw.reshape(ndf_raw.shape[0], -1)
            feats.append(self.ndf_proj(flat))
        return torch.cat(feats, dim=-1)

    def forward_coeff(self, i_feat, z=None, ndf_pca=None, ndf_raw=None):
        x = self._pack_input(i_feat, z, ndf_pca, ndf_raw)
        a = F.softplus(self.out(self.mlp(x)))  # non-negative
        return a  # (B,R)

    def forward_reconstruct(self, i_feat, z=None, ndf_pca=None, ndf_raw=None):
        a = self.forward_coeff(i_feat, z, ndf_pca, ndf_raw)        # (B,R)
        Bmap = F.softplus(self.basis)                              # (R,H,W)
        L2p = torch.einsum('rhw,br->bhw', Bmap, a)                 # (B,H,W)
        L2p = F.relu(L2p) * F.softplus(self.energy_scale)
        return L2p, a

    @torch.no_grad()
    def export_basis(self):
        return F.softplus(self.basis).detach().cpu().numpy()       # (R,H,W)




class L2Dataset(Dataset):
    """
    Each npz is one material baked by your pipeline.
    For simplicity, we iterate over (file, i-index) pairs.
    """
    def __init__(self, npz_list):
        self.files = npz_list
        self.index = []
        for p in self.files:
            d = np.load(p)
            Ni = d['i_angles'].shape[0]
            self.index += [(p, k) for k in range(Ni)]
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        p,k = self.index[idx]
        d = np.load(p)
        z = torch.from_numpy(d['z']).float()                     # (z_dim,) or (0,)
        i = torch.from_numpy(d['i_angles'][k]).float()           # (4,)
        L = torch.from_numpy(d['L2p'][k]).float()                # (H,W)
        w = torch.from_numpy(d['weight_o']).float()              # (H,W)
        # optional NDF features if you decide to use them:
        # Dx = torch.from_numpy(d['Dx']).float()  # (128,)
        # Dy = torch.from_numpy(d['Dy']).float()  # (128,)
        return z, i, L, w

def l2_energy(L, weight):          # L:(B,H,W), weight:(H,W)
    return (L * weight).sum(dim=(-1,-2))  # (B,)

def train(npz_glob="data/l2_baked/*.npz",
          z_dim=64, R=8, H=64, W=32, widths=(128,128),
          lr=1e-3, epochs=200, bs=64, val_ratio=0.1,
          device=None, out_ckpt="data/model/L2LowRank_R8.pth", seed=0):
    import random
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    files = sorted(glob.glob(npz_glob))
    ds = L2Dataset(files)
    # simple split
    n_val = max(1, int(len(ds)*val_ratio))
    idxs  = np.random.permutation(len(ds))
    val_set = torch.utils.data.Subset(ds, idxs[:n_val])
    tr_set  = torch.utils.data.Subset(ds, idxs[n_val:])

    def collate(batch):
        z = torch.stack([b[0] for b in batch],0)
        i = torch.stack([b[1] for b in batch],0)
        L = torch.stack([b[2] for b in batch],0)
        w = batch[0][3]  # same (H,W)
        return z,i,L,w

    tr = DataLoader(tr_set, batch_size=bs, shuffle=True,  num_workers=0, collate_fn=collate)
    va = DataLoader(val_set,batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)

    model = L2LowRankHead(z_dim=z_dim, R=R, H=H, W=W, widths=widths,
                          use_z=True, use_ndf_pca=False, use_ndf_raw=False).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    best = 1e9; best_state=None

    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0
        for z,i,L,w in tr:
            z,i,L,w = z.to(device), i.to(device), L.to(device), w.to(device)
            Lp,_ = model.forward_reconstruct(i_feat=i, z=z)
            mae = torch.mean(torch.abs(Lp - L) * w)
            e_pr = l2_energy(Lp, w); e_gt = l2_energy(L, w)
            loss = mae + 0.1 * torch.mean(torch.abs(e_pr - e_gt))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * z.size(0)
        tr_loss /= len(tr.dataset)

        # val
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for z,i,L,w in va:
                z,i,L,w = z.to(device), i.to(device), L.to(device), w.to(device)
                Lp,_ = model.forward_reconstruct(i_feat=i, z=z)
                mae = torch.mean(torch.abs(Lp - L) * w)
                e_pr = l2_energy(Lp, w); e_gt = l2_energy(L, w)
                va_loss += (mae + 0.1*torch.mean(torch.abs(e_pr - e_gt))).item() * z.size(0)
        va_loss /= len(va.dataset)
        if va_loss < best:
            best = va_loss
            best_state = {'model': model.state_dict(),
                          'cfg': {'z_dim':z_dim,'R':R,'H':H,'W':W,'widths':widths}}
            os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
            torch.save(best_state, out_ckpt)

        if ep==1 or ep%10==0:
            print(f"[ep {ep:03d}] train={tr_loss:.6f}  val={va_loss:.6f}  best={best:.6f}")

    print("saved", out_ckpt)
    return out_ckpt

def export_basis(ckpt_path, out_npy="data/export/l2_basis_R8.npy"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = L2LowRankHead(**ckpt['cfg']); model.load_state_dict(ckpt['model'])
    b = model.export_basis()  # (R,H,W)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, b); print("wrote", out_npy)

if __name__ == "__main__":
    train()