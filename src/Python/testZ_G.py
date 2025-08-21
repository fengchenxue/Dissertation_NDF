import torch, numpy as np
import model_G
import torch.nn as nn
@torch.inference_mode()
def mse_on_loader(model, X, Y, device):
    model.eval()
    bs = 2048
    crit = torch.nn.MSELoss(reduction="mean")
    mses = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        yb = torch.from_numpy(Y[i:i+bs]).float().to(device).view(-1, 1)  # (B,1)
        pred = model(xb)                          # (B,1)
        mses.append(crit(pred, yb).item())
    return float(np.mean(mses))

@torch.inference_mode()
def mse_angle_only(model, X, Y, device, z_dim):
    model.eval()
    bs = 2048
    crit = torch.nn.MSELoss(reduction="mean")
    mses = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        angle = xb[:, :4]
        z0 = torch.zeros((angle.shape[0], z_dim), device=device, dtype=angle.dtype)
        pred = model.head(z0, angle)              # (B,1)
        yb = torch.from_numpy(Y[i:i+bs]).float().to(device).view(-1, 1)
        mses.append(crit(pred, yb).item())
    return float(np.mean(mses))

@torch.inference_mode()
def mse_z_shuffle(model, X, Y, device):
    model.eval()
    bs = 4096
    crit = torch.nn.MSELoss(reduction="mean")
    mses = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        angle = xb[:, :4]
        seq   = xb[:, 4:].view(-1, 2, 128)
        z = model.enc(seq)
        idx = torch.randperm(z.size(0), device=device)
        z_shuf = z[idx]
        pred = model.head(z_shuf, angle)          # (B,1)
        yb = torch.from_numpy(Y[i:i+bs]).float().to(device).view(-1, 1)
        mses.append(crit(pred, yb).item())
    return float(np.mean(mses))

@torch.inference_mode()
def probe_head_weight_usage(model, z_dim: int):
    head = model.head

    fc1 = None
    if isinstance(head.mlp, nn.Sequential):
        for m in head.mlp:
            if isinstance(m, nn.Linear):
                fc1 = m
                break

    if fc1 is None:
        W = head.out.weight.data             # [1, z_dim+4]
        layer_name = "out (no hidden)"
    else:
        W = fc1.weight.data                  # [hidden, z_dim+4]
        layer_name = "fc1"

    zW = W[:, :z_dim]                        # (..., z_dim)
    aW = W[:, z_dim:]                        # (..., 4)

    print(f"[head.{layer_name}] |W_z|_F = {zW.norm().item():.4e}  |W_angle|_F = {aW.norm().item():.4e}")
    if z_dim > 0:
        per_dim = zW.norm(dim=0).cpu().numpy()
        import numpy as np
        print("per-dim z L2:", np.array2string(per_dim, precision=3, separator=', '))


@torch.inference_mode()
def mse_angle_shuffle(model, X, Y, device):
    model.eval()
    bs = 4096
    crit = torch.nn.MSELoss(reduction="mean")
    mses = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        angle = xb[:, :4]
        seq   = xb[:, 4:].view(-1, 2, 128)
        z = model.enc(seq)                   
        idx = torch.randperm(angle.size(0), device=device)
        angle_shuf = angle[idx]              
        pred = model.head(z, angle_shuf)     # (B,1)
        yb = torch.from_numpy(Y[i:i+bs]).float().to(device).view(-1, 1)
        mses.append(crit(pred, yb).item())
    return float(np.mean(mses))

@torch.inference_mode()
def mse_z_only(model, X, Y, device):
    model.eval()
    bs = 2048
    crit = torch.nn.MSELoss(reduction="mean")
    mses = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        seq   = xb[:, 4:].view(-1, 2, 128)
        z = model.enc(seq)
        angle0 = torch.zeros((z.size(0), 4), device=device, dtype=z.dtype)
        pred = model.head(z, angle0)         # (B,1)
        yb = torch.from_numpy(Y[i:i+bs]).float().to(device).view(-1, 1)
        mses.append(crit(pred, yb).item())
    return float(np.mean(mses))

if __name__ == "__main__":
    npz = np.load("data/dataset/dataset_G_100k.npz")
    X, Y = npz["x"], npz["y"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_G.build_model(z_dim=4, head_hidden=(128,64), angle_dim=4).to(device)
    sd = torch.load("data/model/model_G1_z4_100K.pth", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Missing:", missing, "Unexpected:", unexpected)
    model.eval()

    mse_normal = mse_on_loader(model, X, Y, device)
    mse_ang    = mse_angle_only(model, X, Y, device, z_dim=4)
    mse_zshuf  = mse_z_shuffle(model, X, Y, device)
    mse_angsh  = mse_angle_shuffle(model, X, Y, device)
    mse_zonly  = mse_z_only(model, X, Y, device)
    
    print(f"normal   MSE = {mse_normal:.8e}")
    print(f"angle-0  MSE = {mse_ang:.8e}")
    print(f"z-shuffle MSE = {mse_zshuf:.8e}")
    print(f"angle-shuffle MSE = {mse_angsh:.8e}")
    print(f"z-only        MSE = {mse_zonly:.8e}")
    probe_head_weight_usage(model, z_dim=4) 
