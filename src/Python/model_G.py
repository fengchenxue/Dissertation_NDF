import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import inspect
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os, csv, time
import torch.nn.functional as F
import random

class G1FCNN(nn.Module):
    def __init__(self,input_dim):
        super(G1FCNN,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.net(x)


class G1CNN(nn.Module):
    def __init__(self, input_dim):
        super(G1CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048 + 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        x_angle = x[:, :4]               
        x_seq = x[:, 4:].view(-1, 2, 128) 
        
       
        out = self.conv1(x_seq)
        out = nn.functional.relu(out)
        out = self.pool(out)    
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.pool(out)      
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.pool(out)      
        
        batch_size = out.shape[0]
        conv_features = out.view(batch_size, -1) 
        
        fused = torch.cat([conv_features, x_angle], dim=1)
        
        fused = nn.functional.relu(self.fc1(fused))
        fused = nn.functional.relu(self.fc2(fused))
        output = self.fc_out(fused)
        
        return output

class G1CNN_GAP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128 + 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        x_angle = x[:, :4]               
        x_seq = x[:, 4:].view(-1, 2, 128) 
        
       
        out = self.conv1(x_seq)
        out = nn.functional.relu(out)
        out = self.pool(out)    
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.pool(out)      
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.pool(out)      
        
        #batch_size = out.shape[0]
        #conv_features = out.view(batch_size, -1) 

        out = self.gap(out).squeeze(-1)

        #fused = torch.cat([conv_features, x_angle], dim=1)
        fused = torch.cat([out, x_angle], dim=1)

        fused = nn.functional.relu(self.fc1(fused))
        fused = nn.functional.relu(self.fc2(fused))
        output = self.fc_out(fused)
        
        return output



class G1CNN_GAP_S(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2,   out_channels=32,  kernel_size=5, padding=2, stride=2)  # 128 -> 64
        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=64,  kernel_size=5, padding=2, stride=2)  # 64  -> 32
        self.conv3 = nn.Conv1d(in_channels=64,  out_channels=128, kernel_size=3, padding=1, stride=1)  # 32  -> 32
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap   = nn.AdaptiveAvgPool1d(1)

        self.fc1   = nn.Linear(128 + 4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc_out= nn.Linear(64, 1)

    def forward(self, x):
        x_angle = x[:, :4]
        x_seq   = x[:, 4:].view(-1, 2, 128)

        out = torch.relu(self.conv1(x_seq))   # 2x128 -> 32x64
        out = torch.relu(self.conv2(out))     # 32x64 -> 64x32
        out = torch.relu(self.conv3(out))     

        out = self.gap(out).squeeze(-1)
        fused = torch.cat([out, x_angle], dim=1)
        fused = torch.relu(self.fc1(fused))
        fused = torch.relu(self.fc2(fused))
        return self.fc_out(fused)

# Depthwise + Pointwise 1D conv
class DWConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None, act="relu"):
        super().__init__()
        if padding is None:
            padding = (k - 1) // 2
        self.dw  = nn.Conv1d(in_ch, in_ch, k, stride=stride, padding=padding,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True) if act.lower() == "relu" else nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class G1CNN_GAP_DW(nn.Module):
    #width_mult=1.0: same width as "G1CNN_GAP_S"
    #width_mult=2.0: same parameters as "CNN_GAP_S"
    #bound_output=True: use Sigmoid
    def __init__(self, input_dim=260, width_mult=1.0, act="relu", bound_output=False):
        super().__init__()
        c1 = int(round(32 * width_mult))
        c2 = int(round(64 * width_mult))
        c3 = int(round(128 * width_mult))

        self.conv1 = DWConv1d(2,   c1, 5, stride=2, act=act)  # 128 -> 64
        self.conv2 = DWConv1d(c1,  c2, 5, stride=2, act=act)  # 64  -> 32
        self.conv3 = DWConv1d(c2,  c3, 3, stride=1, act=act)  # 32  -> 32

        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.act   = nn.ReLU(inplace=True) if act.lower() == "relu" else nn.SiLU(inplace=True)

        self.fc1   = nn.Linear(c3 + 4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc_out= nn.Linear(64, 1)
        self.bound_output = bound_output

    def forward(self, x):
        x_angle = x[:, :4]
        x_seq   = x[:, 4:].view(-1, 2, 128)

        out = self.conv1(x_seq)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.gap(out).squeeze(-1)               # (B, c3)

        fused = torch.cat([out, x_angle], dim=1)      # (B, c3+4)
        fused = self.act(self.fc1(fused))
        fused = self.act(self.fc2(fused))
        y = self.fc_out(fused)
        return torch.sigmoid(y) if self.bound_output else y

#----------encoder and head for G1Model----------
class G1Head(nn.Module):
    def __init__(self, z_dim=128, angle_dim=4, hidden=(128, 64), act: str = "relu", dropout: float = 0.0):
          super().__init__()
          in_dim = z_dim + angle_dim
          layers = []
          
          def make_act(name):
              return nn.ReLU(inplace=True) if name.lower() == "relu" else nn.SiLU(inplace=True)
          
          last = in_dim
          for h in hidden:
              layers.append(nn.Linear(last, h))
              layers.append(make_act(act))
              if dropout and dropout > 0:
                  layers.append(nn.Dropout(dropout))
                  
              last = h
              
          self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
          self.out = nn.Linear(last, 1)

    def forward(self, z, angle):           # z:(B,z_dim), angle:(B,angle_dim)
        x = torch.cat([z, angle], dim=1)
        x = self.mlp(x)
        return self.out(x)

class _ZeroFeat(nn.Module):
    def forward(self, z):  # z:(B,128)
        return z.new_zeros((z.size(0), 0))

class G1Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.c1  = nn.Conv1d(2,   32, 5, padding=2, stride=1)
        self.p1  = nn.MaxPool1d(kernel_size=2, stride=2)   # 128 -> 64
        self.c2  = nn.Conv1d(32,  64, 5, padding=2, stride=1)
        self.p2  = nn.MaxPool1d(kernel_size=2, stride=2)   # 64  -> 32
        self.c3  = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.p3  = nn.MaxPool1d(kernel_size=2, stride=2)   # 32  -> 16
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        if z_dim == 128: 
            self.proj = nn.Identity()
        elif z_dim == 0:
            self.proj = _ZeroFeat()
        else:
            self.proj = nn.Linear(128, z_dim)

    def forward(self, x_seq):                 # (B,2,128)
        x = torch.relu(self.c1(x_seq)); x = self.p1(x)
        x = torch.relu(self.c2(x));     x = self.p2(x)
        x = torch.relu(self.c3(x));     x = self.p3(x)
        z= self.gap(x).squeeze(-1)        # z:(B,128)
        return self.proj(z)               # (B,z_dim)

class G1Model(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, angle_dim=4):
        super().__init__()
        self.enc = encoder
        self.head = head
        self.angle_dim = angle_dim

    def forward(self, x):                    # x:(B, 4 + 2*128)
        angle = x[:, :self.angle_dim]
        seq   = x[:, self.angle_dim:].view(-1, 2, 128)
        z = self.enc(seq)
        return self.head(z, angle)

def build_model(z_dim=128, head_hidden=(128, 64), angle_dim=4,
                    act="relu", dropout=0.0):
    enc = G1Encoder(z_dim=z_dim)
    head = G1Head(z_dim=z_dim, angle_dim=angle_dim,
                  hidden=head_hidden, act=act, dropout=dropout)
    return G1Model(enc, head, angle_dim)

class G1Dataset(Dataset):
    def __init__(self,x_data,y_data):
        self.x=torch.tensor(x_data,dtype=torch.float32)
        self.y=torch.tensor(y_data,dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def test_environment():
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("optim.lr_scheduler file:", inspect.getfile(optim.lr_scheduler))
    print("ReduceLROnPlateau signature:",inspect.signature(optim.lr_scheduler.ReduceLROnPlateau.__init__))
    print("Torch module file:", torch.__file__)

    # Check if PyTorch is installed and working
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA-capable device detected.")

#-------------auto train models-------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_loaders(npz_path="data/dataset/dataset_G_100k.npz", batch_size=2048, split_seed=42):
    data = np.load(npz_path)
    x, y = data["x"], data["y"]

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.2, random_state=split_seed)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=split_seed)
    train_loader = DataLoader(G1Dataset(x_tr, y_tr), batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(G1Dataset(x_va, y_va), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(G1Dataset(x_te, y_te), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader

@torch.inference_mode()
def eval_mse_on(loader, model, device):
    model.eval()
    s, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        s += F.mse_loss(pred, yb, reduction="sum").item()
        n += yb.numel()
    return s / max(1, n)

def train_once(model, loaders, *, tag, seed=0, train_mode="full",
               lr=1e-3, epochs=500, patience=25,
               enc_ckpt=None, out_dir="data/model"):

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    if train_mode == "head":
        if enc_ckpt and os.path.exists(enc_ckpt):
            full_sd = torch.load(enc_ckpt, map_location=device)
            enc_sd = {k.replace("enc.", ""): v for k, v in full_sd.items() if k.startswith("enc.")}
            model.enc.load_state_dict(enc_sd, strict=False)
        for p in model.enc.parameters():
            p.requires_grad = False
        model.enc.eval()
        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    criterion = torch.nn.MSELoss()

    train_loader, val_loader, test_loader = loaders
    best_val = float("inf"); wait = 0
    ckpt_path = os.path.join(out_dir, f"{tag}_s{seed}_best.pth")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        if train_mode == "head":
            model.enc.eval() 
        tr_sum, ntr = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * yb.numel()
            ntr += yb.numel()
        avg_tr = tr_sum / max(1, ntr)

        val_mse = eval_mse_on(val_loader, model, device)
        scheduler.step(val_mse)
        print(f"[{tag}|seed={seed}] ep={epoch:03d}  train={avg_tr:.8f}  val={val_mse:.8f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_mse < best_val - 1e-12:
            best_val, wait = val_mse, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[{tag}|seed={seed}] early stop at ep {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_mse = eval_mse_on(test_loader, model, device)
    minutes = (time.time() - t0) / 60.0
    print(f"[{tag}|seed={seed}] Test MSE={test_mse:.8f}  time={minutes:.1f} min  -> {ckpt_path}")
    return {"tag": tag, "seed": seed, "val_mse": best_val, "test_mse": test_mse, "minutes": minutes, "ckpt": ckpt_path}

def run_all_experiments():
    input_dim = 260
    seeds = [0, 1, 2, 3, 4]
    
    # 1) model list
    exp_list = [
        ("FCNN",        lambda: G1FCNN(input_dim),                         "full", None),
        ("CNN",         lambda: G1CNN(input_dim),                          "full", None),
        ("CNN_GAP",     lambda: G1CNN_GAP(input_dim),                      "full", None),
        ("CNN_GAP_S",   lambda: G1CNN_GAP_S(input_dim),                    "full", None),
        ("CNN_GAP_DW",      lambda: G1CNN_GAP_DW(input_dim, width_mult=1.0), "full", None),
        ("CNN_GAP_DW_Wide", lambda: G1CNN_GAP_DW(input_dim, width_mult=2.0), "full", None),
    ]
    for z in [128,64, 32, 16, 8, 4, 2, 1, 0]:
        exp_list.append((
            f"Encoder_{z}",
            lambda z=z: build_model(z_dim=z, head_hidden=(128,64), act="relu", dropout=0.0),
            "full", None
        ))

    # 2. CSV
    os.makedirs("data/model", exist_ok=True)
    csv_runs = "data/model/summary_runs.csv"
    csv_agg  = "data/model/summary_agg.csv"

    agg = {}  # tag -> list of test_mse
    with open(csv_runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","seed","val_mse","test_mse","minutes","ckpt"])
        w.writeheader()

        # 3. train
        for tag, ctor, mode, enc_ckpt in exp_list:
            agg[tag] = []
            for s in seeds:
                print(f"\n===== [RUN] {tag}  mode={mode}  seed={s} =====")
                set_seed(s) 
                loaders = make_loaders(npz_path="data/dataset/dataset_G_100k.npz",
                                       batch_size=2048, split_seed=42)
                info = train_once(ctor(), loaders, tag=tag, seed=s, train_mode=mode,
                                  lr=1e-3, epochs=500, patience=25, enc_ckpt=enc_ckpt,
                                  out_dir="data/model")
                w.writerow(info); f.flush()
                agg[tag].append(info["test_mse"])

    # 4. mean and std
    with open(csv_agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","mean_test_mse","std_test_mse","n"])
        w.writeheader()
        for tag, arr in agg.items():
            arr = np.array(arr, dtype=np.float64)
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            w.writerow({"tag": tag, "mean_test_mse": float(arr.mean()), "std_test_mse": std, "n": int(arr.size)})

    print(f"\nAll done.\n - Per-run  -> {csv_runs}\n - Aggregate-> {csv_agg}")

# ===== head-only search for z=64 (freeze encoder, train head) =====
def run_head_search_z64():
    z = 64
    seeds = [0,1,2,3,4]
    # use encoder checkpoints already trained as Encoder_64_s{seed}_best.pth
    def head_tag(hidden, act, dropout):
        h = "x".join(str(u) for u in hidden)
        return f"HeadZ64_{h}_{act}_do{dropout:.2f}"
    
    HEAD_SPECS = [
        ((128,64),      "relu", 0.00),
        ((256,128),     "relu", 0.00),
        ((256,128,64),  "relu", 0.00),
        ((64,),         "relu", 0.00),
        ((128,64),      "relu", 0.05),
        ((128,64),      "relu", 0.10),
        ((128,64),      "silu", 0.00),
        ((192,96),      "silu", 0.05),
    ]

    for hidden, act, dropout in HEAD_SPECS:
        tag = head_tag(hidden, act, dropout)
        for s in seeds:
            enc_ckpt = os.path.join("data", "model", f"Encoder_{z}_s{s}_best.pth")
            if not os.path.exists(enc_ckpt):
                print(f"[SKIP] {tag} seed={s}: missing encoder ckpt -> {enc_ckpt}")
                continue
            loaders = make_loaders(npz_path="data/dataset/dataset_G_100k.npz",
                                   batch_size=2048, split_seed=42)
            model = build_model(z_dim=z, head_hidden=hidden, act=act, dropout=dropout)
            print(f"\n===== [RUN] {tag}  seed={s}  (freeze encoder; train head) =====")
            train_once(model, loaders,
                       tag=tag, seed=s, train_mode="head",
                       lr=1e-3, epochs=500, patience=25,
                       enc_ckpt=enc_ckpt, out_dir="data/model")

if __name__ == "__main__":
    #test_environment()
    #run_all_experiments()
    run_head_search_z64()