from pyexpat import model
from sched import scheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import sys, inspect, torch, torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

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

'''
# Since Depthwise Separable Convolution turned out to be slower and less accurate than CNN_GAP, 
# I decided to abandon it.
class DWConv1d(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None):
        if padding is None: padding = (k - 1) // 2
        super().__init__(
            nn.Conv1d(in_ch, in_ch, k, stride=stride, padding=padding,
                      groups=in_ch, bias=False),   # Depthwise
            nn.Conv1d(in_ch, out_ch, 1, bias=False),   # Pointwise
            nn.ReLU(inplace=True),
        )

class G1CNN_GAP_DW(nn.Module):
    def __init__(self, input_dim=260):
        super().__init__()
        self.conv1 = nn.Conv1d(2,   32, 5, padding=2, stride=2)     # 128->64
        self.conv2 = DWConv1d(32,  64, 5, stride=2)                 # 64 ->32 (DW+PW)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, stride=1)     # 32 ->32
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(128 + 4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc_out= nn.Linear(64, 1)
    def forward(self, x):
        x_angle = x[:, :4]
        x_seq   = x[:, 4:].view(-1, 2, 128)
        out = torch.relu(self.conv1(x_seq))
        out = self.conv2(out)
        out = torch.relu(self.conv3(out))
        out = self.gap(out).squeeze(-1)
        fused = torch.cat([out, x_angle], dim=1)
        fused = torch.relu(self.fc1(fused))
        fused = torch.relu(self.fc2(fused))
        return self.fc_out(fused)
 '''
class G1Head(nn.Module):
    def __init__(self, z_dim=128, angle_dim=4, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + angle_dim, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.out = nn.Linear(64, 1)
    def forward(self, z, angle):              # z:(B,128) angle:(B,4)
        h = torch.cat([z, angle], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.out(h)

class G1Encoder_GAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1  = nn.Conv1d(2,   32, 5, padding=2, stride=1)
        self.p1  = nn.MaxPool1d(kernel_size=2, stride=2)   # 128 -> 64
        self.c2  = nn.Conv1d(32,  64, 5, padding=2, stride=1)
        self.p2  = nn.MaxPool1d(kernel_size=2, stride=2)   # 64  -> 32
        self.c3  = nn.Conv1d(64, 128, 3, padding=1, stride=1)
        self.p3  = nn.MaxPool1d(kernel_size=2, stride=2)   # 32  -> 16
        self.gap = nn.AdaptiveAvgPool1d(1)
    def forward(self, x_seq):                 # (B,2,128)
        x = torch.relu(self.c1(x_seq)); x = self.p1(x)
        x = torch.relu(self.c2(x));     x = self.p2(x)
        x = torch.relu(self.c3(x));     x = self.p3(x)
        return self.gap(x).squeeze(-1)        # z:(B,128)

class G1Model_GAP(nn.Module):
    def __init__(self, angle_dim=4):
        super().__init__()
        self.enc  = G1Encoder_GAP()
        self.head = G1Head(128, angle_dim)   
    def forward(self, x):
        angle = x[:, :4]
        seq   = x[:, 4:].view(-1,2,128)
        z = self.enc(seq)
        return self.head(z, angle)
        '''
class SE1D(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()
        self.fc1 = nn.Linear(C, max(C // r, 4))
        self.fc2 = nn.Linear(max(C // r, 4), C)
    def forward(self, x):                   # x: (B,C,L)
        s = x.mean(dim=-1)                  # GAP over length -> (B,C)
        s = torch.nn.functional.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))      # (B,C)
        return x * s.unsqueeze(-1)
class ResBlock1D(nn.Module):
    def __init__(self, C_in, C_out, k=3, stride=1, norm='bn'):
        super().__init__()
        pad = k // 2
        Norm = nn.BatchNorm1d if norm == 'bn' else nn.GroupNorm
        self.conv1 = nn.Conv1d(C_in,  C_out, k, stride=stride, padding=pad, bias=False)
        self.norm1 = Norm(C_out, num_groups=1) if norm!='bn' else Norm(C_out)
        self.conv2 = nn.Conv1d(C_out, C_out, k, stride=1,      padding=pad, bias=False)
        self.norm2 = Norm(C_out, num_groups=1) if norm!='bn' else Norm(C_out)
        self.act   = nn.SiLU()
        self.se    = SE1D(C_out, r=16)
        self.skip  = (nn.Identity() if (C_in==C_out and stride==1)
                      else nn.Conv1d(C_in, C_out, 1, stride=stride, bias=False))
    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = self.se(y)
        y = self.act(y + self.skip(x))
        return y
class G1Encoder_MSResSE(nn.Module):
    def __init__(self, z_dim_raw=256, norm='bn'):
        super().__init__()
        self.s1 = ResBlock1D(2,   64, k=5, stride=2, norm=norm)   # 128->64
        self.s2 = ResBlock1D(64, 128, k=5, stride=2, norm=norm)   # 64 ->32
        self.s3 = ResBlock1D(128,256, k=3, stride=2, norm=norm)   # 32 ->16
        self.proj = nn.Linear(64+128+256, z_dim_raw)

    def forward(self, x_seq):                # (B,2,128)
        h1 = self.s1(x_seq)                  # (B,64,64)
        h2 = self.s2(h1)                     # (B,128,32)
        h3 = self.s3(h2)                     # (B,256,16)

        g1 = h1.mean(-1)                     # (B,64)
        g2 = h2.mean(-1)                     # (B,128)
        g3 = h3.mean(-1)                     # (B,256)
        z_raw = torch.cat([g1,g2,g3], dim=1) # (B,448)
        z_raw = torch.nn.functional.silu(self.proj(z_raw))     # (B,z_dim_raw)
        return z_raw

class MSResSE_EncoderWithProj(nn.Module):

    def __init__(self, z_raw=256, z=128, norm='bn'):
        super().__init__()
        self.backbone = G1Encoder_MSResSE(z_dim_raw=z_raw, norm=norm)
        self.proj = nn.Linear(z_raw, z)
        self.act = nn.SiLU()
    def forward(self, x_seq):  # x_seq: (B,2,128)
        z_raw = self.backbone(x_seq)        # (B, z_raw)
        return self.act(self.proj(z_raw))   # (B, z)

class G1End2End(nn.Module):

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.enc  = encoder
        self.head = head
    def forward(self, x):                   # x: (B,260)
        angle = x[:, :4]                    # (B,4)
        seq   = x[:, 4:].view(-1, 2, 128)   # (B,2,128)
        z = self.enc(seq)                   # (B,z)
        return self.head(z, angle)          # (B,1)

    @staticmethod
    def build_with_gap(angle_dim=4):
        enc  = G1Encoder_GAP()                  
        head = G1Head(z_dim=128, angle_dim=angle_dim)
        return G1End2End(enc, head)

    @staticmethod
    def build_with_msres(angle_dim=4, z_raw=256, z=128, norm='bn'):
        enc  = MSResSE_EncoderWithProj(z_raw=z_raw, z=z, norm=norm)
        head = G1Head(z_dim=z, angle_dim=angle_dim)
        return G1End2End(enc, head)

'''


@torch.inference_mode()
def infer_many_dirs(model, x_sample, angles):          # x_sample:(1,260) angles:(K,4)
    dev = next(model.parameters()).device
    z = model.enc(x_sample[:,4:].view(1,2,128).to(dev)) 
    z_rep = z.expand(angles.size(0), -1)
    return model.head(z_rep.to(dev), angles.to(dev)).squeeze(-1)  # (K,)

class G1Dataset(Dataset):
    def __init__(self,x_data,y_data):
        self.x=torch.tensor(x_data,dtype=torch.float32)
        self.y=torch.tensor(y_data,dtype=torch.float32).view(-1,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def G1_train():
    #parameters
    batch_size = 256
    learning_rate = 1e-3
    epochs=300
    input_dim=260
    patience=25 # early stopping patience

    # data
    data=np.load("data/dataset/dataset_G_100k.npz")
    x_data= data['x']
    y_data= data['y']
    
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_val,   x_test, y_val,   y_test = train_test_split(x_temp,  y_temp,  test_size=0.5, random_state=42)

    train_loader = DataLoader(G1Dataset(x_train, y_train), batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(G1Dataset(x_val,   y_val),   batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(G1Dataset(x_test,  y_test),  batch_size=batch_size, shuffle=False, pin_memory=True)

    # model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = G1FCNN(input_dim).to(device)
    #model = G1CNN(input_dim).to(device)
    #model = G1CNN_GAP(input_dim).to(device)
    #model = G1CNN_GAP_S(input_dim).to(device)
    #model= G1Model_GAP().to(device)
    model = G1End2End.build_with_msres(angle_dim=4, z_raw=256, z=128).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    #training
    best_val= float('inf')
    wait =0

    for epoch in range(epochs):
        model.train()
        train_loss=0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()*x_batch.size(0)

        avg_train_loss =train_loss / len(train_loader.dataset)
        
        #validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{epochs}]  train={avg_train_loss:.6f}  val={avg_val_loss:.6f}  lr={lr_now:.2e}")

        # Early stopping
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), "data/model/model_G1_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    #test
    model.load_state_dict(torch.load("data/model/model_G1_best.pth", map_location=device))
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss_sum += loss.item() * xb.size(0)
    avg_test = test_loss_sum / len(test_loader.dataset)
    print(f"Test Loss: {avg_test:.6f}")

    return


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



if __name__ == "__main__":
    G1_train()
    #test_environment()