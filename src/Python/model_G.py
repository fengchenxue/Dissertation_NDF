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
    model = G1CNN(input_dim).to(device)
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