from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
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
    learning_rate = 1e-4
    epochs=100
    input_dim=260
    
    # data
    data=np.load("data/dataset/dataset_G_10k.npz")
    x_data= data['x']
    y_data= data['y']
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    train_dataset = G1Dataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = G1Dataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    
    # model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = G1FCNN(input_dim).to(device)
    #model = G1CNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #training
    for epoch in range(epochs):
        model.train()
        train_loss=0
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()*x_batch.size(0)

        avg_train_loss =train_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.6f}")

    torch.save(model.state_dict(), "data/model/model_G1.pth")

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for x_batch, y_batch in test_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0)
        avg_loss = test_loss / len(test_dataset)
        print(f"Test Loss: {avg_loss:.6f}")

    return


def test_environment():
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