import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
data = pd.read_csv('filtered_dataset.csv')
active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()

features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
            'ORtg', 'DRtg', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%',
            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']
target = 'VORP'
class NBAProjectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, player_emb_size, num_players):
        super(NBAProjectionModel, self).__init__()

        # Player embedding
        self.player_embedding = nn.Embedding(num_players, player_emb_size)

        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(input_size + player_emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Residual connection using linear layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)  # Adjusted to match the size
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Normalization layer
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x, player_idx):
        # Ensure player_idx is within the valid range
        if torch.any(player_idx >= self.player_embedding.num_embeddings):
            raise ValueError("Player index out of range")

        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, player_emb], dim=2)
        out, _ = self.lstm(x)
        residual = out[:, -1, :]
        out = self.layer_norm(residual)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out + residual))  # Residual connection
        out = self.fc3(out)

        return out

# Example usage
num_players = data['Player'].nunique()
input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = 1
player_emb_size = 16

model = NBAProjectionModel(input_size, hidden_size, num_layers, output_size, player_emb_size, num_players)

# Update the create_sequences function to include player IDs
def create_sequences(player_data, seq_length=3):
    sequences = []
    for i in range(len(player_data) - seq_length):
        seq = player_data.iloc[i:i + seq_length][features].values
        label = player_data.iloc[i + seq_length][target]
        player_id = player_data.iloc[i + seq_length]['PlayerID']  # Assuming 'PlayerID' is a numeric ID for players
        sequences.append((seq, label, player_id))
    return sequences

sequences = []
seq_length = 3
data['PlayerID'] = data['Player'].factorize()[0]  # Create numeric player IDs
for player in data['Player'].unique():
    player_data = data[data['Player'] == player].sort_values('Season').reset_index(drop=True)
    if len(player_data) > seq_length:
        player_sequences = create_sequences(player_data, seq_length)
        sequences.extend(player_sequences)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for seq, label, player_id in sequences:
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        player_id = torch.tensor([player_id], dtype=torch.long)
        
        optimizer.zero_grad()
        output = model(seq, player_id)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for seq, label, player_id in sequences:
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            player_id = torch.tensor([player_id], dtype=torch.long)
            
            val_output = model(seq, player_id)
            val_loss += criterion(val_output, label).item()
        
    history['loss'].append(total_loss / len(sequences))
    history['val_loss'].append(val_loss / len(sequences))
    history['mae'].append(torch.mean(torch.abs(output - label)).item())
    history['val_mae'].append(torch.mean(torch.abs(val_output - label)).item())

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(sequences)}, Val Loss: {val_loss / len(sequences)}')

def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

# Plot the training history
plot_training_history(history)