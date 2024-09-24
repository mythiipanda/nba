import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Load the data
data = pd.read_csv('filtered_dataset.csv')
active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()
features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
            'ORtg', 'DRtg', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%',
            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']
data['Age_squared'] = data['Age'] ** 2
data['Age_cubed'] = data['Age'] ** 3
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 24, 30, 35, 40, 100], labels=['18-24', '25-30', '31-35', '36-40', '41+'])
data = pd.get_dummies(data, columns=['Age_Group'], drop_first=True)
# Scaling the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
scaled_data = pd.DataFrame(scaled_features, columns=features)

# Add non-scaled columns back to the dataframe
for col in data.columns:
    if col not in features:
        scaled_data[col] = data[col]

data[features] = scaled_data[features]
target = 'VORP'

# Model definition with player embeddings
class NBAProjectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_players, embedding_dim=16):
        super(NBAProjectionModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size//2, num_layers, batch_first=True, bidirectional=True)  
        self.player_embedding = nn.Embedding(num_players, embedding_dim)  # Embedding for players
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
    def attention(self, lstm_out, age_features):
        age_weights = F.softmax(age_features, dim=-1)  # Softmax to get attention weights
        context = torch.bmm(age_weights.unsqueeze(1), lstm_out)  # Weighted sum of LSTM outputs
        return context.squeeze(1)
    def forward(self, x, player_idx):
        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, player_emb], dim=2)
        out, _ = self.lstm(x)
        context = self.attention(out, x[:, :, features.index('Age')])
        out = self.layer_norm(context)
        out = self.layer_norm(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Creating sequences from data
def create_sequences_for_player(player_data, seq_length=3):
    sequences = []
    for i in range(len(player_data) - seq_length + 1):  # Adjusted range for correct sequence count
        seq = player_data.iloc[i:i + seq_length][features].values
        label = player_data.iloc[i + seq_length - 1][target]  # Use the label for the last year in the sequence
        sequences.append((seq, label))
    return sequences

# Preparing data for training
sequences = []
seq_length = 3
player_map = {player: idx for idx, player in enumerate(data['Player'].unique())}  # Map player to index

for player in data['Player'].unique():
    player_data = data[data['Player'] == player].sort_values('Season').reset_index(drop=True)
    if len(player_data) > seq_length:
        player_sequences = create_sequences_for_player(player_data, seq_length)
        sequences.extend([(seq, label, player_map[player]) for seq, label in player_sequences])
sequence_data = []
for seq, label, player_idx in sequences:
    sequence_data.append({
        'Player': player_idx,
        'Sequence': seq.tolist(),
        'Label': label
    })
sequences_df = pd.DataFrame(sequence_data)
sequences_df.to_csv('sequences_with_labels.csv', index=False)

print("Sequences and labels saved to 'sequences_with_labels.csv'")
X, y, players = zip(*sequences)
X = np.array(X)
y = np.array(y)
players = np.array(players)

X_train, X_test, y_train, y_test, players_train, players_test = train_test_split(X, y, players, test_size=0.2, random_state=42)

# Convert data to torch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
players_train = torch.from_numpy(players_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
players_test = torch.from_numpy(players_test).long()

# Model setup
input_size = X_train.shape[2]
hidden_size = 128
num_layers = 4
output_size = 1
num_players = len(player_map)

model = NBAProjectionModel(input_size, hidden_size, num_layers, output_size, num_players)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 1000

# Early stopping implementation
patience = 50
best_val_loss = float('inf')
best_model_state = None
wait = 0

# Training the model
history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train, players_train).squeeze()  # Ensure outputs and targets are the same shape
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test, players_test).squeeze()  # Ensure outputs and targets are the same shape
        val_loss = criterion(val_outputs, y_test)

    history['loss'].append(loss.item())
    history['val_loss'].append(val_loss.item())
    history['mae'].append(torch.mean(torch.abs(outputs - y_train)).item())
    history['val_mae'].append(torch.mean(torch.abs(val_outputs - y_test)).item())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# Load the best model state
model.load_state_dict(best_model_state)

# Plotting the training history
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
    plt.title('Model Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

def project_future_vorp(player_data, model, features, seq_length, player_idx):
    projections = []
    last_seasons = player_data.iloc[-seq_length:][features].values
    last_season = player_data['Season'].max()
    player_name = player_data['Player'].iloc[0]

    # Initialize a trend accumulator
    trend_sum = 0
    num_trends = 0

    for i in range(1, 6):
        input_sequence = last_seasons.reshape(1, seq_length, -1)
        input_sequence = torch.from_numpy(input_sequence).float()
        vorp_prediction = model(input_sequence, torch.tensor([player_idx])).item()
        if projections:
            trend = vorp_prediction - projections[-1]['Projected_VORP']
            trend_sum += trend
            num_trends += 1
            average_trend = trend_sum / num_trends
            vorp_prediction += average_trend
        new_season = last_season + i
        projections.append({
            'Player': player_name,
            'Season': new_season,
            'Projected_VORP': vorp_prediction
        })
        new_row = last_seasons[-1].copy()
        new_row[features.index('Age')] += 1
        last_seasons = np.vstack([last_seasons[1:], new_row])

    return projections

future_projections = []
for player in active_players:
    player_data = data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        player_idx = player_map[player]
        projections = project_future_vorp(player_data, model, features, seq_length, player_idx)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp_improved3.csv', index=False)

print("Projections completed and saved to 'projected_vorp_improved3.csv'")