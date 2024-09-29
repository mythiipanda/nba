import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
from scipy.stats import zscore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

data = pd.read_csv('filtered_dataset.csv')
active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()
data['VORP_diff'] = data.groupby('Player')['VORP'].diff().fillna(0)
features = ['Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
            '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
            'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'ORtg', 'DRtg', 
            'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
            'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 
            'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']
def age_curve(age, peak_age=27, decline_rate=0.5):
    if age < peak_age:
        return 1 + (age - 20) * 0.05
    else:
        return max(0, 1 - decline_rate * (age - peak_age))
def feature_engineering(data):
    data['Age_squared'] = data['Age'] ** 2
    data['Age_cubed'] = data['Age'] ** 3
    data['Age_Curve'] = data['Age'].apply(age_curve)
    data = data.sort_values(['Player', 'Season'])
    rolling_features = ['AGE', 'WS', 'OWS', 'DWS', 'WS/48', 'PER', 'OBPM', 'DBPM']
    for feature in rolling_features:
        data[f'{feature}_rolling_avg'] = data.groupby('Player')[feature].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    return data

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector
class NBAProjectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_players, embedding_dim=16):
        super(NBAProjectionModel, self).__init__()
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionLayer(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, player_idx):
        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, player_emb], dim=2)
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = F.relu(self.fc1(out))
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def create_sequences_for_player(player_data, seq_length=2):
    sequences = []
    for i in range(len(player_data) - seq_length):
        seq = player_data.iloc[i:i + seq_length][features].values
        label = player_data.iloc[i + seq_length]['VORP_diff']
        sequences.append((seq, label))
    return sequences

sequences = []
seq_length = 2
player_map = {player: idx for idx, player in enumerate(data['Player'].unique())}

for player in data['Player'].unique():
    player_data = data[data['Player'] == player].sort_values('Season').reset_index(drop=True)
    if len(player_data) > seq_length:
        player_sequences = create_sequences_for_player(player_data, seq_length)
        sequences.extend([(seq, label, player_map[player]) for seq, label in player_sequences])

def remove_outliers(data, features, contamination=0.01):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(data[features])
    return data[outlier_labels == 1]

data = remove_outliers(data, features + ['VORP_diff'])
# scaler = StandardScaler()
# data[features] = scaler.fit_transform(data[features])
# Save sequences to CSV
sequences_df = pd.DataFrame([
    {
        'sequence': seq.tolist(),
        'label': label,
        'player_idx': player_idx
    }
    for seq, label, player_idx in sequences
])

sequences_df.to_csv('sequences.csv', index=False)
print("Sequences saved to 'sequences.csv'")
X, y, players = zip(*sequences)
X = np.array(X)
y = np.array(y)
players = np.array(players)

X_train, X_test, y_train, y_test, players_train, players_test = train_test_split(X, y, players, test_size=0.2, random_state=42)

input_size = X_train.shape[2]
hidden_size = 128
num_layers = 3
output_size = 1
num_players = len(player_map)

model = NBAProjectionModel(input_size, hidden_size, num_layers, output_size, num_players)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
num_epochs = 1000

patience = 500
best_val_loss = float('inf')
best_model_state = None
wait = 0
fold_best_val_loss = float('inf')
X, player_idxs, y = zip(*sequences)
X = np.array(X)
y = np.array(y)
player_idxs = np.array(player_idxs)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
players_train = torch.from_numpy(players_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
players_test = torch.from_numpy(players_test).long()
model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5, verbose=True)
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train, players_train).squeeze()
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test, players_test).squeeze()
        val_loss = criterion(val_outputs, y_test)
    scheduler.step(val_loss)
    if val_loss < fold_best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

print(f"Best validation loss: {best_val_loss}")
model.load_state_dict(best_model_state)

torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to 'model_weights.pth'")


def project_future_vorp_diff(player_data, model, features, seq_length, player_idx):
    projections = []
    last_seasons = player_data.iloc[-seq_length:][features].values
    last_season = player_data['Season'].max()
    player_name = player_data['Player'].iloc[0]

    # Initialize a trend accumulator for VORP_diff
    trend_sum = 0
    num_trends = 0

    for i in range(1, 6):
        input_sequence = last_seasons.reshape(1, seq_length, -1)
        input_sequence = torch.from_numpy(input_sequence).float()
        vorp_diff_prediction = model(input_sequence, torch.tensor([player_idx])).item()
        if projections:
            trend = vorp_diff_prediction - projections[-1]['Projected_VORP_diff']
            trend_sum += trend
            num_trends += 1
            average_trend = trend_sum / num_trends
            vorp_diff_prediction += average_trend
        
        new_season = last_season + i
        projections.append({
            'Player': player_name,
            'Season': new_season,
            'Projected_VORP_diff': vorp_diff_prediction
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
        projections = project_future_vorp_diff(player_data, model, features, seq_length, player_idx)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp_improved.csv', index=False)

print("Projections completed and saved to 'projected_vorp_improved.csv'")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model.eval()
with torch.no_grad():
    all_outputs = model(torch.from_numpy(X).float(), torch.from_numpy(players).long()).squeeze()
    all_targets = torch.from_numpy(y).float()

mse = mean_squared_error(all_targets, all_outputs)
mae = mean_absolute_error(all_targets, all_outputs)
r2 = r2_score(all_targets, all_outputs)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(all_targets.numpy(), all_outputs.numpy(), alpha=0.5)
plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', lw=2)
plt.xlabel("Actual VORP Difference")
plt.ylabel("Predicted VORP Difference")
plt.title("Actual vs Predicted VORP Difference")
plt.tight_layout()
plt.show()