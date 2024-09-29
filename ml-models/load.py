import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import IsolationForest
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class NBAProjectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, player_emb_size, num_players):
        super(NBAProjectionModel, self).__init__()

        self.player_embedding = nn.Embedding(num_players, player_emb_size)
        self.lstm = nn.LSTM(input_size + player_emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, player_idx):
        if torch.any(player_idx >= num_players):
            raise ValueError(f"Invalid player index found: {player_idx}")
        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, player_emb], dim=2)
        out, _ = self.lstm(x)
        residual = out[:, -1, :]
        out = self.layer_norm(residual)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out + residual))
        out = self.fc3(out)
        return out

# Load the data and preprocess it
data = pd.read_csv('filtered_dataset.csv')
active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()

features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
            'ORtg', 'DRtg', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%',
            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']

data['Age_Curve'] = data['Age'] ** 2

features.append('Age_Curve')

data['VORP_diff'] = data.groupby('Player')['VORP'].diff().fillna(0)

# Remove outliers function
def remove_outliers(df, features):
    iso_forest = IsolationForest(contamination=0.05)
    outlier_labels = iso_forest.fit_predict(df[features])
    return df[outlier_labels == 1]

data = remove_outliers(data, features + ['VORP_diff'])

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])
scaled_data = pd.DataFrame(scaled_features, columns=features)
for col in data.columns:
    if col not in features:
        scaled_data[col] = data[col]

data[features] = scaled_data[features]
target = 'VORP'

player_to_idx = {player: idx for idx, player in enumerate(data['Player'].unique())}
num_players = len(player_to_idx)

input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = 1
player_emb_size = 50

model = NBAProjectionModel(input_size, hidden_size, num_layers, output_size, player_emb_size, num_players)

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth'))

# Define the function to project future VORP differences
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

# Project future VORP differences
future_projections = []
for player in active_players:
    player_data = data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        player_idx = player_to_idx[player]
        projections = project_future_vorp_diff(player_data, model, features, seq_length, player_idx)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp_improved.csv', index=False)

print("Projections completed and saved to 'projected_vorp_improved.csv'")

# Evaluate the model
model.eval()
with torch.no_grad():
    all_outputs = model(torch.from_numpy(X).float(), torch.from_numpy(player_idxs).long()).squeeze()
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