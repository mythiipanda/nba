import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class AdvancedNBAProjectionModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, num_players, 
                 max_age, dropout=0.1, activation="gelu", embedding_dim=64, age_embedding_dim=16):
        super().__init__()
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.age_embedding = nn.Embedding(max_age, age_embedding_dim)
        self.input_projection = nn.Linear(input_size + embedding_dim + age_embedding_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src, player_idx, age):
        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, src.size(1), 1)
        age_emb = self.age_embedding(age).unsqueeze(1).repeat(1, src.size(1), 1)
        combined_emb = torch.cat([player_emb, age_emb], dim=2)
        src = torch.cat([src, combined_emb], dim=2)
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_projection(output[:, -1, :])
        return output

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, num_players, max_age, hidden_size=64, lstm_layers=2, 
                 num_heads=4, dropout=0.1, embedding_dim=32, age_embedding_dim=16):
        super().__init__()
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.age_embedding = nn.Embedding(max_age, age_embedding_dim)
        self.lstm = nn.LSTM(input_size + embedding_dim + age_embedding_dim, hidden_size, lstm_layers, batch_first=True)
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size, input_size + embedding_dim + age_embedding_dim),
            nn.Softmax(dim=-1)
        )
        self.static_enrichment = nn.Linear(hidden_size + embedding_dim + age_embedding_dim, hidden_size)
        self.temporal_self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.gated_residual_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, player_idx, age):
        batch_size, seq_len, _ = x.shape
        player_emb = self.player_embedding(player_idx).unsqueeze(1).repeat(1, seq_len, 1)
        age_emb = self.age_embedding(age).unsqueeze(1).repeat(1, seq_len, 1)
        combined_emb = torch.cat([player_emb, age_emb], dim=2)
        x_with_emb = torch.cat([x, combined_emb], dim=2)
        
        lstm_out, _ = self.lstm(x_with_emb)
        var_weights = self.variable_selection(lstm_out)
        x_selected = x_with_emb * var_weights
        static_context = self.static_enrichment(torch.cat([lstm_out[:, -1, :], combined_emb[:, -1, :]], dim=-1))
        enriched = lstm_out + static_context.unsqueeze(1)
        attn_out, _ = self.temporal_self_attention(enriched.transpose(0, 1), enriched.transpose(0, 1), enriched.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1)
        gated_out = self.gated_residual_network(attn_out)
        output = attn_out + gated_out
        return self.final_layer(output[:, -1, :])
class EnsembleNBAProjectionModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, num_players, 
                 max_age, dropout=0.1, activation="gelu", embedding_dim=64, age_embedding_dim=16, hidden_size=64, lstm_layers=2):
        super().__init__()
        self.transformer_model = AdvancedNBAProjectionModel(input_size, d_model, nhead, num_layers, 
                                                            output_size, num_players, max_age, dropout, 
                                                            activation, embedding_dim, age_embedding_dim)
        self.tft_model = TemporalFusionTransformer(input_size, num_players, max_age, hidden_size, 
                                                   lstm_layers, nhead, dropout, embedding_dim, age_embedding_dim)
        self.ensemble_weights = nn.Parameter(torch.ones(2))

    def forward(self, x, player_idx, age):
        transformer_out = self.transformer_model(x, player_idx, age)
        tft_out = self.tft_model(x, player_idx, age)
        
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = weights[0] * transformer_out + weights[1] * tft_out
        return ensemble_out


def preprocess(data):
    features = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
                'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'ORtg', 'DRtg', 
                'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 
                'WS', 'WS/48', 'OBPM', 'DBPM', 'VORP']
    scaler = StandardScaler()
    data['BPM_diff'] = data.groupby('Player')['BPM'].diff().fillna(0)
    data[features] = scaler.fit_transform(data[features])
    pca = PCA(n_components=0.95)
    pca.fit(data[features])
    importance = np.abs(pca.components_).sum(axis=0)
    features = [f for f, i in zip(features, importance) if i > 0]
    player_mapping = {player: idx for idx, player in enumerate(data['Player'].unique())}
    data['player_idx'] = data['Player'].map(player_mapping)
    features = ['Age', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
            '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 
            'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'ORtg', 'DRtg', 
            'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
            'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 
            'WS', 'WS/48', 'OBPM', 'DBPM', 'VORP']
    return data, features, player_mapping, scaler, pca

def create_sequences(data, features, sequence_length=4):
    X, y, player_idx, age = [], [], [], []
    for player in data['Player'].unique():
        player_data = data[data['Player'] == player].sort_values('Season')
        for i in range(len(player_data) - sequence_length):
            X.append(player_data[features].iloc[i:i+sequence_length].values)
            y.append(player_data['BPM_diff'].iloc[i+sequence_length])
            player_idx.append(player_data['player_idx'].iloc[i+sequence_length])
            age.append(player_data['Age'].iloc[i+sequence_length])
    return np.array(X), np.array(y), np.array(player_idx), np.array(age)

def train_model(model, X_train, y_train, player_idx_train, age_train, X_val, y_val, player_idx_val, age_val, epochs, batch_size, lr, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=math.ceil(len(X_train) / batch_size))
    model.to(device)
    X_train, y_train, player_idx_train, age_train = X_train.to(device), y_train.to(device), player_idx_train.to(device), age_train.to(device)
    X_val, y_val, player_idx_val, age_val = X_val.to(device), y_val.to(device), player_idx_val.to(device), age_val.to(device)
    
    best_val_loss = float('inf')
    patience = 1000000
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            player_idx_batch = player_idx_train[i:i+batch_size]
            age_batch = age_train[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(X_batch, player_idx_batch, age_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X_train) // batch_size)
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val, player_idx_val, age_val)
            val_loss = criterion(val_output.squeeze(), y_val)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model, X_test, y_test, player_idx_test, age_test, device):
    model.eval()
    X_test, y_test, player_idx_test, age_test = X_test.to(device), y_test.to(device), player_idx_test.to(device), age_test.to(device)
    with torch.no_grad():
        predictions = model(X_test, player_idx_test, age_test).squeeze()
        mse = nn.MSELoss()(predictions, y_test)
        mae = nn.L1Loss()(predictions, y_test)
    print(f'Test MSE: {mse.item():.4f}, MAE: {mae.item():.4f}')
    return predictions

def project_future_BPM(player_data, model, features, seq_length, player_idx, age, device):
    projections = []
    last_seasons = player_data.iloc[-seq_length:][features].values
    last_season = player_data['Season'].max()
    player_name = player_data['Player'].iloc[0]
    trend_sum = 0
    num_trends = 0

    for i in range(1, 6):
        input_sequence = last_seasons.reshape(1, seq_length, -1)
        input_sequence = torch.from_numpy(input_sequence).float().to(device)
        BPM_prediction = model(input_sequence, torch.tensor([player_idx]).to(device), torch.tensor([age]).to(device)).item()
        if projections:
            trend = BPM_prediction - projections[-1]['Projected_BPM_diff']
            trend_sum += trend
            num_trends += 1
            average_trend = trend_sum / num_trends
            BPM_prediction += average_trend
        
        new_season = last_season + i
        projections.append({
            'Player': player_name,
            'Season': new_season,
            'Projected_BPM_diff': BPM_prediction,
            'Age': age + i  # Increment age for each future season
        })
        
        new_row = last_seasons[-1].copy()
        new_row[features.index('Age')] += 1
        last_seasons = np.vstack([last_seasons[1:], new_row])

    return projections

def main():
    data = pd.read_csv('filtered_dataset.csv')
    data, features, player_mapping, scaler, pca = preprocess(data)
    X, y, player_idx, age = create_sequences(data, features)

    print(f"Length of X: {len(X)}")
    print(f"Length of y: {len(y)}")
    print(f"Length of player_idx: {len(player_idx)}")
    print(f"Length of age: {len(age)}")

    train_size = int(len(X) * 0.99)
    val_size = int(len(X) * 0.01)
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    player_idx_train, player_idx_val, player_idx_test = player_idx[:train_size], player_idx[train_size:train_size+val_size], player_idx[train_size+val_size:]
    age_train, age_val, age_test = age[:train_size], age[train_size:train_size+val_size], age[train_size+val_size:]
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)
    player_idx_train = torch.LongTensor(player_idx_train)
    player_idx_val = torch.LongTensor(player_idx_val)
    player_idx_test = torch.LongTensor(player_idx_test)
    age_train = torch.LongTensor(age_train)
    age_val = torch.LongTensor(age_val)
    age_test = torch.LongTensor(age_test)

    input_size = len(features)
    d_model = 128
    nhead = 8
    num_layers = 4
    output_size = 1
    num_players = len(player_mapping)
    max_age = int(data['Age'].max() + 1)  # Ensure max_age is an integer
    dropout = 0.1
    activation = "gelu"
    embedding_dim = 64
    age_embedding_dim = 16
    hidden_size = 64
    lstm_layers = 2
    epochs = 50
    batch_size = 64
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnsembleNBAProjectionModel(input_size, d_model, nhead, num_layers, output_size, 
                                   num_players, max_age, dropout, activation, embedding_dim, 
                                   age_embedding_dim, hidden_size, lstm_layers)
    train_model(model, X_train, y_train, player_idx_train, age_train, X_val, y_val, player_idx_val, age_val, epochs, batch_size, learning_rate, device)
    predictions = evaluate_model(model, X_test, y_test, player_idx_test, age_test, device)

    active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()
    seq_length = 4
    future_projections = []
    for player in active_players:
        player_data = data[data['Player'] == player].sort_values('Season')
        if len(player_data) > seq_length:
            player_idx = player_mapping[player]
            age = player_data['Age'].iloc[-1]
            projections = project_future_BPM(player_data, model, features, seq_length, player_idx, age, device)
            future_projections.extend(projections)

    future_df = pd.DataFrame(future_projections)
    future_df.to_csv('projected_bpm_improved.csv', index=False)

    print("Projections completed and saved to 'projected_bpm_improved.csv'")

    model.eval()
    with torch.no_grad():
        all_outputs = model(X_test.to(device), player_idx_test.to(device), age_test.to(device)).squeeze()
        all_targets = y_test.to(device)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets.cpu().numpy(), all_outputs.cpu().numpy(), alpha=0.5)
    plt.plot([all_targets.min().cpu().numpy(), all_targets.max().cpu().numpy()], 
             [all_targets.min().cpu().numpy(), all_targets.max().cpu().numpy()], 'r--', lw=2)
    plt.xlabel("Actual BPM Difference")
    plt.ylabel("Predicted BPM Difference")
    plt.title("Actual vs Predicted BPM Difference")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()