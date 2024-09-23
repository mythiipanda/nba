import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('filtered_dataset.csv')

active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()

features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 
            'ORtg', 'DRtg', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 
            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']

target = 'VORP'

def create_sequences(player_data, seq_length):
    sequences = []
    for i in range(len(player_data) - seq_length):
        seq = player_data.iloc[i:i+seq_length][features + ['Season']].values
        label = player_data.iloc[i+seq_length][target]
        sequences.append((seq, label))
    return sequences

seq_length = 5
sequences = []

for player in data['Player'].unique():
    player_data = data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        player_sequences = create_sequences(player_data, seq_length)
        sequences.extend(player_sequences)

X, y = zip(*sequences)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(seq_length, len(features) + 1)),  # +1 for 'Season'
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

def project_future_vorp(player_data, model, features, seq_length):
    projections = []
    last_seasons = player_data.iloc[-seq_length:][features + ['Season']].values
    last_season = player_data['Season'].max()
    player_name = player_data['Player'].iloc[0]
    for i in range(1, 6):
        input_sequence = last_seasons.reshape(1, seq_length, -1)
        vorp_prediction = model.predict(input_sequence)[0][0]
        new_season = last_season + i
        projections.append({
            'Player': player_name,
            'Season': new_season,
            'Projected_VORP': vorp_prediction
        })
        new_row = last_seasons[-1].copy()
        new_row[features.index('Age')] += 1
        new_row[-1] = new_season
        last_seasons = np.vstack([last_seasons[1:], new_row])    
    return projections

future_projections = []
for player in active_players:
    player_data = data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        projections = project_future_vorp(player_data, model, features, seq_length)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp.csv', index=False)

print("Projections completed and saved to 'projected_vorp.csv'")