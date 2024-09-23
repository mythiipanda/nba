import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import plot_model
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
data = pd.read_csv('filtered_dataset.csv')
active_players = data[data['Season'].isin([2023, 2024])]['Player'].unique()
features = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
            'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 
            'ORtg', 'DRtg', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 
            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM']

def age_curve(age):
    if age < 27:
        return (age - 20) * 1.5
    elif 27 <= age <= 31:
        return 10
    else:
        return 10 - (age - 31) * 0.75

data['Age_Curve'] = data['Age'].apply(age_curve)
features.append('Age_Curve')

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

def create_improved_model(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))(inputs)
    x = LayerNormalization()(x)
    residual = x
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))(x)
    x = LayerNormalization()(x)
    x = Add()([x, residual])
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Add()([x, attention])
    x = LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

improved_model = create_improved_model(seq_length, len(features) + 1)  # +1 for 'Season'
improved_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = improved_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, verbose=1)
loss, mae = improved_model.save('vorp.ckpt')

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
        new_row[features.index('Age_Curve')] = age_curve(new_row[features.index('Age')])
        new_row[-1] = new_season
        last_seasons = np.vstack([last_seasons[1:], new_row])    
    return projections

future_projections = []
for player in active_players:
    player_data = data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        projections = project_future_vorp(player_data, improved_model, features, seq_length)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp_improved.csv', index=False)

print("Projections completed and saved to 'projected_vorp_improved.csv'")
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

plot_model(improved_model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
img = plt.imread('model_architecture.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()