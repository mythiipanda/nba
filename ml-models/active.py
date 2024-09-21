import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class NBAPlayerModel:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.features = ['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast', 
                         'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']

    def preprocess_data(self, data):
        # Debug: Print original season values
        print("Original season values:")
        print(data['season'].unique())
        
        data['season'] = data['season'].apply(lambda x: int(x.split('-')[1]) + 1900 if int(x.split('-')[1]) > 50 else int(x.split('-')[1]) + 2000)
        
        # Debug: Print transformed season values
        print("Transformed season values:")
        print(data['season'].unique())
        
        data = data.sort_values(['player_name', 'season'])
        return data[['player_name', 'season'] + self.features]

    def create_sequences(self, data):
        sequences = []
        targets = []
        for player in data['player_name'].unique():
            player_data = data[data['player_name'] == player][['season'] + self.features]
            if len(player_data) < self.sequence_length + 1:
                continue
            for i in range(len(player_data) - self.sequence_length):
                sequences.append(player_data.iloc[i:i+self.sequence_length][self.features].values)
                targets.append(player_data.iloc[i+self.sequence_length][self.features].values)
        return np.array(sequences), np.array(targets)

    def normalize_data(self, sequences, targets):
        flat_sequences = sequences.reshape(-1, sequences.shape[2])
        self.scaler.fit(np.vstack((flat_sequences, targets)))
        normalized_sequences = self.scaler.transform(flat_sequences).reshape(sequences.shape)
        normalized_targets = self.scaler.transform(targets)
        
        return normalized_sequences, normalized_targets

    def build_model(self, input_shape, output_shape):
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(output_shape)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history

    def predict(self, player_sequence):
        normalized_sequence = self.scaler.transform(player_sequence)
        prediction = self.model.predict(normalized_sequence.reshape(1, self.sequence_length, -1))
        return self.scaler.inverse_transform(prediction)[0]

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = np.mean((self.scaler.inverse_transform(y_test) - self.scaler.inverse_transform(predictions))**2)
        return mse

    def process_and_train(self, data_path):
        data = pd.read_csv(data_path)
        processed_data = self.preprocess_data(data)
        processed_data.to_csv('processed_data_advanced.csv')
        sequences, targets = self.create_sequences(processed_data)
        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created. Check your data and sequence length.")
        normalized_sequences, normalized_targets = self.normalize_data(sequences, targets)
        X_train, X_test, y_train, y_test = train_test_split(normalized_sequences, normalized_targets, test_size=0.2, random_state=42)
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=y_train.shape[1])
        history = self.train(X_train, y_train, X_test, y_test)
        mse = self.evaluate(X_test, y_test)
        print(f"Mean Squared Error: {mse}")
        return history

if __name__ == "__main__":
    model = NBAPlayerModel()
    try:
        history = model.process_and_train('all_seasons.csv')
        data = pd.read_csv('all_seasons.csv')
        player_data = model.preprocess_data(data)
        active_players = player_data[player_data['season'].isin([2022, 2023])]['player_name'].unique()
        # print(f"Active players: {len(active_players)}")
        # print(f"Active players list: {active_players}")
        active_player_data = player_data[player_data['player_name'].isin(active_players)]
        
        predictions = []
        for player_name in active_player_data['player_name'].unique():
            player_sequence = active_player_data[active_player_data['player_name'] == player_name].iloc[-5:][model.features].values
            if len(player_sequence) == 5:
                next_season = active_player_data[active_player_data['player_name'] == player_name]['season'].max()
                for _ in range(10):  # Project 10 more seasons
                    next_season += 1
                    prediction = model.predict(player_sequence)
                    prediction_dict = {feature: prediction[i] for i, feature in enumerate(model.features)}
                    prediction_dict['player_name'] = player_name
                    prediction_dict['season'] = next_season
                    predictions.append(prediction_dict)
                    # print(f"Predicted next season for {player_name}: {prediction_dict}")
                    new_row = np.array([prediction])
                    player_sequence = np.vstack([player_sequence[1:], new_row])
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv('projected_next_10_seasons.csv', index=False)
        print("Predictions saved to projected_next_10_seasons.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")