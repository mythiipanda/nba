def project_future_vorp(player_data, model, features, seq_length):
    projections = []
    last_seasons = player_data.iloc[-seq_length:].values
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
    player_data = poly_data[data['Player'] == player].sort_values('Season')
    if len(player_data) > seq_length:
        projections = project_future_vorp(player_data, improved_model, poly_feature_names.tolist() + ['Age_Curve', 'Age_Squared', 'Experience', 'Experience_Squared'], seq_length)
        future_projections.extend(projections)

future_df = pd.DataFrame(future_projections)
future_df.to_csv('projected_vorp_improved3.csv', index=False)

print("Projections completed and saved to 'projected_vorp_improved3.csv'")

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