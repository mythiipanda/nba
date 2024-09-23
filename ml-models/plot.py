import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
data = pd.read_csv('projected_next_10_seasons.csv')
filtered_data = data.groupby('Player').head(5)
os.makedirs('plots', exist_ok=True)
for Player in filtered_data['Player'].unique():
    player_data = filtered_data[filtered_data['Player'] == Player]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=player_data, x='Season', y='VORP', marker='o')
    plt.title(f'Projected VORP for {Player} (Next 5 Years)')
    plt.xlabel('Season')
    plt.ylabel('VORP')
    plt.savefig(f'plots/{Player}_projected_net_rating.png')
    plt.close()

print("Plots saved in the /plots directory.")
