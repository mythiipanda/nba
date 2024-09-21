import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
data = pd.read_csv('projected_next_10_seasons.csv')
filtered_data = data.groupby('player_name').head(5)
os.makedirs('plots', exist_ok=True)
for player_name in filtered_data['player_name'].unique():
    player_data = filtered_data[filtered_data['player_name'] == player_name]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=player_data, x='season', y='net_rating', marker='o')
    plt.title(f'Projected Net Rating for {player_name} (Next 5 Years)')
    plt.xlabel('Season')
    plt.ylabel('Net Rating')
    plt.savefig(f'plots/{player_name}_projected_net_rating.png')
    plt.close()

print("Plots saved in the /plots directory.")
