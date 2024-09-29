import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('filtered_dataset.csv')
projections = pd.read_csv('projected_vorp_improved.csv')

# Get the last known VORP for each player
last_known_vorp = data.groupby('Player').last()[['Season', 'VORP']].reset_index()
last_known_vorp = last_known_vorp.rename(columns={'Season': 'Last_Season', 'VORP': 'Last_VORP'})

# Merge the last known VORP with the projections
projections = projections.merge(last_known_vorp, on='Player', how='left')

# Calculate the projected VORP
projections['Projected_VORP'] = projections.groupby('Player').apply(lambda x: x['Last_VORP'] + x['Projected_VORP_diff'].cumsum()).reset_index(level=0, drop=True)

# Select top 10 players by projected VORP in the final projected season
final_season = projections['Season'].max()
top_players = projections[projections['Season'] == final_season].nlargest(10, 'Projected_VORP')['Player'].tolist()

# Filter data for top players
filtered_data = data[data['Player'].isin(top_players)]
filtered_projections = projections[projections['Player'].isin(top_players)]

# Plotting
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")
colors = sns.color_palette("husl", len(top_players))

for player, color in zip(top_players, colors):
    player_data = filtered_data[filtered_data['Player'] == player]
    player_projections = filtered_projections[filtered_projections['Player'] == player]
    
    # Plot historical data
    plt.plot(player_data['Season'], player_data['VORP'], marker='o', linestyle='-', color=color, label=player)
    
    # Plot projections
    plt.plot(player_projections['Season'], player_projections['Projected_VORP'], marker='s', linestyle='--', color=color)

plt.title('Top 10 Players: Historical VORP and Projections', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('VORP', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('vorp_projections.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print the projected VORP for the next 5 seasons for all players
print("Projected VORP for All Players (Next 5 Seasons):")
for player in projections['Player'].unique():
    player_projections = projections[projections['Player'] == player].sort_values('Season')
    print(f"\n{player}:")
    for _, row in player_projections.iterrows():
        print(f"  Season {row['Season']}: {row['Projected_VORP']:.2f}")

# Save the complete projections to a CSV file
projections.to_csv('complete_vorp_projections.csv', index=False)
print("\nComplete projections saved as 'complete_vorp_projections.csv'")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
data = pd.read_csv('projected_vorp_improved2.csv')
filtered_data = data.groupby('Player').head(5)
os.makedirs('plots', exist_ok=True)
for Player in filtered_data['Player'].unique():
    player_data = filtered_data[filtered_data['Player'] == Player]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=player_data, x='Season', y='Projected_VORP', marker='o')
    plt.title(f'Projected VORP for {Player} (Next 5 Years)')
    plt.xlabel('Season')
    plt.ylabel('VORP')
    plt.savefig(f'plots/{Player}_projected_net_rating.png')
    plt.close()

print("Plots saved in the /plots directory.")