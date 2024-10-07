import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('filtered_dataset.csv')
projections = pd.read_csv('projected_bpm_improved.csv')

# Get the last known bpm for each player
last_known_bpm = data.groupby('Player').last()[['Season', 'BPM']].reset_index()
last_known_bpm = last_known_bpm.rename(columns={'Season': 'Last_Season', 'BPM': 'Last_BPM'})

# Merge the last known bpm with the projections
projections = projections.merge(last_known_bpm, on='Player', how='left')

# Calculate the projected bpm
projections['Projected_BPM'] = projections.groupby('Player').apply(lambda x: x['Last_BPM'] + x['Projected_BPM_diff'].cumsum()).reset_index(level=0, drop=True)

# Select top 10 players by projected bpm in the final projected season
final_season = projections['Season'].max()
top_players = projections[projections['Season'] == final_season].nlargest(10, 'Projected_BPM')['Player'].tolist()

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
    plt.plot(player_data['Season'], player_data['BPM'], marker='o', linestyle='-', color=color, label=player)
    
    # Plot projections
    plt.plot(player_projections['Season'], player_projections['Projected_BPM'], marker='s', linestyle='--', color=color)

plt.title('Top 10 Players: Historical BPM and Projections', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('BPM', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('bpm_projections.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print the projected bpm for the next 5 seasons for all players
print("Projected BPM for All Players (Next 5 Seasons):")
for player in projections['Player'].unique():
    player_projections = projections[projections['Player'] == player].sort_values('Season')
    print(f"\n{player}:")
    for _, row in player_projections.iterrows():
        print(f"  Season {row['Season']}: {row['Projected_BPM']:.2f}")

# Save the complete projections to a CSV file
projections.to_csv('complete_bpm_projections.csv', index=False)
print("\nComplete projections saved as 'complete_bpm_projections.csv'")

# Additional plotting for the next 5 seasons for all players
import os
os.makedirs('plots', exist_ok=True)
for player in projections['Player'].unique():
    player_projections = projections[projections['Player'] == player].sort_values('Season').head(5)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=player_projections, x='Season', y='Projected_BPM', marker='o')
    plt.title(f'Projected BPM for {player} (Next 5 Years)')
    plt.xlabel('Season')
    plt.ylabel('BPM')
    plt.savefig(f'plots/{player}_projected_bpm.png')
    plt.close()

print("Plots saved in the /plots directory.")