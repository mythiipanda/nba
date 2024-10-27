from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import pandas as pd
import time
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGODB_URL = os.getenv('MONGODB_URL')

# Connect to MongoDB
client = MongoClient(MONGODB_URL)
db = client['nba_stats']
active_players_collection = db['active_players_2024-2025']

# Function to safely fetch player career stats with retries
def fetch_player_career_stats(player_id, retries=3):
    for attempt in range(retries):
        try:
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
            return career_stats.get_data_frames()[0]
        except Exception as e:
            print(f"Attempt {attempt + 1} for player ID {player_id} failed: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch stats for player ID {player_id} after several attempts.")
                return None

# Fetch all active players
all_players = players.get_active_players()

# Initialize a list to hold active players' stats for the 2024-2025 season
active_players_stats = []

# Loop through all active players and fetch their stats for the 2024-2025 season
for player in all_players:
    player_id = player['id']
    career_stats_df = fetch_player_career_stats(player_id)
    
    if career_stats_df is not None and not career_stats_df.empty:
        # Check if the player has stats for the 2024-2025 season
        season_stats = career_stats_df[career_stats_df['SEASON_ID'] == '2024-25']
        
        if not season_stats.empty:
            # Convert the 2024-2025 season stats to dictionary and append to the list
            season_stats_dict = season_stats.iloc[0].to_dict()
            season_stats_dict['player_name'] = player['full_name']
            active_players_stats.append(season_stats_dict)
    
    # Sleep to avoid rate limits
    time.sleep(0.5)

# Convert to DataFrame and insert into MongoDB
df_active_players_stats = pd.DataFrame(active_players_stats)

# Insert data into MongoDB
active_players_collection.insert_many(df_active_players_stats.to_dict('records'))

# Save to CSV for reference
df_active_players_stats.to_csv('active_players_stats_2024-2025.csv', index=False)
print("Active players' stats for the 2024-2025 season have been fetched and saved.")
