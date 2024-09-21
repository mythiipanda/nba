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
players_collection = db['players_adv_all']

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

# Fetch all players
all_players = players.get_players()

# Initialize a list to hold all career stats
all_career_stats = []

# Loop through all players and fetch their career stats
for player in all_players:
    player_id = player['id']
    career_stats_df = fetch_player_career_stats(player_id)
    
    if career_stats_df is not None and not career_stats_df.empty:
        # Convert the latest season's stats to dictionary
        latest_season_stats = career_stats_df.iloc[-1].to_dict()
        latest_season_stats['player_name'] = player['full_name']
        all_career_stats.append(latest_season_stats)
    
    # Sleep to avoid rate limits
    time.sleep(0.5)

# Convert to DataFrame and insert into MongoDB
df_all_career_stats = pd.DataFrame(all_career_stats)

# Insert data into MongoDB
players_collection.insert_many(df_all_career_stats.to_dict('records'))

# Save to CSV for reference
df_all_career_stats.to_csv('all_players_career_stats.csv', index=False)
print("All players' career stats have been fetched and saved.")
