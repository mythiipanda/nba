from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import pandas as pd
import numpy as np
import time
from pymongo import MongoClient

# Connect to MongoDB (replace with your connection string)
client = MongoClient('mongodb+srv://mythii:mythii@cluster0.xonkd.mongodb.net/nba-stats?retryWrites=true&w=majority')
db = client['nba_stats']
players_collection = db['players']

# Convert numpy and pandas types to native Python types
def convert_to_python_type(data):
    if isinstance(data, (np.int64, np.float64)):
        return data.item()
    return data

# Fetch NBA players and their stats from nba_api and store in MongoDB
def fetch_and_store_nba_players():
    all_players = players.get_players()

    for player in all_players:
        if not player['is_active']:
            continue  # Skip inactive players

        # Check if the player already exists in the database
        existing_player = players_collection.find_one({'name': player['full_name']})
        if existing_player:
            print(f"Player {player['full_name']} already exists in the database. Skipping...")
            continue  # Skip if the player already exists

        try:
            stats = playercareerstats.PlayerCareerStats(player_id=player['id']).get_data_frames()[0]
            latest_stats = stats.iloc[-1]  # Get the latest season's stats

            player_data = {
                "name": player['full_name'],
                "team": "N/A",  # You can add team info if necessary
                "points": convert_to_python_type(latest_stats['PTS']),
                "rebounds": convert_to_python_type(latest_stats['REB']),
                "assists": convert_to_python_type(latest_stats['AST']),
                "fg_pct": convert_to_python_type(latest_stats['FG_PCT']),
                "three_pt_pct": convert_to_python_type(latest_stats['FG3_PCT']),
                "free_throw_pct": convert_to_python_type(latest_stats['FT_PCT']),
                "is_active": player['is_active']
            }

            # Insert the player data into MongoDB
            players_collection.update_one(
                {'name': player['full_name']},
                {"$set": player_data},
                upsert=True
            )
            print(f"Stored data for {player['full_name']}.")
            time.sleep(5)  # Avoid hitting rate limits
        except Exception as e:
            print(f"Error fetching stats for {player['full_name']}: {e}")

if __name__ == '__main__':
    fetch_and_store_nba_players()
    print("Player data fetched and stored in MongoDB successfully!")
