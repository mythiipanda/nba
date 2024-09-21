from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import LeagueLeaders
import pandas as pd
import numpy as np
import time
from pymongo import MongoClient

# Connect to MongoDB (replace with your connection string)
client = MongoClient('mongodb+srv://mythii:mythii@cluster0.xonkd.mongodb.net/nba-stats?retryWrites=true&w=majority')
db = client['nba_stats']
players_collection = db['players_adv']

# Fetch league leaders data for the 2023-2024 season
league_leaders = LeagueLeaders(
    league_id='00',  # nba 00, g_league 20, wnba 10
    season='2023-24',  # change year(s) if needed
    per_mode48='PerGame',  # "Totals", "Per48", "PerGame"
)
df_league_leaders = league_leaders.get_data_frames()[0]

# Convert DataFrame to dictionary
league_leaders_dict = df_league_leaders.to_dict('records')

# Insert data into MongoDB
players_collection.insert_many(league_leaders_dict)

print("League leaders data for the 2023-2024 season saved to MongoDB.")