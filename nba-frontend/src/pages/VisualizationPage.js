import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const VisualizationPage = () => {
  const [playerData, setPlayerData] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const observer = useRef();

  // Fetch data with pagination and search
  const fetchData = async (reset = false) => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/advanced-players`, {
        params: { page, name: searchQuery, limit: 20 },  // limit the results to 20 per page
      });

      if (reset) {
        setPlayerData(response.data.players);
      } else {
        setPlayerData((prevData) => [...prevData, ...response.data.players]);
      }

      setHasMore(response.data.players.length > 0);
      setPage((prevPage) => prevPage + 1);
    } catch (error) {
      setError('Error fetching player data');
      console.error('Error fetching player data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData(true);
  }, [searchQuery]);

  useEffect(() => {
    fetchData();
  }, []);

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
    setPage(1);  // Reset page to 1 when search query changes
    setHasMore(true);  // Reset hasMore to true when search query changes
  };

  const lastPlayerElementRef = useCallback((node) => {
    if (isLoading) return;
    if (observer.current) observer.current.disconnect();
    observer.current = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && hasMore) {
        fetchData();
      }
    });
    if (node) observer.current.observe(node);
  }, [isLoading, hasMore]);

  const PlayerStatsRow = ({ player }) => (
    <div className="border rounded-lg p-4 mb-4 hover:bg-gray-50 transition-colors duration-200">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <h3 className="font-bold text-lg">{player.player_name}</h3>
          <p className="text-gray-600 text-sm">
            {player.TEAM_ABBREVIATION} | {player.SEASON_ID}
          </p>
        </div>
        <div>
          <p className="text-sm">Games: {player.GP}</p>
          <p className="text-sm">Minutes: {player.MIN}</p>
        </div>
        <div>
          <p className="text-sm">Points: {player.PTS}</p>
          <p className="text-sm">Rebounds: {player.REB}</p>
        </div>
        <div>
          <p className="text-sm">Assists: {player.AST}</p>
          <p className="text-sm">FG%: {(player.FG_PCT * 100).toFixed(1)}%</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-6 border-b">
          <h2 className="text-2xl font-bold mb-4">Advanced Player Statistics</h2>
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Search players (min. 2 characters)..."
              value={searchQuery}
              onChange={handleSearchChange}
              className="w-full max-w-md px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        <div className="p-6">
          {error && (
            <div className="text-red-500 mb-4 p-4 bg-red-50 rounded-lg">
              {error}
            </div>
          )}
          {isLoading && (
            <div className="text-center py-8 text-gray-600">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              Loading...
            </div>
          )}
          <div>
            {playerData.length === 0 && !isLoading ? (
              <div className="text-center py-8 text-gray-600">
                No players found
              </div>
            ) : (
              playerData.map((player, index) => (
                <div ref={index === playerData.length - 1 ? lastPlayerElementRef : null} key={`${player.PLAYER_ID}-${player.SEASON_ID}-${index}`}>
                  <PlayerStatsRow player={player} />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizationPage;