import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';

const PlayerPage = () => {
  const [players, setPlayers] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: 'name', direction: 'ascending' });

  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/api/players');
      setPlayers(response.data);
    } catch (error) {
      console.error('Error fetching player data:', error);
    }
  };

  const sortedPlayers = [...players].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? -1 : 1;
    }
    if (a[sortConfig.key] > b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? 1 : -1;
    }
    return 0;
  });

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key === key) {
      return sortConfig.direction === 'ascending' ? <FaSortUp /> : <FaSortDown />;
    }
    return <FaSort />;
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-gray-900 mb-8 text-center">Player Stats</h1>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white mx-auto border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b cursor-pointer text-left" onClick={() => requestSort('name')}>
                Name {getSortIcon('name')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('team')}>
                Team {getSortIcon('team')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('points')}>
                Points {getSortIcon('points')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('assists')}>
                Assists {getSortIcon('assists')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('rebounds')}>
                Rebounds {getSortIcon('rebounds')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('free_throw_pct')}>
                Free Throw % {getSortIcon('free_throw_pct')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('fg_pct')}>
                Field Goal % {getSortIcon('fg_pct')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('three_pt_pct')}>
                Three Point % {getSortIcon('three_pt_pct')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((player, index) => (
              <tr key={index} className="hover:bg-gray-100">
                <td className="py-2 px-4 border-b">{player.name}</td>
                <td className="py-2 px-4 border-b text-center">{player.team}</td>
                <td className="py-2 px-4 border-b text-center">{player.points}</td>
                <td className="py-2 px-4 border-b text-center">{player.assists}</td>
                <td className="py-2 px-4 border-b text-center">{player.rebounds}</td>
                <td className="py-2 px-4 border-b text-center">{(player.free_throw_pct * 100).toFixed(1)}%</td>
                <td className="py-2 px-4 border-b text-center">{(player.fg_pct * 100).toFixed(1)}%</td>
                <td className="py-2 px-4 border-b text-center">{(player.three_pt_pct * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PlayerPage;