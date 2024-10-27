import React, { useEffect, useState } from 'react';
import axios from 'axios';
import PlayerRow from '../components/PlayerRow';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';
import {
  ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS
} from 'react-nba-logos';

const PlayerPage = () => {
  const [players, setPlayers] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: 'PTS', direction: 'descending' });
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/players`);
      console.log(response.data); // Log the fetched data
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

  const filteredPlayers = sortedPlayers.filter(player =>
    player.player_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

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

  const getTeamColor = (team) => {
    const teamColors = {
      'ATL': '#F5B7B1', 'BOS': '#D0E7E2', 'BKN': '#BDC3C7', 'CHA': '#B2E1D7', 'CHI': '#F9B2B2',
      'CLE': '#E5B6C4', 'DAL': '#A8D8E7', 'DEN': '#B2C9E4', 'DET': '#F9B2B2', 'GSW': '#A0C3E8',
      'HOU': '#F9B2B2', 'IND': '#B2C9E4', 'LAC': '#F9B2B2', 'LAL': '#D7B3E0', 'MEM': '#A4C8E1',
      'MIA': '#F8BBD0', 'MIL': '#C4E1D5', 'MIN': '#A4D3E2', 'NOP': '#F9B2B2', 'NYK': '#FFE0B2',
      'OKC': '#A8D8E7', 'ORL': '#A4D8E1', 'PHI': '#A0C3E8', 'PHX': '#F0C3B1', 'POR': '#F5B7B1',
      'SAC': '#E6B3E0', 'SAS': '#BDC3C7', 'TOR': '#F2B2B2', 'UTA': '#B2C9E4', 'WAS': '#F2B2B2',
    };
    return teamColors[team] || '#FFFFFF';
  };

  const getTeamLogo = (team) => {
    const teamLogos = {
      'ATL': <ATL size={38} />, 'BOS': <BOS size={38} />, 'BKN': <BKN size={38} />, 'CHA': <CHA size={38} />,
      'CHI': <CHI size={38} />, 'CLE': <CLE size={38} />, 'DAL': <DAL size={38} />, 'DEN': <DEN size={38} />,
      'DET': <DET size={38} />, 'GSW': <GSW size={38} />, 'HOU': <HOU size={38} />, 'IND': <IND size={38} />,
      'LAC': <LAC size={38} />, 'LAL': <LAL size={38} />, 'MEM': <MEM size={38} />, 'MIA': <MIA size={38} />,
      'MIL': <MIL size={38} />, 'MIN': <MIN size={38} />, 'NOP': <NOP size={38} />, 'NYK': <NYK size={38} />,
      'OKC': <OKC size={38} />, 'ORL': <ORL size={38} />, 'PHI': <PHI size={38} />, 'PHX': <PHX size={38} />,
      'POR': <POR size={38} />, 'SAC': <SAC size={38} />, 'SAS': <SAS size={38} />, 'TOR': <TOR size={38} />,
      'UTA': <UTA size={38} />, 'WAS': <WAS size={38} />,
    };
    return teamLogos[team] || null;
  };
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-gray-900 mb-8 text-center">Player Stats (Per Game)</h1>
      <input
        type="text"
        placeholder="Search by player name"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="mb-4 p-2 border border-gray-300 rounded"
      />
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white mx-auto border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b cursor-pointer text-left" onClick={() => requestSort('player_name')}>
                Name {getSortIcon('player_name')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('TEAM')}>
                Team {getSortIcon('TEAM')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('GP')}>
                GP {getSortIcon('GP')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('MIN')}>
                MIN {getSortIcon('MIN')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FGM')}>
                FGM {getSortIcon('FGM')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FGA')}>
                FGA {getSortIcon('FGA')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FG_PCT')}>
                FG% {getSortIcon('FG_PCT')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FG3M')}>
                3PM {getSortIcon('FG3M')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FG3A')}>
                3PA {getSortIcon('FG3A')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FG3_PCT')}>
                3P% {getSortIcon('FG3_PCT')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FTM')}>
                FTM {getSortIcon('FTM')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FTA')}>
                FTA {getSortIcon('FTA')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('FT_PCT')}>
                FT% {getSortIcon('FT_PCT')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('OREB')}>
                OREB {getSortIcon('OREB')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('DREB')}>
                DREB {getSortIcon('DREB')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('REB')}>
                REB {getSortIcon('REB')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('AST')}>
                AST {getSortIcon('AST')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('STL')}>
                STL {getSortIcon('STL')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('BLK')}>
                BLK {getSortIcon('BLK')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('TOV')}>
                TOV {getSortIcon('TOV')}
              </th>
              <th className="py-2 px-4 border-b cursor-pointer text-center" onClick={() => requestSort('PTS')}>
                PTS {getSortIcon('PTS')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredPlayers.map((player, index) => (
              <PlayerRow key={index} player={player} getTeamColor={getTeamColor} getTeamLogo={getTeamLogo} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PlayerPage;