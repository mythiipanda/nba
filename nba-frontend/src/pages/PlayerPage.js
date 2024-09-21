import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';
import {
  ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS
} from 'react-nba-logos';

const PlayerPage = () => {
  const [players, setPlayers] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: 'PTS', direction: 'descending' });

  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/players`);
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

  const getTeamColor = (team) => {
    const teamColors = {
      'ATL': '#F5B7B1', // Atlanta Hawks
      'BOS': '#D0E7E2', // Boston Celtics
      'BKN': '#BDC3C7', // Brooklyn Nets
      'CHA': '#B2E1D7', // Charlotte Hornets
      'CHI': '#F9B2B2', // Chicago Bulls
      'CLE': '#E5B6C4', // Cleveland Cavaliers
      'DAL': '#A8D8E7', // Dallas Mavericks
      'DEN': '#B2C9E4', // Denver Nuggets
      'DET': '#F9B2B2', // Detroit Pistons
      'GSW': '#A0C3E8', // Golden State Warriors
      'HOU': '#F9B2B2', // Houston Rockets
      'IND': '#B2C9E4', // Indiana Pacers
      'LAC': '#F9B2B2', // Los Angeles Clippers
      'LAL': '#D7B3E0', // Los Angeles Lakers
      'MEM': '#A4C8E1', // Memphis Grizzlies
      'MIA': '#F8BBD0', // Miami Heat
      'MIL': '#C4E1D5', // Milwaukee Bucks
      'MIN': '#A4D3E2', // Minnesota Timberwolves
      'NOP': '#F9B2B2', // New Orleans Pelicans
      'NYK': '#FFE0B2', // New York Knicks
      'OKC': '#A8D8E7', // Oklahoma City Thunder
      'ORL': '#A4D8E1', // Orlando Magic
      'PHI': '#A0C3E8', // Philadelphia 76ers
      'PHX': '#F0C3B1', // Phoenix Suns
      'POR': '#F5B7B1', // Portland Trail Blazers
      'SAC': '#E6B3E0', // Sacramento Kings
      'SAS': '#BDC3C7', // San Antonio Spurs
      'TOR': '#F2B2B2', // Toronto Raptors
      'UTA': '#B2C9E4', // Utah Jazz
      'WAS': '#F2B2B2', // Washington Wizards
    };
    return teamColors[team] || '#FFFFFF'; // Default to white if team color is not found
  };

  const getTeamLogo = (team) => {
    const teamLogos = {
      'ATL': <ATL size={38} />, // Atlanta Hawks
      'BOS': <BOS size={38} />, // Boston Celtics
      'BKN': <BKN size={38} />, // Brooklyn Nets
      'CHA': <CHA size={38} />, // Charlotte Hornets
      'CHI': <CHI size={38} />, // Chicago Bulls
      'CLE': <CLE size={38} />, // Cleveland Cavaliers
      'DAL': <DAL size={38} />, // Dallas Mavericks
      'DEN': <DEN size={38} />, // Denver Nuggets
      'DET': <DET size={38} />, // Detroit Pistons
      'GSW': <GSW size={38} />, // Golden State Warriors
      'HOU': <HOU size={38} />, // Houston Rockets
      'IND': <IND size={38} />, // Indiana Pacers
      'LAC': <LAC size={38} />, // Los Angeles Clippers
      'LAL': <LAL size={38} />, // Los Angeles Lakers
      'MEM': <MEM size={38} />, // Memphis Grizzlies
      'MIA': <MIA size={38} />, // Miami Heat
      'MIL': <MIL size={38} />, // Milwaukee Bucks
      'MIN': <MIN size={38} />, // Minnesota Timberwolves
      'NOP': <NOP size={38} />, // New Orleans Pelicans
      'NYK': <NYK size={38} />, // New York Knicks
      'OKC': <OKC size={38} />, // Oklahoma City Thunder
      'ORL': <ORL size={38} />, // Orlando Magic
      'PHI': <PHI size={38} />, // Philadelphia 76ers
      'PHX': <PHX size={38} />, // Phoenix Suns
      'POR': <POR size={38} />, // Portland Trail Blazers
      'SAC': <SAC size={38} />, // Sacramento Kings
      'SAS': <SAS size={38} />, // San Antonio Spurs
      'TOR': <TOR size={38} />, // Toronto Raptors
      'UTA': <UTA size={38} />, // Utah Jazz
      'WAS': <WAS size={38} />, // Washington Wizards
    };
    return teamLogos[team] || null; // Default to null if team logo is not found
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-gray-900 mb-8 text-center">Player Stats (Per Game)</h1>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white mx-auto border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b cursor-pointer text-left" onClick={() => requestSort('PLAYER')}>
                Name {getSortIcon('PLAYER')}
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
            {sortedPlayers.map((player, index) => (
              <tr key={index} className="hover:bg-gray-100" style={{ backgroundColor: getTeamColor(player.TEAM) }}>
                <td className="py-2 px-4 border-b flex items-center">
                  {getTeamLogo(player.TEAM)}
                  <span className="ml-2">{player.PLAYER}</span>
                </td>
                <td className="py-2 px-4 border-b text-center">{player.TEAM}</td>
                <td className="py-2 px-4 border-b text-center">{player.GP}</td>
                <td className="py-2 px-4 border-b text-center">{player.MIN}</td>
                <td className="py-2 px-4 border-b text-center">{player.FGM}</td>
                <td className="py-2 px-4 border-b text-center">{player.FGA}</td>
                <td className="py-2 px-4 border-b text-center">{(player.FG_PCT * 100).toFixed(1)}%</td>
                <td className="py-2 px-4 border-b text-center">{player.FG3M}</td>
                <td className="py-2 px-4 border-b text-center">{player.FG3A}</td>
                <td className="py-2 px-4 border-b text-center">{(player.FG3_PCT * 100).toFixed(1)}%</td>
                <td className="py-2 px-4 border-b text-center">{player.FTM}</td>
                <td className="py-2 px-4 border-b text-center">{player.FTA}</td>
                <td className="py-2 px-4 border-b text-center">{(player.FT_PCT * 100).toFixed(1)}%</td>
                <td className="py-2 px-4 border-b text-center">{player.OREB}</td>
                <td className="py-2 px-4 border-b text-center">{player.DREB}</td>
                <td className="py-2 px-4 border-b text-center">{player.REB}</td>
                <td className="py-2 px-4 border-b text-center">{player.AST}</td>
                <td className="py-2 px-4 border-b text-center">{player.STL}</td>
                <td className="py-2 px-4 border-b text-center">{player.BLK}</td>
                <td className="py-2 px-4 border-b text-center">{player.TOV}</td>
                <td className="py-2 px-4 border-b text-center">{player.PTS}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PlayerPage;