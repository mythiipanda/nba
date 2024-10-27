import React from 'react';

const PlayerRow = ({ player, getTeamColor, getTeamLogo }) => {
  return (
    <tr className="hover:bg-gray-100" style={{ backgroundColor: getTeamColor(player.TEAM_ABBREVIATION) }}>
      <td className="py-2 px-4 border-b flex items-center">
        {getTeamLogo(player.TEAM_ABBREVIATION)}
        <span className="ml-2">{player.player_name}</span>
      </td>
      <td className="py-2 px-4 border-b text-center">{player.TEAM_ABBREVIATION}</td>
      <td className="py-2 px-4 border-b text-center">{player.GP}</td>
      <td className="py-2 px-4 border-b text-center">{(player.MIN / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FGM / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FGA / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FG_PCT * 100).toFixed(1)}%</td>
      <td className="py-2 px-4 border-b text-center">{(player.FG3M / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FG3A / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FG3_PCT * 100).toFixed(1)}%</td>
      <td className="py-2 px-4 border-b text-center">{(player.FTM / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FTA / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.FT_PCT * 100).toFixed(1)}%</td>
      <td className="py-2 px-4 border-b text-center">{(player.OREB / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.DREB / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.REB / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.AST / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.STL / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.BLK / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.TOV / player.GP).toFixed(1)}</td>
      <td className="py-2 px-4 border-b text-center">{(player.PTS / player.GP).toFixed(1)}</td>
    </tr>
  );
};

export default PlayerRow;