import React from 'react';
import { Link } from 'react-router-dom';

const PlayerRow = ({ player }) => {
  return (
    <tr>
      <td><Link to={`/player/${player.PLAYER_ID}`}>{player.PLAYER_NAME}</Link></td>
      <td>{player.TEAM_ABBREVIATION}</td>
      <td>{player.PTS}</td>
      <td>{player.REB}</td>
      <td>{player.AST}</td>
      <td>{player.STL}</td>
      <td>{player.BLK}</td>
    </tr>
  );
};

export default PlayerRow;
