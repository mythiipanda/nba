import React from 'react';

const PlayerCard = ({ player }) => {
  return (
    <div className="border border-gray-300 rounded-lg p-4 flex space-x-4">
      <img src={player.imageUrl} alt={player.name} className="w-24 h-24 rounded-full" />
      <div>
        <h2 className="text-xl font-bold">{player.name}</h2>
        <p>Position: {player.position}</p>
        <p>Team: {player.team}</p>
      </div>
    </div>
  );
};

export default PlayerCard;
