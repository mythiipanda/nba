import React from 'react';
import { NavLink } from 'react-router-dom';

const Header = () => {
  return (
    <header className="bg-white shadow-sm">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <span className="text-2xl font-bold text-indigo-600">NBA Analytics</span>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <NavLink 
                to="/" 
                className={({ isActive }) => 
                  isActive ? "border-indigo-500 text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium" 
                           : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium"
                }
              >
                Home
              </NavLink>
              <NavLink 
                to="/player-stats" 
                className={({ isActive }) => 
                  isActive ? "border-indigo-500 text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium" 
                           : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium"
                }
              >
                Player Stats
              </NavLink>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;