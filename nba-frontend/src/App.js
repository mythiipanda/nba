// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import PlayerPage from './pages/PlayerPage';
import Header from './components/Header';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Header />
        <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/player-stats" element={<PlayerPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;