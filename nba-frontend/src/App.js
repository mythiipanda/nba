// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import HomePage from './pages/HomePage';
import PlayerPage from './pages/PlayerPage';
import AuthPage from './pages/AuthPage';
import Header from './components/Header';
import PrivateRoute from './components/PrivateRoute';

const App = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const auth = getAuth();
    const unregisterAuthObserver = onAuthStateChanged(auth, (user) => {
      setUser(user);
    });

    return () => unregisterAuthObserver(); // Cleanup subscription on unmount
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Header user={user} />
        <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<AuthPage />} />
            <Route path="/player-stats" element={<PrivateRoute element={PlayerPage} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;