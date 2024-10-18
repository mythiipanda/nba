// src/components/AuthButton.js
import React from 'react';
import { getAuth, signOut } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';

const AuthButton = ({ user }) => {
  const navigate = useNavigate();
  const auth = getAuth();

  const handleLogout = () => {
    signOut(auth).then(() => {
      navigate('/login');
    });
  };

  return (
    <div className="auth-button">
      {user ? (
        <button
          onClick={handleLogout}
          className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium"
        >
          Logout
        </button>
      ) : (
        <button
          onClick={() => navigate('/login')}
          className="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-lg font-medium"
        >
          Login
        </button>
      )}
    </div>
  );
};

export default AuthButton;