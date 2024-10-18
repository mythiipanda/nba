import React, { useEffect, useState } from 'react';
import { getAuth, onAuthStateChanged, GoogleAuthProvider, EmailAuthProvider } from 'firebase/auth';
import * as firebaseui from 'firebaseui';
import 'firebaseui/dist/firebaseui.css';
import { auth } from '../firebaseConfig';

const FirebaseAuth = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const uiConfig = {
      signInFlow: 'popup', // Use popup for sign-in flow
      signInSuccessUrl: '/', // Redirect URL after successful sign-in
      signInOptions: [
        GoogleAuthProvider.PROVIDER_ID,
        EmailAuthProvider.PROVIDER_ID,
      ],
      callbacks: {
        signInFailure: function(error) {
          if (error.code === 'auth/email-already-in-use') {
            // Handle email already in use error
            alert('This email is already in use. Please reset your password to recover your account.');
          }
        }
      },
      // Terms of service url.
      tosUrl: 'https://www.example.com/terms-of-service',
      // Privacy policy url.
      privacyPolicyUrl: 'https://www.example.com/privacy-policy'
    };

    const ui = firebaseui.auth.AuthUI.getInstance() || new firebaseui.auth.AuthUI(auth);
    ui.start('#firebaseui-auth-container', uiConfig);

    const unregisterAuthObserver = onAuthStateChanged(auth, (user) => {
      setUser(user);
    });

    return () => unregisterAuthObserver(); // Cleanup subscription on unmount
  }, []);

  return (
    <div>
      <div id="firebaseui-auth-container"></div>
      {user && (
        <div>
          <h2>Welcome, {user.displayName}</h2>
          <p>Email: {user.email}</p>
        </div>
      )}
    </div>
  );
};

export default FirebaseAuth;