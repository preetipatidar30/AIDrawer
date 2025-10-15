// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyArB_ZjoFnb3Ed48v2w0f--qzK35yYF_H8",
  authDomain: "ai-drawer-3e411.firebaseapp.com",
  projectId: "ai-drawer-3e411",
  storageBucket: "ai-drawer-3e411.appspot.com",
  messagingSenderId: "891017383053",
  appId: "1:891017383053:web:8623e8efd8003810a966c4",
  measurementId: "G-MDY0H4LSL2"
};


// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const goggleAuthProvider = new GoogleAuthProvider();

// Initialize Cloud Firestore and get a reference to the service
const db = getFirestore(firebaseApp);

export { auth, goggleAuthProvider, db };
