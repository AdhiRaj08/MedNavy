// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import DoctorList from './components/DoctorList';
import Navbar from './components/Navbar';
import About from './components/About';

const App = () => {
  return (
    <Router>
      <div>
        <Navbar />
        <Routes>
          <Route exact path="/" element={<Home/>} />
          <Route path="/doctors" element={<DoctorList/>} />
          <Route path="/about" element={<About/>} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
