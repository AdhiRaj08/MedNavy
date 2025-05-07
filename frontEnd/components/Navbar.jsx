import React from 'react';
import { Link } from 'react-router-dom';
import './style.css';
import logo from './R.jpeg';

const Navbar = () => {
  return (
    <nav className="navbar">
      <ul>
        <li>
        <img src={logo} alt="Togo" className='togo' />
        </li>
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/About">About</Link>
        </li>
        <li>
          <Link to="/doctors">Doctors List</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
