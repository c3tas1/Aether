import React, { useState } from 'react';
import './Navbar.css'; // Import your CSS file
import { NavLink } from 'react-router-dom';

function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <NavLink to="/">Arkham</NavLink>
      </div>
      <div className='navbar-div'>
        <div className={`navbar-menu ${isMenuOpen ? 'open' : ''}`}>
            <ul className="navbar-list">
            <li className="navbar-item">
                <NavLink to="/">Home</NavLink>
            </li>
            <li className="navbar-item">
                <NavLink to="/upload">Upload</NavLink>
            </li>
            <li className="navbar-item">
                <NavLink to="/annotate">Annotate</NavLink>
            </li>
            <li className="navbar-item">
                <NavLink to="/inference">Inference</NavLink>
            </li>
            <li className="navbar-item">
                <NavLink to="/about">About</NavLink>
            </li>
            <li className="navbar-item">
                <NavLink to="/contact">Contact</NavLink>
            </li>
            </ul>
            <div className="navbar-actions">
            <button className="navbar-button">Login</button>
            <button className="navbar-button navbar-button-primary">
                Sign Up
            </button>
            </div>
        </div>
      </div>
      <button className="navbar-toggle" onClick={toggleMenu}>
        {/* Replace with your menu icon */}
        <span className="navbar-toggle-icon">☰</span>
      </button>
    </nav>
  );
}

export default Navbar;
