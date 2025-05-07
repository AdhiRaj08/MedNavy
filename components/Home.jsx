import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './style.css';
import logo from './img2.jpeg';

const Home = () => {
  const [symptoms, setSymptoms] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent the default form submission behavior
    navigate('/doctors', { state: { symptoms } }); // Navigate to the DoctorList page with symptoms
  };

  return (
    <div>
    <img src={logo} alt="Logo" />
    <div className="container">
      
    <div className='check'>
      <h1>Welcome to MediNavi!</h1>
      <p>Please enter your symptoms below:</p>
      <form onSubmit={handleSubmit}>
        <textarea
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder="Enter your symptoms"
          rows="6"
          cols="48"
          required
          style = {{color:'blue', fontFamily:'cursive', fontSize:'15px', backgroundColor:'hsl(0, 0%, 33%, 0%)'}}
        />
        <br />
        <div className='about-cta'>
        <button style={{marginLeft:'10px'}} type="submit">Get Doctors</button>
        </div>
      </form>
    </div>
    </div>
    </div>
  );
};

export default Home;
