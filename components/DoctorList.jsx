import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './style.css';
import pogo from './img1.jpg';

const DoctorList = () => {
  const location = useLocation();
  const navigate = useNavigate(); // Initialize useNavigate
  const { symptoms } = location.state || { symptoms: '' };
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDoctors = async () => {
      try {
        const response = await axios.post('http://localhost:5000/api/doctors', { symptoms });
        setDoctors(response.data);
      } catch (err) {
        setError(1);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchDoctors();
  }, [symptoms]);

  const handleClose = () => {
    setDoctors([]);
    navigate('/');
  };

  if (loading) {
    return <div className='check' style = {{color:'darkblue', fontSize:'50px',fontFamily:'cursive', border:'none'}}>Loading...</div>;
  }

  if (error) {
    return <div> <img src={pogo} alt="Pogo" className='Pogo' /><div className='check' style = {{color:'red', fontSize:'50px',fontFamily:'roboto', border:'none' }}>Failed to fetch doctors!!<br/>Please try again later.</div></div>;
  }

  return (
    <div>
    <div className='container'>
    <div className='list'>
    <div style={{ padding: '20px' }}>
      <h1 style={{fontFamily:'sans', fontSize:'50px', color:'black', textDecoration:'underline'}}>Doctors List</h1>
      <h2 style={{ textAlign: 'justify', paddingLeft: '92px', fontFamily: 'sans', fontSize: '24px' }}>
        <span style={{ fontWeight: 'bold', textDecoration:'underline'}}>Symptoms Entered:</span>
        <span style={{ color: 'darkblue', marginLeft: '8px', fontSize:'20px', fontFamily:'cursive' }}>{symptoms}</span>
      </h2>
      <h3 style={{paddingLeft:'20px', textDecoration:'underline'}}>Recommended Doctors</h3>
      <div className="table-container">
      {doctors.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Specialization</th>
            </tr>
          </thead>
          <tbody>
            {doctors.map((doctor, index) => (
              <tr key={index}>
                <td>{doctor[0]}</td>
                <td>{doctor[1]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <tr>
            <td colSpan="2">No doctors found for the entered symptoms.</td>
        </tr>
      )}
    </div>
    </div>
    <div className='about-cta' style={{margin:'10px'}}>
      <button  onClick={handleClose}>Close</button>
      </div>
    </div>
    </div>
    </div>
  );
};

export default DoctorList;
