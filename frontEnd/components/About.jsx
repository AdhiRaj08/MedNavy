import { useNavigate } from 'react-router-dom';
import React from 'react';
import './style.css';

const About = () => {
    
    const navigate = useNavigate(); // Initialize useNavigate
    const handleClose = () => {
        navigate('/'); // Navigate back to the main page
        };
  return (
    <div className="about-container">
      {/* Header Section */}
      <header className="about-header">
        <h1>About Us</h1>
        <p>Learn more about our mission, vision, and values</p>
      </header>

      <section className="about-content">
        <h2>Our Mission</h2>
        <p>
          We are dedicated to providing top-notch solutions and making a difference. Our mission is to
          deliver high-quality products that make life easier for our users.
        </p>

        <h2>Our Vision</h2>
        <p>
          We envision a future where technology serves humanity in meaningful ways. Our goal is to be a
          leading innovator in our industry, pushing the boundaries of what's possible.
        </p>

        <h2>Our Team</h2>
        <p>
          We have a diverse team of talented individuals who are passionate about what they do. Each team
          member brings unique skills and experiences that contribute to our success.
        </p>
      </section>

      <div className="about-cta">
        <button onClick={handleClose}>Contact Us</button>
      </div>
    </div>
  );
};

export default About;
