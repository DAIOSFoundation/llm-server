import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ loading }) => {
  if (!loading) {
    return null;
  }

  return (
    <div className="progress-bar-container">
      <div className="progress-bar"></div>
    </div>
  );
};

export default ProgressBar;
