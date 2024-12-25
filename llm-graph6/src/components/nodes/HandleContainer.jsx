import React from 'react';

function HandleContainer({ children, isPulsating }) {
  return (
    <div className={isPulsating ? 'pulsate' : ''}>
      {children}
    </div>
  );
}

export default HandleContainer; 