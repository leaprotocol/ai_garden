import React, { useState, useEffect } from 'react';
import { Handle } from 'reactflow';

function OutputNode({ id, data }) {
  const [receivedEvents, setReceivedEvents] = useState([]);
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    console.log(`OutputNode ${id}: Data changed:`, data);
    console.log('Last event:', data?.lastEvent);
    
    if (data?.lastEvent && isActive) {
      console.log(`OutputNode ${id}: Processing new event:`, data.lastEvent);
      setReceivedEvents(prev => [...prev, {
        ...data.lastEvent,
        receivedAt: new Date().toISOString()
      }]);
    }
  }, [data?.lastEvent, isActive, id]);

  const clearEvents = () => {
    console.log(`OutputNode ${id}: Clearing events`);
    setReceivedEvents([]);
  };

  return (
    <div className="output-node">
      <Handle type="target" position="left" id="input" />
      <div className="node-header">
        Output Node
        <button onClick={() => setIsActive(prev => !prev)} style={{marginLeft: '10px'}}>
          {isActive ? 'Deactivate' : 'Activate'}
        </button>
        <button onClick={clearEvents} style={{marginLeft: '10px'}}>
          Clear
        </button>
      </div>
      <div className="output-content" style={{padding: '10px'}}>
        {isActive ? (
          receivedEvents.length === 0 ? (
            <div>No events received yet</div>
          ) : (
            <div style={{maxHeight: '200px', overflowY: 'auto'}}>
              {receivedEvents.map((event, index) => (
                <div key={index} style={{marginBottom: '5px', fontSize: '12px'}}>
                  {event.formattedTime}: Event from {event.source}
                </div>
              ))}
            </div>
          )
        ) : (
          <div>Output is deactivated</div>
        )}
      </div>
    </div>
  );
}

export default OutputNode; 