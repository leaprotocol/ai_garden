import React, { memo, useState, useEffect } from 'react';
import Node, { BaseNodeProps } from './Node';

interface OutputNodeProps extends BaseNodeProps {
  data: {
    lastEvent?: any;
    isActive?: boolean;
    onActiveChange?: (active: boolean) => void;
  } & BaseNodeProps['data'];
}

interface Event {
  source: string;
  formattedTime: string;
  receivedAt: string;
  [key: string]: any;
}

const OutputNode = memo(({ data, ...props }: OutputNodeProps) => {
  const [receivedEvents, setReceivedEvents] = useState<Event[]>([]);
  const [isActive, setIsActive] = useState(data.isActive ?? true);

  useEffect(() => {
    console.log(`OutputNode: Data changed:`, data);
    console.log('Last event:', data?.lastEvent);
    
    if (data?.lastEvent && isActive) {
      console.log(`OutputNode: Processing new event:`, data.lastEvent);
      setReceivedEvents(prev => [...prev, {
        ...data.lastEvent,
        receivedAt: new Date().toISOString()
      }]);
    }
  }, [data?.lastEvent, isActive]);

  const handleActiveToggle = () => {
    const newState = !isActive;
    setIsActive(newState);
    data.onActiveChange?.(newState);
  };

  const handleClear = () => {
    console.log('OutputNode: Clearing events');
    setReceivedEvents([]);
  };

  return (
    <Node {...props} data={data}>
      <div className="node-content">
        <div className="output-controls">
          <button 
            onClick={handleActiveToggle}
            className={`toggle-button ${isActive ? 'active' : ''}`}
          >
            {isActive ? 'Deactivate' : 'Activate'}
          </button>
          <button 
            onClick={handleClear}
            className="clear-button"
          >
            Clear
          </button>
        </div>
        <div className="output-content">
          {isActive ? (
            receivedEvents.length === 0 ? (
              <div className="no-events">No events received yet</div>
            ) : (
              <div className="events-list">
                {receivedEvents.map((event, index) => (
                  <div key={index} className="event-item">
                    {event.formattedTime}: Event from {event.source}
                  </div>
                ))}
              </div>
            )
          ) : (
            <div className="inactive-message">Output is deactivated</div>
          )}
        </div>
      </div>
    </Node>
  );
});

OutputNode.displayName = 'OutputNode';

export default OutputNode; 