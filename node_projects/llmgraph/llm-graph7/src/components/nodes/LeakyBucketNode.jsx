import React, { useCallback, useEffect, useState } from 'react';
import { Handle } from 'reactflow';
import { createEvent, EventTypes } from '../../types/events';

export default function LeakyBucketNode({ id, data }) {
  const [events, setEvents] = useState([]);
  const [isDebug, setIsDebug] = useState(false);
  const [bucketSize, setBucketSize] = useState(data.bucketSize || 5);
  const [leakInterval, setLeakInterval] = useState(data.leakInterval || 5000);
  const [isActive, setIsActive] = useState(true);

  const debugLog = useCallback((...args) => {
    if (isDebug) {
      console.log(`[LeakyBucket ${id}]`, ...args);
    }
  }, [isDebug, id]);

  // Handle incoming events
  const handleEvent = useCallback((evt) => {
    if (!isActive) return;
    
    debugLog('Received event:', evt);
    
    setEvents(currentEvents => {
      // Only add if we haven't reached bucket size
      if (currentEvents.length < bucketSize) {
        // Create a new bucket event from the incoming event
        const bucketEvent = createEvent(
          EventTypes.CHUNK,
          id,
          evt.data,
          {
            parent: evt.id,
            seq: currentEvents.length + 1
          }
        );
        debugLog('Added event to bucket:', bucketEvent);
        return [...currentEvents, bucketEvent];
      }
      debugLog('Bucket full, dropping event');
      return currentEvents;
    });
  }, [id, bucketSize, isActive, debugLog]);

  // Leak events periodically
  useEffect(() => {
    if (!isActive) return;

    const leakTimer = setInterval(() => {
      setEvents(currentEvents => {
        if (currentEvents.length > 0) {
          const [nextEvent, ...remainingEvents] = currentEvents;
          debugLog('Leaking event:', nextEvent);
          
          // Send the event through the socket
          if (data.socket) {
            data.socket.send(JSON.stringify(nextEvent));
          }
          
          return remainingEvents;
        }
        return currentEvents;
      });
    }, leakInterval);

    return () => clearInterval(leakTimer);
  }, [leakInterval, isActive, data.socket, debugLog]);

  // Handle debug toggle
  const toggleDebug = useCallback(() => {
    setIsDebug(prev => !prev);
  }, []);

  return (
    <div className="node leaky-bucket-node">
      <Handle type="target" position="left" />
      
      <div className="node-content">
        <h4>Leaky Bucket</h4>
        <div className="node-controls">
          <label>
            Bucket Size:
            <input
              type="number"
              value={bucketSize}
              onChange={e => setBucketSize(parseInt(e.target.value))}
              min="1"
            />
          </label>
          <label>
            Leak Interval (ms):
            <input
              type="number"
              value={leakInterval}
              onChange={e => setLeakInterval(parseInt(e.target.value))}
              min="100"
              step="100"
            />
          </label>
          <label>
            Active:
            <input
              type="checkbox"
              checked={isActive}
              onChange={e => setIsActive(e.target.checked)}
            />
          </label>
          <button onClick={toggleDebug}>
            {isDebug ? '🐛 Debug On' : '🪲 Debug Off'}
          </button>
        </div>
        <div className="node-status">
          Events in bucket: {events.length} / {bucketSize}
        </div>
      </div>
      
      <Handle type="source" position="right" />
    </div>
  );
} 