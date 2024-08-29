import blessed from 'blessed';

export function setupBlessedUI() {
    const screen = blessed.screen({
        smartCSR: true,
        title: 'WebSocket Server Monitor'
    });

    const statusBox = blessed.box({
        top: 0,
        left: 0,
        width: '100%',
        height: '10%',
        content: 'Server is starting...',
        tags: true,
        border: {
            type: 'line'
        },
        style: {
            border: {
                fg: 'green'
            }
        }
    });

    const clientsBox = blessed.list({
        top: '10%',
        left: 0,
        width: '100%',
        height: '30%',
        label: 'Connected Clients',
        tags: true,
        border: {
            type: 'line'
        },
        style: {
            selected: {
                bg: 'blue'
            },
            border: {
                fg: 'green'
            }
        }
    });

    const logBox = blessed.log({
        top: '40%',
        left: 0,
        width: '100%',
        height: '60%',
        label: 'Server Logs',
        tags: true,
        border: {
            type: 'line'
        },
        style: {
            border: {
                fg: 'green'
            }
        }
    });

    screen.append(statusBox);
    screen.append(clientsBox);
    screen.append(logBox);
    logBox.focus();
    screen.render();

    return { screen, logBox, statusBox, clientsBox };
}

export function updateUIOnConnection(clientsBox, sessionId) {
    clientsBox.add(`Session: ${sessionId}`);
    clientsBox.select(clientsBox.items.length - 1); // Ensure the new item is highlighted
    clientsBox.screen.render();
}

export function updateUIOnDisconnection(clientsBox, sessionId) {
    const index = clientsBox.getItemIndex(`Session: ${sessionId}`);
    if (index >= 0) {
        clientsBox.removeItem(index);
        clientsBox.screen.render();
    }
}

export function updateUILog(logBox, message) {
    logBox.log(message);
    logBox.screen.render();
}
