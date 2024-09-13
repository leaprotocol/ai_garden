import blessed from 'blessed';

// Create a blessed screen object
const screen = blessed.screen({
    smartCSR: true,
    title: 'Client Summarization Monitor'
});

// Array to keep track of each thread's UI box
let threadBoxes = [];

// Create a global console log box
const consoleLogBox = blessed.log({
    bottom: 0,
    left: 0,
    width: '100%',
    height: '50%',  // Occupies the bottom 20% of the screen
    label: 'Console Logs',
    tags: true,
    border: {
        type: 'line'
    },
    style: {
        fg: 'white',
        border: {
            fg: '#f0f0f0'
        }
    },
    scrollable: true,  // Allow scrolling if content exceeds box height
    alwaysScroll: true,
    scrollbar: {
        ch: ' ',        // Scrollbar character
        inverse: true   // Scrollbar style
    },
    padding: {
        left: 1,
        right: 1
    },
    wrap: true // Ensures text wraps within the box
});

// Function to create a UI box for each thread with Markdown support
export function createThreadBox(threadId) {
    const box = blessed.box({
        top: `${(threadId - 1) * 25}%`,  // Adjusted for up to 3 threads, giving space for the console log box
        left: 0,
        width: '100%',
        height: '25%',  // Height adjusted to fit 3 threads plus the console log
        label: `Thread ${threadId} Output`,
        tags: true,
        border: {
            type: 'line'
        },
        style: {
            fg: 'white',
            border: {
                fg: '#f0f0f0'
            }
        },
        scrollable: true,  // Allow scrolling if content exceeds box height
        alwaysScroll: true,
        scrollbar: {
            ch: ' ',        // Scrollbar character
            inverse: true   // Scrollbar style
        },
        padding: {
            left: 1,
            right: 1
        },
        wrap: true // Ensures text wraps within the box
    });

    // Bind keys to scroll up and down
    box.key(['up', 'k'], function(ch, key) {
        box.scroll(-1);
        screen.render();
    });

    box.key(['down', 'j'], function(ch, key) {
        box.scroll(1);
        screen.render();
    });

    box.key(['pageup'], function(ch, key) {
        box.scroll(-box.height);
        screen.render();
    });

    box.key(['pagedown'], function(ch, key) {
        box.scroll(box.height);
        screen.render();
    });

    screen.append(box);
    threadBoxes.push(box);
    screen.render();
    return box;
}

// Function to log messages to the global console log box
export function logToConsole(message) {
    consoleLogBox.log(message);
    screen.render();
}


// Bind keys to scroll up and down
consoleLogBox.key(['up', 'k'], function(ch, key) {
    consoleLogBox.scroll(-1);
    screen.render();
});

consoleLogBox.key(['down', 'j'], function(ch, key) {
    consoleLogBox.scroll(1);
    screen.render();
});

consoleLogBox.key(['pageup'], function(ch, key) {
    consoleLogBox.scroll(-consoleLogBox.height);
    screen.render();
});

consoleLogBox.key(['pagedown'], function(ch, key) {
    consoleLogBox.scroll(consoleLogBox.height);
    screen.render();
});
consoleLogBox.on('click', function() {
    consoleLogBox.focus();
});
// Append the console log box to the screen
screen.append(consoleLogBox);

export { screen };
