import { screen, createThreadBox, logToConsole } from './ui.js';
import { createThread } from './websocketThread.js';
import {bazosctrlc, exampleText1, lslah} from "./example_texts.js";


// Example usage
const threads = [
    {
        threadId: 1,
        systemPrompt: `You are a helpful chatbot.`,
        userPrompt: `What is universe?`
    },
    {
        threadId: 2,
        systemPrompt: `You are a helpful chatbot.`,
        userPrompt: `What is universe?`
    }
];


threads.forEach(thread => {
    let template = `
<start_of_turn>system
System:
${thread.systemPrompt}
<end_of_turn>
<start_of_turn>user 
User:
${thread.userPrompt}
<end_of_turn>
<start_of_turn>assistant
`
    logToConsole(`Starting thread ${thread.threadId}...`);
    createThread(thread.threadId, template, '');
    logToConsole(`Thread ${thread.threadId} started.`);
});

// Allow exiting the UI with 'q', 'Ctrl+C', or 'Esc'
screen.key(['q', 'C-c', 'Esc'], () => process.exit(0));
