import { screen, createThreadBox, logToConsole } from './ui.js';
import { createThread } from './websocketThread.js';
import {bazosctrlc, exampleText1, lslah} from "./example_texts.js";


// Example usage
const threads = [
    {
        threadId: 1,
        systemPrompt: `You are an unhelpful chatbot.`,
        userPrompt: `What is universe?`,
        options:{
            model: `smollm:135m-instruct-v0.2-q4_K_M`,
            num_thread: 1,
        }
    },
    {
        threadId: 2,
        systemPrompt: `You are an unhelpful chatbot.`,
        userPrompt: `What is universe?`,
        options:{
            model: `gemma2:2b-instruct-q4_K_M`,
            num_thread: 1,
        }

    },
    {
        threadId: 3,
        systemPrompt: `You are an unhelpful chatbot.`,
        userPrompt: `What is universe?`,
        options:{
            model: `qwen2.5:0.5b-instruct-q4_K_M`,
            num_thread: 1,
        }
    },
    {
        threadId: 4,
        systemPrompt: `You are an unhelpful chatbot.`,
        userPrompt: `What is universe?`,
        options:{
            model: `llama3.2:1b-instruct-q4_K_M`,
            num_thread: 1,
        }
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
    createThread(thread.threadId, template, thread.options);
    logToConsole(`Thread ${thread.threadId} started.`);
});

// Allow exiting the UI with 'q', 'Ctrl+C', or 'Esc'
screen.key(['q', 'C-c', 'Esc'], () => process.exit(0));
