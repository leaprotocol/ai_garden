import { screen, createThreadBox, logToConsole } from './ui.js';
import { createThread } from './websocketThread.js';
import {bazosctrlc, bookSmalll, exampleText1, lslah} from "./example_texts.js";

//todo: function, returns promise, resolves into a response from llm


// Example usage
const threads = [
    {
        threadId: 1,
        systemPrompt: ` You are analyzing a directory with the following files:
        
        ${lslah.slice(0,4000)}
        
        Based on this information, please suggest the next shell commands to execute. These commands outputs will be provided to LLM for processing. LLM has context window of 8K tokens. Provide the commands as a list. Output nothing else than the list.`
    },
    {
        threadId: 2,
        systemPrompt: ` You are analyzing a directory with the following files:
        
        ${lslah.slice(0,4000)}
        
        Based on this information, please suggest the next shell commands to execute. These commands outputs will be provided to LLM for processing. LLM has context window of 8K tokens. Provide the commands as a list. Output nothing else than the list.`
    },
    {
        threadId: 3,
        systemPrompt: ` You are analyzing a directory with the following files:
        
        ${lslah.slice(0,4000)}
        
        Based on this information, please suggest the next shell commands to execute. These commands outputs will be provided to LLM for processing. LLM has context window of 8K tokens. Provide the commands as a list. Output nothing else than the list.`
    }
];

let timeStart = Date.now();
for (const thread of threads) {

    let template = `
${thread.systemPrompt?`<start_of_turn>system
System:
${thread.systemPrompt}
<end_of_turn>`:``}
${thread.userPrompt?`<start_of_turn>user
User:
${thread.userPrompt}
<end_of_turn>
<start_of_turn>assistant`:``}
`
    let tmp = createThread(thread.threadId, template);
    tmp.then(logToConsole)

}


let timeEnd = Date.now();
let timeDiff = timeEnd - timeStart;
logToConsole(timeDiff);
logToConsole(`Per request: ${timeDiff/threads.length} ms`);

// Allow exiting the UI with 'q', 'Ctrl+C', or 'Esc'
screen.key(['q', 'C-c', 'Esc'], () => process.exit(0));
