import { screen, createThreadBox, logToConsole } from './ui.js';
import { createThread } from './websocketThread.js';
import {bazosctrlc, exampleText1, lslah} from "./example_texts.js";

//todo: function, returns promise, resolves into a response from llm


// Example usage
const threads = [
    {
        threadId: 1,
        systemPrompt: `You are an assistant for generating linux commands for analyzing user's content. Answer only the linux command`,
        userPrompt: `This is ${lslah.slice(0,500)}`
    },
    {
        threadId: 2,
        systemPrompt: `You are an assistant for generating linux commands for analyzing user's content. Answer only the linux command`,
        userPrompt: `This is ${lslah}`
    }
];

for (const thread of threads) {
/*
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
*/
}
let resultArray = [];
let timeStart = Date.now();
let noRequests = 50;
for (let i = 0; i < noRequests; i++) {
    let template = `On a scale of 0.0 to 1.0, how bad was winston churchill. answer with nothing else but the number itself`
    let tmp = await createThread(i.threadId*1000, template);
    logToConsole(tmp);
    resultArray.push(tmp.result);
}
let timeEnd = Date.now();
logToConsole(resultArray);
let timeDiff = timeEnd - timeStart;
logToConsole(timeDiff);
logToConsole(`Per request: ${timeDiff/noRequests} ms`);

// Allow exiting the UI with 'q', 'Ctrl+C', or 'Esc'
screen.key(['q', 'C-c', 'Esc'], () => process.exit(0));
