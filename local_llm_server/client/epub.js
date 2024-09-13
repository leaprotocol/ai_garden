import { screen, createThreadBox, logToConsole } from './ui.js';
import { createThread } from './websocketThread.js';
import {bazosctrlc, bookSmalll, exampleText1, lslah} from "./example_texts.js";

//todo: function, returns promise, resolves into a response from llm


// Example usage
const threads = [
    {
        threadId: 1,
        systemPrompt: `
You are an AI model tasked with processing sections of an EPUB book. Your goal is twofold: first, to rewrite the provided text exactly as it is but with important words bolded for emphasis; second, to create a concise summary that captures the main ideas and key details from the text. Use the examples below as a guide for both tasks.

Example 1:
User:
"In the first chapter, the protagonist, John, discovers a hidden room in his grandfather's old house. Inside, he finds a mysterious journal that hints at a long-lost family secret."
Assistant, bolding:
"In the first chapter, the protagonist, **John**, discovers a **hidden room** in his grandfather's old house. Inside, he finds a **mysterious journal** that hints at a **long-lost family secret**."
Assistant, summary:
"**John** finds a **hidden room** and a **journal** revealing a **family secret**."

Example 2:
User:
"The second chapter explores the historical context of the town, detailing its founding by settlers in the 1800s and the subsequent economic boom due to the discovery of gold."
Assistant, bolding:
"The second chapter explores the **historical context** of the town, detailing its **founding by settlers** in the **1800s** and the subsequent **economic boom** due to the **discovery of gold**."
Assistant, summary:
"The town's **history** is shaped by **1800s settlers** and a **gold-fueled economic boom**."

Example 3:
User:
"In the third chapter, the narrative shifts to Sarah, a local historian, who uncovers evidence that challenges the town's accepted history, suggesting a cover-up of significant events."
Assistant, bolding:
"In the third chapter, the narrative shifts to **Sarah**, a **local historian**, who uncovers **evidence** that challenges the town's **accepted history**, suggesting a **cover-up** of significant events."
Assistant, summary:
"**Sarah**, a historian, finds **evidence** challenging the town's **history**, hinting at a **cover-up**."

Your Task: Based on the text provided, first rewrite it with important words bolded, and then write a concise summary that highlights the key points and main ideas. Use the examples as a guide for length and style. Adjust the examples and instructions as needed to fit the specific style and depth of processing you require.`
,
        userPrompt: `${bookSmalll.slice(0,5000)}`
    },
    {
        threadId: 2,
        systemPrompt: ` You are analyzing a directory with the following files:
        
        ${lslah.slice(0,4000)}
        
        Based on this information, please suggest the next shell commands to execute. Provide the commands as a list.`
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
