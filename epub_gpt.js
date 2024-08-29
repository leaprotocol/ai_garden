import { promises as fs } from 'node:fs';
import { createCompletion, loadModel } from "gpt4all";
import chalk from 'chalk';

async function loadBookStructure() {
  try {
    const jsonData = await fs.readFile('book_structure.json', 'utf-8');
    const bookStructure = JSON.parse(jsonData);
    console.log('Loaded book structure successfully.');
    return bookStructure;
  } catch (error) {
    console.error('Error reading or parsing the JSON file:', error);
  }
}

async function processBookStructureWithLLM(bookStructure) {
  const model = await loadModel(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    {
      modelPath: "/home/undefined/gpt4all/models/",
      verbose: true,
      device: "cpu",
      nCtx: 4096,
    }
  );

  const chat = await model.createChatSession({
    temperature: 0.5,
    systemPrompt: "### System:\nYou are a helpful assistant that summarizes book content clearly and concisely.\n\n",
  });

  let accumulatedContext = "";  // Store accumulated context here

  const processChapter = async (chapter, index = 0) => {
    const title = chapter.title || `Chapter ${index + 1}`;

    if (!chapter.content || chapter.content.trim() === "") {
      console.log(chalk.yellow(`Skipping empty content for: ${title}`));
    } else if (chapter.content.includes('<svg') || chapter.content.includes('<img')) {
      const trimmedContent = chapter.content.slice(0, 100).replace(/\s+/g, ' ').trim();
      console.log(chalk.yellow(`Skipping non-text content for: ${title}, Content: "${trimmedContent}..."`));
    } else {
      const contentToProcess = chapter.content.replace(/<\/?[^>]+(>|$)/g, "").trim();
      if (!contentToProcess) {
        console.log(chalk.yellow(`No textual content found after cleanup for: ${title}`));
      } else {
        const truncatedContent = contentToProcess.slice(0, 2000);

        // Combine accumulated context with the current chapter's content
        const prompt = `${accumulatedContext}\n\n### New Chapter Content:\n${truncatedContent}`;

        // Ensure the prompt doesn't exceed the context window
        const contextLimit = 4096 - 1000;  // Reserve some tokens for the response
        const finalPrompt = prompt.length > contextLimit ? prompt.slice(-contextLimit) : prompt;

        // Start timing token generation
        const startTime = Date.now();

        const res = await createCompletion(chat, finalPrompt, { nPredict: 50 });

        // End timing
        const endTime = Date.now();
        const timeTaken = (endTime - startTime) / 1000; // Time in seconds

        // Calculate speed (tokens per second)
        const tokenCount = res?.usage?.total_tokens || 0; // Assume 'usage' provides token info
        const speed = tokenCount > 0 ? (tokenCount / timeTaken).toFixed(2) : "N/A";

        // Assuming you have a token cost (e.g., $0.00001 per token)
        const tokenCost = 0.00001; // Replace with actual cost if known
        const cost = (tokenCount * tokenCost).toFixed(6);

        const responseMessage = res?.choices?.[0]?.message || "No response received";

        // Log everything on one line with different colors
        console.log(chalk.blue(`Processing chapter: ${title}`) + ' | ' +
          chalk.green(`Input Length: ${truncatedContent.length}`) + ' | ' +
          chalk.whiteBright(`Input: ${truncatedContent.slice(0,100)}`) + ' | ' +
          chalk.cyan(`Time taken: ${timeTaken}s`) + ' | ' +
          chalk.magenta(`Tokens generated: ${tokenCount}`) + ' | ' +
          chalk.yellow(`Speed: ${speed} tokens/s`) + ' | ' +
          chalk.red(`Cost: $${cost}`) + ' | ' +
          chalk.white(`LLM Response: ${JSON.stringify(responseMessage)}`));

        // Accumulate context for the next chapter
        accumulatedContext += `\n### Summary of ${title}:\n${responseMessage}`;
      }
    }

    for (const [i, subchapter] of chapter.subchapters.entries()) {
      await processChapter(subchapter, i);
    }
  };

  for (const [index, chapter] of bookStructure.entries()) {
    await processChapter(chapter, index);
  }

  model.dispose();
  console.log("Processing complete.");
}

async function main() {
  const bookStructure = await loadBookStructure();
  if (bookStructure) {
    await processBookStructureWithLLM(bookStructure);
  }
}

main();
