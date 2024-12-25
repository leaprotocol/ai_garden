#!/usr/bin/env node

import path from 'path';
import { promises as fs } from 'fs';
import chalk from 'chalk';
import { Ollama } from 'ollama';
import { getOptions } from './config.js';

function parseArguments() {
  const targetDir = process.argv[2] || process.cwd();
  return getOptions(targetDir);
}

async function processPhotosWithLLM(directoryPath, options = {}) {
  const ollama = new Ollama({
    host: options.host,
    headers: options.headers
  });

  const processPhotoMultipleTimes = async (photoPath, numRuns = 3) => {
    try {
      console.log(`Starting analysis of ${photoPath}`);
      const imageData = await fs.readFile(photoPath);
      const base64Image = imageData.toString('base64');
      
      const results = [];
      
      for (let run = 0; run < numRuns; run++) {
        try {
          console.log(`Run ${run + 1}/${numRuns} for ${path.basename(photoPath)}`);
          const startTime = Date.now();
          
          const response = await ollama.generate({
            model: options.model,
            prompt: options.prompt,
            images: [base64Image],
            stream: false,
            raw: options.raw,
            options: {
              temperature: options.temperature,
              seed: 42 + run
            }
          });

          const endTime = Date.now();
          const timeTaken = (endTime - startTime) / 1000;

          if (!response || !response.response) {
            throw new Error('Empty response from Ollama');
          }

          console.log(`Raw response for ${path.basename(photoPath)} run ${run + 1}: ${response.response}`);
          
          const scores = response.response.split(',').map(Number);
          
          if (scores.some(isNaN)) {
            throw new Error('Invalid score format received');
          }

          results.push({
            scores,
            timeTaken,
            rawResponse: response.response
          });

          // Add a small delay between runs to prevent overwhelming the server
          await new Promise(resolve => setTimeout(resolve, 500));

        } catch (runError) {
          console.error(`Error in run ${run + 1} for ${path.basename(photoPath)}:`, runError);
          // Continue with next run instead of failing completely
          continue;
        }
      }

      if (results.length === 0) {
        throw new Error(`No successful runs for ${path.basename(photoPath)}`);
      }

      // Calculate averages from successful runs
      const avgScores = results[0].scores.map((_, index) => {
        const sum = results.reduce((acc, result) => acc + result.scores[index], 0);
        return (sum / results.length).toFixed(2);
      });

      const avgTimeTaken = (results.reduce((acc, result) => acc + result.timeTaken, 0) / results.length).toFixed(2);

      console.log(
        chalk.blue(`Successfully processed ${path.basename(photoPath)}`) + '\n' +
        chalk.yellow(`Temperature: ${options.temperature}`) + '\n' +
        chalk.cyan(`Average time taken: ${avgTimeTaken}s`) + '\n' +
        chalk.green('Individual responses:') + '\n' +
        results.map((result, i) => 
          chalk.white(`Run ${i + 1}: ${result.rawResponse} (${result.timeTaken.toFixed(2)}s)`)
        ).join('\n') + '\n' +
        chalk.magenta(`Average scores: ${avgScores.join(', ')}`)
      );

      return {
        photo: path.basename(photoPath),
        averageScores: avgScores,
        individualRuns: results.map(r => ({
          scores: r.scores,
          timeTaken: r.timeTaken.toFixed(2)
        })),
        averageTimeTaken: avgTimeTaken,
        temperature: options.temperature
      };

    } catch (error) {
      console.error(chalk.red(`Error processing ${path.basename(photoPath)}:`, error.message));
      return null;
    }
  };

  try {
    const files = await fs.readdir(directoryPath);
    const photoFiles = files.filter(file => 
      ['.jpg', '.jpeg', '.png'].includes(path.extname(file).toLowerCase())
    );

    console.log(`Processing ${photoFiles.length} photos...`);

    // Shuffle the photo files array
    const shuffledPhotos = [...photoFiles].sort(() => Math.random() - 0.5);
    
    // Process photos sequentially
    const results = [];
    
    for (const photoFile of shuffledPhotos) {
      console.log(`Processing photo ${shuffledPhotos.indexOf(photoFile) + 1}/${shuffledPhotos.length}`);
      const result = await processPhotoMultipleTimes(path.join(directoryPath, photoFile));
      if (result) {
        results.push(result);
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    if (results.length === 0) {
      throw new Error('No photos were successfully analyzed');
    }

    console.log(`Successfully analyzed ${results.length}/${photoFiles.length} photos`);
    return results;

  } catch (error) {
    console.error('Error in processPhotosWithLLM:', error);
    throw error;
  }
}

async function processPhotoMultipleTimes(photoPath, options = {}) {
  const ollama = new Ollama({
    host: options.host,
    headers: options.headers
  });

  try {
    console.log(`Starting analysis of ${photoPath}`);
    const imageData = await fs.readFile(photoPath);
    const base64Image = imageData.toString('base64');
    
    const results = [];
    
    for (let run = 0; run < 3; run++) {
      try {
        console.log(`Run ${run + 1}/3 for ${path.basename(photoPath)}`);
        const startTime = Date.now();
        
        const response = await ollama.generate({
          model: options.model,
          prompt: options.prompt,
          images: [base64Image],
          stream: false,
          raw: options.raw,
          options: {
            temperature: options.temperature,
            seed: 42 + run
          }
        });

        const endTime = Date.now();
        const timeTaken = (endTime - startTime) / 1000;

        if (!response || !response.response) {
          throw new Error('Empty response from Ollama');
        }

        console.log(`Raw response for ${path.basename(photoPath)} run ${run + 1}: ${response.response}`);
        
        const scores = response.response.split(',').map(Number);
        
        if (scores.some(isNaN)) {
          throw new Error('Invalid score format received');
        }

        results.push({
          scores,
          timeTaken,
          rawResponse: response.response
        });

        await new Promise(resolve => setTimeout(resolve, 500));

      } catch (runError) {
        console.error(`Error in run ${run + 1} for ${path.basename(photoPath)}:`, runError);
        continue;
      }
    }

    if (results.length === 0) {
      throw new Error(`No successful runs for ${path.basename(photoPath)}`);
    }

    const avgScores = results[0].scores.map((_, index) => {
      const sum = results.reduce((acc, result) => acc + result.scores[index], 0);
      return (sum / results.length).toFixed(2);
    });

    const avgTimeTaken = (results.reduce((acc, result) => acc + result.timeTaken, 0) / results.length).toFixed(2);

    return {
      photo: path.basename(photoPath),
      averageScores: avgScores,
      individualRuns: results.map(r => ({
        scores: r.scores,
        timeTaken: r.timeTaken.toFixed(2)
      })),
      averageTimeTaken: avgTimeTaken,
      temperature: options.temperature
    };

  } catch (error) {
    console.error(chalk.red(`Error processing ${path.basename(photoPath)}:`, error.message));
    return null;
  }
}

async function analyzePhotoSet(photoPaths, question, options = {}) {
  const ollama = new Ollama({
    host: options.host,
    headers: options.headers
  });

  try {
    console.log(chalk.blue(`\nStarting analysis of ${photoPaths.length} photos with question: "${question}"`));
    
    const messages = [];
    
    // System message
    const systemMessage = {
      role: 'system',
      content: 'You are analyzing a set of photos. Each photo will be shown separately, and you should consider all of them to answer the final question. For each photo, tell me how many photos have you seen and how many of them were similar to this one.'
    };
    messages.push(systemMessage);
    console.log(chalk.cyan('\nSystem Message:'));
    console.log(systemMessage.content);

    // Process each photo
    for (let i = 0; i < photoPaths.length; i++) {
      console.log(chalk.yellow(`\nProcessing Photo ${i + 1} of ${photoPaths.length}: ${path.basename(photoPaths[i])}`));
      
      const imageData = await fs.readFile(photoPaths[i]);
      const base64Image = imageData.toString('base64');
      
      const userMessage = {
        role: 'user',
        content: `Photo ${i + 1} of ${photoPaths.length}:`,
        images: [base64Image]
      };
      messages.push(userMessage);
      console.log(messages);
      
      console.log(chalk.green('Asking model to describe the photo...'));
      const response = await ollama.chat({
        model: options.model,
        messages: [...messages],
        stream: false,
        options: {
          temperature: options.temperature
        }
      });
      
      const assistantMessage = {
        role: 'assistant',
        content: response.message.content
      };
      messages.push(assistantMessage);
      
      console.log(chalk.magenta('Model response:'));
      console.log(assistantMessage.content);
    }

    // Ask final question
    console.log(chalk.yellow('\nAsking final question:', question));
    messages.push({
      role: 'user',
      content: question
    });

    const startTime = Date.now();
    
    const finalResponse = await ollama.chat({
      model: options.model,
      messages: messages,
      stream: false,
      options: {
        temperature: options.temperature
      }
    });

    const endTime = Date.now();
    const timeTaken = (endTime - startTime) / 1000;

    console.log(chalk.blue('\nFinal Analysis:'));
    console.log(chalk.green('Question:', question));
    console.log(chalk.magenta('Response:'));
    console.log(finalResponse.message.content);
    console.log(chalk.cyan(`Time taken: ${timeTaken.toFixed(2)}s`));

    // Log full conversation for debugging
    console.log(chalk.yellow('\nFull conversation history:'));
    messages.forEach((msg, idx) => {
      console.log(chalk.cyan(`\n[Message ${idx + 1}] Role: ${msg.role}`));
      console.log(msg.content);
      if (msg.images) console.log('<image data present>');
    });

    return {
      photos: photoPaths.map(p => path.basename(p)),
      question,
      response: finalResponse.message.content,
      timeTaken: timeTaken.toFixed(2),
      conversation: messages.map(m => ({
        role: m.role,
        content: m.content,
        hasImage: !!m.images
      }))
    };

  } catch (error) {
    console.error(chalk.red(`Error analyzing photo set:`, error.message));
    throw error;
  }
}

async function analyzePhotosSequential(photoPaths, question, options = {}) {
  const ollama = new Ollama({
    host: options.host || 'http://127.0.0.1:11434',
    headers: options.headers
  });
  
  const messages = [];
  messages.push({
    role: 'system',
    content: `You are analyzing a sequence of photos. Each photo will be shown separately.
    Rate these aspects 0-10:

Technical: (sharpness, exposure, noise)
0: unusable/corrupted
10: perfect technical execution

Composition: (balance, framing)
0: random/accidental
10: masterful composition

Artistic: (emotion, uniqueness)
0: no impact
10: museum-worthy

People: (expressions, poses)
0: awkward/unflattering
10: natural/flattering

Overall:
0: poor
10: exceptional

Everytime, output five numbers only, separated by commas.`

  });

  for (const photoPath of photoPaths) {
    const imageData = await fs.readFile(photoPath);
    const base64Image = imageData.toString('base64');
    
    messages.push({
      role: 'user',
      content: `Analyzing photo: ${path.basename(photoPath)}`,
      images: [base64Image]
    });

    const response = await ollama.chat({
      model: options.model || 'llava',
      messages: [...messages],
      stream: false
    });

    messages.push(response.message);
  }

  messages.push({
    role: 'user', 
    content: question
  });

  const finalResponse = await ollama.chat({
    model: options.model || 'llava',
    messages: messages,
    stream: false
  });

  return {
    question,
    response: finalResponse.message.content,
    conversation: messages.map(m => ({
      role: m.role,
      content: m.content,
      hasImage: !!m.images
    }))
  };
}

async function analyzePhotosBatch(photoPaths, question, options = {}) {
  const ollama = new Ollama({
    host: options.host || 'http://127.0.0.1:11434',
    headers: options.headers
  });

  try {
    console.log(chalk.blue(`\nStarting batch analysis of ${photoPaths.length} photos with question: "${question}"`));
    
    const images = await Promise.all(
      photoPaths.map(async (photoPath) => {
        const imageData = await fs.readFile(photoPath);
        return imageData.toString('base64');
      })
    );
    console.log(images.length);

    const startTime = Date.now();
    
    const response = await ollama.chat({
      model: options.model,
      messages: [
        {
          role: 'system',
          content: 'User will provide a set of photos and a question. You will analyze the photos and answer the question.'
        },
        {
          role: 'user',
          images: images
        },
        {
          role: 'user',
          content: question,
        }
      ],
      stream: false,
      options: {
        //temperature: options.temperature
      }
    });

    const endTime = Date.now();
    const timeTaken = (endTime - startTime) / 1000;
    console.log(response);
    return {
      photos: photoPaths.map(p => path.basename(p)),
      question,
      response: response.message.content,
      timeTaken: timeTaken.toFixed(2),
      conversation: [
        {
          role: 'system',
          content: 'You are analyzing multiple photos at once. Remember how many there were. Consider all photos together to answer the question.',
          hasImage: false
        },
        {
          role: 'user',
          content: question,
          hasImage: true
        },
        {
          role: 'assistant',
          content: response.message.content,
          hasImage: false
        }
      ]
    };

  } catch (error) {
    console.error(chalk.red(`Error in batch analysis:`, error.message));
    throw error;
  }
}

async function curatePhotoSet(photoPaths, purpose, options = {}) {
  const ollama = new Ollama({
    host: options.host || 'http://127.0.0.1:11434',
    headers: options.headers
  });

  const initialQuestion = `Rate each photo's suitability for ${purpose} from 0-10. ${
    options.customRequirements ? 
    `Additional requirements: ${options.customRequirements}. ` : 
    ''
  }Output only numbers separated by commas, nothing else.`;
  
  const batchResult = await analyzePhotosBatch(photoPaths, initialQuestion, options);
  
  // Extract just the numbers from the response
  const numbersOnly = batchResult.response.match(/\d+/g);
  const scores = numbersOnly ? numbersOnly.map(Number) : [];
  
  const rankedPhotos = photoPaths
    .map((photoPath, index) => ({
      path: photoPath.split('/').pop(),
      score: scores[index] || 0
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, options.count || 20);
    
  const detailQuestion = `Rate this photo for ${purpose}${
    options.customRequirements ? 
    ` with these requirements: ${options.customRequirements}.` : 
    '.'
  } Rate on:
    Impact: (attention-grabbing)
    Message: (clarity of purpose)
    Quality: (technical aspects)
    Audience: (target appeal)
    Output only four numbers 0-10, comma-separated.`;
    
  const detailedResults = await analyzePhotosSequential(
    rankedPhotos.map(p => photoPaths.find(path => path.endsWith(p.path))),
    detailQuestion,
    options
  );

  return {
    purpose,
    selectedPhotos: rankedPhotos,
    detailedAnalysis: detailedResults,
    recommendedCount: Math.min(options.count || 10, rankedPhotos.length)
  };
}

// Only run as CLI if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  const options = parseArguments();
  processPhotosWithLLM(options.targetDir, options)
    .then(() => process.exit(0))
    .catch(error => {
      console.error(error);
      process.exit(1);
    });
}
export { processPhotosWithLLM, processPhotoMultipleTimes, analyzePhotoSet, analyzePhotosSequential, analyzePhotosBatch, curatePhotoSet };