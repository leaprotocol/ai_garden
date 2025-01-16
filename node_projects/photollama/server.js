import express from 'express';
import { processPhotosWithLLM, processPhotoMultipleTimes, analyzePhotoSet, analyzePhotosSequential, analyzePhotosBatch, curatePhotoSet } from './photo_analyzer.js';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';
import { getOptions } from './config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = 3002;

// Get target directory from command line args, default to 'photos' if not provided
const targetDir = process.argv[2] || path.join(__dirname, 'photos');

// Cache for analysis results
let analysisResults = [];
let isAnalyzing = false;
let isPaused = false;

// Start analysis immediately
async function runAnalysis() {
  try {
    console.log('Starting photo analysis...');
    isAnalyzing = true;
    const options = getOptions(targetDir);

    const files = await fs.readdir(targetDir);
    const photoFiles = files.filter(file => 
      ['.jpg', '.jpeg', '.png'].includes(path.extname(file).toLowerCase())
    );

    for (const photoFile of photoFiles) {
      if (isPaused) {
        console.log('Analysis paused...');
        while (isPaused && isAnalyzing) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
        if (!isAnalyzing) break;
        console.log('Analysis resumed...');
      }

      const result = await processPhotoMultipleTimes(path.join(targetDir, photoFile), options);
      if (result) {
        analysisResults.push(result);
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log(`Analysis complete. Processed ${analysisResults.length} photos.`);
    isAnalyzing = false;
  } catch (error) {
    console.error('Analysis failed:', error);
    isAnalyzing = false;
  }
}

// Serve the HTML file from the correct directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve photos from the target directory
app.use('/photos', express.static(targetDir));

// Serve index.html for the root path
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Endpoint to get current results
app.get('/api/results', (req, res) => {
  res.json({
    results: analysisResults,
    isAnalyzing,
    total: analysisResults.length
  });
});
app.post('/api/analysis/toggle', (req, res) => {
  isPaused = !isPaused;
  res.json({ isPaused });
});
// Endpoint to analyze a set of photos
app.post('/api/analyze-set', express.json(), async (req, res) => {
  try {
    const { photos, question, mode, purpose, count } = req.body;
    
    if (!Array.isArray(photos) || photos.length === 0) {
      return res.status(400).json({ error: 'Photos array required' });
    }

    const photoPaths = photos.map(photo => path.join(targetDir, photo));
    const options = getOptions(targetDir);
    
    let result;
    if (mode === 'curate') {
      result = await curatePhotoSet(photoPaths, purpose, {
        ...options,
        count: count || 10,
        customRequirements: question
      });
    } else if (mode === 'sequential') {
      result = await analyzePhotosSequential(photoPaths, question, options);
    } else {
      result = await analyzePhotosBatch(photoPaths, question, options);
    }
    
    res.json(result);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Add pause/resume endpoint
app.post('/api/analysis/toggle', (req, res) => {
  isPaused = !isPaused;
  res.json({ isPaused });
});

// Start the server and analysis
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Looking for photos in: ${targetDir}`);
  runAnalysis();
}); 