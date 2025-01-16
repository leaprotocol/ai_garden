import { createReadStream } from 'fs';
import { stat, readdir } from 'fs/promises';
import { join } from 'path';

export async function processSource(path, options) {
  const { sourceType, chunkSize = 1000, onProgress } = options;
  const chunks = [];

  async function processFile(path) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      let content = '';
      
      const stream = createReadStream(path, { encoding: 'utf8' });
      
      stream.on('data', (chunk) => {
        content += chunk;
        while (content.length >= chunkSize) {
          chunks.push(content.slice(0, chunkSize));
          content = content.slice(chunkSize);
        }
      });
      
      stream.on('end', () => {
        if (content.length > 0) {
          chunks.push(content);
        }
        resolve(chunks);
      });
      
      stream.on('error', reject);
    });
  }

  async function processDirectory(path, currentDepth = 0) {
    const entries = await readdir(path, { withFileTypes: true });
    const chunks = [];
    let processed = 0;

    for (const entry of entries) {
      const fullPath = join(path, entry.name);
      
      if (entry.isDirectory()) {
        chunks.push(...await processDirectory(fullPath, currentDepth + 1));
      } else if (entry.isFile()) {
        chunks.push(...await processFile(fullPath));
      }
      
      processed++;
      onProgress?.(Math.floor((processed / entries.length) * 100));
    }

    return chunks;
  }

  const stats = await stat(path);
  if (stats.isDirectory()) {
    return processDirectory(path);
  } else {
    return processFile(path);
  }
} 