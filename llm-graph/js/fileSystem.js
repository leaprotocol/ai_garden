async function processSource(path, options) {
  const { sourceType, chunkSize, maxDepth, mode, onProgress } = options;
  
  async function processEpub(path) {
    const epub = new EPub(path);
    return new Promise((resolve) => {
      epub.parse();
      epub.on("end", () => {
        const chunks = [];
        let processed = 0;
        
        epub.flow.forEach((chapter) => {
          epub.getChapter(chapter.id, (error, text) => {
            if (!error) {
              const content = text.replace(/<[^>]*>/g, ' ');
              for (let i = 0; i < content.length; i += chunkSize) {
                chunks.push({
                  type: 'epub',
                  chapter: chapter.title,
                  content: content.slice(i, i + chunkSize)
                });
              }
            }
            processed++;
            onProgress(Math.floor((processed / epub.flow.length) * 100));
            if (processed === epub.flow.length) resolve(chunks);
          });
        });
      });
    });
  }

  async function processPhotos(path, depth = 0) {
    if (depth > maxDepth) return [];
    const chunks = [];
    const files = await fs.readdir(path);
    let processed = 0;

    for (const file of files) {
      const fullPath = `${path}/${file}`;
      const stat = await fs.stat(fullPath);
      
      if (stat.isDirectory()) {
        chunks.push(...await processPhotos(fullPath, depth + 1));
      } else if (/\.(jpg|jpeg|png|gif)$/i.test(file)) {
        const buffer = await fs.readFile(fullPath);
        chunks.push({
          type: 'photo',
          path: fullPath,
          size: stat.size,
          data: buffer
        });
      }
      processed++;
      onProgress(Math.floor((processed / files.length) * 100));
    }
    return chunks;
  }

  async function processProject(path, depth = 0) {
    if (depth > maxDepth) return [];
    const chunks = [];
    const files = await fs.readdir(path);
    let processed = 0;

    for (const file of files) {
      const fullPath = `${path}/${file}`;
      const stat = await fs.stat(fullPath);
      
      if (stat.isDirectory()) {
        chunks.push(...await processProject(fullPath, depth + 1));
      } else {
        const content = await fs.readFile(fullPath, 'utf-8');
        for (let i = 0; i < content.length; i += chunkSize) {
          chunks.push({
            type: 'file',
            path: fullPath,
            content: content.slice(i, i + chunkSize)
          });
        }
      }
      processed++;
      onProgress(Math.floor((processed / files.length) * 100));
    }
    return chunks;
  }

  async function processShell(command) {
    const { exec } = require('child_process');
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) reject(error);
        resolve([{
          type: 'shell',
          command,
          output: stdout,
          error: stderr
        }]);
      });
    });
  }

  switch (sourceType) {
    case 'epub': return processEpub(path);
    case 'photos': return processPhotos(path);
    case 'project': return processProject(path);
    case 'shell': return processShell(path);
    default: throw new Error(`Unknown source type: ${sourceType}`);
  }
} 