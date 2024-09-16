import { readdir, stat, readFile } from 'fs/promises';
import path from 'path';

const totalSummary = [];
let totalCharacters = 0;

// Helper function to calculate size of file metadata (path + size information)
function calculateMetadataSize(filePath, size) {
    return filePath.length + String(size).length + 50; // Approximate size for metadata display
}

// Function to summarize a directory tree with BFS
async function summarizeDirBFS(startDir, charLimit) {
    const queue = [{ dir: startDir, depth: 1 }];
    const processedItems = [];

    // First pass: Traverse and estimate character sizes
    while (queue.length > 0) {
        const { dir, depth } = queue.shift();

        try {
            const items = await readdir(dir);

            for (const item of items) {
                const itemPath = path.join(dir, item);
                const statObj = await stat(itemPath);

                if (statObj.isDirectory()) {
                    queue.push({ dir: itemPath, depth: depth + 1 });
                    const dirMetadataSize = calculateMetadataSize(itemPath, 0);
                    processedItems.push({ path: itemPath, type: 'dir', size: dirMetadataSize });
                    totalCharacters += dirMetadataSize;
                } else if (statObj.isFile()) {
                    const fileSizeKB = (statObj.size / 1024).toFixed(2);
                    const fileMetadataSize = calculateMetadataSize(itemPath, fileSizeKB);
                    processedItems.push({ path: itemPath, type: 'file', size: fileMetadataSize, fileSizeKB });
                    totalCharacters += fileMetadataSize;
                }

                if (totalCharacters > charLimit) {
                    return processedItems; // Early exit if we exceed character limit
                }
            }
        } catch (err) {
            console.error(`Error reading directory "${dir}": ${err.message}`);
        }
    }

    return processedItems;
}

// Function to process and output the summary
async function processSummary(processedItems, charLimit, charPerFileContent) {
    let currentCharCount = 0;

    for (const item of processedItems) {
        if (currentCharCount + item.size > charLimit) {
            console.log(`Skipping remaining items. Character limit exceeded.`);
            break;
        }

        if (item.type === 'dir') {
            console.log(`Directory: "${item.path}"`);
        } else if (item.type === 'file') {
            console.log(`File: "${item.path}" (size: ${item.fileSizeKB} KB)`);

            // Process file content if under character limit
            const fileContent = await readFile(item.path, 'utf8');
            const trimmedContent = fileContent.slice(0, charPerFileContent);
            const escapedContent = trimmedContent.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
            console.log(`Content (first ${charPerFileContent} characters): "${escapedContent}"`);
        }

        currentCharCount += item.size;
    }
}

// Configuration
const charLimit = parseInt(process.argv[2], 10) || 5000; // Max characters to output
const charPerFileContent = 100; // Characters to print per file content
const targetDir = process.argv[3] || process.cwd(); // Directory from command line or current directory

(async () => {
    console.log(`Starting BFS traversal in directory: "${targetDir}" with a character limit of ${charLimit}`);

    // First pass: Get estimated items and their sizes
    const processedItems = await summarizeDirBFS(targetDir, charLimit);

    // Second pass: Output files and directories based on the character limit
    await processSummary(processedItems, charLimit, charPerFileContent);
})();
