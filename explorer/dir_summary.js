#!/usr/bin/env node

import { promises as fs } from 'fs';
import path from 'path';

// Function to escape special XML characters
function escapeXml(unsafe) {
    if (typeof unsafe !== 'string') {
        unsafe = String(unsafe);
    }
    // Remove null characters
    unsafe = unsafe.replace(/[\x00-\x1F\x7F-\x9F\u2000-\u200F\u2028-\u202F\u205F-\u206F\uFEFF\u0000-\u001F]/g, '');

    return unsafe.replace(/[<>&'"]/g, function (c) {
        switch (c) {
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '&': return '&amp;';
            case "'": return '&apos;';
            case '"': return '&quot;';
            default: return c;
        }
    });
}

// Function to parse command-line arguments
function parseArguments() {
    const args = process.argv.slice(2);
    const options = {
        targetDir: process.cwd(),
        includeContent: false,
        contentLength: 2000,
        maxItemsPerFolder: Infinity,
        maxDepth: Infinity,
        maxCharsPerFolder: Infinity,
        maxTotalChars: Infinity,
        ignorePaths: [],
        excludeContentTypes: [],
    };

    let i = 0;
    while (i < args.length) {
        const arg = args[i];
        switch (arg) {
            case '--include-content':
                options.includeContent = true;
                i++;
                break;
            case '--content-length':
                options.contentLength = parseInt(args[i + 1], 10);
                i += 2;
                break;
            case '--max-items-per-folder':
                options.maxItemsPerFolder = parseInt(args[i + 1], 10);
                i += 2;
                break;
            case '--max-depth':
                options.maxDepth = parseInt(args[i + 1], 10);
                i += 2;
                break;
            case '--max-chars-per-folder':
                options.maxCharsPerFolder = parseInt(args[i + 1], 10);
                i += 2;
                break;
            case '--max-total-chars':
                options.maxTotalChars = parseInt(args[i + 1], 10);
                i += 2;
                break;
            case '--ignore-path':
                if (args[i + 1]) {
                    options.ignorePaths.push(args[i + 1]);
                    i += 2;
                } else {
                    console.error('Error: --ignore-path requires a value.');
                    process.exit(1);
                }
                break;
            case '--exclude-content-types':
                if (args[i + 1]) {
                    options.excludeContentTypes = args[i + 1].split(',').map(ext => ext.trim().toLowerCase());
                    i += 2;
                } else {
                    console.error('Error: --exclude-content-types requires a comma-separated list of extensions.');
                    process.exit(1);
                }
                break;
            default:
                if (!arg.startsWith('--') && !options.targetDirSet) {
                    options.targetDir = arg;
                    options.targetDirSet = true;
                    i++;
                } else {
                    console.error(`Unknown argument: ${arg}`);
                    process.exit(1);
                }
                break;
        }
    }

    // Normalize ignore paths
    options.ignorePaths = options.ignorePaths.map(ignorePath => path.resolve(ignorePath));

    //console.log(options);
    return options;
}

// Function to check if a path should be ignored
function isIgnored(itemPath, options) {
    return options.ignorePaths.some(ignorePath => itemPath.startsWith(ignorePath));
}

// Main function to generate XML summary
async function dirToXml(dirPath, options, state, indentLevel = 2) {
    if (state.totalChars >= options.maxTotalChars) {
        return `<!-- Skipping due to max total characters limit (${state.totalChars}/${options.maxTotalChars}) -->\n`;
    }
    if (state.currentDepth > options.maxDepth) {
        return `<!-- Skipping due to max depth (${state.currentDepth}/${options.maxDepth}) -->\n`;
    }

    const indent = ' '.repeat(indentLevel);
    let xmlOutput = '';
    let folderChars = 0;
    let itemsProcessed = 0;

    try {
        let items = await fs.readdir(dirPath);
        items = items.slice(0, options.maxItemsPerFolder); // Limit items per folder

        for (const item of items) {
            const itemPath = path.join(dirPath, item);
            const resolvedItemPath = path.resolve(itemPath);

            // Check if the item should be ignored
            if (isIgnored(resolvedItemPath, options)) {
                xmlOutput += `${indent}<!-- Skipped "${escapeXml(item)}" due to ignore path -->\n`;
                continue;
            }

            let itemStat;
            try {
                itemStat = await fs.stat(itemPath);
            } catch (err) {
                xmlOutput += `${indent}<!-- Error accessing "${escapeXml(item)}": ${escapeXml(err.message)} -->\n`;
                continue;
            }

            let itemXml = '';
            if (itemStat.isDirectory()) {
                state.currentDepth++;
                itemXml += `${indent}<directory name="${escapeXml(item)}">\n`;
                const childXml = await dirToXml(itemPath, options, state, indentLevel + 2);
                itemXml += childXml;
                itemXml += `${indent}</directory>\n`;
                state.currentDepth--;
            } else if (itemStat.isFile()) {
                const sizeKB = (itemStat.size / 1024).toFixed(2);
                itemXml += `${indent}<file name="${escapeXml(item)}" size="${sizeKB} KB"`;

                if (options.includeContent && !options.excludeContentTypes.includes(path.extname(item).toLowerCase())) {
                    try {
                        const content = await fs.readFile(itemPath, 'utf8');
                        const snippet = content.slice(0, options.contentLength);
                        itemXml += `>\n${indent}  <content>${escapeXml(snippet)}</content>\n`;
                        if (content.length > options.contentLength) {
                            itemXml += `${indent}  <!-- Content truncated -->\n`;
                        }
                        itemXml += `${indent}</file>\n`;
                    } catch (err) {
                        xmlOutput += `${indent}<!-- Error reading file "${escapeXml(item)}": ${escapeXml(err.message)} -->\n`;
                        itemXml += ` />\n`;
                    }
                } else {
                    itemXml += ` />\n`;
                    if (options.excludeContentTypes.includes(path.extname(item).toLowerCase())) {
                        itemXml += `${indent}<!-- Skipped content due to excluded type "${path.extname(item)}" -->\n`;
                    }
                }
            }

            const itemChars = itemXml.length;
            folderChars += itemChars;
            state.totalChars += itemChars;

            if (folderChars > options.maxCharsPerFolder) {
                xmlOutput += `${indent}<!-- Max characters per folder exceeded (${folderChars}/${options.maxCharsPerFolder}) -->\n`;
                break;
            }

            if (state.totalChars > options.maxTotalChars) {
                xmlOutput += `${indent}<!-- Max total characters exceeded (${state.totalChars}/${options.maxTotalChars}) -->\n`;
                break;
            }

            xmlOutput += itemXml;
            itemsProcessed++;

            if (itemsProcessed >= options.maxItemsPerFolder) {
                xmlOutput += `${indent}<!-- Max items per folder reached -->\n`;
                break;
            }
        }
    } catch (err) {
        xmlOutput += `${indent}<!-- Error reading directory "${escapeXml(dirPath)}": ${escapeXml(err.message)} -->\n`;
    }

    return xmlOutput;
}

(async () => {
    const options = parseArguments();
    const state = {
        totalChars: 0,
        currentDepth: 1,
    };

    const xmlContent = await dirToXml(options.targetDir, options, state);
    console.log('<?xml version="1.0" encoding="UTF-8"?>');
    console.log(`<directorySummary path="${escapeXml(options.targetDir)}" length=${xmlContent.length}>`);
    console.log(xmlContent);
    console.log('</directorySummary>');
})();
