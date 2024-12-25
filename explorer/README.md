Certainly! I've created a flexible Node.js script that allows you to specify:

- **Max items listed per folder** (`--max-items-per-folder`)
- **Max levels (depth) reached** (`--max-depth`)
- **Max characters per folder** (`--max-chars-per-folder`)
- **Max total characters** (`--max-total-chars`)
- **Include file content snippets** (`--include-content`)
- **Content snippet length** (`--content-length`)

This script recursively traverses a directory and outputs an XML summary, respecting all specified limits.

---

### **Flexible Directory to XML Script (`dirToXml.js`)**

```javascript
#!/usr/bin/env node

import { promises as fs } from 'fs';
import path from 'path';

// Function to escape special XML characters
function escapeXml(unsafe) {
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
        contentLength: 100,
        maxItemsPerFolder: Infinity,
        maxDepth: Infinity,
        maxCharsPerFolder: Infinity,
        maxTotalChars: Infinity,
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

    return options;
}

// Main function to generate XML summary
async function dirToXml(dirPath, options, state, indentLevel = 2) {
    if (state.totalChars >= options.maxTotalChars) {
        return '';
    }
    if (state.currentDepth > options.maxDepth) {
        return '';
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
            let itemStat;

            try {
                itemStat = await fs.stat(itemPath);
            } catch (err) {
                console.error(`Error accessing "${itemPath}": ${err.message}`);
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

                if (options.includeContent) {
                    try {
                        const content = await fs.readFile(itemPath, 'utf8');
                        const snippet = content.slice(0, options.contentLength);
                        itemXml += `>\n${indent}  <content>${escapeXml(snippet)}</content>\n${indent}</file>\n`;
                    } catch (err) {
                        console.error(`Error reading file "${itemPath}": ${err.message}`);
                        itemXml += ` />\n`;
                    }
                } else {
                    itemXml += ` />\n`;
                }
            }

            const itemChars = itemXml.length;
            folderChars += itemChars;
            state.totalChars += itemChars;

            if (folderChars > options.maxCharsPerFolder) {
                xmlOutput += `${indent}<!-- Max characters per folder exceeded -->\n`;
                break;
            }

            if (state.totalChars > options.maxTotalChars) {
                xmlOutput += `${indent}<!-- Max total characters exceeded -->\n`;
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
        console.error(`Error reading directory "${dirPath}": ${err.message}`);
    }

    return xmlOutput;
}

(async () => {
    const options = parseArguments();
    const state = {
        totalChars: 0,
        currentDepth: 1,
    };

    console.log('<?xml version="1.0" encoding="UTF-8"?>');
    console.log(`<directorySummary path="${escapeXml(options.targetDir)}">`);
    const xmlContent = await dirToXml(options.targetDir, options, state);
    console.log(xmlContent);
    console.log('</directorySummary>');
})();
```

---

### **Usage Instructions**

#### **Command-Line Arguments**

- **Target Directory**: `[directoryPath]`
    - **Description**: Specify the path of the directory to summarize.
    - **Default**: Current working directory.

- **Include File Content**: `--include-content`
    - **Description**: Include snippets of file contents in the summary.

- **Content Snippet Length**: `--content-length [number]`
    - **Description**: Number of characters to include from each file's content.
    - **Default**: `100`

- **Max Items per Folder**: `--max-items-per-folder [number]`
    - **Description**: Maximum number of items (files/directories) to list per folder.
    - **Default**: Unlimited

- **Max Depth**: `--max-depth [number]`
    - **Description**: Maximum depth to traverse into subdirectories.
    - **Default**: Unlimited

- **Max Characters per Folder**: `--max-chars-per-folder [number]`
    - **Description**: Maximum number of characters to output per folder.
    - **Default**: Unlimited

- **Max Total Characters**: `--max-total-chars [number]`
    - **Description**: Maximum total number of characters to output.
    - **Default**: Unlimited

#### **Examples**

1. **Basic Usage**:

   ```bash
   node dirToXml.js /path/to/directory
   ```

2. **Include File Content Snippets**:

   ```bash
   node dirToXml.js /path/to/directory --include-content
   ```

3. **Limit Content Snippet Length**:

   ```bash
   node dirToXml.js /path/to/directory --include-content --content-length 200
   ```

4. **Limit Max Items per Folder**:

   ```bash
   node dirToXml.js /path/to/directory --max-items-per-folder 10
   ```

5. **Limit Max Depth**:

   ```bash
   node dirToXml.js /path/to/directory --max-depth 3
   ```

6. **Limit Max Characters per Folder**:

   ```bash
   node dirToXml.js /path/to/directory --max-chars-per-folder 5000
   ```

7. **Limit Max Total Characters**:

   ```bash
   node dirToXml.js /path/to/directory --max-total-chars 20000
   ```

8. **Combining Multiple Options**:

   ```bash
   node dirToXml.js /path/to/directory --include-content --content-length 100 --max-items-per-folder 5 --max-depth 2 --max-chars-per-folder 1000 --max-total-chars 5000
   ```

#### **Saving Output to a File**

To save the output to an XML file:

```bash
node dirToXml.js /path/to/directory [options] > output.xml
```

#### **Notes**

- If the script reaches any of the specified limits, it inserts comments in the XML indicating that the limit has been reached, e.g., `<!-- Max items per folder reached -->`.
- The `--max-depth` option counts the root directory as level `1`.
- The script handles errors gracefully, logging them to the console and continuing execution.

---

### **Script Explanation**

#### **1. Importing Modules**

```javascript
import { promises as fs } from 'fs';
import path from 'path';
```

- **fs.promises**: Provides promise-based versions of filesystem functions.
- **path**: Module for handling file paths.

#### **2. Command-Line Argument Parsing**

The `parseArguments` function processes the command-line arguments and sets default values for options.

- **Options Object**:
    - `targetDir`: The directory to summarize.
    - `includeContent`: Whether to include file content snippets.
    - `contentLength`: Number of characters to include from file contents.
    - `maxItemsPerFolder`: Maximum items to list per folder.
    - `maxDepth`: Maximum directory depth to traverse.
    - `maxCharsPerFolder`: Maximum characters per folder.
    - `maxTotalChars`: Maximum total characters in the output.

- **Argument Parsing Logic**:
    - Iterates over `process.argv`.
    - Updates `options` based on recognized arguments.
    - Exits with an error message if an unknown argument is encountered.

#### **3. Main Function: `dirToXml`**

This asynchronous function recursively traverses the directory tree and builds the XML output.

- **Parameters**:
    - `dirPath`: The current directory path.
    - `options`: Object containing user-specified options.
    - `state`: Object to keep track of total characters and current depth.
    - `indentLevel`: Controls the indentation for the XML output.

- **Features**:
    - **Max Depth**: Stops recursion if the current depth exceeds `maxDepth`.
    - **Max Items per Folder**: Limits the number of items processed in each folder.
    - **Max Characters per Folder**: Stops processing items in a folder if the character limit is reached.
    - **Max Total Characters**: Stops the entire process if the total character limit is reached.
    - **Include Content**: Includes file content snippets if specified.

- **State Management**:
    - Uses the `state` object to track:
        - `totalChars`: Total characters output so far.
        - `currentDepth`: Current depth in the directory tree.

#### **4. Outputting the XML**

The script prints the XML declaration and opens the root `<directorySummary>` element before calling `dirToXml`. After the recursive processing, it closes the root element.

---

### **Example Output**

Assuming the following directory structure:

```
/path/to/directory
├── file1.txt
├── file2.txt
├── subdir1
│   ├── file3.txt
│   └── file4.txt
└── subdir2
    ├── file5.txt
    └── subdir3
        └── file6.txt
```

Command:

```bash
node dirToXml.js /path/to/directory --include-content --content-length 50 --max-items-per-folder 2 --max-depth 3 --max-chars-per-folder 500 --max-total-chars 1500
```

```bash
node dir_summary.js ../../clona/repo/clona/ --ignore-path ../../clona/repo/clona/venv --ignore-path ../../clona/repo/clona/.idea --ignore-path ../../clona/repo/clona/.git --include-content --max-items-per-folder 30 --max-depth 4 --exclude-content-types .jpg,.png,.cr2,.zip,.pyc --content-length 8000 > output
```

Possible Output:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<directorySummary path="/path/to/directory">
  <file name="file1.txt" size="1.23 KB">
    <content>First 50 characters of file1.txt content...</content>
  </file>
  <file name="file2.txt" size="0.56 KB">
    <content>First 50 characters of file2.txt content...</content>
  </file>
  <!-- Max items per folder reached -->
  <directory name="subdir1">
    <file name="file3.txt" size="0.78 KB">
      <content>First 50 characters of file3.txt content...</content>
    </file>
    <file name="file4.txt" size="1.00 KB">
      <content>First 50 characters of file4.txt content...</content>
    </file>
    <!-- Max characters per folder exceeded -->
  </directory>
  <directory name="subdir2">
    <!-- Max depth reached -->
  </directory>
  <!-- Max total characters exceeded -->
</directorySummary>
```

---

### **Conclusion**

This script provides a highly flexible way to generate an XML summary of a directory's contents, with fine-grained control over various limits such as items per folder, recursion depth, and character counts. By adjusting the command-line options, you can tailor the output to your specific needs.

**If you have any further requests or need additional customization, feel free to ask!**