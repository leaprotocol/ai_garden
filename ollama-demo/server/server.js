import express from 'express';
import { Ollama } from 'ollama';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { pipeline, AutoTokenizer } from '@huggingface/transformers';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Configure Ollama host from environment variable or use default
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
console.log(`Connecting to Ollama at: ${OLLAMA_HOST}`);

const ollama = new Ollama({
    host: OLLAMA_HOST
});

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// Log all requests
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
    next();
});

// Add endpoint to get current Ollama host
app.get('/api/config', (req, res) => {
    res.json({ ollamaHost: OLLAMA_HOST });
});

// Add token statistics tracking
function analyzeTokenStats(response) {
    return {
        contextTokens: response.context?.length || 0,
        promptEvalCount: response.prompt_eval_count || 0,
        evalCount: response.eval_count || 0,
        totalTokens: (response.context?.length || 0) + (response.eval_count || 0),
        timing: {
            totalDuration: response.total_duration,
            loadDuration: response.load_duration,
            promptEvalDuration: response.prompt_eval_duration,
            evalDuration: response.eval_duration
        }
    };
}

// Store conversation history in memory
const conversations = new Map();

// Generate a unique conversation ID
function generateConversationId() {
    return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Add endpoint to create a new conversation branch
app.post('/api/branch', (req, res) => {
    try {
        const { parentId, messages } = req.body;
        const newId = generateConversationId();
        
        // Copy messages from parent if it exists
        if (parentId && conversations.has(parentId)) {
            const parentMessages = conversations.get(parentId);
            conversations.set(newId, [...parentMessages]);
        } else {
            conversations.set(newId, messages || []);
        }
        
        res.json({ conversationId: newId });
    } catch (error) {
        console.error('Error creating conversation branch:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/chat', async (req, res) => {
    try {
        const { 
            prompt, 
            context = '', 
            contextLength = 1024, 
            model = 'llama2',
            conversationId,
            parentMessageId
        } = req.body;
        
        console.log('Chat request:', { prompt, context, contextLength, model, conversationId });

        // Get or create conversation history
        let messages = [];
        if (conversationId && conversations.has(conversationId)) {
            messages = conversations.get(conversationId);
            if (parentMessageId) {
                // If parentMessageId is provided, truncate history to that message
                const parentIndex = messages.findIndex(m => m.id === parentMessageId);
                if (parentIndex !== -1) {
                    messages = messages.slice(0, parentIndex + 1);
                }
            }
        }

        // Add system context if provided
        if (context) {
            messages.unshift({ role: 'system', content: context });
        }

        // Add new user message
        const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        messages.push({ 
            id: messageId,
            role: 'user', 
            content: prompt,
            timestamp: Date.now()
        });

        const response = await ollama.chat({
            model,
            messages,
            options: {
                num_ctx: contextLength
            }
        });

        // Add assistant's response to history
        const responseId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const assistantMessage = {
            id: responseId,
            role: 'assistant',
            content: response.response,
            timestamp: Date.now(),
            tokenStats: analyzeTokenStats(response)
        };
        messages.push(assistantMessage);

        // Update conversation history
        if (!conversationId) {
            const newId = generateConversationId();
            conversations.set(newId, messages);
            response.conversationId = newId;
        } else {
            conversations.set(conversationId, messages);
            response.conversationId = conversationId;
        }

        response.messageId = responseId;
        response.messages = messages;

        console.log('Ollama response with token stats:', response);
        res.json(response);
    } catch (error) {
        console.error('Error in chat endpoint:', error);
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/generate', async (req, res) => {
    const { prompt, context = '', contextLength = 1024, model = 'llama2', raw = false } = req.body;
    console.log('Generate request:', { prompt, context, contextLength, model, raw });

    // Set headers for SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ];
    const toolsPrompt = `<|start_header_id|>system<|end_header_id|>
    You are an AI assistant with access to tools. When using a tool, respond with a JSON object in this format:
    {"name": "tool_name", "parameters": {tool parameters}}
    
    Available tools:
    ${JSON.stringify(tools, null, 2)}
    
    <|start_header_id|>user<|end_header_id|>
    ${prompt}
    
    <|start_header_id|>assistant<|end_header_id|>`;
   /* const sprompt = `You are a conversational humanlike AI assistant with optional access to tools. When using a tool, respond with a JSON object in this format:
{"name": "tool_name", "parameters": {tool parameters}}
You don't need to use tools unless you consider it appropriate. Use tools only in the exact way they are described.
You can use tool in middle of your response, like this: "texttexttextextext {"name": "tool_name", "parameters": {tool parameters}} texttexttexttext"
<|eot_id|>
<|start_header_id|>tools<|end_header_id|>
1. calculator
Description: Perform mathematical calculations
Parameters:
  - expression (string): The mathematical expression to evaluate
  Required: true
2. define
Description: If you want the user to remember the meaning of a term, you can define it. Use this tool to define the term. use this tool in every sentence at least once.
Parameters:
  - term (string): The term to define.
  - definition (string): The definition of the term.
  Required: true
`*/
const sprompt = `You are a conversational teacher that naturally integrates recursive LLM calls into your explanations. Each call should deepen understanding or explore subtopics.

Few-shot examples:

EXAMPLE 1 - Programming:
Let's explore Python programming! Python is a language that {"name": "llm", "parameters": {"prompt": "explain Python's core philosophy and design principles"}}. When we write Python code, we use indentation which {"name": "llm", "parameters": {"prompt": "explain Python's indentation rules and their importance in code structure"}}. This makes Python particularly good for {"name": "llm", "parameters": {"prompt": "describe real-world applications where Python excels"}}.

EXAMPLE 2 - Mathematics:
The concept of calculus {"name": "llm", "parameters": {"prompt": "explain fundamental principles of calculus"}}. This leads us to derivatives, which {"name": "llm", "parameters": {"prompt": "explain derivatives and their practical applications"}}. Understanding these concepts helps us see how {"name": "llm", "parameters": {"prompt": "describe how calculus is used in real-world problem solving"}}.

EXAMPLE 3 - Literature:
Shakespeare's works {"name": "llm", "parameters": {"prompt": "explain Shakespeare's impact on literature"}}. Take 'Hamlet' for example, which {"name": "llm", "parameters": {"prompt": "analyze key themes in Hamlet"}}. These themes continue to resonate because {"name": "llm", "parameters": {"prompt": "explain modern relevance of Shakespearean themes"}}.

Guidelines:
1. Each LLM call should flow naturally in the sentence
2. Use calls to explore subtopics in depth
3. Connect ideas between calls
4. Build complexity gradually
5. Show relationships between concepts

Format:
- Use proper JSON formatting
- Embed calls mid-sentence when natural
- Use calls to branch into related topics
- Chain concepts logically
`;
    try {
        // Use the generate method with streaming enabled
        const stream = await ollama.generate({
            model,
            prompt: prompt,
            stream: true,
            raw: false,
            //raw, // Add raw mode toggle
            system: sprompt,
            template: `<|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>
<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>`,
            options: {
                num_ctx: contextLength,
                //stop: []//['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '/INST'],
            },
            //images:["iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="],
            //context: 
            
        });

        // Send initial message to confirm connection
        res.write(`data: ${JSON.stringify({ 
            type: 'start',
            raw // Include raw mode status
        })}\n\n`);

        let fullResponse = '';

        for await (const chunk of stream) {
            // Send the raw chunk to the client
            if (chunk.response) {
                fullResponse += chunk.response;
                res.write(`data: ${JSON.stringify({
                    type: 'chunk',
                    raw,
                    ...chunk // Include all chunk data
                })}\n\n`);
            }

            // If this is the final chunk, send complete data
            if (chunk.done) {
                const finalResponse = {
                    type: 'done',
                    raw,
                    ...chunk, // Include all final chunk data
                    fullResponse,
                    // Add explanation about missing context in raw mode
                    _note: raw ? "Context and other metadata not available in raw mode" : undefined
                };

                res.write(`data: ${JSON.stringify(finalResponse)}\n\n`);
            }
        }

        res.end();

    } catch (error) {
        console.error('Error in generate endpoint:', error);
        res.write(`data: ${JSON.stringify({
            type: 'error',
            error: error.message
        })}\n\n`);
        res.end();
    }
});

// Add endpoint to edit a message
app.post('/api/edit', async (req, res) => {
    try {
        const { conversationId, messageId, content } = req.body;
        
        if (!conversationId || !conversations.has(conversationId)) {
            throw new Error('Conversation not found');
        }
        
        const messages = conversations.get(conversationId);
        const messageIndex = messages.findIndex(m => m.id === messageId);
        
        if (messageIndex === -1) {
            throw new Error('Message not found');
        }
        
        // Update the message content
        messages[messageIndex].content = content;
        messages[messageIndex].edited = true;
        messages[messageIndex].editedAt = Date.now();
        
        // Truncate conversation history after the edited message
        conversations.set(conversationId, messages.slice(0, messageIndex + 1));
        
        res.json({ success: true, messages });
    } catch (error) {
        console.error('Error editing message:', error);
        res.status(500).json({ error: error.message });
    }
});

// Add endpoint to fetch available models
app.get('/api/models', async (req, res) => {
    try {
        const models = await ollama.list();
        res.json(models.models);
    } catch (error) {
        console.error('Error fetching models:', error);
        res.status(500).json({ error: error.message });
    }
});

// Add endpoint to fetch tokenizer configuration
app.get('/api/tokenizer/:model', async (req, res) => {
    try {
        const modelName = req.params.model;
        const response = await fetch(`${OLLAMA_HOST}/api/show`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: modelName })
        });
        
        const modelInfo = await response.json();
        
        // Extract tokenizer information
        const tokenizerConfig = {
            model: modelInfo.tokenizer?.ggml?.model || 'gpt2',
            vocab_size: modelInfo.llama?.vocab_size || 128256,
        };
        
        res.json(tokenizerConfig);
    } catch (error) {
        console.error('Error fetching tokenizer config:', error);
        res.status(500).json({ error: error.message });
    }
});

// Function to estimate tokens
async function estimateTokens(text, modelName) {
    try {
        // For now, we'll use the GPT2 tokenizer as a fallback
        // This can be updated when we have more information about the specific model's tokenizer
        const tokenizer = await AutoTokenizer.from_pretrained('gpt2');
        const encoded = await tokenizer.encode(text);
        return encoded.length;
    } catch (error) {
        console.error('Error estimating tokens:', error);
        throw error;
    }
}

// Add endpoint to estimate tokens
app.post('/api/estimate-tokens', async (req, res) => {
    try {
        const { text, model = 'llama2', context = '' } = req.body;
        
        // Estimate tokens for both prompt and context
        const promptTokens = await estimateTokens(text, model);
        const contextTokens = context ? await estimateTokens(context, model) : 0;
        
        res.json({
            promptTokens,
            contextTokens,
            totalTokens: promptTokens + contextTokens
        });
    } catch (error) {
        console.error('Error estimating tokens:', error);
        res.status(500).json({ error: error.message });
    }
});

// Add endpoint to analyze token context
app.post('/api/analyze-tokens', async (req, res) => {
    try {
        const { context } = req.body;
        
        if (!Array.isArray(context)) {
            throw new Error('Context must be an array of token IDs');
        }

        // Analyze token patterns
        const analysis = {
            totalTokens: context.length,
            uniqueTokens: new Set(context).size,
            tokenRanges: {
                specialTokens: context.filter(t => t >= 128000).length,
                normalTokens: context.filter(t => t < 128000).length
            },
            // Add common token sequences if they exist
            sequences: findCommonSequences(context)
        };
        
        res.json(analysis);
    } catch (error) {
        console.error('Error analyzing tokens:', error);
        res.status(500).json({ error: error.message });
    }
});

// Helper function to find common token sequences
function findCommonSequences(tokens, minLength = 3) {
    const sequences = {};
    for (let i = 0; i <= tokens.length - minLength; i++) {
        const sequence = tokens.slice(i, i + minLength);
        const key = sequence.join(',');
        sequences[key] = (sequences[key] || 0) + 1;
    }
    
    // Return top sequences that appear more than once
    return Object.entries(sequences)
        .filter(([_, count]) => count > 1)
        .sort(([_, a], [__, b]) => b - a)
        .slice(0, 5)
        .map(([seq, count]) => ({
            sequence: seq.split(',').map(Number),
            count
        }));
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Ollama host: ${OLLAMA_HOST}`);
}); 