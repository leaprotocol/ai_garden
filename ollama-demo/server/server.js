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

    try {
        // Use the generate method with streaming enabled
        const stream = await ollama.generate({
            model,
            prompt: context ? `${context}\n\n${prompt}` : prompt,
            stream: true,
            raw: false,
            //raw, // Add raw mode toggle
            system: '',
            template: '{{ .Prompt }}{{ .Response }}',   //TODO this is interesting 
            options: {
                num_ctx: contextLength
            }
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