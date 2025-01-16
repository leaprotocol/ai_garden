// llama.js
import 'dotenv/config';

// Configuration
const LIVE = true;

// Class for handling the live API requests
class ApiService {
  async fetchData() {
    const response = await fetch("https://api.fireworks.ai/inference/v1/chat/completions", {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.FIREWORKS_AI_API_KEY}`
      },
      body: JSON.stringify({
        model: "accounts/fireworks/models/llama-v3p1-8b-instruct",
        //model: "accounts/fireworks/models/llama-v3p1-405b-instruct",
        //model: "accounts/fireworks/models/mixtral-moe-8x7b-instruct",
        max_tokens: 16384,
        top_p: 1,
        top_k: 40,
        presence_penalty: 0,
        frequency_penalty: 0,
        temperature: 0.6,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Universe is"
              }
            ]
          }
        ]
      })
    });

    return await response.json();
  }
}

// Class for handling mock data
class MockApiService {
  fetchData() {
    return {
      id: '12345678-abcd-1234-efgh-56789ijklmno',
      object: 'chat.completion',
      created: 1724506464,
      model: 'accounts/fireworks/models/llama-v3p1-405b-instruct',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'A vast and complex topic! The universe is:\n' +
              '\n' +
              '* **All of space and time**: The universe is the totality of all existence, encompassing all matter, energy, space, and time.\n' +
              '* **Estimated to be around 13.8 billion years old**: Based on observations of the cosmic microwave background radiation and other evidence, scientists believe the universe began as a singularity around 13.8 billion years ago and has been expanding and evolving ever since.\n' +
              '* **Composed of billions of galaxies**: The universe contains an estimated 100-400 billion galaxies, each containing billions of stars, planets, and other celestial objects.\n' +
              "* **Made up of ordinary and dark matter**: The universe is thought to be composed of about 5% ordinary matter (the type we can see and interact with) and 95% dark matter and dark energy (mysterious forms of matter and energy that we can't directly observe).\n" +
              '* **Still expanding**: The universe is still expanding, with galaxies moving away from each other at incredible speeds.\n' +
              '* ** Governed by physical laws**: The universe is governed by a set of physical laws, including the laws of physics, gravity, and electromagnetism, which shape the behavior of matter and energy.\n' +
              '* **Full of mysteries and unanswered questions**: Despite significant advances in our understanding of the universe, there is still much that remains unknown, including the nature of dark matter and dark energy, the origins of the universe, and the potential for life beyond Earth.\n' +
              '\n' +
              'What aspect of the universe would you like to explore further?'
          }
        }
      ],
      usage: {
        prompt_tokens: 15,
        total_tokens: 320,
        completion_tokens: 305
      }
    };
  }
}

// Main class that uses either the ApiService or MockApiService
class ApiController {
  constructor(service) {
    this.service = service;
  }

  async callApi() {
    try {
      const data = await this.service.fetchData();
      console.log(data.choices.map(c => c.message));
      console.log(data);
      return data;
    } catch (error) {
      console.error('Error:', error);
    }
  }
}

// Determine which service to use based on the LIVE flag
const apiService = LIVE ? new ApiService() : new MockApiService();
const apiController = new ApiController(apiService);

// Execute the API call
apiController.callApi();
