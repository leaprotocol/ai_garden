import 'dotenv/config';

import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});
(async () => {
  const result = await llm.invoke("Hello, world!");
  console.log(result);
})();
