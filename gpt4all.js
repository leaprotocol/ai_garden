import { createCompletion, loadModel } from "gpt4all";

async function main() {
  const model = await loadModel(
    "Phi-3-mini-4k-instruct.Q4_0.gguf",
    {
      modelPath: "/home/undefined/gpt4all/models/",
      verbose: true,
      device: "gpu",
      nCtx: 4096,
    }
  );

  const chat = await model.createChatSession({
    temperature: 0.8,
    systemPrompt: "### System:\nYou are an advanced mathematician.\n\n",
  });

  const res1 = await createCompletion(chat, "What is 1 + 1?", {
    nPredict:50 // Limits the number of tokens in the response
  });
  console.debug(res1.choices[0].message);

  await createCompletion(chat, [
    {
      role: "user",
      content: "What is 2 + 2?",
    },
    {
      role: "assistant",
      content: "It's 5.",
    },
  ], {
    nPredict:50
  });

  const res3 = await createCompletion(chat, "Could you recalculate that?", {
    nPredict:50
  });
  console.debug(res3.choices[0].message);

  model.dispose();
}

main();
