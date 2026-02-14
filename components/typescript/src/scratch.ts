import "dotenv/config";
import { createAgent, AIMessage, ToolMessage } from "langchain";
import { HumanMessage } from "@langchain/core/messages";

const agent = createAgent({
  model: "openai:gpt-4o-mini",
  tools: [],
  systemPrompt: "You are a helpful assistant. Be concise.",
});

async function run() {
  console.log("Testing agent invocation...");
  try {
    const result = await agent.invoke({
      messages: [new HumanMessage("Say hello in one sentence.")],
    });
    console.log("Success! Response:", result.messages.at(-1)?.content);
  } catch (e) {
    console.error("Error:", e);
  }
}

run();
