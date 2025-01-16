import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import time
from transformers.utils import logging as transformers_logging
from rich.console import Console
from rich.text import Text
from demo4 import CacherClientV4
import asyncio

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reward_model")

# Enable HF download progress bars
# transformers_logging.set_verbosity_info()
transformers_logging.enable_progress_bar()

class RewardModelDemo:
    def __init__(self, cacher_client: CacherClientV4, model_name="OpenAssistant/reward-model-deberta-v3-large-v2"):
        self.cacher_client = cacher_client
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    async def get_reward_score(self, prompt: str, response: str):
        """Calculate reward score for a single prompt-response pair using Cacher API"""
        start_time = time.time()
        logger.debug(f"Calculating reward score for prompt: '{prompt[:50]}...'")

        raw_input_text = f"User: {prompt}\n\nAssistant: {response}"
        logger.debug(f"Raw input text: '{raw_input_text}'")

        # Create state with Cacher API
        state = await self.cacher_client.create_state(self.model_name, raw_input_text)
        state_id = state["state_id"]
        logger.debug(f"Created state with ID: {state_id}")

        tokens = []
        rewards = []
        cumulative_rewards = []
        token_start_positions = []
        current_position = 0
        cumulative_reward = 0.0

        # Generate tokens and calculate rewards
        for i in range(512):  # Max 512 tokens
            try:
                generation_result = await self.cacher_client.generate_token(state_id, max_tokens=1)
                logger.info(f"generation_result: {generation_result}")
                if not generation_result or 'text' not in generation_result:
                    logger.warning("No text generated. Stopping generation.")
                    break

                generated_token = generation_result['text']
                
                # Decode the generated token to get its ID
                decoded_token_ids = self.tokenizer.encode(generated_token, add_special_tokens=False)
                if not decoded_token_ids:
                    logger.warning(f"Could not decode token: '{generated_token}'. Stopping.")
                    break
                decoded_token_id = decoded_token_ids[0]

                # Get reward from the model (assuming it's available in the response)
                reward = generation_result.get('reward', 0.0)  # Replace 'reward' with actual key if different
                cumulative_reward += reward

                # Find exact position in raw text
                start_pos = raw_input_text.find(generated_token, current_position)
                if start_pos != -1:
                    token_start_positions.append(start_pos)
                    current_position = start_pos + len(generated_token)
                    logger.debug(f"Found token '{generated_token}' at position {start_pos}")
                else:
                    token_start_positions.append(-1)
                    logger.debug(f"Token '{generated_token}' not found in raw text")

                tokens.append(generated_token)
                rewards.append(reward)
                cumulative_rewards.append(cumulative_reward)

                logger.info(f"Token: '{generated_token}', Reward: {reward:.4f}, Cumulative: {cumulative_reward:.4f}")

            except Exception as e:
                logger.error(f"Error during generation: {e}")
                break

        elapsed_time = time.time() - start_time
        tokens_per_second = len(tokens) / elapsed_time if tokens else 0.0
        average_reward = cumulative_reward / len(tokens) if tokens else 0.0

        logger.info(f"Final average reward per token: {average_reward:.4f} (took {elapsed_time:.2f}s, {tokens_per_second:.2f} tokens/s)")
        return average_reward, tokens_per_second, tokens, rewards, token_start_positions

    def visualize_rewards(self, tokens, rewards, reward_bounds=None, token_start_positions=None, input_text=None):
        """Visualize text with tokens colored by their reward scores using rich"""
        min_reward, max_reward = reward_bounds if reward_bounds else (min(rewards), max(rewards))
        reward_range = max_reward - min_reward

        logger.info(f"Reward range: {min_reward:.4f} to {max_reward:.4f}")

        def get_color_for_reward(reward):
            normalized = (reward - min_reward) / reward_range if reward_range != 0 else 0
            red = int(255 * (1 - normalized))
            green = int(255 * normalized)
            return f"rgb({red},{green},0)"

        text = Text()
        for token, reward, start_position in zip(tokens, rewards, token_start_positions):
            if token:
                color = get_color_for_reward(reward)
                if start_position > 0 and len(input_text) > start_position - 1 and input_text[start_position - 1] == ' ':
                    text.append(" ", style=color)
                text.append(token, style=color)

        console = Console()
        console.print("\nVisualized text with rewards:")
        console.print("─" * 80)
        console.print(text)
        console.print("─" * 80)

        legend = Text("\nReward scale: ")
        steps = 5
        for i in range(steps + 1):
            reward = min_reward + (i * reward_range / steps)
            color = get_color_for_reward(reward)
            legend.append(f" {reward:.4f} ", style=color)
        console.print(legend)

        console.print("\nDetokenized sentence:")
        console.print(" ".join(tokens))  # Simplified detokenization

def get_reward_bounds(tokens_list, rewards_list):
    """Calculate global min/max rewards across multiple responses"""
    all_rewards = []
    for rewards in rewards_list:
        all_rewards.extend(rewards)
    return min(all_rewards), max(all_rewards)

async def main():
    logger.info("Starting reward model demo with Cacher API")
    start_time = time.time()
    cacher_client = CacherClientV4()
    try:
        demo = RewardModelDemo(cacher_client)

        prompt = "When does rain occur?"
        good_response = "Rain occurs when the atmosphere is saturated with water vapor and the air is cooled to the point where the water vapor condenses into liquid droplets."
        bad_response = "Rain is when water falls from the sky because clouds get too heavy or something."

        logger.info("\nTesting with single response pair...")
        logger.info("Good Response:")
        good_prompt = f"User: {prompt}\n\nAssistant: {good_response}"
        good_score, good_tps, good_tokens, good_rewards, good_token_start_positions = await demo.get_reward_score(prompt, good_response)

        logger.info("Bad Response:")
        bad_prompt = f"User: {prompt}\n\nAssistant: {bad_response}"
        bad_score, bad_tps, bad_tokens, bad_rewards, bad_token_start_positions = await demo.get_reward_score(prompt, bad_response)

        reward_bounds = get_reward_bounds([good_tokens, bad_tokens], [good_rewards, bad_rewards])

        logger.info("\nGood Response Visualization:")
        demo.visualize_rewards(good_tokens, good_rewards, reward_bounds, good_token_start_positions, good_prompt)

        logger.info("\nBad Response Visualization:")
        demo.visualize_rewards(bad_tokens, bad_rewards, reward_bounds, bad_token_start_positions, bad_prompt)

        logger.info("Final Results:")
        logger.info(f"Good response score: {good_score:.4f} ({good_tps:.2f} tokens/s)")
        logger.info(f"Bad response score: {bad_score:.4f} ({bad_tps:.2f} tokens/s)")
        logger.info(f"Score difference: {good_score - bad_score:.4f}")

    except Exception as e:
        logger.error(f"Error during demo execution: {str(e)}", exc_info=True)
        raise
    finally:
        await cacher_client.close()
        logger.info(f"Demo completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())