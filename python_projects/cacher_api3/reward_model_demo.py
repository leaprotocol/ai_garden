import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import time
from transformers.utils import logging as transformers_logging
from rich.console import Console
from rich.text import Text

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
    def __init__(self, model_name="OpenAssistant/reward-model-deberta-v3-large-v2"):
        start_time = time.time()
        logger.info(f"Initializing RewardModelDemo with model: {model_name}")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        logger.info("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()
        logger.debug(f"Model initialization completed in {time.time() - start_time:.2f}s")

    def get_reward_score(self, prompt: str):
        """Calculate reward score for a single prompt-response pair"""
        start_time = time.time()
        logger.debug(f"Calculating reward score for prompt: '{prompt[:50]}...'")

        # Create raw input text once and use it consistently
        raw_input_text = prompt
        logger.debug(f"Raw input text: '{raw_input_text}'")

        tokens = []
        rewards = []
        cumulative_rewards = []
        token_start_positions = []

        # Store the raw text for consistent indexing
        current_position = 0

        inputs = self.tokenizer(raw_input_text, return_tensors="pt", truncation=True, max_length=512)
        total_tokens = len(inputs['input_ids'][0])
        logger.debug(f"Input token length: {total_tokens}")

        cumulative_reward = 0.0

        # Inference
        with torch.no_grad():
            for i in range(total_tokens):
                current_input = inputs['input_ids'][:, :i+1]
                logger.info(f"Current input: {current_input}")
                outputs = self.model(current_input)
                reward = outputs.logits.item()

                cumulative_reward += reward

                # Get the raw token text without any cleaning
                decoded_token = self.tokenizer.decode([inputs['input_ids'][0][i]], clean_up_tokenization_spaces=False)

                # Find exact position in raw text
                start_pos = raw_input_text.find(decoded_token, current_position)
                if start_pos != -1:
                    token_start_positions.append(start_pos)
                    current_position = start_pos + len(decoded_token)
                    logger.debug(f"Found token '{decoded_token}' at position {start_pos}")
                else:
                    # Special tokens won't be found in raw text
                    token_start_positions.append(-1)
                    logger.debug(f"Special token '{decoded_token}' not in raw text")

                tokens.append(decoded_token)
                rewards.append(reward)
                cumulative_rewards.append(cumulative_reward)

                # Log token and reward information
                logger.info(f"Token: '{decoded_token}', Reward: {reward:.4f}, Cumulative: {cumulative_reward:.4f}")

        elapsed_time = time.time() - start_time
        tokens_per_second = len(tokens) / elapsed_time

        # Calculate average reward per token
        average_reward = cumulative_reward / total_tokens if total_tokens > 0 else 0.0

        logger.info(f"Final average reward per token: {average_reward:.4f} (took {elapsed_time:.2f}s, {tokens_per_second:.2f} tokens/s)")
        return average_reward, tokens_per_second, tokens, rewards, token_start_positions

    def visualize_rewards(self, tokens, rewards, reward_bounds=None, token_start_positions=None, input_text=None):
        """Visualize text with tokens colored by their reward scores using rich"""
        # Use provided reward bounds or calculate from current rewards
        min_reward, max_reward = reward_bounds if reward_bounds else (min(rewards), max(rewards))
        reward_range = max_reward - min_reward

        logger.info(f"Reward range: {min_reward:.4f} to {max_reward:.4f}")

        def get_color_for_reward(reward):
            # Normalize reward to 0-1 range
            normalized = (reward - min_reward) / reward_range if reward_range != 0 else 0

            # Convert to RGB (red for low rewards, green for high rewards)
            red = int(255 * (1 - normalized))
            green = int(255 * normalized)
            return f"rgb({red},{green},0)"

        # Create rich Text object for visualization
        text = Text()
        last_end_position = 0
        encode_input = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        detokenized_sentence = self.tokenizer.decode(encode_input['input_ids'][0], clean_up_tokenization_spaces=False)
        print(detokenized_sentence)
        print(tokens)

        for token, reward, start_position in zip(tokens, rewards, token_start_positions):
            if token:
                color = get_color_for_reward(reward)
                if start_position > 0 and len(input_text) > start_position - 1 and input_text[start_position - 1] == ' ':
                    text.append(" ", style=color)

                text.append(token, style=color)

        # Print visualization
        console = Console()
        console.print("\nVisualized text with rewards:")
        console.print("─" * 80)
        console.print(text)
        console.print("─" * 80)

        # Print legend
        legend = Text("\nReward scale: ")
        steps = 5
        for i in range(steps + 1):
            reward = min_reward + (i * reward_range / steps)
            color = get_color_for_reward(reward)
            legend.append(f" {reward:.4f} ", style=color)

        console.print(legend)

        # Print detokenized sentence
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        detokenized_sentence = self.tokenizer.decode(inputs['input_ids'][0], clean_up_tokenization_spaces=True)
        console.print("\nDetokenized sentence:")
        console.print(detokenized_sentence)

def get_reward_bounds(tokens_list, rewards_list):
    """Calculate global min/max rewards across multiple responses"""
    all_rewards = []
    for rewards in rewards_list:
        all_rewards.extend(rewards)
    return min(all_rewards), max(all_rewards)

def main():
    logger.info("Starting reward model demo")
    start_time = time.time()

    try:
        demo = RewardModelDemo()

        prompt = "When does rain occur?"
        good_response = "Rain occurs when the atmosphere is saturated with water vapor and the air is cooled to the point where the water vapor condenses into liquid droplets."
        bad_response = "Rain is when water falls from the sky because clouds get too heavy or something."

        logger.info("\nTesting with single response pair...")
        logger.info("Good Response:")
        good_prompt = f"User: {prompt}\n\nAssistant: {good_response}"
        good_score, good_tps, good_tokens, good_rewards, good_token_start_positions = demo.get_reward_score(good_prompt)

        logger.info("Bad Response:")
        bad_prompt = f"User: {prompt}\n\nAssistant: {bad_response}"
        bad_score, bad_tps, bad_tokens, bad_rewards, bad_token_start_positions = demo.get_reward_score(bad_prompt)

        # Calculate global reward bounds
        reward_bounds = get_reward_bounds([good_tokens, bad_tokens], [good_rewards, bad_rewards])

        # Visualize both responses with the same reward scale
        logger.info("\nGood Response Visualization:")
        demo.visualize_rewards(good_tokens, good_rewards, reward_bounds, good_token_start_positions, good_prompt)

        logger.info("\nBad Response Visualization:")
        demo.visualize_rewards(bad_tokens, bad_rewards, reward_bounds, bad_token_start_positions, bad_prompt)

        # Print final results
        logger.info("Final Results:")
        logger.info(f"Good response score: {good_score:.4f} ({good_tps:.2f} tokens/s)")
        logger.info(f"Bad response score: {bad_score:.4f} ({bad_tps:.2f} tokens/s)")
        logger.info(f"Score difference: {good_score - bad_score:.4f}")

    except Exception as e:
        logger.error(f"Error during demo execution: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info(f"Demo completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()