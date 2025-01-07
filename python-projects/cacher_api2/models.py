import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from cacher_api2.utils import get_logger

logger = get_logger(__name__)

MODEL_CONFIG = {
    "SmolLM": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "device": "cpu",
        "load_in_8bit": False,
    },
    # Add more models here...
}
class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_model(self, model_id: str):
        logger.info(f"Loading model: {model_id}")
        if model_id not in MODEL_CONFIG:
            raise ValueError(f"Model {model_id} not found in config")

        config = MODEL_CONFIG[model_id]
        model_name = os.environ.get("MODEL_NAME", config["name"])
        device = os.environ.get("DEVICE", config["device"])

        # Correctly determine load_in_8bit
        load_in_8bit_str = os.environ.get("LOAD_IN_8BIT")
        if load_in_8bit_str is not None:
            load_in_8bit = load_in_8bit_str.lower() == "true"
        else:
            load_in_8bit = config.get("load_in_8bit", False)

        logger.debug(f"Model name: {model_name}, Device: {device}, Load in 8bit: {load_in_8bit}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        if device == "cpu":
            model.to(device)

        self.models[model_id] = model
        self.tokenizers[model_id] = tokenizer
        logger.info(f"Model {model_id} loaded successfully")

    def get_model(self, model_id: str) -> PreTrainedModel:
        if model_id not in self.models:
            self.load_model(model_id)
        return self.models[model_id]

    def get_tokenizer(self, model_id: str) -> PreTrainedTokenizer:
        if model_id not in self.tokenizers:
            self.load_model(model_id)
        return self.tokenizers[model_id]