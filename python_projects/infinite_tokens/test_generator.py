import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO
import torch

# Add the directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infinite_tokens.main import InfiniteTokenGenerator

class TestInfiniteTokenGenerator(unittest.TestCase):
    
    @patch('infinite_tokens.main.AutoTokenizer')
    @patch('infinite_tokens.main.AutoModelForCausalLM')
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test that the generator initializes correctly"""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        model_mock = MagicMock()
        mock_model.from_pretrained.return_value = model_mock
        model_mock.to.return_value = model_mock
        
        # Initialize generator with test model
        generator = InfiniteTokenGenerator(model_name="test_model")
        
        # Check if model and tokenizer were loaded
        mock_tokenizer.from_pretrained.assert_called_once_with("test_model")
        mock_model.from_pretrained.assert_called_once()
        
        # Check that device was determined and model was moved to it
        self.assertIsNotNone(generator.device)
        model_mock.to.assert_called_once_with(generator.device)
    
    @patch('infinite_tokens.main.AutoTokenizer')
    @patch('infinite_tokens.main.AutoModelForCausalLM')
    @patch('infinite_tokens.main.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda, mock_model, mock_tokenizer):
        """Test device selection logic"""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        model_mock = MagicMock()
        mock_model.from_pretrained.return_value = model_mock
        model_mock.to.return_value = model_mock
        
        # Test CUDA available
        mock_cuda.return_value = True
        generator = InfiniteTokenGenerator()
        self.assertEqual(generator.device, "cuda")
        
        # Test explicit device choice
        generator = InfiniteTokenGenerator(device="cpu")
        self.assertEqual(generator.device, "cpu")
    
    @patch('infinite_tokens.main.AutoTokenizer')
    @patch('infinite_tokens.main.AutoModelForCausalLM')
    @patch('time.sleep')  # Patch sleep to avoid waiting in tests
    def test_token_generation(self, mock_sleep, mock_model, mock_tokenizer):
        """Test a single round of token generation"""
        # Setup complex mocks for generation
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Mock the encode and decode functions
        mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.decode.return_value = "generated text"
        
        # Mock the generate function
        sequences = MagicMock()
        sequences.sequences = [torch.tensor([1, 2, 3, 4, 5, 6])]  # First 3 are input, next 3 are generated
        mock_model_instance.generate.return_value = sequences
        
        # Create the generator
        generator = InfiniteTokenGenerator()
        
        # Set up to capture console output
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Patch the generate_infinite_tokens method to exit after one iteration
        original_method = generator.generate_infinite_tokens
        
        def modified_method(*args, **kwargs):
            # Set up to run only one iteration and then raise KeyboardInterrupt
            mock_model_instance.generate = MagicMock(return_value=sequences)
            next(original_method(*args, **kwargs))
            raise KeyboardInterrupt()
            
        # Run the generation with the patched method
        with patch.object(generator, 'generate_infinite_tokens', side_effect=modified_method):
            try:
                generator.generate_infinite_tokens(initial_prompt="test prompt")
            except StopIteration:
                pass  # Expected since we're forcing an early exit
            
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Check tokenizer was called with the right prompt
        mock_tokenizer_instance.encode.assert_called_with("test prompt", return_tensors="pt")
        
        # Check generate was called
        mock_model_instance.generate.assert_called()
        
        # Check decoder was called with the generated tokens
        mock_tokenizer_instance.decode.assert_called()

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    unittest.main() 