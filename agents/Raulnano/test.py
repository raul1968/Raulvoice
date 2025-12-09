import torch
import argparse
import json
from pathlib import Path

from train import O1Model, tokenize, detokenize, vocab_size


BASE_DIR = Path(__file__).resolve().parent

def load_config(config_path="config.json"):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(model_path, config):
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer model parameters from the state dict
    ckpt_vocab_size = state_dict['embed.weight'].shape[0]
    d_model = state_dict['embed.weight'].shape[1]
    num_layers = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('transformer_layers.')]) + 1
    nhead = state_dict['transformer_layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
    
    print(f"Inferred model parameters: d_model={d_model}, num_layers={num_layers}, nhead={nhead}, ckpt_vocab={ckpt_vocab_size}, tokenizer_vocab={vocab_size}")
    
    # Create the model with tokenizer vocab; adapt checkpoint weights when vocab sizes differ
    model = O1Model(vocab_size, d_model, nhead, num_layers)
    model.to(device)

    filtered_state = {}
    for key, val in state_dict.items():
        if key in model.state_dict() and model.state_dict()[key].shape == val.shape:
            filtered_state[key] = val

    if filtered_state:
        model.load_state_dict(filtered_state, strict=False)

    if ckpt_vocab_size != vocab_size:
        print(f"Warning: checkpoint vocab ({ckpt_vocab_size}) != tokenizer vocab ({vocab_size}); partially loading and padding weights.")
        with torch.no_grad():
            # Copy overlapping rows for embedding and decoders
            min_vocab = min(ckpt_vocab_size, vocab_size)
            if 'embed.weight' in state_dict:
                model.embed.weight[:min_vocab].copy_(state_dict['embed.weight'][:min_vocab])
            if 'completion_decoder.weight' in state_dict:
                model.completion_decoder.weight[:min_vocab].copy_(state_dict['completion_decoder.weight'][:min_vocab])
            if 'completion_decoder.bias' in state_dict:
                model.completion_decoder.bias[:min_vocab].copy_(state_dict['completion_decoder.bias'][:min_vocab])
            if 'reasoning_decoder.weight' in state_dict:
                model.reasoning_decoder.weight[:min_vocab].copy_(state_dict['reasoning_decoder.weight'][:min_vocab])
            if 'reasoning_decoder.bias' in state_dict:
                model.reasoning_decoder.bias[:min_vocab].copy_(state_dict['reasoning_decoder.bias'][:min_vocab])

    model.to(device)
    model.eval()
    return model

def chat_with_model(model, config):
    print("Welcome to the O1 Model Arithmetic Solver!")
    print("You can ask arithmetic questions like:")
    print("- Calculate the sum of 5 and 7")
    print("- Calculate the difference between 15 and 8")
    print("- Calculate the product of 6 and 4")
    print("- Calculate the quotient of 20 and 5")
    print("Type 'quit' to exit.")
    
    max_new_tokens = config['generation']['max_new_tokens']
    
    while True:
        try:
            user_input = input("\nEnter your question: ")
            if user_input.lower() == 'quit':
                break
            
            device = next(model.parameters()).device
            input_ids = torch.tensor([tokenize(user_input)], device=device)
            completion_tokens, reasoning_tokens, subtasks = model.generate_completion(input_ids, max_new_tokens=max_new_tokens)
            
            print("\nModel's thought process:")
            print("Reasoning:", detokenize(reasoning_tokens))
            print("Subtasks:")
            for i, subtask in enumerate(subtasks, 1):
                print(f"  {i}. {detokenize(subtask)}")
            
            print("\nModel's response:")
            print(detokenize(completion_tokens))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test O1 Nano Model")
    parser.add_argument("--model", type=str, default="o1_model.pth", help="Path to model file")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    try:
        model = load_model(args.model, config)
        print(f"Model loaded successfully. Number of layers: {len(model.transformer_layers)}")
        chat_with_model(model, config)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        print("Make sure you have trained the model and saved it with the correct filename.")
    except Exception as e:
        print(f"An error occurred: {e}")
