import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
import random
import json
import logging
from tqdm import tqdm
import os

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Set device - use RTX 4050 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print("WARNING: CUDA not available. Training will be slow on CPU.")

# Constants from config
CONTEXT_WINDOW_SIZE = config['model']['context_window_size']
MAX_OUTPUT_TOKENS_MINI = config['model']['max_output_tokens_mini']

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            x = x.squeeze(2)
        
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class O1Model(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, is_mini=False):
        super(O1Model, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.completion_decoder = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
        self.is_mini = is_mini

    def forward(self, src):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        elif src.dim() == 3:
            src = src.squeeze(1)
        
        if src.size(1) == 0:
            print(f"Warning: Empty input tensor. Shape: {src.shape}")
            batch_size = src.size(0)
            return torch.zeros(batch_size, 1, self.vocab_size), torch.zeros(batch_size, 1)
        
        src = self.embed(src)
        src = self.pos_encoder(src)
        
        for layer in self.transformer_layers:
            src = layer(src)
        
        completion_logits = self.completion_decoder(src)
        values = self.value_head(src).squeeze(-1)
        
        return completion_logits, values

# Simplified vocabulary
vocab = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2, 'Step:': 3, '+': 4, '-': 5, '*': 6, '/': 7, '=': 8,
    '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18,
    'Calculate': 19, 'the': 20, 'sum': 21, 'of': 22, 'and': 23,
    'difference': 24, 'between': 25, 'product': 26, 'quotient': 27,
    'First,': 28, 'Next,': 29, 'Finally,': 30, 'result': 31, 'is': 32,
}
vocab_size = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}

def tokenize(text):
    return [vocab.get(token, vocab['<pad>']) for token in text.strip().split()]

def detokenize(indices):
    return ' '.join([inv_vocab.get(idx, ' ') for idx in indices])

# Simplified training function for quick test (10 epochs)
def train_o1_model_quick(model, optimizer, num_epochs=10, batch_size=2):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    max_state_length = 40
    
    for epoch in tqdm(range(num_epochs), desc="Quick Training"):
        # Generate a dummy batch
        states = []
        actions = []
        rewards = []
        log_probs = []
        values_list = []
        
        for _ in range(batch_size):
            # Create dummy input
            text = "Calculate the sum of 5 and 3"
            input_ids = torch.tensor([tokenize(text)]).to(device)
            
            # Forward pass
            with torch.no_grad():
                logits, values_ = model(input_ids)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Prepare batch data
            action_sequence = torch.full((1, max_state_length), vocab['<pad>'], dtype=torch.long, device=device)
            action_sequence[0, 0] = action.item()
            
            # Pad state
            state = input_ids
            if state.size(1) < max_state_length:
                padding = torch.full((1, max_state_length - state.size(1)), vocab['<pad>'], dtype=state.dtype, device=device)
                state = torch.cat([state, padding], dim=1)
            elif state.size(1) > max_state_length:
                state = state[:, :max_state_length]
            
            states.append(state)
            actions.append(action_sequence)
            log_probs.append(log_prob)
            values_list.append(values_[:, -1])
            reward_val = 1.0 if action.item() != vocab['<pad>'] else 0.0
            rewards.append(torch.tensor([reward_val], device=device))
        
        # Concatenate batch tensors
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        log_probs = torch.cat([lp.unsqueeze(0) if lp.dim() == 0 else lp for lp in log_probs], dim=0)
        values = torch.cat([v.unsqueeze(0) if v.dim() == 0 else v for v in values_list], dim=0)
        
        # Simple training step
        logits, values_pred = model(states)
        
        # Loss: cross-entropy + value loss
        batch_size, seq_len = actions.shape
        logits_flat = logits.view(-1, vocab_size)
        actions_flat = actions.view(-1)
        
        min_length = min(logits_flat.size(0), actions_flat.size(0))
        logits_flat = logits_flat[:min_length]
        actions_flat = actions_flat[:min_length]
        
        non_pad_mask = actions_flat != vocab['<pad>']
        logits_flat = logits_flat[non_pad_mask]
        actions_flat = actions_flat[non_pad_mask]
        
        loss = F.cross_entropy(logits_flat, actions_flat)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(f"Quick training completed ({num_epochs} epochs)")

if __name__ == "__main__":
    # Model parameters
    d_model = config['model']['d_model']
    nhead = config['model']['nhead']
    num_layers = 2  # Use fewer layers for quick test
    
    # Initialize the model
    model = O1Model(vocab_size, d_model, nhead, num_layers, is_mini=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training with 10 epochs
    train_o1_model_quick(model, optimizer, num_epochs=10, batch_size=2)
    
    # Save the model
    output_path = os.path.join(os.path.dirname(__file__), 'o1_model_quick.pth')
    torch.save(model.state_dict(), output_path)
    print(f"Quick test model saved to {output_path}")
