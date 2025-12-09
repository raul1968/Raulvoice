import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
import random
import json
import argparse
import logging
import tempfile
from pathlib import Path
from typing import List

import sentencepiece as spm
from tqdm import tqdm

# Load configuration
import os
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
MAX_OUTPUT_TOKENS_PREVIEW = config['model']['max_output_tokens_preview']
MAX_OUTPUT_TOKENS_MINI = config['model']['max_output_tokens_mini']

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)


# SentencePiece tokenizer setup
TOKENIZER_PREFIX = Path(__file__).with_name("o1_tokenizer")
TOKENIZER_MODEL_PATH = TOKENIZER_PREFIX.with_suffix(".model")
TOKENIZER_VOCAB_PATH = TOKENIZER_PREFIX.with_suffix(".vocab")


class SentencePieceTokenizer:
    """Wraps SentencePiece with on-demand training and safe encode/decode helpers."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_path = TOKENIZER_MODEL_PATH
        self.vocab_path = TOKENIZER_VOCAB_PATH
        tokenizer_cfg = cfg.get('tokenizer', {})
        self.vocab_size = tokenizer_cfg.get('vocab_size', cfg['model'].get('vocab_size', 2000))
        self.model_type = tokenizer_cfg.get('model_type', 'bpe')
        self.character_coverage = tokenizer_cfg.get('character_coverage', 0.9995)
        self.sp = None
        self.vocab = {}
        self._ensure_model()

    def _collect_corpus_files(self) -> List[Path]:
        data_root = Path(__file__).resolve().parents[1] / 'data'
        candidates: List[Path] = []
        text_dir = data_root / 'text_data'
        if text_dir.exists():
            candidates.extend(text_dir.glob('*.txt'))
        user_log = data_root / 'user_interactions.txt'
        if user_log.exists():
            candidates.append(user_log)
        return candidates

    def _train_model(self):
        corpus_files = self._collect_corpus_files()
        # Bootstrap with a minimal corpus if none exists
        if not corpus_files:
            bootstrap_path = TOKENIZER_PREFIX.with_name('tokenizer_bootstrap.txt')
            bootstrap_path.write_text("hello world\nthis is a bootstrap corpus for o1\n", encoding='utf-8')
            corpus_files = [bootstrap_path]

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp:
            corpus_text = []
            for path in corpus_files:
                try:
                    text = path.read_text(encoding='utf-8', errors='ignore')
                    tmp.write(text + "\n")
                    corpus_text.append(text)
                except Exception:
                    continue
            tmp_path = tmp.name

        # Downscale vocab if corpus is tiny to avoid sentencepiece fatal errors
        alphabet = set("".join(corpus_text))
        adaptive_vocab = min(self.vocab_size, max(64, len(alphabet) * 4))
        target_vocab_size = adaptive_vocab

        # Retry with halved vocab on failure until a floor is reached
        last_error = None
        while target_vocab_size >= 64:
            try:
                spm.SentencePieceTrainer.train(
                    input=tmp_path,
                    model_prefix=str(TOKENIZER_PREFIX),
                    vocab_size=target_vocab_size,
                    model_type=self.model_type,
                    character_coverage=self.character_coverage,
                    pad_id=0,
                    pad_piece='<pad>',
                    bos_id=1,
                    bos_piece='<sos>',
                    eos_id=2,
                    eos_piece='<eos>',
                    unk_id=3,
                    unk_piece='<unk>',
                    user_defined_symbols=['<subtask>'],
                    train_extremely_large_corpus=False,
                )
                break
            except RuntimeError as exc:  # handle vocab too high errors
                last_error = exc
                target_vocab_size = max(64, target_vocab_size // 2)
                continue

        Path(tmp_path).unlink(missing_ok=True)

        if target_vocab_size < 64 and last_error:
            raise last_error

        # Reload with the actual vocab size produced
        if self.model_path.exists():
            self._load_model()
            self.vocab_size = self.sp.get_piece_size() if self.sp else target_vocab_size
            self.vocab = {self.sp.id_to_piece(i): i for i in range(self.vocab_size)} if self.sp else self.vocab
            return

    def _load_model(self):
        sp = spm.SentencePieceProcessor()
        sp.load(str(self.model_path))
        self.vocab_size = sp.get_piece_size()
        self.vocab = {sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self.sp = sp

    def _ensure_model(self):
        if self.model_path.exists() and self.vocab_path.exists():
            self._load_model()
            return
        self._train_model()
        self._load_model()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id() if self.sp else 0

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id() if self.sp else 2

    def encode(self, text: str):
        if self.sp is None:
            self._ensure_model()
        tokens = self.sp.encode(text or "", out_type=int, add_bos=True, add_eos=True)
        return tokens if tokens else [self.sp.bos_id()]

    def decode(self, indices):
        if self.sp is None:
            self._ensure_model()
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().view(-1).tolist() if indices.numel() > 0 else []
        flat_ids = []
        for idx in indices:
            if isinstance(idx, (list, tuple)):
                flat_ids.extend(idx)
            else:
                flat_ids.append(int(idx))
        filtered = [i for i in flat_ids if i != self.pad_id]
        return self.sp.decode(filtered) if filtered else ""


tokenizer = SentencePieceTokenizer(config)
vocab = tokenizer.vocab
vocab_size = tokenizer.vocab_size


def tokenize(text):
    return tokenizer.encode(text)


def detokenize(indices):
    return tokenizer.decode(indices)

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
        # Ensure x has the correct shape (batch_size, seq_len, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif x.dim() == 4:
            x = x.squeeze(2)  # Remove extra dimension if present
        
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
        self.reasoning_decoder = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
        self.subtask_head = nn.Linear(d_model, 1)
        self.is_mini = is_mini
        self.max_reasoning_tokens = config['model']['max_reasoning_tokens']

    def forward(self, src, reasoning_tokens=None, generate_reasoning=True):
        if src.dim() == 1:
            src = src.unsqueeze(0)
        elif src.dim() == 3:
            src = src.squeeze(1)
        
        if src.size(1) == 0:
            print(f"Warning: Empty input tensor in forward pass. Shape: {src.shape}")
            batch_size = src.size(0)
            return torch.zeros(batch_size, 1, self.vocab_size), torch.zeros(batch_size, 1, self.vocab_size), torch.zeros(batch_size, 1)
        
        src = self.embed(src)  # [batch, seq, d_model]
        if reasoning_tokens is not None and reasoning_tokens.numel() > 0:
            # reasoning_tokens is 1D; embed and ensure [1, seq, d_model]
            if reasoning_tokens.dim() == 1:
                reasoning_embeddings = self.embed(reasoning_tokens).unsqueeze(0)  # [1, seq, d_model]
            else:
                reasoning_embeddings = self.embed(reasoning_tokens.squeeze())
                if reasoning_embeddings.dim() == 2:
                    reasoning_embeddings = reasoning_embeddings.unsqueeze(0)
            # Expand batch dimension if needed
            if reasoning_embeddings.size(0) == 1 and src.size(0) > 1:
                reasoning_embeddings = reasoning_embeddings.expand(src.size(0), -1, -1)
            src = torch.cat([src, reasoning_embeddings], dim=1)
        
        src = self.pos_encoder(src)
        
        for layer in self.transformer_layers:
            src = layer(src)
        
        completion_logits = self.completion_decoder(src)
        values = self.value_head(src).squeeze(-1)
        
        if generate_reasoning:
            reasoning_logits = self.reasoning_decoder(src)
            return completion_logits, reasoning_logits, values
        else:
            return completion_logits, values

    def generate_completion(self, input_ids, max_new_tokens):
        num_paths = config['generation']['num_paths']
        max_tokens = MAX_OUTPUT_TOKENS_MINI if self.is_mini else MAX_OUTPUT_TOKENS_PREVIEW
        max_new_tokens = min(max_new_tokens, max_tokens)
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
        
        paths = []
        for _ in range(num_paths):
            generated = input_ids.clone()
            # Initialize reasoning_tokens on the same device as input_ids
            reasoning_tokens = torch.tensor([], dtype=torch.long, device=input_ids.device)
            completion_tokens = []
            subtasks = []
            
            for _ in range(max_new_tokens):
                if generated.size(1) + reasoning_tokens.size(0) >= CONTEXT_WINDOW_SIZE:
                    break
                
                completion_logits, reasoning_logits, values = self(generated, reasoning_tokens)
                
                if completion_logits.numel() == 0:
                    print(f"Warning: completion_logits is empty. Input shape: {generated.shape}")
                    break
                
                next_token_logits = completion_logits[:, -1, :]
                next_token = self.sample_token(next_token_logits)
                
                reasoning_token = self.sample_token(reasoning_logits[:, -1, :])  # [batch]
                # reasoning_token is [batch], take first element and ensure it's on correct device
                reasoning_token_scalar = reasoning_token[0] if reasoning_token.dim() > 0 else reasoning_token
                reasoning_token_scalar = reasoning_token_scalar.to(reasoning_tokens.device)  # Ensure device match
                reasoning_tokens = torch.cat([reasoning_tokens, reasoning_token_scalar.unsqueeze(0)])
                
                if reasoning_tokens.size(0) > self.max_reasoning_tokens:
                    reasoning_tokens = reasoning_tokens[-self.max_reasoning_tokens:]
                
                last_hidden = self.embed(generated[:, -1])
                subtask_prob = torch.sigmoid(self.subtask_head(last_hidden))
                if subtask_prob > config['generation']['subtask_threshold']:
                    subtask = self.generate_subtask(generated, reasoning_tokens)
                    subtasks.append(subtask)
                    subtask_token = vocab['<subtask>']
                    generated = torch.cat([generated, torch.tensor([[subtask_token]]).to(generated.device)], dim=1)
                else:
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    completion_tokens.append(next_token.item())
                
                if self.should_revise_reasoning():
                    generated, reasoning_tokens = self.revise_reasoning(generated, reasoning_tokens)
                
                if next_token.item() == vocab['<eos>']:
                    break
            
            paths.append((completion_tokens, reasoning_tokens.tolist(), subtasks))
        
        if not paths:
            print("Warning: No valid paths generated")
            return [], [], []
        
        rewards = [self.compute_reward(p[0], p[1], p[2]) for p in paths]
        best_path = paths[rewards.index(max(rewards))]
        
        return best_path[0], best_path[1], best_path[2]

    def sample_token(self, logits):
        temperature = config['generation']['temperature']
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def add_reasoning_token(self, token):
        self.reasoning_buffer.append(token)
        if len(self.reasoning_buffer) > self.max_reasoning_tokens:
            self.reasoning_buffer.pop(0)

    def should_revise_reasoning(self):
        # Implement logic to decide if reasoning should be revised
        return random.random() < config['generation']['revision_probability']

    def revise_reasoning(self, generated, reasoning_tokens):
        # Implement logic to revise reasoning
        # For demonstration, we'll just remove the last few tokens from both
        return generated[:, :-5], reasoning_tokens[:-5]

    def generate_subtask(self, context, reasoning_tokens):
        subtask_tokens = []
        max_subtask_length = config['generation']['max_subtask_length']
        for _ in range(max_subtask_length):  # Max subtask length
            logits, _, _ = self(context, reasoning_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            subtask_tokens.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(1)], dim=1)
            if next_token.item() == vocab['<eos>']:
                break
        return subtask_tokens

    def compute_reward(self, completion_tokens, reasoning_tokens, subtasks):
        completion_reward = len(completion_tokens) * 0.1
        reasoning_reward = len(set(reasoning_tokens)) * 0.2
        subtask_reward = len(subtasks) * 0.5
        coherence_reward = self.compute_coherence(completion_tokens)
        process_reward = self.compute_process_reward(reasoning_tokens)
        return completion_reward + reasoning_reward + subtask_reward + coherence_reward + process_reward

    def compute_coherence(self, tokens):
        # Simple coherence check (can be made more sophisticated)
        return sum(1 for i in range(len(tokens)-1) if tokens[i] + 1 == tokens[i+1]) * 0.1

    def compute_process_reward(self, reasoning_tokens):
        # Implement a more sophisticated process reward
        unique_tokens = len(set(reasoning_tokens))
        return unique_tokens * 0.1  # Reward diverse reasoning

class PPO:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.clip_epsilon = config['training']['clip_epsilon']
        self.value_coef = config['training']['value_coef']
        self.entropy_coef = config['training']['entropy_coef']

    def compute_advantages(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        gamma = config['training']['gamma']
        lambda_ = config['training']['lambda']
        
        # Make sure to only iterate through the valid range
        for t in reversed(range(len(rewards))):
            if t + 1 < len(values):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
            else:
                delta = rewards[t] - values[t]
                
            advantages[t] = delta + gamma * lambda_ * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values[:len(advantages)]
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, old_values):
        # Reshape states if necessary
        if states.dim() == 2:
            batch_size, seq_len = states.shape
            states = states.unsqueeze(0)  # Add a dimension to make it [1, batch_size, seq_len]
        else:
            num_steps, batch_size, seq_len = states.shape
        
        # Flatten other tensors
        actions_flat = actions.view(-1)
        old_log_probs_flat = old_log_probs.view(-1)
        advantages, returns = self.compute_advantages(rewards, old_values)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        
        for _ in range(config['training']['ppo_epochs']):  # PPO epochs
            logits, _, values = self.model(states.view(-1, seq_len))
            
            # Focus on the logits of the last token in the sequence
            next_token_logits = logits[:, -1, :]
            new_probs = F.softmax(next_token_logits, dim=-1)
            dist = Categorical(new_probs)
            
            # Ensure actions_flat matches the shape of new_probs
            actions_flat_truncated = actions_flat[:new_probs.size(0)]
            old_log_probs_flat_truncated = old_log_probs_flat[:new_probs.size(0)]
            advantages_flat_truncated = advantages_flat[:new_probs.size(0)]
            returns_flat_truncated = returns_flat[:new_probs.size(0)]
            
            # Calculate new log probabilities
            new_log_probs = dist.log_prob(actions_flat_truncated)
            
            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs_flat_truncated)
            surr1 = ratio * advantages_flat_truncated
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat_truncated
            
            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Extract the value of the last token in each sequence
            values_last = values[:, -1].view(-1)
            critic_loss = nn.MSELoss()(values_last, returns_flat_truncated)
            
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Update the compute_reward function
def compute_reward(state, target_result):
    """Compute reward for generated tokens. Handles variable-sized inputs."""
    try:
        # Handle different state formats
        if isinstance(state, list):
            tokens = state
        elif isinstance(state, torch.Tensor):
            if state.dim() == 1:
                tokens = state.cpu().numpy().tolist()
            elif state.dim() == 2:
                # Take first row if batch
                tokens = state[0, :].cpu().numpy().tolist()
            else:
                return 0.0
        else:
            return 0.0
        
        # Detokenize and check result
        generated_text = detokenize(tokens)
        if "result is" in generated_text:
            result_str = generated_text.split("result is")[-1].strip()
            try:
                result = int(result_str) if result_str.isdigit() else float(result_str)
                if abs(result - target_result) < 1e-6:
                    return 1.0
                elif abs(result - target_result) < 5:
                    return 0.5
                elif abs(result - target_result) < 10:
                    return 0.2
                else:
                    return -0.2
            except (ValueError, TypeError):
                return 0.0
        else:
            return 0.0
    except Exception as e:
        return -0.5

# Generate arithmetic problems
def generate_arithmetic_problem():
    operations = ['+', '-', '*', '/']
    op = random.choice(operations)
    
    max_num = config['data']['max_problem_numbers']
    min_num = config['data']['min_problem_numbers']
    
    while True:
        if op in ['+', '-']:
            a, b = random.randint(min_num, max_num), random.randint(min_num, max_num)
        else:
            a, b = random.randint(min_num, 10), random.randint(min_num, 10)
        
        if op == '+':
            result = a + b
            problem = f"Calculate the sum of {a} and {b}"
        elif op == '-':
            result = a - b
            problem = f"Calculate the difference between {a} and {b}"
        elif op == '*':
            result = a * b
            problem = f"Calculate the product of {a} and {b}"
        else:
            if b != 0:  # Avoid division by zero
                result = a // b
                problem = f"Calculate the quotient of {a} and {b}"
            else:
                continue  # Try again if b is zero
        
        if problem and result:
            return problem, result

# Generate reasoning chain
def generate_reasoning_chain(problem, result):
    words = problem.split()
    operation = words[3]  # "sum", "difference", "product", or "quotient"
    
    if operation == "sum":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we add these numbers: {a} + {b}. "
        chain += f"Finally, we get the result: The sum is {result}."
    elif operation == "difference":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we subtract the second number from the first: {a} - {b}. "
        chain += f"Finally, we get the result: The difference is {result}."
    elif operation == "product":
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we multiply these numbers: {a} * {b}. "
        chain += f"Finally, we get the result: The product is {result}."
    else:  # quotient
        a, b = map(int, words[-3::2])
        chain = f"Step: First, we identify the numbers: {a} and {b}. "
        chain += f"Next, we divide the first number by the second: {a} / {b}. "
        chain += f"Finally, we get the result: The quotient is {result}."
    
    return chain

# Initialize agents globally to avoid re-initialization overhead and warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agents.agent_roo import AgentRoo
from agents.agent_ragu import AgentRagu
from agents.Agent_Rahulio import AgentRahulio
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
_roo_agent = None
_ragu_agent = None
_rahulio_agent = None

def get_agents():
    global _roo_agent, _ragu_agent, _rahulio_agent
    if _roo_agent is None:
        _roo_agent = AgentRoo(device_str)
        _ragu_agent = AgentRagu(device_str)
        _rahulio_agent = AgentRahulio(device_str)
    return _roo_agent, _ragu_agent, _rahulio_agent

# Modify collect_trajectories to use arithmetic problems
def collect_trajectories(model, batch_size):
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    max_state_length = 40

    # --- AGENT PIPELINE SETUP ---
    roo, ragu, rahulio = get_agents()
    
    import glob
    import random
    text_files = glob.glob(os.path.join(os.path.dirname(__file__), '../../data/text_data/*.txt'))
    # Include user_interactions.txt if it exists
    user_log = os.path.join(os.path.dirname(__file__), '../../data/user_interactions.txt')
    if os.path.exists(user_log):
        text_files.append(user_log)

    for _ in range(batch_size):
        # Sample a random text file and line
        file_path = random.choice(text_files)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        text = random.choice(lines)
        # AGENT PIPELINE: Roo -> Ragu -> Rahulio
        _, roo_embeddings = roo.process(text)
        _, ragu_embeddings = ragu.process_vectors(roo_embeddings)
        _, features = rahulio.process(ragu_embeddings)
        # Use mean of features as prefix embedding and align to model's d_model
        prefix = features.mean(dim=0, keepdim=True)  # [1, num_features]
        if prefix.size(-1) != model.d_model:
            diff = model.d_model - prefix.size(-1)
            if diff > 0:
                padding = torch.zeros(1, diff, device=prefix.device)
                prefix = torch.cat([prefix, padding], dim=-1)
            else:
                prefix = prefix[:, :model.d_model]
        # Tokenize text for O1Model
        tokens = tokenize(text)
        if len(tokens) > 4000:
            tokens = tokens[:4000]
        input_ids = torch.tensor([tokens]).to(device)
        # O1Model: embed tokens, prepend prefix
        with torch.no_grad():
            token_embeds = model.embed(input_ids)
            # Concatenate prefix as first token
            token_embeds = torch.cat([prefix.unsqueeze(1), token_embeds], dim=1)  # [1, seq+1, d_model]
            # Forward pass with custom embeddings (simulate as if prefix is a token)
            src = model.pos_encoder(token_embeds)
            for layer in model.transformer_layers:
                src = layer(src)
            logits = model.completion_decoder(src)
            values_ = model.value_head(src).squeeze(-1)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        # Prepare action sequence and state
        action_sequence = torch.full((1, max_state_length), vocab['<pad>'], dtype=torch.long, device=device)
        action_sequence[0, 0] = action.item()
        # Pad state to max_state_length
        state = input_ids
        if state.size(1) < max_state_length:
            padding = torch.full((1, max_state_length - state.size(1)), vocab['<pad>'], dtype=state.dtype, device=device)
            state = torch.cat([state, padding], dim=1)
        elif state.size(1) > max_state_length:
            state = state[:, :max_state_length]
        states.append(state)
        actions.append(action_sequence)
        log_probs.append(log_prob)
        values.append(values_[:, -1])
        # Dummy reward (can be improved): reward for non-empty output
        reward_val = 1.0 if action.item() != vocab['<pad>'] else 0.0
        rewards.append(torch.tensor([reward_val], device=device))

    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0)
    log_probs = torch.cat([lp.unsqueeze(0) if lp.dim() == 0 else lp for lp in log_probs], dim=0)
    values = torch.cat([v.unsqueeze(0) if v.dim() == 0 else v for v in values], dim=0)

    return states, actions, rewards, log_probs, values

# Update the training function
def train_o1_model(model, optimizer, num_epochs, batch_size):
    ppo = PPO(model, optimizer)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Generate a batch of arithmetic problems
        states, actions, rewards, old_log_probs, values = collect_trajectories(model, batch_size)
        
        # Supervised learning step
        sl_loss = supervised_finetuning_loss(model, (states, actions))
        optimizer.zero_grad()
        sl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['clip_grad_norm'])
        optimizer.step()
    
        # Reinforcement learning step
        ppo.update(states, actions, old_log_probs, rewards, values)
    
        # Evaluation and logging
        if epoch % 10 == 0:
            metrics = evaluate_model(model, batch_size)
            log_metrics(metrics, epoch, logger)

        # Dynamic curriculum learning
        if epoch % 50 == 0:
            adjust_problem_difficulty(epoch)

def log_metrics(metrics, epoch, logger):
    logger.info(f"Epoch {epoch} Metrics: {metrics}")

def supervised_finetuning_loss(model, batch):
    states, actions = batch
    logits, _ = model(states, generate_reasoning=False)
    
    # Reshape logits to [batch_size * sequence_length, vocab_size]
    batch_size, seq_length, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    
    # Reshape actions to [batch_size * sequence_length]
    target_ids = actions.view(-1)
    
    # Ensure logits and target_ids have the same length
    min_length = min(logits.size(0), target_ids.size(0))
    logits = logits[:min_length]
    target_ids = target_ids[:min_length]
    
    # Compute loss only on non-padded tokens
    non_pad_mask = target_ids != vocab['<pad>']
    logits = logits[non_pad_mask]
    target_ids = target_ids[non_pad_mask]
    
    loss = F.cross_entropy(logits, target_ids)
    return loss

# Update evaluation function
def evaluate_model(model, batch_size):
    model.eval()
    total_reward = 0
    valid_samples = 0
    with torch.no_grad():
        for _ in range(batch_size):
            try:
                problem, result = generate_arithmetic_problem()
                input_ids = torch.tensor([tokenize(problem)]).to(device)
                if input_ids.numel() == 0:
                    continue
                completion_tokens, reasoning_tokens, subtasks = model.generate_completion(input_ids, max_new_tokens=50)
                if completion_tokens:
                    reward = compute_reward(completion_tokens, result)
                    total_reward += float(reward)
                    valid_samples += 1
            except Exception as e:
                pass
    model.train()
    avg_reward = total_reward / valid_samples if valid_samples > 0 else 0
    return {"average_reward": avg_reward, "valid_samples": valid_samples}

def adjust_problem_difficulty(epoch):
    # Implement dynamic difficulty adjustment based on model performance
    global problem_difficulty
    if epoch < 100:
        problem_difficulty = "easy"
    elif epoch < 300:
        problem_difficulty = "medium"
    else:
        problem_difficulty = "hard"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train O1 Nano Model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--output", type=str, default="o1_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    args = parser.parse_args()
    
    # Load config if different
    if args.config != "config.json":
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Model parameters from config
    d_model = config['model']['d_model']
    nhead = config['model']['nhead']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    
    # Initialize the model
    model = O1Model(vocab_size, d_model, nhead, num_layers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training parameters
    num_epochs = args.epochs if args.epochs is not None else config['training']['num_epochs']
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    
    # Train the model
    train_o1_model(model, optimizer, num_epochs, batch_size)
    
    # Save the model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")