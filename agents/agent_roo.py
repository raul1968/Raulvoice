import torch
import torch.nn as nn
import json
import os
from datetime import datetime
# Import Librarian for file repair
import sys
sys.path.append(os.path.dirname(__file__))
from Librarian_Agent import LibrarianAgent

class AgentRoo:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Vocabulary management
        self.vocab_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'json_data', 'vocabulary.json')
        self.word2idx = {}
        self.load_vocab()
        
        # Embedding layer
        # We use a large fixed size for the embedding matrix to avoid resizing frequently,
        # but we track the actual used vocab size.
        self.max_vocab_size = 50000 
        self.d_model = 64
        self.embed = nn.Embedding(self.max_vocab_size, self.d_model).to(device)
        
        # Progress file path
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'roo_progress.json')
        self.load_progress()
        
        # Auto-encoder for reconstruction loss (simple linear projection)
        self.decoder = nn.Linear(self.d_model, self.max_vocab_size).to(device)
        
        # Shared memory/history
        self.message_history = []  # Track conversations
        self.vector_cache = {}     # Cache frequently used vectors
        self.attention_weights = None  # For attention mechanisms

    def remember_interaction(self, sender, vector, timestamp):
        """Store important interactions."""
        memory_entry = {
            'sender': sender,
            'vector': vector,
            'timestamp': timestamp,
            'importance': torch.norm(vector).item() if vector is not None else 0.0 # Simple importance metric
        }
        self.message_history.append(memory_entry)
        
        # Keep only recent memories
        if len(self.message_history) > 100:
            self.message_history = self.message_history[-100:]

    @property
    def vocab_size(self):
        # Return actual learned vocabulary count. 
        # If 0, it means we haven't learned anything yet (or just started fresh).
        return len(self.word2idx)

    def load_vocab(self):
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'r') as f:
                    self.word2idx = json.load(f)
            except:
                self.word2idx = {}
        else:
            self.word2idx = {}

    def save_vocab(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)
        with open(self.vocab_file, 'w') as f:
            json.dump(self.word2idx, f, indent=4)
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted progress file {self.progress_file}. Summoning Librarian...")
                try:
                    librarian = LibrarianAgent()
                    self.progress = {"processed_files": {}, "total_processed": 0}
                    librarian.fix_corrupt_file(self.progress_file, self.progress)
                except Exception as e:
                    print(f"Critical: Librarian unavailable ({e}). Resetting manually.")
                    self.progress = {"processed_files": {}, "total_processed": 0}
        else:
            self.progress = {"processed_files": {}, "total_processed": 0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def process(self, text, filepath=None):
        # Tokenize simply (split words, map to ids)
        tokens = text.lower().split()[:50]  # Limit tokens
        
        # Update vocabulary
        new_words = False
        token_ids = []
        for token in tokens:
            if token not in self.word2idx:
                if len(self.word2idx) < self.max_vocab_size:
                    self.word2idx[token] = len(self.word2idx)
                    new_words = True
            
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                # Fallback for out of vocab (if max reached)
                token_ids.append(hash(token) % self.max_vocab_size)
        
        if new_words:
            self.save_vocab()

        if token_ids:
            # Convert to tensor
            tensor = torch.tensor(token_ids).to(self.device)
            
            # --- Incremental Training Step ---
            self.embed.train()
            self.decoder.train()
            optimizer = torch.optim.Adam(list(self.embed.parameters()) + list(self.decoder.parameters()), lr=0.001)
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.embed(tensor)
            
            # Try to reconstruct the token IDs from embeddings
            # Output shape: [seq_len, max_vocab_size]
            logits = self.decoder(embeddings)
            
            # Loss: CrossEntropy between logits and original token_ids
            loss = nn.functional.cross_entropy(logits, tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            self.embed.eval()
            self.decoder.eval()
            # ---------------------------------

            # Simple processing: return embeddings
            result = f"Roo (Perception): Processed {len(tokens)} tokens on {self.device}, loss: {loss.item():.4f}, embeddings shape: {embeddings.shape}"
            
            # Save progress if filepath provided
            if filepath:
                self.progress["processed_files"][filepath] = {
                    "timestamp": datetime.now().isoformat(),
                    "tokens_processed": len(tokens),
                    "loss": loss.item(),
                    "embeddings_shape": list(embeddings.shape)
                }
                self.progress["total_processed"] += 1
                self.save_progress()
            
            return result, embeddings.detach()  # Return string and detached embeddings tensor
        else:
            return "Roo (Perception): No tokens to process", None

    def process_file(self, filepath):
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                text = json.dumps(data)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
            return self.process(text, filepath)
        except Exception as e:
            return f"Roo (Perception): Failed to process file {filepath}: {e}", None

    def self_diagnose(self):
        """Run self-diagnostics."""
        diagnostics = {
            'agent': self.__class__.__name__,
            'message_history_len': len(getattr(self, 'message_history', [])),
            'vector_cache_size': len(getattr(self, 'vector_cache', {})),
            'gradient_norm': self._get_gradient_norm(),
            'memory_usage': self._get_memory_usage(),
            'last_error': getattr(self, 'last_error', None)
        }
        
        # Check for issues
        issues = []
        if len(getattr(self, 'message_history', [])) == 0:
            issues.append("No message history")
        if self._get_gradient_norm() > 1000:
            issues.append("Exploding gradients")
        
        diagnostics['issues'] = issues
        return diagnostics

    def _get_gradient_norm(self):
        """Calculate total gradient norm."""
        total_norm = 0
        try:
            if hasattr(self, 'parameters'):
                 for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            elif hasattr(self, '__dict__'):
                for attr in self.__dict__.values():
                    if isinstance(attr, torch.nn.Module):
                        for p in attr.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                    elif isinstance(attr, torch.nn.Parameter):
                        if attr.grad is not None:
                            param_norm = attr.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
        except Exception:
            return 0.0
        return total_norm ** 0.5

    def _get_memory_usage(self):
        """Estimate memory usage of the agent."""
        import sys
        usage = sys.getsizeof(self)
        if hasattr(self, 'message_history'):
            usage += sys.getsizeof(self.message_history)
        if hasattr(self, 'vector_cache'):
            usage += sys.getsizeof(self.vector_cache)
        return usage
