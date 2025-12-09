import torch
import torch.nn as nn
import json
import os

class AgentAttentionCoordinator(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.attention_weights = nn.Parameter(torch.randn(num_agents, num_agents))
        
    def compute_attention(self, agent_vectors):
        """Compute attention between agents."""
        # agent_vectors: list of vectors from each agent
        vectors = torch.stack(agent_vectors)
        
        # Self-attention between agents
        attention_scores = torch.matmul(vectors, vectors.T)
        attention_scores = attention_scores * self.attention_weights
        
        # Softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Weighted sum
        attended = torch.matmul(attention_probs, vectors)
        
        return attended, attention_probs

class AgentLinguist:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.vector_to_symbol = {} # Map symbol -> vector (for searching)
        self.symbol_to_vector = {} # Map symbol -> vector (for decoding)
        self.symbol_counter = 0
        
        # Persistence
        self.vocab_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'linguist_vocab.json')
        self.load_vocab()
        
    def load_vocab(self):
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'r') as f:
                    data = json.load(f)
                    self.symbol_counter = data.get("counter", 0)
                    # Load symbols and convert lists back to tensors
                    raw_symbols = data.get("symbols", {})
                    for sym, vec_list in raw_symbols.items():
                        vec = torch.tensor(vec_list).to(self.device)
                        self.symbol_to_vector[sym] = vec
                        # We don't strictly need vector_to_symbol for loading since we iterate values
                        # But for consistency we can rebuild it if needed, or just use symbol_to_vector.values()
            except Exception as e:
                print(f"Linguist: Error loading vocab: {e}")
                self.symbol_counter = 0
                self.symbol_to_vector = {}

    def save_vocab(self):
        data = {
            "counter": self.symbol_counter,
            "symbols": {k: v.tolist() for k, v in self.symbol_to_vector.items()}
        }
        try:
            with open(self.vocab_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Linguist: Error saving vocab: {e}")

    def encode_vector(self, vector, threshold=0.8):
        """Convert vector to emergent symbol if similar enough."""
        # Ensure vector is on correct device and flattened
        if vector.dim() > 1:
            vector = vector.view(-1)
        vector = vector.to(self.device)
            
        # Normalize for cosine similarity
        vector_norm = vector / (torch.norm(vector) + 1e-8)
        
        # Search for existing symbol
        best_sim = -1.0
        best_sym = None
        
        for symbol, stored_vector in self.symbol_to_vector.items():
            # stored_vector should already be normalized or we normalize here
            stored_norm = stored_vector / (torch.norm(stored_vector) + 1e-8)
            similarity = torch.dot(vector_norm, stored_norm)
            
            if similarity > best_sim:
                best_sim = similarity
                best_sym = symbol
        
        if best_sim > threshold:
            return best_sym  # Return existing symbol
        
        # Create new symbol
        new_symbol = f"SYM_{self.symbol_counter}"
        self.symbol_to_vector[new_symbol] = vector # Store original vector
        self.symbol_counter += 1
        
        self.save_vocab() # Persist new word
        return new_symbol
    
    def decode_symbol(self, symbol):
        """Convert symbol back to vector."""
        return self.symbol_to_vector.get(symbol)

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

