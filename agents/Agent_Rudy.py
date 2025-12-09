import torch
import torch.nn as nn
import json
import os

class AgentRudy:
    def __init__(self, device=None, num_intents=10, intent_dim=16, num_dialog_acts=5, dialog_dim=16, routing_iterations=3):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.num_dialog_acts = num_dialog_acts
        self.dialog_dim = dialog_dim
        self.routing_iterations = routing_iterations
        # Weight matrix for predictions: [num_dialog_acts, intent_dim, dialog_dim]
        self.W = nn.Parameter(torch.randn(num_dialog_acts, intent_dim, dialog_dim)).to(device)
        # Encoder and Decoder for creating a new vector language
        self.encoder = nn.Linear(dialog_dim, dialog_dim).to(device)
        self.decoder = nn.Linear(dialog_dim, dialog_dim).to(device)
        # Progress file
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'rudy_progress.json')
        self.load_progress()
        
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
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted progress file {self.progress_file}, resetting to default.")
                self.progress = {"processed_intents": 0, "last_dialog_acts": 0}
        else:
            self.progress = {"processed_intents": 0, "last_dialog_acts": 0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def squash(self, x, dim=-1):
        # Squashing function
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2)
        return scale * x / norm
    
    def dynamic_routing(self, u):
        # u: [batch, num_intents, intent_dim]
        batch_size = u.shape[0]
        # Initialize routing coefficients
        b = torch.zeros(batch_size, self.num_intents, self.num_dialog_acts).to(self.device)
        
        for r in range(self.routing_iterations):
            # Softmax to get coupling coefficients
            c = torch.softmax(b, dim=-1)  # [batch, num_intents, num_dialog_acts]
            
            # Predictions: u_hat = W @ u using einsum
            # u: [batch, num_intents, intent_dim]
            # W: [num_dialog_acts, intent_dim, dialog_dim]
            # u_hat: [batch, num_intents, num_dialog_acts, dialog_dim]
            u_hat = torch.einsum('bim,dmk->bidk', u, self.W)
            
            # Weighted sum: s = sum c_i * u_hat_i
            s = torch.einsum('bid,bidk->bdk', c, u_hat)  # [batch, num_dialog_acts, dialog_dim]
            
            # Squash to get v
            v = self.squash(s, dim=-1)  # [batch, num_dialog_acts, dialog_dim]
            
            # Agreement: b += u_hat @ v
            if r < self.routing_iterations - 1:
                agreement = torch.einsum('bidk,bdk->bid', u_hat, v)  # [batch, num_intents, num_dialog_acts]
                b = b + agreement
        
        return v  # [batch, num_dialog_acts, dialog_dim]
    
    def process(self, intent_outputs, filepath=None):
        # Input: intent_outputs from previous agent [num_intents, intent_dim]
        if intent_outputs is not None and intent_outputs.shape[0] > 0:
            # Prepare for routing: [batch=1, num_intents, intent_dim]
            u = intent_outputs.unsqueeze(0)  # [1, num_intents, intent_dim]
            
            # Dynamic routing to dialog acts
            dialog_capsules = self.dynamic_routing(u)  # [1, num_dialog_acts, dialog_dim]
            
            # Output: [num_dialog_acts, dialog_dim]
            dialog_outputs = dialog_capsules.squeeze(0)
            
            # Update progress
            self.progress["processed_intents"] += 1
            self.progress["last_dialog_acts"] = self.num_dialog_acts
            self.save_progress()
            
            result = f"Rudy (Additional Capsule Layers): Routed {self.num_intents} intents to {self.num_dialog_acts} dialog act capsules."
            return result, dialog_outputs
        else:
            return "Rudy: No intent outputs to process", None
    
    def process_vectors(self, vectors):
        # Rudy likes to speak with other agents in vector or create a new language
        if vectors is not None:
            # Encode dialog vectors into a new language
            encoded = self.encoder(vectors)
            # Decode and modulate
            new_language_vectors = self.decoder(encoded) + torch.randn_like(encoded) * 0.1
            result = f"Rudy: Communicating in dialog vectors, created new language for {vectors.shape[0]} dialog capsules."
            return result, new_language_vectors
        return "Rudy: No vectors to process", None

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
