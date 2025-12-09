import torch
import torch.nn as nn
import json
import os

class AgentBoss:
    def __init__(self, device=None, num_capsules=10, capsule_dim=16, num_intents=10, intent_dim=16, routing_iterations=3):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_intents = num_intents
        self.intent_dim = intent_dim
        self.routing_iterations = routing_iterations
        # Weight matrix for predictions: [num_intents, capsule_dim, intent_dim]
        self.W = nn.Parameter(torch.randn(num_intents, capsule_dim, intent_dim)).to(device)
        # Encoder and Decoder for creating a new vector language
        self.encoder = nn.Linear(intent_dim, intent_dim).to(device)
        self.decoder = nn.Linear(intent_dim, intent_dim).to(device)
        # Progress file
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'boss_progress.json')
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
                self.progress = {"processed_capsules": 0, "last_intents": 0}
        else:
            self.progress = {"processed_capsules": 0, "last_intents": 0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def squash(self, x, dim=-1):
        # Squashing function
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2)
        return scale * x / norm
    
    def dynamic_routing(self, u):
        # u: [batch, num_capsules, capsule_dim]
        batch_size = u.shape[0]
        # Initialize routing coefficients
        b = torch.zeros(batch_size, self.num_capsules, self.num_intents).to(self.device)

        for r in range(self.routing_iterations):
            # Softmax to get coupling coefficients
            c = torch.softmax(b, dim=-1)  # [batch, num_capsules, num_intents]

            # Predictions: u_hat = W @ u using einsum
            # u: [batch, num_capsules, capsule_dim]
            # W: [num_intents, capsule_dim, intent_dim]
            # u_hat: [batch, num_capsules, num_intents, intent_dim]
            u_hat = torch.einsum('bim,jmk->bijk', u, self.W)

            # Weighted sum: s = sum c_i * u_hat_i
            s = torch.einsum('bij,bijk->bjk', c, u_hat)  # [batch, num_intents, intent_dim]

            # Squash to get v
            v = self.squash(s, dim=-1)  # [batch, num_intents, intent_dim]

            # Agreement: b += u_hat @ v
            if r < self.routing_iterations - 1:
                agreement = torch.einsum('bijk,bjk->bij', u_hat, v)  # [batch, num_capsules, num_intents]
                b = b + agreement

        return v  # [batch, num_intents, intent_dim]
    
    def process(self, capsule_outputs, filepath=None):
        # Input: capsule_outputs from previous agent [seq_len, num_capsules, capsule_dim]
        if capsule_outputs is not None and capsule_outputs.shape[0] > 0:
            # Prepare for routing: [batch=1, seq_len, num_capsules, capsule_dim] -> treat seq_len as batch or flatten
            # For simplicity, average over seq_len or process per position
            # Here, average capsules over seq_len
            u = capsule_outputs.mean(dim=0).unsqueeze(0)  # [1, num_capsules, capsule_dim]
            
            # Dynamic routing
            intent_capsules = self.dynamic_routing(u)  # [1, num_intents, intent_dim]
            
            # Output: [num_intents, intent_dim]
            intent_outputs = intent_capsules.squeeze(0)
            
            # Update progress
            self.progress["processed_capsules"] += 1
            self.progress["last_intents"] = self.num_intents
            self.save_progress()
            
            result = f"Boss (Routing Capsule Layer): Routed {self.num_capsules} capsules to {self.num_intents} intent capsules."
            return result, intent_outputs
        else:
            return "Boss: No capsule outputs to process", None
    
    def process_vectors(self, vectors):
        # Boss likes to speak with other agents in vector or create a new language
        if vectors is not None:
            # Encode intent vectors into a new language
            encoded = self.encoder(vectors)
            # Decode and modulate
            new_language_vectors = self.decoder(encoded) + torch.randn_like(encoded) * 0.1
            result = f"Boss: Communicating in intent vectors, created new language for {vectors.shape[0]} intent capsules."
            return result, new_language_vectors
        return "Boss: No vectors to process", None

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
