import torch
import torch.nn as nn
import json
import os

class AgentMajorRoo:
    def __init__(self, device=None, num_dialog_acts=5, dialog_dim=16, vocab_size=1000, d_model=64, max_seq_len=100):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_dialog_acts = num_dialog_acts
        self.dialog_dim = dialog_dim
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Decoder: reconstruct input from capsules
        # First, project capsules to hidden
        self.fc1 = nn.Linear(num_dialog_acts * dialog_dim, 512).to(device)
        self.fc2 = nn.Linear(512, 256).to(device)
        self.fc3 = nn.Linear(256, max_seq_len * d_model).to(device)  # Reconstruct embeddings
        # For text reconstruction, perhaps to tokens
        self.token_decoder = nn.Linear(d_model, vocab_size).to(device)  # Per position
        # Encoder and Decoder for creating a new vector language
        self.encoder = nn.Linear(d_model, d_model).to(device)
        self.decoder = nn.Linear(d_model, d_model).to(device)
        # Progress file
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'majorroo_progress.json')
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
                self.progress = {"processed_dialogs": 0, "last_reconstruction_loss": 0.0}
        else:
            self.progress = {"processed_dialogs": 0, "last_reconstruction_loss": 0.0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def reconstruct(self, dialog_outputs):
        # dialog_outputs: [num_dialog_acts, dialog_dim]
        # Flatten
        x = dialog_outputs.view(-1)  # [num_dialog_acts * dialog_dim]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reconstructed_embeddings = self.fc3(x).view(self.max_seq_len, self.d_model)  # [max_seq_len, d_model]
        return reconstructed_embeddings
    
    def process(self, dialog_outputs, original_embeddings=None, filepath=None):
        # Input: dialog_outputs from previous agent [num_dialog_acts, dialog_dim]
        # Optional: original_embeddings for loss calculation [seq_len, d_model]
        if dialog_outputs is not None and dialog_outputs.shape[0] > 0:
            # Reconstruct
            reconstructed = self.reconstruct(dialog_outputs)
            
            loss = 0.0
            if original_embeddings is not None:
                # Compute reconstruction loss (MSE on embeddings)
                # Pad or truncate to match
                seq_len = min(original_embeddings.shape[0], self.max_seq_len)
                rec = reconstructed[:seq_len]
                orig = original_embeddings[:seq_len]
                loss = nn.functional.mse_loss(rec, orig).item()
            
            # Update progress
            self.progress["processed_dialogs"] += 1
            self.progress["last_reconstruction_loss"] = loss
            self.save_progress()
            
            result = f"MajorRoo (Decoder/Reconstruction): Reconstructed input with loss {loss:.4f}."
            return result, reconstructed
        else:
            return "MajorRoo: No dialog outputs to reconstruct", None
    
    def process_vectors(self, vectors):
        # MajorRoo likes to speak with other agents in vector or create a new language
        if vectors is not None:
            # Encode reconstructed vectors into a new language
            encoded = self.encoder(vectors)
            # Decode and modulate
            new_language_vectors = self.decoder(encoded) + torch.randn_like(encoded) * 0.1
            result = f"MajorRoo: Communicating in reconstructed vectors, created new language for {vectors.shape[0]} vectors."
            return result, new_language_vectors
        return "MajorRoo: No vectors to process", None

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
