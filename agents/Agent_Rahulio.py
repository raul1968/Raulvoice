import torch
import torch.nn as nn
import json
import os

class AgentRahulio:
    def __init__(self, device=None, d_model=64, num_features=64, kernel_size=3):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # 1D Convolutional Layer for text feature extraction
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=num_features, kernel_size=kernel_size, padding=1).to(device)
        # Optional: add activation or pooling
        self.relu = nn.ReLU()
        # Encoder and Decoder for creating a new vector language
        self.encoder = nn.Linear(num_features, num_features).to(device)
        self.decoder = nn.Linear(num_features, num_features).to(device)
        # Progress file (optional, for consistency)
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'rahulio_progress.json')
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
                self.progress = {"processed_sequences": 0, "last_features": 0}
        else:
            self.progress = {"processed_sequences": 0, "last_features": 0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def process(self, embeddings, filepath=None):
        # Input: embeddings tensor from previous agent [seq_len, d_model]
        if embeddings is not None and embeddings.shape[0] > 0:
            # Prepare for Conv1d: [batch=1, channels=d_model, seq_len]
            x = embeddings.unsqueeze(0).transpose(1, 2).to(self.device)
            # Apply 1D convolution
            conv_out = self.conv1d(x)
            features = self.relu(conv_out)
            # Output: [1, num_features, out_seq_len] -> feature vectors [out_seq_len, num_features]
            feature_vectors = features.squeeze(0).transpose(0, 1)
            
            # Update progress
            self.progress["processed_sequences"] += 1
            self.progress["last_features"] = feature_vectors.shape[0]
            self.save_progress()
            
            result = f"Rahulio (Feature Extraction): Captured {feature_vectors.shape[0]} feature vectors from {embeddings.shape[0]} input embeddings."
            return result, feature_vectors
        else:
            return "Rahulio: No embeddings to process", None
    
    def process_vectors(self, vectors):
        # Rahulio likes to speak with other agents in vector or create a new language
        if vectors is not None:
            # Encode vectors into a new language
            encoded = self.encoder(vectors)
            # Optionally decode back or modulate
            new_language_vectors = self.decoder(encoded) + torch.randn_like(encoded) * 0.1  # Add some noise for "new language"
            result = f"Rahulio: Communicating in vectors, created new language for {vectors.shape[0]} vectors."
            return result, new_language_vectors
        return "Rahulio: No vectors to process", None

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
