import torch
import torch.nn as nn
import json
import os
import logging
from datetime import datetime

class AgentRagu:
    def __init__(self, device=None, model_path=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Determine model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'Raulnano', 'o1_model.pth')
        self.resized_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ragu_resized.pth')
            
        # Default values
        self.vocab_size = 10000  
        self.d_model = 64
        
        # Prefer resized weights if available
        if os.path.exists(self.resized_path):
            try:
                state_dict = torch.load(self.resized_path, map_location=self.device)
                if 'weight' in state_dict:
                    self.vocab_size = state_dict['weight'].shape[0]
                    self.d_model = state_dict['weight'].shape[1]
                    logging.info(f"AgentRagu: Loaded resized embeddings vocab={self.vocab_size}")
                    self.embed = nn.Embedding(self.vocab_size, self.d_model).to(device)
                    self.embed.load_state_dict(state_dict)
                else:
                    raise RuntimeError("resized weights missing 'weight'")
            except Exception as e:
                logging.error(f"AgentRagu: Failed to load resized embeddings {self.resized_path}: {e}")
                self._load_from_model(model_path)
        else:
            self._load_from_model(model_path)

        # If embed not created (e.g., fallback), create default
        if not hasattr(self, 'embed'):
            self.embed = nn.Embedding(self.vocab_size, self.d_model).to(device)
        
        # Make it trainable
        self.optimizer = torch.optim.Adam(self.embed.parameters(), lr=0.001)
        # Progress file
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'ragu_progress.json')
        self.load_progress()
        
        # Shared memory/history
        self.message_history = []  # Track conversations
        self.vector_cache = {}     # Cache frequently used vectors
        self.attention_weights = None  # For attention mechanisms

    def _load_from_model(self, model_path):
        # Try to load model to get actual vocab size
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if 'embed.weight' in state_dict:
                    self.vocab_size = state_dict['embed.weight'].shape[0]
                    self.d_model = state_dict['embed.weight'].shape[1]
                    logging.info(f"AgentRagu: Actual vocabulary size from model = {self.vocab_size}")
                    if self.vocab_size < 11049:
                        logging.warning(f"Model vocab size ({self.vocab_size}) < Roo's vocab (11049)")
                        logging.warning("Will need to resize embeddings!")
                # Initialize embed from model weights if present
                if 'embed.weight' in state_dict:
                    self.embed = nn.Embedding(self.vocab_size, self.d_model).to(self.device)
                    self.embed.load_state_dict({'weight': state_dict['embed.weight']})
            except Exception as e:
                logging.error(f"AgentRagu: Failed to load model from {model_path}: {e}")

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
                self.progress = {"trained_steps": 0, "last_loss": 0.0}
        else:
            self.progress = {"trained_steps": 0, "last_loss": 0.0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def process(self, text, filepath=None):
        # Tokenize simply
        tokens = text.lower().split()[:100]  # More tokens
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        if token_ids:
            tensor = torch.tensor(token_ids).to(self.device)
            embeddings = self.embed(tensor)
            # Simple training: minimize norm or something (placeholder)
            loss = embeddings.norm()  # Dummy loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress
            self.progress["trained_steps"] += 1
            self.progress["last_loss"] = loss.item()
            self.save_progress()
            
            # Return vector representation (mean embedding)
            mean_embed = embeddings.mean(dim=0)
            result = f"Ragu (Embedding): Processed {len(tokens)} tokens, updated embeddings. Mean vector norm: {mean_embed.norm().item():.2f}"
            return result, mean_embed  # Return string and vector
        else:
            return "Ragu (Embedding): No tokens to process", None
    
    def process_vectors(self, vectors):
        # Receive vectors from other agents, refine them
        if vectors is not None:
            # For example, add to own embedding space or something
            # Placeholder: just return refined vector
            refined = vectors + torch.randn_like(vectors) * 0.1  # Add noise as "new language"
            result = f"Ragu: Received and refined vector, norm: {refined.norm().item():.2f}"
            return result, refined
        return "Ragu: No vectors to process", None

    def update_embeddings(self, token_ids, context_vectors, learning_rate=0.01):
        """Update embeddings based on context."""
        with torch.enable_grad():
            token_tensor = torch.tensor(token_ids).to(self.device)
            
            # Get current embeddings
            embeddings = self.embed(token_tensor)
            
            # Ensure context vectors match shape (or mean pool them)
            if context_vectors.shape != embeddings.shape:
                # Simple broadcasting or resizing logic if needed
                # For now, assume context_vectors is a target we want to move towards
                # If context is single vector, expand it
                if context_vectors.dim() == 1:
                    context_vectors = context_vectors.expand_as(embeddings)
            
            # Simple learning: move embeddings toward context
            loss = torch.nn.functional.mse_loss(embeddings, context_vectors)
            
            # Update
            optimizer = torch.optim.SGD(self.embed.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log update
            self.progress["trained_steps"] += 1
            self.progress["last_loss"] = loss.item()
            self.save_progress()
            
            return embeddings.detach()

    def learn_from_feedback(self, token_ids, positive_feedback=True):
        """Learn from user feedback."""
        # Handle single int or list
        if isinstance(token_ids, int):
            token_ids = [token_ids]
            
        token_tensor = torch.tensor(token_ids, dtype=torch.long).to(self.device)
        
        # Get embedding
        embedding = self.embed(token_tensor)
        
        if positive_feedback:
            # Strengthen this embedding
            updated = embedding * 1.1
        else:
            # Weaken this embedding
            updated = embedding * 0.9
        
        # Update
        with torch.no_grad():
            self.embed.weight[token_tensor] = updated
            
        return f"Ragu: Learned from feedback for {len(token_ids)} tokens."

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
