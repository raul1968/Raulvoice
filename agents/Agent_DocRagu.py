import torch
import torch.nn as nn
import json
import os
import random

class AgentDocRagu:
    def __init__(self, device=None, num_dialog_acts=5, dialog_dim=16, num_responses=10, response_dim=64):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_dialog_acts = num_dialog_acts
        self.dialog_dim = dialog_dim
        self.num_responses = num_responses
        self.response_dim = response_dim
        # For retrieval: candidate responses (simple list)
        self.candidate_responses = [
            "Hello! How can I help you?",
            "I'm sorry, I didn't understand that.",
            "That's interesting! Tell me more.",
            "What do you mean by that?",
            "I love coffee too!",
            "Animation is amazing!",
            "Let's talk about something else.",
            "I'm learning from our conversation.",
            "Can you provide more details?",
            "Thanks for chatting!"
        ]
        # For generative: simple linear to response embedding
        self.response_generator = nn.Linear(num_dialog_acts * dialog_dim, response_dim).to(device)
        
        # Transformer Decoder for better generation
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=response_dim, nhead=4, batch_first=True),
            num_layers=2
        ).to(device)
        
        # For classification: softmax on capsule lengths
        self.intent_classifier = nn.Linear(num_dialog_acts, num_responses).to(device)
        # Encoder and Decoder for creating a new vector language
        self.encoder = nn.Linear(response_dim, response_dim).to(device)
        self.decoder = nn.Linear(response_dim, response_dim).to(device)
        # Progress file
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'docragu_progress.json')
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
                self.progress = {"generated_responses": 0, "last_intent": 0}
        else:
            self.progress = {"generated_responses": 0, "last_intent": 0}
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)
    
    def generate_with_transformers(self, capsule_vectors, max_len=30):
        """Improved response generation using attention."""
        # Ensure capsule_vectors is [batch, seq, dim]
        if capsule_vectors.dim() == 2:
            capsule_vectors = capsule_vectors.unsqueeze(0)
            
        # Add attention mechanism (Self-Attention on capsules)
        # Simple dot-product attention
        # [batch, seq, dim] x [batch, dim, seq] -> [batch, seq, seq]
        attention_scores = torch.matmul(capsule_vectors, capsule_vectors.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        self.attention_weights = attention_weights # Store for visualization/debugging
        
        attended = torch.matmul(attention_weights, capsule_vectors)
        
        # Use transformer decoder
        # Target memory (attended capsules) acts as memory for decoder
        # We need a "tgt" sequence to generate. For now, we can use the attended vector as a seed.
        # In a real sequence generation, we'd loop. Here we just refine the vector.
        output = self.transformer_decoder(attended, attended)
        
        # Flatten back if needed or take mean
        return output.mean(dim=1).squeeze(0)

    def generate_response(self, dialog_outputs):
        # dialog_outputs: [num_dialog_acts, dialog_dim]
        # For retrieval: similarity matching (simple: random for now)
        response_text = random.choice(self.candidate_responses)
        
        # For generative: generate response embedding
        # OLD: x = dialog_outputs.view(-1)
        # OLD: response_embedding = self.response_generator(x)
        
        # NEW: Use Transformer
        # Project dialog_outputs to response_dim if needed
        if dialog_outputs.shape[-1] != self.response_dim:
            # Simple projection to match dimensions
            projection = nn.Linear(dialog_outputs.shape[-1], self.response_dim).to(self.device)
            projected_inputs = projection(dialog_outputs)
        else:
            projected_inputs = dialog_outputs
            
        response_embedding = self.generate_with_transformers(projected_inputs)
        
        # For classification: intent from lengths
        lengths = torch.norm(dialog_outputs, dim=1)  # [num_dialog_acts]
        intent_logits = self.intent_classifier(lengths)
        intent_prob = torch.softmax(intent_logits, dim=0)
        intent_idx = torch.argmax(intent_prob).item()
        
        return response_text, response_embedding, intent_idx
    
    def process(self, dialog_outputs, filepath=None):
        # Input: dialog_outputs from previous agent [num_dialog_acts, dialog_dim]
        if dialog_outputs is not None and dialog_outputs.shape[0] > 0:
            # Generate response
            response_text, response_embedding, intent_idx = self.generate_response(dialog_outputs)
            
            # Update progress
            self.progress["generated_responses"] += 1
            self.progress["last_intent"] = intent_idx
            self.save_progress()
            
            result = f"DocRagu (Response Generation): Generated response '{response_text[:50]}...' with intent {intent_idx}."
            return result, response_text
        else:
            return "DocRagu: No dialog outputs to generate response", None
    
    def process_vectors(self, vectors):
        # DocRagu likes to speak with other agents in vector or create a new language
        if vectors is not None:
            # Encode response vectors into a new language
            encoded = self.encoder(vectors)
            # Decode and modulate
            new_language_vectors = self.decoder(encoded) + torch.randn_like(encoded) * 0.1
            result = f"DocRagu: Communicating in response vectors, created new language for {vectors.shape[0]} vectors."
            return result, new_language_vectors
        return "DocRagu: No vectors to process", None

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
