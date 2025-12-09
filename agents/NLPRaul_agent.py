import json
import os
import torch
import torch.nn as nn
import re

class NLPRaulAgent:
    def __init__(self):
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'nlp_raul_progress.json')
        self.load_progress()
        # Simple NLP model using PyTorch
        self.vocab_size = 10000  # Simple vocab
        self.embed_dim = 64  # Match system d_model
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.simple_tokenizer = self.build_simple_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding.to(self.device)
        # O1 integration (lazy loaded)
        self.o1_model = None
        self.o1_config = None
        self.o1_model_path = os.path.join(os.path.dirname(__file__), 'Raulnano', 'o1_model.pth')
        self.o1_config_path = os.path.join(os.path.dirname(__file__), 'Raulnano', 'config.json')

    def build_simple_tokenizer(self):
        # Simple tokenizer: split on spaces, lowercase, remove punctuation
        def tokenize(text):
            text = re.sub(r'[^\w\s]', '', text.lower())
            return text.split()
        return tokenize

    def text_to_indices(self, text):
        tokens = self.simple_tokenizer(text)
        # Simple vocab mapping (placeholder)
        indices = [hash(token) % self.vocab_size for token in tokens[:50]]  # Limit to 50 tokens
        return torch.tensor(indices, dtype=torch.long).to(self.device)

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except json.JSONDecodeError:
                self.progress = {"processed_texts": 0, "embeddings_generated": 0}
        else:
            self.progress = {"processed_texts": 0, "embeddings_generated": 0}

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)

    def load_o1(self):
        """Lazy-load the O1 model for inference."""
        if self.o1_model is not None:
            return self.o1_model
        try:
            # Import here to avoid heavy load at startup
            from agents.Raulnano.train import O1Model, vocab, tokenize, detokenize, vocab_size  # type: ignore
            self.o1_vocab = vocab
            self.o1_tokenize = tokenize
            self.o1_detokenize = detokenize
            self.o1_vocab_size = vocab_size
            print(f"NLPRaul: Successfully imported O1 vocab (size={vocab_size})")
        except Exception as e:
            print(f"NLPRaul: Failed to import O1 modules: {e}")
            # Set fallback tokenizer and vocab
            self.o1_tokenize = self._fallback_tokenize
            self.o1_detokenize = self._fallback_detokenize
            self.o1_vocab_size = 100
            self.o1_vocab = {}
            return None

        try:
            if not os.path.exists(self.o1_model_path):
                print(f"NLPRaul: O1 model not found at {self.o1_model_path}. Will initialize fresh.")
                # Create fresh model
                self.o1_model = O1Model(self.o1_vocab_size, 128, 8, 4)
                self.o1_model.to(self.device)
                self.o1_model.eval()
                return self.o1_model
            
            state_dict = torch.load(self.o1_model_path, map_location=self.device)
            # Infer params from state
            checkpoint_vocab_size = state_dict['embed.weight'].shape[0]
            d_model = state_dict['embed.weight'].shape[1]
            
            # Check if checkpoint vocab size matches current vocab size
            if checkpoint_vocab_size != self.o1_vocab_size:
                print(f"NLPRaul: Vocab size mismatch: checkpoint has {checkpoint_vocab_size}, current is {self.o1_vocab_size}. Initializing fresh model.")
                self.o1_model = O1Model(self.o1_vocab_size, d_model, 8, 4)
                self.o1_model.to(self.device)
                self.o1_model.eval()
                return self.o1_model
            
            num_layers = max(int(k.split('.')[1]) for k in state_dict if k.startswith('transformer_layers.')) + 1
            nhead = state_dict['transformer_layers.0.self_attn.in_proj_weight'].shape[0] // (3 * d_model)
            self.o1_model = O1Model(self.o1_vocab_size, d_model, nhead, num_layers)
            self.o1_model.load_state_dict(state_dict, strict=False)
            self.o1_model.to(self.device)
            self.o1_model.eval()
            print(f"NLPRaul: Loaded O1 model with d_model={d_model}, layers={num_layers}, nhead={nhead}")
        except Exception as e:
            print(f"NLPRaul: Failed to load O1 model ({e}). Initializing fresh.")
            try:
                self.o1_model = O1Model(self.o1_vocab_size, 128, 8, 4)
                self.o1_model.to(self.device)
                self.o1_model.eval()
            except Exception as init_e:
                print(f"NLPRaul: Failed to init fresh model: {init_e}")
                self.o1_model = None
        return self.o1_model

    def _fallback_tokenize(self, text):
        """Simple fallback tokenizer when vocab import fails."""
        tokens = text.strip().lower().split()
        # Map to simple IDs
        token_ids = [hash(t) % 100 for t in tokens]
        return token_ids if token_ids else [1]  # Return <sos> if empty

    def _fallback_detokenize(self, indices):
        """Simple fallback detokenizer."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist() if indices.numel() > 0 else []
        return f"<generated {len(indices)} tokens>"

    def run_o1_completion(self, text, max_new_tokens=50):
        model = self.load_o1()
        if model is None:
            return "O1 unavailable"
        
        # Ensure tokenizer is available
        if not hasattr(self, 'o1_tokenize') or self.o1_tokenize is None:
            return "O1 tokenizer unavailable"
        
        try:
            # Safely tokenize the input
            try:
                tokens = self.o1_tokenize(text)
                if tokens is None or len(tokens) == 0:
                    # Fallback: use simple tokenization
                    tokens = [1]  # <sos>
            except Exception as te:
                print(f"NLPRaul: Tokenization error: {te}")
                tokens = [1]  # <sos> fallback
            
            # Create input tensor with explicit dtype and device
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            # Ensure model is on the same device as input
            model = model.to(self.device)
            
            completion_tokens, reasoning_tokens, subtasks = model.generate_completion(input_ids, max_new_tokens=max_new_tokens)
            
            # Safely detokenize
            try:
                completion = self.o1_detokenize(completion_tokens) if completion_tokens else ""
                reasoning = self.o1_detokenize(reasoning_tokens) if reasoning_tokens else ""
                return f"{completion} (reasoning: {reasoning})" if completion else "O1 generated no output"
            except Exception as de:
                print(f"NLPRaul: Detokenization error: {de}")
                return f"O1 generated response (decode error: {str(de)[:50]})"
        except Exception as e:
            print(f"NLPRaul: O1 inference error: {e}")
            return f"O1 error: {str(e)[:50]}"
            print(f"NLPRaul: O1 inference error: {e}")
            return f"O1 error: {str(e)[:50]}"

    def process(self, input_text, filepath=None):
        if input_text:
            indices = self.text_to_indices(input_text)
            with torch.no_grad():
                embeddings = self.embedding(indices)
                # Mean pooling
                final_embedding = embeddings.mean(dim=0)
            self.progress["processed_texts"] += 1
            self.progress["embeddings_generated"] += 1
            self.save_progress()
            o1_resp = self.run_o1_completion(input_text)
            result = f"NLPRaul: Processed text '{input_text[:50]}...' into embeddings. O1: {o1_resp}"
            return result, final_embedding
        return "NLPRaul: No input to process.", None

    def process_vectors(self, vectors):
        if vectors is not None:
            # Modulate vectors with simple NLP insights
            refined_vectors = vectors + torch.randn_like(vectors) * 0.1
            result = f"NLPRaul: Refined vectors with NLP context for {vectors.shape[0]} dimensions."
            return result, refined_vectors
        return "NLPRaul: No vectors to refine.", None

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
