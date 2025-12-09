import torch
import logging
import os

class VocabularySynchronizer:
    def __init__(self):
        self.master_vocab = {}
        self.token_counts = {}
        self.vocab_size = 0
        
    def synchronize_agents(self, agent_roo, agent_ragu, agent_rahulio):
        """Synchronize vocabulary across all agents."""
        # Get actual vocabulary from each agent
        vocab_sizes = {
            'Roo': self._get_roo_vocab_size(agent_roo),
            'Ragu': self._get_ragu_vocab_size(agent_ragu),
            'Rahulio': self._get_rahulio_vocab_size(agent_rahulio)
        }
        
        logging.info(f"Current vocab sizes: {vocab_sizes}")
        
        # Find maximum needed size
        max_needed = max(vocab_sizes.values())
        logging.info(f"Maximum vocabulary needed: {max_needed}")
        
        # Check for mismatches
        mismatches = []
        for agent, size in vocab_sizes.items():
            if size != max_needed:
                mismatches.append(f"{agent}: {size} (should be {max_needed})")
        
        if mismatches:
            logging.warning(f"Vocabulary mismatches detected: {mismatches}")
            return self._fix_mismatches(max_needed, agent_roo, agent_ragu, agent_rahulio)
        else:
            logging.info("All agents have synchronized vocabulary!")
            return True
    
    def _get_roo_vocab_size(self, agent_roo):
        """Get actual vocabulary size from Roo."""
        try:
            if hasattr(agent_roo, 'word2idx'):
                return len(agent_roo.word2idx)
            elif hasattr(agent_roo, 'vocab'):
                return len(agent_roo.vocab)
            else:
                # Fallback
                return 11049
        except:
            return 11049
    
    def _get_ragu_vocab_size(self, agent_ragu):
        """Get embedding size from Ragu."""
        try:
            # Check for direct embed or model.embed
            if hasattr(agent_ragu, 'embed'):
                return agent_ragu.embed.weight.shape[0]
            elif hasattr(agent_ragu, 'model') and hasattr(agent_ragu.model, 'embed'):
                return agent_ragu.model.embed.weight.shape[0]
            return 10000
        except:
            return 10000
    
    def _get_rahulio_vocab_size(self, agent_rahulio):
        """Get vocabulary size from Rahulio."""
        try:
            if hasattr(agent_rahulio, 'embedding'):
                return agent_rahulio.embedding.weight.shape[0]
            return 10000
        except:
            return 10000
    
    def _fix_mismatches(self, target_size, agent_roo, agent_ragu, agent_rahulio):
        """Resize all agents to target vocabulary size."""
        logging.info(f"Resizing all agents to vocabulary size: {target_size}")
        
        # Resize Ragu
        success_ragu = self._resize_ragu_embeddings(agent_ragu, target_size)
        
        # Resize Rahulio
        success_rahulio = self._resize_rahulio_embeddings(agent_rahulio, target_size)
        
        # Update Roo's expectations (Roo usually drives the vocab, so this might be a no-op or just logging)
        success_roo = self._update_roo_vocab(agent_roo, target_size)
        
        return success_ragu and success_rahulio and success_roo
    
    def _resize_ragu_embeddings(self, agent_ragu, new_size):
        """Resize Ragu's embedding layer."""
        try:
            embed_layer = None
            if hasattr(agent_ragu, 'embed'):
                embed_layer = agent_ragu.embed
            elif hasattr(agent_ragu, 'model') and hasattr(agent_ragu.model, 'embed'):
                embed_layer = agent_ragu.model.embed
            
            if embed_layer:
                old_weight = embed_layer.weight
                old_size, d_model = old_weight.shape
                
                if new_size > old_size:
                    # Expand embeddings
                    new_weight = torch.nn.Parameter(
                        torch.randn(new_size, d_model).to(old_weight.device) * 0.02
                    )
                    new_weight.data[:old_size] = old_weight.data
                    embed_layer.weight = new_weight
                    logging.info(f"Expanded Ragu embeddings: {old_size} -> {new_size}")
                elif new_size < old_size:
                    # Truncate
                    new_weight = torch.nn.Parameter(old_weight[:new_size])
                    embed_layer.weight = new_weight
                    logging.info(f"Truncated Ragu embeddings: {old_size} -> {new_size}")
                
                # Update optimizer if needed (usually need to re-init optimizer)
                if hasattr(agent_ragu, 'optimizer'):
                     agent_ragu.optimizer = torch.optim.Adam(embed_layer.parameters(), lr=0.001)

                return True
            return False
        except Exception as e:
            logging.error(f"Failed to resize Ragu: {e}")
            return False
    
    def _resize_rahulio_embeddings(self, agent_rahulio, new_size):
        """Resize Rahulio's embedding layer."""
        try:
            if hasattr(agent_rahulio, 'embedding'):
                old_weight = agent_rahulio.embedding.weight
                old_size, d_model = old_weight.shape
                
                if new_size > old_size:
                    new_weight = torch.nn.Parameter(
                        torch.randn(new_size, d_model).to(old_weight.device) * 0.02
                    )
                    new_weight.data[:old_size] = old_weight.data
                    agent_rahulio.embedding.weight = new_weight
                    logging.info(f"Expanded Rahulio embeddings: {old_size} -> {new_size}")
                elif new_size < old_size:
                    new_weight = torch.nn.Parameter(old_weight[:new_size])
                    agent_rahulio.embedding.weight = new_weight
                    logging.info(f"Truncated Rahulio embeddings: {old_size} -> {new_size}")
                return True
            return True # If no embedding, nothing to resize
        except Exception as e:
            logging.error(f"Failed to resize Rahulio: {e}")
            return False
    
    def _update_roo_vocab(self, agent_roo, target_size):
        # Roo is the source, so usually we don't resize him unless we are shrinking to a target?
        # But here we took max, so Roo is likely the max.
        return True
