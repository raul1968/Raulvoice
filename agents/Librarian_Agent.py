import json
import os
import hashlib
import shutil
import torch
import re
import math

class LibrarianAgent:
    def __init__(self):
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'librarian_progress.json')
        self.load_progress()
        # Point librarian to text_data so reviews/chunking operate on that corpus
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'text_data')
        self.archive_dir = os.path.join(self.data_dir, 'archive')
        os.makedirs(self.archive_dir, exist_ok=True)
        self.chunks_dir = os.path.join(self.data_dir, 'chunks')
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.max_active_files = 100  # first 100 files stay active
        self.chunk_size_chars = 5000  # target chunk size for large text files
        self.file_hashes = set()  # To track unique files
        self.patterns_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'librarian_patterns.json')
        self.load_patterns()

    def fix_corrupt_file(self, filepath, default_content=None):
        """
        Archives the corrupt file and optionally creates a new one with default content.
        """
        print(f"Librarian: Detected corruption in {filepath}. Initiating repair protocol...")
        if os.path.exists(filepath):
            self.archive_file(filepath)
        
        if default_content is not None:
            try:
                with open(filepath, 'w') as f:
                    json.dump(default_content, f, indent=4)
                print(f"Librarian: Restored {filepath} with default configuration.")
            except Exception as e:
                print(f"Librarian: Failed to restore {filepath}: {e}")
        return default_content

    def load_patterns(self):
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.unwanted_patterns = patterns_data.get('unwanted_patterns', [])
                    self.learned_patterns = patterns_data.get('learned_patterns', {})
            except json.JSONDecodeError:
                print(f"Warning: Corrupted patterns file {self.patterns_file}.")
                self.set_default_patterns()
                # Self-heal
                self.fix_corrupt_file(self.patterns_file, {
                    'unwanted_patterns': self.unwanted_patterns,
                    'learned_patterns': self.learned_patterns
                })
        else:
            self.set_default_patterns()

    def set_default_patterns(self):
        self.unwanted_patterns = [
            r':\w+:',  # Emotes like :smile:
            r'\b(non-consumable|item)\b',  # Non-consumable items
        ]
        self.learned_patterns = {}

    def save_patterns(self):
        patterns_data = {
            'unwanted_patterns': self.unwanted_patterns,
            'learned_patterns': self.learned_patterns
        }
        with open(self.patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=4)

    def learn_patterns(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        word_freq = {}
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        text = data.get('cleaned_text', '')
                        words = re.findall(r'\b\w+\b', text.lower())
                        for word in words:
                            word_freq[word] = word_freq.get(word, 0) + 1
                    except Exception:
                        continue
        
        # Find potential unwanted patterns: words that appear very frequently or are emotes
        total_words = sum(word_freq.values())
        threshold = total_words * 0.01  # 1% of total words
        learned = []
        for word, count in word_freq.items():
            if count > threshold and len(word) > 3:  # Skip short words
                if re.match(r'^\w+$', word):  # Alphanumeric
                    learned.append(r'\b' + re.escape(word) + r'\b')
        
        # Update learned patterns
        self.learned_patterns = {pattern: word_freq.get(re.sub(r'\\b|\\', '', pattern), 0) for pattern in learned}
        self.save_patterns()
        print(f"Librarian: Learned {len(learned)} new patterns from data.")

    def categorize_file(self, filepath):
        """Categorize file by size: small (<50KB), medium (<500KB), large (>500KB)"""
        try:
            size = os.path.getsize(filepath)
            if size < 50 * 1024:
                return "small"
            elif size < 500 * 1024:
                return "medium"
            else:
                return "large"
        except OSError:
            return "unknown"

    def chunk_file(self, filepath, chunk_size=None):
        """Split large files into smaller chunks (default 5k chars)."""
        if chunk_size is None:
            chunk_size = self.chunk_size_chars
        try:
            base_name = os.path.basename(filepath)
            name, ext = os.path.splitext(base_name)
            
            chunk_files = []
            
            if ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Assuming data is a list or dict we can split? 
                # If it's a complex structure, simple splitting is hard.
                # Fallback: if it has a 'text' field, split that.
                content = json.dumps(data)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Split content string
            total_len = len(content)
            num_chunks = max(1, math.ceil(total_len / chunk_size))
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total_len)
                chunk_content = content[start:end]
                
                chunk_filename = f"{name}_chunk_{i}{ext}"
                chunk_path = os.path.join(self.chunks_dir, chunk_filename)
                
                # Save chunk
                if ext == '.json':
                    # Try to make it valid JSON if possible, or just save as text wrapped
                    try:
                        # This is risky for arbitrary JSON. 
                        # Better to save as a text wrapper for the agent to process raw.
                        with open(chunk_path, 'w', encoding='utf-8') as f:
                            f.write(chunk_content) 
                    except:
                        pass
                else:
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk_content)
                
                chunk_files.append(chunk_path)
                
            print(f"Librarian: Chunked {filepath} into {len(chunk_files)} parts.")
            return chunk_files
            
        except Exception as e:
            print(f"Librarian: Error chunking {filepath}: {e}")
            return [filepath]

    def add_pattern(self, pattern):
        if pattern not in self.unwanted_patterns:
            self.unwanted_patterns.append(pattern)
            self.save_patterns()
            print(f"Librarian: Added pattern {pattern}")

    def remove_pattern(self, pattern):
        if pattern in self.unwanted_patterns:
            self.unwanted_patterns.remove(pattern)
            self.save_patterns()
            print(f"Librarian: Removed pattern {pattern}")

    def check_log_file(self, log_path=None):
        if log_path is None:
            log_path = os.path.join(self.data_dir, 'user_interactions.txt')
        if not os.path.exists(log_path):
            print(f"Librarian: Log file {log_path} does not exist.")
            return True
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    json.loads(line)
            print(f"Librarian: Log file {log_path} is valid.")
            return True
        except Exception as e:
            print(f"Librarian: Log file {log_path} is corrupted: {e}")
            self.progress["corrupted_files"] += 1
            self.save_progress()
            return False

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted progress file {self.progress_file}.")
                self.progress = {"files_reviewed": 0, "duplicates_found": 0, "files_cleaned": 0, "files_archived": 0, "corrupted_files": 0}
                # Self-heal
                self.fix_corrupt_file(self.progress_file, self.progress)
        else:
            self.progress = {"files_reviewed": 0, "duplicates_found": 0, "files_cleaned": 0, "files_archived": 0, "corrupted_files": 0}

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)

    def review_file(self, filepath):
        self.progress["files_reviewed"] += 1
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Librarian: Corrupted file {filepath}, archiving.")
            self.progress["corrupted_files"] += 1
            self.archive_file(filepath)
            return None

        # Check for duplicates based on cleaned_text hash
        if 'cleaned_text' in data:
            text_hash = hashlib.md5(data['cleaned_text'].encode()).hexdigest()
            if text_hash in self.file_hashes:
                print(f"Librarian: Duplicate file {filepath}, archiving.")
                self.progress["duplicates_found"] += 1
                self.archive_file(filepath)
                return None
            self.file_hashes.add(text_hash)

        # Clean unwanted content
        cleaned = False
        all_patterns = self.unwanted_patterns + list(self.learned_patterns.keys())
        if 'cleaned_text' in data:
            original_text = data['cleaned_text']
            for pattern in all_patterns:
                data['cleaned_text'] = re.sub(pattern, '', data['cleaned_text'], flags=re.IGNORECASE)
            if data['cleaned_text'] != original_text:
                cleaned = True

        # Filter entities if any unwanted
        if 'entities' in data:
            data['entities'] = [e for e in data['entities'] if not any(re.search(p, str(e), re.IGNORECASE) for p in all_patterns)]
            cleaned = True

        if cleaned:
            # Rewrite the file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            self.progress["files_cleaned"] += 1
            print(f"Librarian: Cleaned file {filepath}.")

        return data

    def archive_file(self, filepath):
        archive_path = os.path.join(self.archive_dir, os.path.basename(filepath))
        shutil.move(filepath, archive_path)
        self.progress["files_archived"] += 1
        print(f"Librarian: Archived {filepath} to {archive_path}.")

    def manage_text_corpus(self, limit=None):
        """Keep the first N text files active (chunked to 5k), archive the rest when digested."""
        if limit is None:
            limit = self.max_active_files

        # collect txt files (top-level data dir)
        txt_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        txt_files.sort()

        active = txt_files[:limit]
        inactive = txt_files[limit:]

        # Chunk active files if large
        for filepath in active:
            try:
                size = os.path.getsize(filepath)
                if size > self.chunk_size_chars:
                    chunk_paths = self.chunk_file(filepath, self.chunk_size_chars)
                    # Archive original after successful chunking so only chunks stay active
                    if chunk_paths:
                        self.archive_file(filepath)
            except Exception as e:
                print(f"Librarian: manage_text_corpus chunk error for {filepath}: {e}")

        # Archive inactive files that are no longer needed
        for filepath in inactive:
            try:
                self.archive_file(filepath)
            except Exception as e:
                print(f"Librarian: manage_text_corpus archive error for {filepath}: {e}")

    def scan_and_process_files(self):
        # Manage active text corpus: keep first N files active, chunk to 5k chars
        self.manage_text_corpus()

        # Scan perception_tokens and other json folders
        folders_to_scan = ['perception_tokens', 'json_data']
        for folder in folders_to_scan:
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.json'):
                        filepath = os.path.join(folder_path, file)
                        self.review_file(filepath)

        self.save_progress()

    def process(self, feature_vectors, filepath=None):
        # Coordinate with Sgt_Rock: Assume permission granted or request via vectors
        if filepath:
            data = self.review_file(filepath)
            if data:
                # Create vectors from cleaned data, e.g., encode text
                # Simple: use length or something as vector
                vector_dim = 10  # Example
                vectors = torch.randn(1, vector_dim) * len(data.get('cleaned_text', ''))
                result = f"Librarian: Reviewed and cleaned file {filepath}."
                return result, vectors
            else:
                return "Librarian: File archived or corrupted.", None
        else:
            # Periodic scan
            self.scan_and_process_files()
            result = f"Librarian: Scanned and processed files. Reviewed: {self.progress['files_reviewed']}, Archived: {self.progress['files_archived']}."
            # Return status vectors
            status_vector = torch.tensor([self.progress['files_reviewed'], self.progress['duplicates_found'], self.progress['files_cleaned'], self.progress['files_archived']], dtype=torch.float32)
            return result, status_vector

    def process_vectors(self, vectors):
        # Communicate with agents via vectors: Modulate vectors with librarian status
        if vectors is not None:
            # Add librarian's progress influence
            progress_factor = sum(self.progress.values()) / 100.0
            new_vectors = vectors * (1 + progress_factor) + torch.randn_like(vectors) * 0.01
            result = f"Librarian: Communicating data management status via vectors, influencing {vectors.shape[0]} vectors."
            return result, new_vectors
        return "Librarian: No vectors to process", None

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
