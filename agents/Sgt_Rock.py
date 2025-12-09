import json
import os
import time
import threading
import gc
import importlib
from datetime import datetime, timedelta
import torch

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

class SgtRock:
    def __init__(self):
        self.progress_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'sgt_rock_progress.json')
        self.load_progress()
        self.agents = {}  # dict of loaded agent instances
        self.monitoring = False
        self.memory_threshold = 80.0  # percent
        self.idle_review_interval = 300  # seconds
        self.last_review_time = time.time()
        self.training_files = []  # list of files for training
        self.training_schedule = []  # list of (agent, files) tuples
        self.o1_interval_seconds = 3600
        
        # New attributes for smart scheduling
        self.last_activity_time = time.time()
        self.user_idle_threshold = 300  # 5 minutes
        self.cpu_threshold = 30.0       # 30% CPU usage limit
        self.chunk_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks')
        self.processed_chunks = set(self.progress.get("processed_chunks", []))

    def update_activity(self):
        """Call this when user interacts with the app"""
        self.last_activity_time = time.time()

    def schedule_o1_training(self, interval_seconds=3600):
        self.o1_interval_seconds = interval_seconds
        
        def training_loop():
            # Initial check delay to let app startup finish
            time.sleep(10)
            
            while True:
                try:
                    now_dt = datetime.now()
                    last_run_str = self.progress.get("o1_last_run")
                    
                    seconds_since_last = float('inf')
                    if last_run_str:
                        try:
                            last_run = datetime.fromisoformat(last_run_str)
                            seconds_since_last = (now_dt - last_run).total_seconds()
                        except ValueError:
                            pass
                    
                    # Criteria
                    time_due = seconds_since_last >= interval_seconds
                    user_idle = (time.time() - self.last_activity_time) > self.user_idle_threshold
                    
                    system_idle = True
                    if PSUTIL_AVAILABLE:
                        # Non-blocking CPU check
                        if psutil.cpu_percent(interval=0.1) > self.cpu_threshold:
                            system_idle = False
                    
                    # Force run if very overdue (e.g. 4x interval) to prevent starvation
                    force_run = seconds_since_last >= (interval_seconds * 4)
                    
                    if (time_due and user_idle and system_idle) or force_run:
                        self.progress["o1_last_run"] = now_dt.isoformat()
                        self.save_progress()
                        
                        epochs = 10 # Small incremental update
                        reason = "Idle"
                        if force_run:
                            reason = "Force (Overdue)"
                            
                        print(f"Sgt_Rock: Starting incremental O1Model training ({reason}) - 10 epochs.")
                        # Run training with reduced epochs
                        os.system(f'python agents/Raulnano/train.py --epochs {epochs}')
                    
                except Exception as e:
                    print(f"Sgt_Rock Scheduler Error: {e}")
                
                # Check every minute
                time.sleep(60)
                
        t = threading.Thread(target=training_loop, daemon=True)
        t.start()

    def _idle_window(self):
        """Check if user and system are idle enough to run background work."""
        user_idle = (time.time() - self.last_activity_time) > self.user_idle_threshold
        system_idle = True
        if PSUTIL_AVAILABLE:
            if psutil.cpu_percent(interval=0.1) > self.cpu_threshold:
                system_idle = False
        return user_idle and system_idle

    def _collect_new_chunks(self):
        """Return list of chunk file paths that haven't been processed yet."""
        if not os.path.isdir(self.chunk_dir):
            return []
        new_chunks = []
        for name in os.listdir(self.chunk_dir):
            path = os.path.join(self.chunk_dir, name)
            if os.path.isfile(path) and name not in self.processed_chunks:
                new_chunks.append(path)
        return new_chunks

    def _mark_chunks_processed(self, chunk_paths):
        for path in chunk_paths:
            name = os.path.basename(path)
            self.processed_chunks.add(name)
        self.progress["processed_chunks"] = list(self.processed_chunks)
        self.save_progress()

    def start_chunk_scheduler(self, interval_seconds=300):
        """Periodically schedule chunk training for other agents during idle windows."""
        def loop():
            time.sleep(5)
            while True:
                try:
                    if not self._idle_window():
                        time.sleep(interval_seconds)
                        continue

                    chunk_batch = self._collect_new_chunks()
                    if chunk_batch:
                        # Reuse existing divided scheduling to spread work across agents.
                        self.schedule_divided_training(chunk_batch)
                        self._mark_chunks_processed(chunk_batch)
                except Exception as e:
                    print(f"Sgt_Rock chunk scheduler error: {e}")
                time.sleep(interval_seconds)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def get_training_batch(self, data_dir):
        """
        Returns a list of files to train based on current time and load.
        Collaborates with Librarian to categorize and chunk files.
        """
        # Lazy load Librarian to avoid circular imports if any
        from agents.Librarian_Agent import LibrarianAgent
        librarian = LibrarianAgent()
        
        now = datetime.now()
        is_night_owl = now.hour >= 1 and now.hour < 6  # 1 AM to 6 AM
        
        # Check system load
        is_idle = True
        if PSUTIL_AVAILABLE:
            if psutil.cpu_percent(interval=0.1) > self.cpu_threshold:
                is_idle = False
        
        batch_files = []
        total_batch_size = 0
        max_batch_size = 1000 * 1024  # 1000KB limit
        
        # Scan files
        candidates = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.json', '.txt')):
                    filepath = os.path.join(root, file)
                    candidates.append(filepath)
        
        # Prioritize and filter
        for filepath in candidates:
            category = librarian.categorize_file(filepath)
            
            should_train = False
            if category == "small":
                should_train = True # Always train small files quickly
            elif category == "medium":
                if is_idle or is_night_owl:
                    should_train = True
            elif category == "large":
                if is_night_owl:
                    # Chunk large files if it's night time
                    chunks = librarian.chunk_file(filepath)
                    # Add chunks to candidates (simplified: just add to batch if fits)
                    for chunk in chunks:
                        chunk_size = os.path.getsize(chunk)
                        if total_batch_size + chunk_size <= max_batch_size:
                            batch_files.append(chunk)
                            total_batch_size += chunk_size
                    continue # Skip adding the original large file
            
            if should_train:
                try:
                    size = os.path.getsize(filepath)
                    if total_batch_size + size <= max_batch_size:
                        batch_files.append(filepath)
                        total_batch_size += size
                    else:
                        # Batch full
                        break
                except OSError:
                    pass
                    
        return batch_files

    def get_o1_status(self):
        last_run = self.progress.get("o1_last_run")
        if last_run is None:
            return "O1-nano training has not run yet."
        try:
            last_dt = datetime.fromisoformat(last_run)
            next_dt = last_dt + timedelta(seconds=self.o1_interval_seconds)
            return f"O1-nano last ran at {last_dt.strftime('%Y-%m-%d %H:%M:%S')} (next scheduled around {next_dt.strftime('%Y-%m-%d %H:%M:%S')})."
        except ValueError:
            return f"O1-nano last run recorded as {last_run}."

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted progress file {self.progress_file}, resetting to default.")
                self.progress = {"memory_checks": 0, "agents_loaded": 0, "training_sessions": 0, "reviews_done": 0, "o1_last_run": None, "processed_chunks": []}
        else:
            self.progress = {"memory_checks": 0, "agents_loaded": 0, "training_sessions": 0, "reviews_done": 0, "o1_last_run": None, "processed_chunks": []}
        self.progress.setdefault("o1_last_run", None)
        self.progress.setdefault("processed_chunks", [])

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=4)

    def get_memory_usage(self):
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent
        else:
            # Fallback: use torch for GPU or resource for CPU
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    if reserved > 0:
                        return (allocated / reserved) * 100.0
                    else:
                        return 0.0
                except Exception:
                    pass
            if RESOURCE_AVAILABLE:
                try:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    # Approximate percentage (this is rough)
                    # resource gives KB, but we can estimate
                    return min(usage.ru_maxrss / 1024 / 1024 * 10, 100.0)  # Rough estimate
                except Exception:
                    pass
            # Final fallback
            gc.collect()
            return 50.0  # Placeholder if all else fails

    def get_detailed_memory_info(self):
        info = {}
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            info['cpu_percent'] = mem.percent
            info['cpu_used_gb'] = mem.used / (1024**3)
            info['cpu_total_gb'] = mem.total / (1024**3)
        if torch.cuda.is_available():
            try:
                info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
                info['gpu_percent'] = (torch.cuda.memory_allocated() / torch.cuda.memory_reserved()) * 100 if torch.cuda.memory_reserved() > 0 else 0
            except Exception:
                pass
        if RESOURCE_AVAILABLE:
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                info['cpu_rss_mb'] = usage.ru_maxrss / 1024
            except Exception:
                pass
        return info

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_memory)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

    def monitor_memory(self):
        while self.monitoring:
            mem_usage = self.get_memory_usage()
            self.progress["memory_checks"] += 1
            if mem_usage > self.memory_threshold:
                self.unload_agents_to_free_memory()
            time.sleep(10)  # Check every 10 seconds

    def unload_agents_to_free_memory(self):
        # Unload least recently used agents
        if self.agents:
            agent_to_unload = min(self.agents.keys(), key=lambda k: self.agents[k].get('last_used', 0))
            del self.agents[agent_to_unload]
            print(f"Sgt_Rock: Unloaded agent {agent_to_unload} due to high memory usage.")

    def load_agent(self, agent_name):
        if agent_name not in self.agents:
            try:
                module = importlib.import_module(f'agents.{agent_name}')
                agent_class = getattr(module, agent_name.replace('_', ''))
                self.agents[agent_name] = {'instance': agent_class(), 'last_used': time.time()}
                self.progress["agents_loaded"] += 1
                self.save_progress()
                print(f"Sgt_Rock: Loaded agent {agent_name}.")
            except (ImportError, AttributeError) as e:
                print(f"Sgt_Rock: Failed to load agent {agent_name}: {e}")

    def unload_agent(self, agent_name):
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"Sgt_Rock: Unloaded agent {agent_name}.")

    def communicate_with_agents(self, message):
        # Communicate with loaded agents
        responses = []
        for agent_name, data in self.agents.items():
            agent = data['instance']
            data['last_used'] = time.time()
            # Assuming agents have a communicate method or process
            if hasattr(agent, 'process'):
                response, _ = agent.process(None, message)
                responses.append(f"{agent_name}: {response}")
        return responses

    def process(self, feature_vectors, filepath=None):
        # For training tasks
        if filepath:
            self.training_files.append(filepath)
            num_files = len(self.training_files)
            if num_files <= 10:
                # All agents train on them
                self.schedule_training_all_agents(self.training_files)
            elif num_files >= 50:
                # Divide work and schedule over time
                self.schedule_divided_training(self.training_files)
            self.progress["training_sessions"] += 1
            self.save_progress()
            result = f"Sgt_Rock: Scheduled training for {num_files} files."
            return result, feature_vectors  # Pass through
        else:
            # Check if idle
            if time.time() - self.last_review_time > self.idle_review_interval:
                self.review_past_data()
            return "Sgt_Rock: Monitoring resources.", feature_vectors

    def schedule_training_all_agents(self, files):
        # Load all agents and assign all files
        agent_names = ['Agent_Raul', 'Agent_Boss', 'Agent_DocRagu', 'Agent_MajorRoo', 'Agent_Rahulio', 'Agent_Rudy', 'Librarian_Agent', 'NLPRaul_agent']  # Example
        for agent_name in agent_names:
            self.load_agent(agent_name)
            # Assume agents have a train method or something, but since not, perhaps just note
            print(f"Sgt_Rock: Assigned all {len(files)} files to {agent_name}.")

    def schedule_divided_training(self, files):
        # Divide files among agents and schedule over time
        agent_names = ['Agent_Raul', 'Agent_Boss', 'Agent_DocRagu', 'Agent_MajorRoo', 'Agent_Rahulio', 'Agent_Rudy', 'Librarian_Agent', 'NLPRaul_agent']
        num_agents = len(agent_names)
        if not files:
            return
        chunk_size = max(1, len(files) // num_agents)
        for i, agent_name in enumerate(agent_names):
            start = i * chunk_size
            end = start + chunk_size if i < num_agents - 1 else len(files)
            agent_files = files[start:end]
            if not agent_files:
                continue
            self.training_schedule.append((agent_name, agent_files))
            # Schedule over time, perhaps delay
            threading.Timer(i * 60, self.train_agent, args=(agent_name, agent_files)).start()  # Example delay

    def train_agent(self, agent_name, files):
        self.load_agent(agent_name)
        # Simulate training
        print(f"Sgt_Rock: Training {agent_name} on {len(files)} files.")

    def review_past_data(self):
        # Review past data for deep training
        # Load some past data and process
        print("Sgt_Rock: Reviewing past data for deep training.")
        self.progress["reviews_done"] += 1
        self.save_progress()
        self.last_review_time = time.time()

    def process_vectors(self, vectors):
        # Vector communication
        if vectors is not None:
            # Modulate vectors for communication
            # Simple example: add noise or something
            new_vectors = vectors + torch.randn_like(vectors) * 0.05
            result = f"Sgt_Rock: Communicating resource status via vectors for {vectors.shape[0]} vectors."
            return result, new_vectors
        return "Sgt_Rock: No vectors to process", None

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
