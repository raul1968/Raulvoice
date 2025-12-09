from tkinter import scrolledtext, Entry, Button
from tkinter import ttk
import tkinter as tk
from agents.agent_roo import AgentRoo
from agents.agent_ragu import AgentRagu
from agents.Agent_Rahulio import AgentRahulio
from agents.Agent_Raul import AgentRaul
from agents.Agent_Boss import AgentBoss
from agents.Agent_Rudy import AgentRudy
from agents.Agent_MajorRoo import AgentMajorRoo
from agents.Agent_DocRagu import AgentDocRagu
from agents.Sgt_Rock import SgtRock
from agents.Librarian_Agent import LibrarianAgent
from agents.NLPRaul_agent import NLPRaulAgent
from agents.Agent_Linguist import AgentLinguist
from agents.vocab_sync import VocabularySynchronizer
import os
import json
from datetime import datetime
import time
import math
import torch
import re
import math
import webbrowser

class RaulChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("Raul AI Assistant")
        
        # Set device - use RTX 4050 GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using device: {self.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = torch.device('cpu')
            print(f"Using device: {self.device}")
            print("WARNING: CUDA not available. Training will be slow on CPU.")
        
        # Status label and progress bar above dialogue
        self.status_label = tk.Label(root, text="Status: Idle")
        self.status_label.pack()
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack()
        
        # Dialogue box
        self.dialogue = scrolledtext.ScrolledText(root, height=20, width=50)
        self.dialogue.pack()
        
        # Input box
        self.input_entry = Entry(root, width=50)
        self.input_entry.pack()
        self.input_entry.bind("<Return>", self.send_message)
        
        self.send_button = Button(root, text="Send", command=self.send_message)
        self.send_button.pack()
        
        # Agents
        self.agents = {
            "Roo": AgentRoo(self.device),
            "Ragu": AgentRagu(self.device),
            "Rahulio": AgentRahulio(self.device),
            "Raul": AgentRaul(self.device),
            "Boss": AgentBoss(self.device),
            "Rudy": AgentRudy(self.device),
            "MajorRoo": AgentMajorRoo(self.device),
            "DocRagu": AgentDocRagu(self.device),
            "SgtRock": SgtRock(),
            "Librarian": LibrarianAgent(),
            "NLPRaul": NLPRaulAgent(),
            "Linguist": AgentLinguist(self.device),
            # Add other agents here
        }
        
        # Synchronize vocabulary and persist resized Ragu weights if needed
        try:
            sync = VocabularySynchronizer()
            sync.synchronize_agents(self.agents["Roo"], self.agents["Ragu"], self.agents["Rahulio"])
            # Persist Ragu resized weights so they load on next startup
            try:
                ragu_weight_path = os.path.join(os.path.dirname(__file__), 'data', 'ragu_resized.pth')
                torch.save({'weight': self.agents["Ragu"].embed.weight.detach().cpu()}, ragu_weight_path)
                print(f"Saved Ragu resized embeddings to {ragu_weight_path}")
            except Exception as save_e:
                print(f"Warning: Could not save resized Ragu weights: {save_e}")
        except Exception as e:
            print(f"Vocabulary synchronization failed: {e}")

        self.agents["SgtRock"].start_monitoring()
        # Sgt Rock keeps the O1-nano training loop alive so the hive mind can draw on its arithmetic reasoning updates.
        self.agents["SgtRock"].schedule_o1_training()
        # Schedule background chunk processing for other agents during idle periods
        self.agents["SgtRock"].start_chunk_scheduler()
        
        # Info box on top
        self.info_text = tk.Text(root, height=6, width=60)
        self.info_text.pack()
        
        # Initial stats update
        self.update_info_box()
        
        # Long-term memory persisted to disk
        self.memory_file = os.path.join('data', 'memory.json')
        self.memory = self.load_memory()
        # Track last arbitration choices for user follow-up
        self.last_arbitration = None
        # Feature flag for O1 reasoning (default enabled)
        self.use_o1 = True
        # Training state tracking
        self.train_state_path = os.path.join('data', 'train_state.json')
        self.train_state = self.load_train_state()
        
    def _o1_low_quality(self, text: str) -> bool:
        if not text:
            return True
        text = str(text).strip()
        if not text:
            return True
        alpha = sum(1 for c in text if c.isalpha())
        ratio = alpha / max(len(text), 1)
        tokens = text.lower().split()
        unique_tokens = len(set(tokens))
        # Heuristic: very short, low alphabetic density, or low token diversity -> likely gibberish
        if len(text) < 15:
            return True
        if ratio < 0.35:
            return True
        if unique_tokens < 3:
            return True
        return False

    def get_agent_stats(self):
        total_params = 0
        total_vocab = 0
        
        for name, agent in self.agents.items():
            # Count parameters
            if hasattr(agent, 'parameters'):
                try:
                    total_params += sum(p.numel() for p in agent.parameters() if p.requires_grad)
                except:
                    pass
            elif hasattr(agent, 'model') and hasattr(agent.model, 'parameters'):
                try:
                    total_params += sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
                except:
                    pass
            # Check for individual layers if agent is not an nn.Module
            else:
                for attr_name in dir(agent):
                    try:
                        attr = getattr(agent, attr_name)
                        if isinstance(attr, torch.nn.Module):
                            total_params += sum(p.numel() for p in attr.parameters() if p.requires_grad)
                    except:
                        pass
            
            # Count vocab
            if hasattr(agent, 'vocab_size'):
                total_vocab += agent.vocab_size
                
        return total_params, total_vocab

    def update_info_box(self, status="Idle"):
        params, vocab = self.get_agent_stats()
        
        # Format parameters
        if params > 1_000_000:
            param_str = f"{params/1_000_000:.2f}M"
        elif params > 1_000:
            param_str = f"{params/1_000:.2f}K"
        else:
            param_str = str(params)
            
        # Get specific details
        # Roo's vocab is dynamic (property), so accessing it gets the current count
        roo_vocab = getattr(self.agents["Roo"], 'vocab_size', 0)
        
        # Ragu's vocab is static (attribute), so it won't change unless we make it dynamic
        ragu_vocab = getattr(self.agents["Ragu"], 'vocab_size', 0)
        
        rahulio = self.agents.get("Rahulio")
        if rahulio and hasattr(rahulio, 'conv1d'):
            rahulio_features = rahulio.conv1d.out_channels
        else:
            rahulio_features = 0
            
        # Rahulio's "processed sequences" is a better dynamic metric than static features
        rahulio_processed = 0
        if rahulio and hasattr(rahulio, 'progress'):
             rahulio_processed = rahulio.progress.get("processed_sequences", 0)

        info_lines = [
            f"Parameters: {param_str}",
            f"Total Vocab: {vocab}",
            f"Details: Roo {roo_vocab} words, Ragu {ragu_vocab} slots, Rahulio {rahulio_processed} seqs",
            f"Agent Status: {status}",
            "AI Assistant Mode: Active",
            "O1 Nano: Scheduled via Sgt Rock (agents/Raulnano/train.py)",
        ]
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info_lines) + "\n")

    def send_message(self, event=None):
        # Notify Sgt Rock of user activity
        if "SgtRock" in self.agents:
            self.agents["SgtRock"].update_activity()

        user_input = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self.dialogue.insert(tk.END, f"You: {user_input}\n")
        self.dialogue.see(tk.END) # Auto-scroll
        
        # Process in background thread
        import threading
        threading.Thread(target=self.process_and_respond, args=(user_input,), daemon=True).start()

    def process_and_respond(self, user_input):
        try:
            # Process with agents
            response = self.process_input(user_input)
            # Update dialogue safely
            def update_dialogue():
                self.dialogue.insert(tk.END, f"Raul: {response}\n")
                self.dialogue.see(tk.END) # Auto-scroll
            self.root.after(0, update_dialogue)
        except Exception as e:
            self.root.after(0, lambda: self.dialogue.insert(tk.END, f"Error: {e}\n"))
        
        # Update memory
        entry = {"timestamp": datetime.now().isoformat(), "user": user_input, "raul": response}
        self.memory.append(entry)
        self.save_memory()
        
        # Log conversation for future training
        self.log_conversation(user_input, response)

    def log_conversation(self, user_input, raul_response):
        log_path = os.path.join('data', 'user_interactions.txt')
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'raul': raul_response
        }
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Failed to log conversation: {e}")

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load memory file {self.memory_file}: {e}")
        return []

    def save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Failed to save memory: {e}")

    def load_train_state(self):
        if os.path.exists(self.train_state_path):
            try:
                with open(self.train_state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load train state {self.train_state_path}: {e}")
        return {}

    def save_train_state(self):
        try:
            os.makedirs(os.path.dirname(self.train_state_path), exist_ok=True)
            with open(self.train_state_path, 'w', encoding='utf-8') as f:
                json.dump(self.train_state, f, indent=2)
        except Exception as e:
            print(f"Failed to save train state: {e}")

    def _fmt_time(self, iso_str):
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return iso_str

    def _get_o1_last_trained(self):
        if self.train_state.get("o1_last_trained"):
            return self._fmt_time(self.train_state["o1_last_trained"])
        checkpoint = os.path.join("agents", "Raulnano", "o1_model.pth")
        if os.path.exists(checkpoint):
            ts = datetime.fromtimestamp(os.path.getmtime(checkpoint)).strftime("%Y-%m-%d %H:%M:%S")
            return f"Checkpoint modified: {ts}"
        return "No O1 training record found."

    def _get_raul_last_trained(self):
        if self.train_state.get("raul_agent_last_trained"):
            return self._fmt_time(self.train_state["raul_agent_last_trained"])
        return "No Raul-agent training record found."

    def process_input(self, input_text):
        if input_text.lower() == "train text":
            self.train_data()
            return "Training initiated by Agent Roo. Files scanned and updated for incremental learning."
        
        if input_text.lower() == "train agent roo":
            self.train_agent_roo()
            return "Agent Roo training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent ragu":
            self.train_agent_ragu()
            return "Agent Ragu training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent rahulio":
            self.train_agent_rahulio()
            return "Agent Rahulio training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent raul":
            self.train_agent_raul()
            return "Agent Raul training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent boss":
            self.train_agent_boss()
            return "Agent Boss training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent rudy":
            self.train_agent_rudy()
            return "Agent Rudy training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent majorroo":
            self.train_agent_majorroo()
            return "Agent MajorRoo training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "train agent docragu":
            self.train_agent_docragu()
            return "Agent DocRagu training initiated on JSON and TXT files in data directory."
        
        if input_text.lower() == "help":
            return self.get_help()

        if input_text.lower() == "choose capsule":
            if self.last_arbitration:
                return f"You chose capsule: {self.last_arbitration.get('capsule','(none)')}"
            return "No recent arbitration to choose from."

        if input_text.lower() == "choose o1":
            if self.last_arbitration:
                return f"You chose O1: {self.last_arbitration.get('o1','(none)')}"
            return "No recent arbitration to choose from."

        if input_text.lower() == "ask more info":
            if self.last_arbitration:
                original = self.last_arbitration.get('user_input','(unknown question)')
                return (
                    "I need a bit more detail to resolve the difference. "
                    f"What specifically should I focus on about: '{original}'?"
                )
            return "No recent arbitration to clarify."

        if input_text.lower() == "o1 on":
            self.use_o1 = True
            return "O1 enabled."

        if input_text.lower() == "o1 off":
            self.use_o1 = False
            return "O1 disabled."

        if input_text.lower().startswith("recall memory"):
            parts = input_text.split()
            n = 5
            if len(parts) > 2:
                try:
                    n = int(parts[2])
                except ValueError:
                    pass
            recent = self.memory[-n:]
            if not recent:
                return "Memory is empty."
            lines = [f"{i+1}. [{item.get('timestamp','')}] You: {item.get('user','')} | Raul: {item.get('raul','')}" for i, item in enumerate(recent)]
            return "Recent memory:\n" + "\n".join(lines)
        
        if input_text.lower() == "librarian review":
            return self.librarian_review()
        
        if input_text.lower() == "librarian learn patterns":
            self.agents["Librarian"].learn_patterns()
            return "Librarian: Learning patterns from data directory."
        
        if input_text.lower().startswith("librarian add pattern"):
            parts = input_text.split(" ", 3)
            if len(parts) > 3:
                pattern = parts[3]
                self.agents["Librarian"].add_pattern(pattern)
                return f"Librarian: Added pattern '{pattern}'"
            else:
                return "Usage: librarian add pattern <regex>"
        
        if input_text.lower().startswith("librarian remove pattern"):
            parts = input_text.split(" ", 3)
            if len(parts) > 3:
                pattern = parts[3]
                self.agents["Librarian"].remove_pattern(pattern)
                return f"Librarian: Removed pattern '{pattern}'"
            else:
                return "Usage: librarian remove pattern <regex>"
        
        if input_text.lower().startswith("nlp process"):
            parts = input_text.split(" ", 2)
            if len(parts) > 2:
                text_to_process = parts[2]
                return self.nlp_process(text_to_process)
            else:
                return "Usage: nlp process <text>"
        
        if input_text.lower().startswith("calculate"):
            parts = input_text.split(" ", 1)
            if len(parts) > 1:
                expr = parts[1]
                return self.calculate(expr)
            else:
                return "Usage: calculate <expression>"
        
        if input_text.lower() == "check vocabulary":
            try:
                # Get sizes
                roo_size = 11049  # From your data
                
                try:
                    ragu_size = self.agents["Ragu"].embed.weight.shape[0]
                except:
                    try:
                        ragu_size = self.agents["Ragu"].model.embed.weight.shape[0]
                    except:
                        ragu_size = "Unknown (error)"
                    
                try:
                    rahulio_size = self.agents["Rahulio"].embedding.weight.shape[0]
                except:
                    rahulio_size = "Unknown (error)"
                
                # Check for issues
                issues = []
                if isinstance(ragu_size, int) and ragu_size < roo_size:
                    issues.append(f"Ragu can't embed {roo_size - ragu_size} words!")
                if isinstance(rahulio_size, int) and rahulio_size < roo_size:
                    issues.append(f"Rahulio can't process {roo_size - rahulio_size} words!")
                
                message = (
                    f"Vocabulary Status:\n"
                    f"• AgentRoo: {roo_size} actual words\n"
                    f"• AgentRagu: {ragu_size} embedding slots\n"
                    f"• AgentRahulio: {rahulio_size} vocabulary size\n\n"
                )
                
                if issues:
                    message += "🚨 ISSUES DETECTED:\n" + "\n".join(f"  - {issue}" for issue in issues)
                    message += "\n\nType 'fix vocabulary' to resize embeddings."
                else:
                    message += "✅ All agents synchronized!"
                
                return message
                
            except Exception as e:
                return f"Vocabulary check error: {e}"

        if input_text.lower() == "fix vocabulary":
            try:
                from agents.vocab_sync import VocabularySynchronizer
                sync = VocabularySynchronizer()

                # Determine actual sizes
                roo_size = 11049
                if hasattr(self.agents["Roo"], "word2idx"):
                    try:
                        roo_size = len(self.agents["Roo"].word2idx)
                    except Exception:
                        pass

                # Run synchronizer to align all agents to the max-needed size
                sync.synchronize_agents(self.agents["Roo"], self.agents["Ragu"], self.agents["Rahulio"])

                # Read back sizes after sync
                try:
                    ragu_size = self.agents["Ragu"].embed.weight.shape[0]
                except Exception:
                    ragu_size = "Unknown"
                try:
                    rahulio_size = self.agents["Rahulio"].embedding.weight.shape[0]
                except Exception:
                    rahulio_size = "Unknown"

                target_size = "max-aligned"
                return (
                    f"✅ Vocabulary synchronized.\n"
                    f"Roo: {roo_size}\nRagu: {ragu_size}\nRahulio: {rahulio_size}\n"
                    f"Restart GUI if you want to reload resized weights."
                )

            except Exception as e:
                return f"Resize failed: {e}"

        if input_text.lower().startswith("search"):
            parts = input_text.split(" ", 1)
            if len(parts) > 1:
                query = parts[1]
                return self.search_web(query)
            else:
                return "Usage: search <query>"
        
        lowered = input_text.lower()

        if lowered == "time" or "what time" in lowered:
            return self.get_time()

        if lowered == "date" or lowered.startswith("what is the date") or lowered.startswith("what's the date") or lowered.startswith("what date"):
            return self.get_date()

        if lowered in {
            "when was o1 trained",
            "when was o1 last trained",
            "o1 last trained",
            "when was 01 trained",
            "when was 01 last trained",
            "01 last trained",
        }:
            return f"O1 last trained: {self._get_o1_last_trained()}"

        if lowered in {
            "when was raul trained",
            "when was raul agent last trained",
            "raul last trained",
            "when was the raul agent last trained",
        }:
            return f"Raul agent last trained: {self._get_raul_last_trained()}"
        
        if input_text.lower().startswith("note"):
            parts = input_text.split(" ", 1)
            if len(parts) > 1:
                note = parts[1]
                return self.create_note(note)
            else:
                return "Usage: note <text>"
        
        if input_text.lower().startswith("list files"):
            parts = input_text.split(" ", 2)
            directory = parts[2] if len(parts) > 2 else "data"
            return self.list_files(directory)
        
        if input_text.lower().startswith("explain code"):
            parts = input_text.split(" ", 2)
            if len(parts) > 2:
                code = parts[2]
                return self.explain_code(code)
            else:
                return "Usage: explain code <code_snippet>"

        if input_text.lower() == "o1 status":
            return self.agents["SgtRock"].get_o1_status()
        
        # Normal processing: full agent chain for Tay-like response
        response = self.generate_tay_response(input_text)
        return response

    def generate_tay_response(self, input_text):
        # Full agent chain for generating Tay-like response
        try:
            # Roo -> Ragu -> Rahulio -> Raul -> Boss -> Rudy -> MajorRoo -> DocRagu
            try:
                roo_output, roo_embeddings = self.agents["Roo"].process(input_text)
            except Exception as e:
                return f"Roo error: {e}"

            # Debug: log shape
            if roo_embeddings is not None and hasattr(roo_embeddings, 'shape'):
                print(f"Roo embeddings shape: {getattr(roo_embeddings, 'shape', None)}")

            if roo_embeddings is not None:
                try:
                    ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                except Exception as e:
                    return f"Ragu error: {e}"

                if ragu_embeddings is not None and hasattr(ragu_embeddings, 'shape'):
                    print(f"Ragu embeddings shape: {getattr(ragu_embeddings, 'shape', None)}")

                if ragu_embeddings is not None:
                    try:
                        rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings)
                    except Exception as e:
                        return f"Rahulio error: {e}"

                    if feature_vectors is not None and hasattr(feature_vectors, 'shape'):
                        print(f"Rahulio feature_vectors shape: {getattr(feature_vectors, 'shape', None)}")

                    if feature_vectors is not None:
                        try:
                            raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors)
                        except Exception as e:
                            return f"Raul error: {e}"

                        if capsule_outputs is not None and hasattr(capsule_outputs, 'shape'):
                            print(f"Raul capsule_outputs shape: {getattr(capsule_outputs, 'shape', None)}")

                        if capsule_outputs is not None:
                            # --- LINGUIST INTEGRATION START ---
                            # Automatically check if these capsule vectors represent new concepts
                            try:
                                # We take the mean of the capsule outputs to get a single "concept vector" for this turn
                                # capsule_outputs shape: [batch, num_capsules, capsule_dim]
                                if capsule_outputs.dim() >= 2:
                                    concept_vector = torch.mean(capsule_outputs, dim=1)
                                    if concept_vector.dim() > 1:
                                        concept_vector = concept_vector.mean(dim=0) # Flatten batch if needed
                                    
                                    symbol = self.agents["Linguist"].encode_vector(concept_vector)
                                    # If a NEW symbol was created (starts with SYM_), log it quietly
                                    # We check if the symbol counter increased or just by name pattern if we track it
                                    # For now, just print debug info to console
                                    print(f"Linguist: Mapped concept to {symbol}")
                            except Exception as e:
                                print(f"Linguist observation error: {e}")
                            # --- LINGUIST INTEGRATION END ---

                            try:
                                boss_output, intent_outputs = self.agents["Boss"].process(capsule_outputs)
                            except Exception as e:
                                return f"Boss error: {e}"

                            if intent_outputs is not None and hasattr(intent_outputs, 'shape'):
                                print(f"Boss intent_outputs shape: {getattr(intent_outputs, 'shape', None)}")

                            if intent_outputs is not None:
                                try:
                                    rudy_output, dialog_outputs = self.agents["Rudy"].process(intent_outputs)
                                except Exception as e:
                                    return f"Rudy error: {e}"

                                if dialog_outputs is not None and hasattr(dialog_outputs, 'shape'):
                                    print(f"Rudy dialog_outputs shape: {getattr(dialog_outputs, 'shape', None)}")

                                if dialog_outputs is not None:
                                    try:
                                        majorroo_output, reconstructed = self.agents["MajorRoo"].process(dialog_outputs, roo_embeddings)
                                    except Exception as e:
                                        return f"MajorRoo error: {e}"

                                    if reconstructed is not None and hasattr(reconstructed, 'shape'):
                                        print(f"MajorRoo reconstructed shape: {getattr(reconstructed, 'shape', None)}")

                                    if reconstructed is not None:
                                        try:
                                            docragu_output, response_text = self.agents["DocRagu"].process(dialog_outputs)
                                        except Exception as e:
                                            return f"DocRagu error: {e}"

                                        # Add personality
                                        personality = ""
                                        if "coffee" in input_text.lower():
                                            personality += "I love coffee! "
                                        if "animation" in input_text.lower():
                                            personality += "Animation is amazing! "
                                        if "?" in input_text:
                                            personality += "That's an interesting question. "

                                        # Cross-check with O1 for agreement and offer arbitration (gated)
                                        o1_resp = ""
                                        if self.use_o1:
                                            try:
                                                o1_resp = self.agents["NLPRaul"].run_o1_completion(input_text)
                                            except Exception as e:
                                                o1_resp = f"O1 error: {e}"

                                        capsule_reply = f"{personality}{response_text}"

                                        if self.use_o1 and o1_resp and isinstance(o1_resp, str) and o1_resp.strip() and o1_resp.strip() != capsule_reply.strip():
                                            o1_low_quality = self._o1_low_quality(o1_resp)
                                            if o1_low_quality:
                                                full_response = f"{capsule_reply} (Using capsule; O1 still training. If this seems off, share more detail.)"
                                            else:
                                                full_response = capsule_reply
                                            self.last_arbitration = {
                                                "capsule": capsule_reply,
                                                "o1": o1_resp,
                                                "user_input": input_text,
                                                "o1_low_quality": o1_low_quality,
                                            }
                                        else:
                                            full_response = capsule_reply
                                            self.last_arbitration = None

                                        # Update memory with agent insights
                                        self.memory.append({"user": input_text, "raul": full_response, "agents": [roo_output, ragu_output, rahulio_output, raul_output, boss_output, rudy_output, majorroo_output, docragu_output], "o1": o1_resp})

                                        return full_response
                                    else:
                                        return "I'm still learning... MajorRoo couldn't reconstruct."
                                else:
                                    return "I'm processing your message... Rudy is thinking."
                            else:
                                return "Got it, Boss is routing intents."
                        else:
                            return "Raul is forming capsules."
                    else:
                        return "Rahulio is extracting features."
                else:
                    return "Ragu is refining embeddings."
            else:
                return "Roo couldn't process the input."
        except Exception as e:
            # If the error is about tensor shape mismatch, show a more specific message
            if "expand" in str(e) and "number of sizes provided" in str(e):
                return f"Sorry, a tensor shape mismatch occurred in the agent chain: {e}\nPlease check the input data or agent implementations."
            return f"Sorry, an error occurred: {e}. I'm still learning!"

    def train_data(self):
        data_dir = "data"
        for root, dirs, files in os.walk(data_dir):
            if 'fine_tune' in root.replace('\\', '/').lower():
                continue
            for file in files:
                if file.endswith('.json'):
                    filepath = os.path.join(root, file)
                    # Process with Agent Roo
                    roo_output = self.agents["Roo"].process_file(filepath)
                    # Update file for incremental training
                    self.update_file_for_training(filepath)
                    # Log to dialogue
                    self.dialogue.insert(tk.END, f"Training: {roo_output}\n")

    def update_file_for_training(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Add metadata
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["reviewed_by_roo"] = datetime.now().isoformat()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.dialogue.insert(tk.END, f"Error updating {filepath}: {e}\n")

    def train_agent_roo(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Roo...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process with Agent Roo in chunks so large files are fully covered
                            chunk_size = 5000  # chars per chunk
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            last_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                                last_output = roo_output
                                
                                # Optional: Pass to Ragu for vector communication
                                if roo_embeddings is not None:
                                    ragu_output, refined_vectors = self.agents["Ragu"].process_vectors(roo_embeddings)
                                    last_output += f" | {ragu_output}"
                            
                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent Roo trained on {filepath} ({total_chunks} chunks): {last_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent ca Roo... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Roo training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/roo_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Roo training started in background."

    def train_agent_ragu(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Ragu...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process with Agent Ragu in chunks to cover whole file
                            chunk_size = 5000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            last_output = ""
                            
                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                ragu_output = self.agents["Ragu"].process(text_chunk, filepath)
                                last_output = ragu_output
                            
                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent Ragu trained on {filepath} ({total_chunks} chunks): {last_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent Ragu... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Ragu training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/ragu_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Ragu training started in background."

    def train_agent_rahulio(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Rahulio...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process in chunks: Ragu -> Rahulio
                            chunk_size = 4000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            last_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                ragu_output, ragu_embeddings = self.agents["Ragu"].process(text_chunk, filepath)
                                
                                # Then process with Agent Rahulio if embeddings available
                                if ragu_embeddings is not None:
                                    rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                    last_output = f"{ragu_output} | {rahulio_output}"
                                else:
                                    last_output = f"{ragu_output} | Rahulio: No embeddings to process"
                            
                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent Rahulio trained on {filepath} ({total_chunks} chunks): {last_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent Rahulio... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Rahulio training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/rahulio_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Rahulio training started in background."

    def train_agent_raul(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Raul...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            
            # Force full scan for manual training command, ignoring Sgt Rock's batching
            # so user sees all new files immediately.
            files_to_train = []
            for root, dirs, files in os.walk(data_dir):
                # Skip json_data to avoid heavy/irrelevant JSON blobs
                dirs[:] = [d for d in dirs if d.lower() != "json_data"]
                for file in files:
                    if file.endswith(('.json', '.txt')):
                        files_to_train.append(os.path.join(root, file))

            total_files = len(files_to_train)
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for filepath in files_to_train:
                if 'fine_tune' in filepath.replace('\\', '/').lower():
                    continue
                try:
                    file = os.path.basename(filepath)
                    if file.endswith('.json'):
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        # Extract text from JSON (simple: stringify)
                        text_content = json.dumps(data)
                    else:  # .txt
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            text_content = f.read()
                    
                    # Process chain: Roo -> Ragu -> Rahulio -> Raul in chunks to cover whole file
                    chunk_size = 3000
                    total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                    combined_output = ""

                    for i in range(0, len(text_content), chunk_size):
                        text_chunk = text_content[i:i+chunk_size]
                        roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                        
                        if roo_embeddings is not None:
                            ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                            
                            if ragu_embeddings is not None:
                                rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                
                                if feature_vectors is not None:
                                    raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors, filepath)
                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output}"
                                else:
                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | Raul: No feature vectors to process"
                            else:
                                combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings to process | Raul: No feature vectors"
                        else:
                            combined_output = f"{roo_output} | Ragu: No embeddings | Rahulio: No embeddings | Raul: No feature vectors"
                    
                    # Log (use after for thread safety)
                    # Reduced verbosity: only log every 10 files or errors
                    if trained_files % 10 == 0:
                        def log_message():
                            self.dialogue.insert(tk.END, f"Agent Raul training progress: {trained_files}/{total_files} files processed...\n")
                            self.dialogue.see(tk.END) # Auto-scroll
                        self.root.after(0, log_message)
                    
                    trained_files += 1
                    
                    # Update progress and ETA
                    elapsed = time.time() - start_time
                    progress_ratio = trained_files / total_files if total_files > 0 else 0
                    if progress_ratio > 0:
                        eta = (elapsed / progress_ratio) - elapsed
                        eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                    else:
                        eta_str = "Calculating..."
                    self.root.after(0, lambda: self.progress.step(1))
                    self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent Raul... ETA: {eta_str}"))
                    
                    # Update file metadata
                    if file.endswith('.json'):
                        self.update_file_for_training(filepath)
                        
                except Exception as e:
                    self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Raul training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/raul_progress.json\n")
                self.dialogue.see(tk.END) # Auto-scroll
                self.train_state["raul_agent_last_trained"] = datetime.now().isoformat()
                self.save_train_state()
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Raul training started in background."

    def train_agent_boss(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Boss...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process chain: Roo -> Ragu -> Rahulio -> Raul -> Boss in chunks
                            chunk_size = 3000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            combined_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                                
                                if roo_embeddings is not None:
                                    ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                                    
                                    if ragu_embeddings is not None:
                                        rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                        
                                        if feature_vectors is not None:
                                            raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors, filepath)
                                            
                                            if capsule_outputs is not None:
                                                boss_output, intent_outputs = self.agents["Boss"].process(capsule_outputs, filepath)
                                                combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output}"
                                            else:
                                                combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | Raul: No feature vectors | Boss: No capsule outputs"
                                        else:
                                            combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings to process | Raul: No feature vectors | Boss: No capsule outputs"
                                    else:
                                        combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs"
                                else:
                                    combined_output = f"{roo_output} | Ragu: No embeddings | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs"
                            
                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent Boss trained on {filepath}: {combined_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent Boss... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Boss training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/boss_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Boss training started in background."

    def train_agent_rudy(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent Rudy...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process chain: Roo -> Ragu -> Rahulio -> Raul -> Boss -> Rudy in chunks
                            chunk_size = 3000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            combined_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                                
                                if roo_embeddings is not None:
                                    ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                                    
                                    if ragu_embeddings is not None:
                                        rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                        
                                        if feature_vectors is not None:
                                            raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors, filepath)
                                            
                                            if capsule_outputs is not None:
                                                boss_output, intent_outputs = self.agents["Boss"].process(capsule_outputs, filepath)
                                                
                                                if intent_outputs is not None:
                                                    rudy_output, dialog_outputs = self.agents["Rudy"].process(intent_outputs, filepath)
                                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output}"
                                                else:
                                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | Rudy: No intent outputs"
                                            else:
                                                combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs"
                                        else:
                                            combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings to process | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs"
                                    else:
                                        combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs"
                                else:
                                    combined_output = f"{roo_output} | Ragu: No embeddings | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs"
                            
                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent Rudy trained on {filepath}: {combined_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent Rudy... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent Rudy training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/rudy_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent Rudy training started in background."

    def train_agent_majorroo(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent MajorRoo...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                            
                            # Process chain: Roo -> Ragu -> Rahulio -> Raul -> Boss -> Rudy -> MajorRoo in chunks
                            chunk_size = 3000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            combined_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                                
                                if roo_embeddings is not None:
                                    ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                                    
                                    if ragu_embeddings is not None:
                                        rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                        
                                        if feature_vectors is not None:
                                            raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors, filepath)
                                            
                                            if capsule_outputs is not None:
                                                boss_output, intent_outputs = self.agents["Boss"].process(capsule_outputs, filepath)
                                                
                                                if intent_outputs is not None:
                                                    rudy_output, dialog_outputs = self.agents["Rudy"].process(intent_outputs, filepath)
                                                    
                                                    if dialog_outputs is not None:
                                                        majorroo_output, reconstructed = self.agents["MajorRoo"].process(dialog_outputs, roo_embeddings, filepath)
                                                        combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output} | {majorroo_output}"
                                                    else:
                                                        combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output} | MajorRoo: No dialog outputs"
                                                else:
                                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | Rudy: No intent outputs | MajorRoo: No dialog outputs"
                                            else:
                                                combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs"
                                        else:
                                            combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs"
                                    else:
                                        combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs"
                                else:
                                    combined_output = f"{roo_output} | Ragu: No embeddings | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs"
                            

                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent MajorRoo trained on {filepath}: {combined_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent MajorRoo... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent MajorRoo training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/majorroo_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent MajorRoo training started in background."

    def train_agent_docragu(self):
        # Update info box to show training
        self.update_info_box(status="Training Agent DocRagu...")
        
        import threading
        
        def train_thread():
            data_dir = "data"
            # Count total files first
            total_files = sum(1 for root, dirs, files in os.walk(data_dir) for file in files if file.endswith(('.json', '.txt')))
            trained_files = 0
            start_time = time.time()
            
            # Set progress bar max
            self.root.after(0, lambda: self.progress.config(maximum=total_files))
            self.root.after(0, lambda: self.progress.config(value=0))
            
            for root, dirs, files in os.walk(data_dir):
                if 'fine_tune' in root.replace('\\', '/').lower():
                    continue
                for file in files:
                    if file.endswith('.json') or file.endswith('.txt'):
                        filepath = os.path.join(root, file)
                        try:
                            if file.endswith('.json'):
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                # Extract text from JSON (simple: stringify)
                                text_content = json.dumps(data)
                            else:  # .txt
                                with open(filepath, 'r') as f:
                                    text_content = f.read()
                            
                            # Process chain: Roo -> Ragu -> Rahulio -> Raul -> Boss -> Rudy -> MajorRoo -> DocRagu in chunks
                            chunk_size = 3000
                            total_chunks = max(1, math.ceil(len(text_content) / chunk_size))
                            combined_output = ""

                            for i in range(0, len(text_content), chunk_size):
                                text_chunk = text_content[i:i+chunk_size]
                                roo_output, roo_embeddings = self.agents["Roo"].process(text_chunk, filepath)
                                
                                if roo_embeddings is not None:
                                    ragu_output, ragu_embeddings = self.agents["Ragu"].process_vectors(roo_embeddings)
                                    
                                    if ragu_embeddings is not None:
                                        rahulio_output, feature_vectors = self.agents["Rahulio"].process(ragu_embeddings, filepath)
                                        
                                        if feature_vectors is not None:
                                            raul_output, capsule_outputs = self.agents["Raul"].process(feature_vectors, filepath)
                                            
                                            if capsule_outputs is not None:
                                                boss_output, intent_outputs = self.agents["Boss"].process(capsule_outputs, filepath)
                                                
                                                if intent_outputs is not None:
                                                    rudy_output, dialog_outputs = self.agents["Rudy"].process(intent_outputs, filepath)
                                                    
                                                    if dialog_outputs is not None:
                                                        majorroo_output, reconstructed = self.agents["MajorRoo"].process(dialog_outputs, roo_embeddings, filepath)
                                                        
                                                        if reconstructed is not None:
                                                            docragu_output, response = self.agents["DocRagu"].process(dialog_outputs, filepath)
                                                            combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output} | {majorroo_output} | {docragu_output}"
                                                        else:
                                                            combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output} | {majorroo_output} | DocRagu: No reconstructed"
                                                    else:
                                                        combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | {rudy_output} | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                                                else:
                                                    combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | {boss_output} | Rudy: No intent outputs | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                                            else:
                                                combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | {raul_output} | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                                        else:
                                            combined_output = f"{roo_output} | {ragu_output} | {rahulio_output} | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                                    else:
                                        combined_output = f"{roo_output} | {ragu_output} | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                                else:
                                    combined_output = f"{roo_output} | Ragu: No embeddings | Rahulio: No embeddings | Raul: No feature vectors | Boss: No capsule outputs | Rudy: No intent outputs | MajorRoo: No dialog outputs | DocRagu: No reconstructed"
                            

                            # Log (use after for thread safety)
                            def log_message():
                                self.dialogue.insert(tk.END, f"Agent DocRagu trained on {filepath}: {combined_output[:100]}...\n")
                            self.root.after(0, log_message)
                            
                            trained_files += 1
                            
                            # Update progress and ETA
                            elapsed = time.time() - start_time
                            progress_ratio = trained_files / total_files if total_files > 0 else 0
                            if progress_ratio > 0:
                                eta = (elapsed / progress_ratio) - elapsed
                                eta_str = f"{int(eta // 60)} min {int(eta % 60)} sec"
                            else:
                                eta_str = "Calculating..."
                            self.root.after(0, lambda: self.progress.step(1))
                            self.root.after(0, lambda eta_str=eta_str: self.status_label.config(text=f"Training Agent DocRagu... ETA: {eta_str}"))
                            
                            # Update file metadata
                            if file.endswith('.json'):
                                self.update_file_for_training(filepath)
                                
                        except Exception as e:
                            self.root.after(0, lambda e=e, filepath=filepath: self.dialogue.insert(tk.END, f"Error training on {filepath}: {e}\n"))
            
            def completion_message():
                self.dialogue.insert(tk.END, f"Agent DocRagu training completed on {trained_files} files.\n")
                self.dialogue.insert(tk.END, "Progress saved to data/docragu_progress.json\n")
                self.update_info_box(status="Idle")
                self.root.after(0, lambda: self.status_label.config(text="Status: Idle"))
                self.root.after(0, lambda: self.progress.config(value=0))
            self.root.after(0, completion_message)
        
        threading.Thread(target=train_thread, daemon=True).start()
        return "Agent DocRagu training started in background."

    def librarian_review(self):
        # Trigger librarian review in background
        import threading
        def review_thread():
            try:
                librarian_output, _ = self.agents["Librarian"].process("review_all")
                self.root.after(0, lambda: self.dialogue.insert(tk.END, f"Librarian: {librarian_output}\n"))
            except Exception as e:
                self.root.after(0, lambda: self.dialogue.insert(tk.END, f"Librarian Error: {e}\n"))
        threading.Thread(target=review_thread, daemon=True).start()
        return "Librarian review initiated. Checking for duplicates, corruption, and archiving processed files."

    def nlp_process(self, text):
        # Process text with NLP agent
        try:
            nlp_output, embeddings = self.agents["NLPRaul"].process(text)

            # Feed embeddings to Linguist to mint/recall a symbol
            symbol = None
            if embeddings is not None:
                try:
                    symbol = self.agents["Linguist"].encode_vector(embeddings)
                except Exception as e:
                    symbol = f"(linguist error: {e})"

            symbol_str = f" | Linguist: {symbol}" if symbol else ""
            return f"{nlp_output}{symbol_str} Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}"
        except Exception as e:
            return f"NLP Error: {e}"

    def calculate(self, expression):
        # Safe calculator for math expressions
        try:
            # Allow only safe operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"__builtins__": {}})
            result = eval(expression, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    def search_web(self, query):
        # Simple web search by opening browser
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Opened web search for: {query}"
        except Exception as e:
            return f"Web search error: {e}"

    def get_time(self):
        # Get current time
        now = datetime.now()
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    def get_date(self):
        # Get current date
        today = datetime.now()
        return f"Today's date: {today.strftime('%Y-%m-%d')}"

    def create_note(self, note):
        # Create a simple note
        note_path = os.path.join('data', 'notes.txt')
        try:
            with open(note_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()}: {note}\n")
            return "Note saved."
        except Exception as e:
            return f"Note error: {e}"

    def list_files(self, directory="data"):
        # List files in directory
        try:
            files = os.listdir(directory)
            return f"Files in {directory}: {', '.join(files)}"
        except Exception as e:
            return f"List files error: {e}"

    def explain_code(self, code_snippet):
        # Basic code explanation using agents
        try:
            # Use the agent chain to process the code as text
            response = self.generate_tay_response(f"Explain this code: {code_snippet}")
            return f"Code explanation: {response}"
        except Exception as e:
            return f"Code explanation error: {e}"

    def get_help(self):
        # List available commands
        help_text = """
Available commands:
- help: Show this help
- train text: Train on data files
- train agent [name]: Train specific agent (roo, ragu, rahulio, raul, boss, rudy, majorroo, docragu)
- librarian review: Review and manage files
- librarian learn patterns: Learn unwanted patterns from data
- librarian add pattern <regex>: Add a cleaning pattern
- librarian remove pattern <regex>: Remove a cleaning pattern
- nlp process <text>: Process text with NLP
- calculate <expression>: Calculate math expression
- search <query>: Search web for query
- time: Get current time
- note <text>: Save a note
- list files [directory]: List files in directory
- explain code <code>: Explain code snippet
- Any other text: Chat with the AI assistant
- o1 status: Show the last scheduled O1-nano retraining time
- Sgt Rock runs agents/Raulnano/train.py on a schedule to keep the O1-nano reasoning agent refreshed for the hive mind
- recall memory [n]: Show last n memory entries (default 5)
        """
        return help_text.strip()

# Add other agents as needed

if __name__ == "__main__":
    root = tk.Tk()
    app = RaulChatbot(root)
    root.mainloop()