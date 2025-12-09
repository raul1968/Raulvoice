# Slow Growth: Compartmentalized Capsule Networks for Efficient, Subconscious-Like Chatbot Learning

**Abstract**
Capsule networks (CapsNets) preserve spatial and semantic hierarchies that are often lost in pooling layers of convolutional or transformer models, but their dynamic routing procedure imposes significant computational cost that scales poorly with vocabulary size and context length. This paper introduces a "Slow Growth" architecture that mitigates these costs by decomposing the system into specialized agents and scheduling capsule updates during idle resource windows. A Librarian agent standardizes and archives large inputs as bounded chunks, and a scheduler opportunistically dispatches these chunks without interrupting foreground responsiveness. We further add agreement-gated responses between the capsule core and an external O1 nano reasoner, persistent memory of disagreements, and symbolic grounding so the system can arbitrate conflicting outputs, retain corrective feedback, and refine over time without monopolizing compute.

## 1. Introduction
The dominant paradigm in Large Language Model (LLM) training is high-throughput, synchronous optimization, utilizing 100% of available GPU/TPU resources to minimize loss rapidly. While effective for transformers, this approach is prohibitive for Capsule Networks. CapsNets, introduced by Sabour et al., use vector-based "capsules" rather than scalar neurons to encode entity properties (pose, texture, semantic role). The "routing-by-agreement" mechanism, which iteratively refines connections between lower and higher-level capsules, is computationally expensive. In a chatbot context, where real-time responsiveness is critical, the latency introduced by dynamic routing renders monolithic CapsNets impractical.

We argue that the bottleneck is not the architecture itself, but the monolithic execution model. By decomposing the network into autonomous agents—each representing a specialized capsule or processing stage—and adopting an asynchronous, "slow growth" learning strategy, we can maintain the rich semantic representation of CapsNets while respecting hardware constraints.

## 2. The Computational Bottleneck of Monolithic CapsNets
In a standard CapsNet, every forward pass requires an iterative routing loop (typically 3 iterations) to determine part-whole relationships. For a vocabulary $V$ and embedding dimension $D$, the routing complexity can approach $O(k \cdot V \cdot D^2)$, where $k$ is the number of routing iterations.
*   **Processor Saturation:** Training a monolithic CapsNet on a large corpus requires massive parallelization, saturating GPU cores and blocking user interaction.
*   **Memory Bandwidth:** Storing routing coefficients for every capsule pair consumes significant VRAM, limiting batch sizes and context windows.
*   **Latency:** Inference becomes sluggish, breaking the illusion of a responsive conversational partner.

## 3. The "Slow Growth" Philosophy
"Slow Growth" reframes learning not as a race to convergence, but as a continuous, background metabolic process. Just as the human subconscious processes memories and complex associations during sleep or low-activity periods, our architecture decouples data ingestion from deep semantic integration.

### 3.1. Agent-Based Compartmentalization
Instead of a single monolithic network, we define a set of specialized agents—effectively macro-capsules—each with a bounded role:
*   **Agent Roo (Scanner):** Lightweight front-end that ingests raw text and performs initial tokenization/embedding.
*   **Agent Ragu (Refiner):** Receives embeddings and refines them, analogous to a primary capsule layer.
*   **Agent Rahulio (Feature Extractor):** Extracts higher-order features, serving as a convolutional precursor to the capsule core.
*   **Agent Raul (Capsule Core):** Central routing unit that focuses on semantic routing after upstream feature preparation.
*   **Librarian (Document Ingestion and Chunking):** Monitors the corpus, identifies large inputs, and converts them into standardized chunks (e.g., 5,000 characters) while archiving raw originals.
*   **Sgt. Rock (Scheduler):** Resource-aware dispatcher that monitors CPU/GPU usage and user activity to schedule work without disrupting foreground responsiveness.

### 3.2. Asynchronous Digestion
The workflow shifts from synchronous batch processing to an asynchronous pipeline:
1.  **Ingestion:** The user provides a large document. The Librarian immediately archives it and slices it into chunks in a staging area (`data/chunks`).
2.  **Idle Detection:** Sgt. Rock detects a "low-stress" window (e.g., user is reading, or away).
3.  **Distributed Routing:** Sgt. Rock assigns specific chunks to specific agents. Agent Raul might process Chunk A while Agent Boss (Intent Router) processes the output of Chunk B from a previous cycle.
4.  **Integration:** The agents update their internal weights (capsule parameters) incrementally. There is no "epoch" in the traditional sense; the network grows its understanding continuously over time.

### 3.3. Librarian Chunking Strategy (Chunk Size and Stability)
- **Mode:** Hybrid semantic+fixed-length. The Librarian targets ~5,000-character chunks as the default unit, but will split on sentence/paragraph boundaries when possible to keep semantic cohesion.
- **Why 5,000 chars:** This size is small enough to keep routing coefficients and capsule activations in GPU memory without thrashing, yet large enough to preserve local discourse structure so routing-by-agreement can form stable part–whole bindings.
- **Routing quality vs chunk size:**
	- Smaller chunks (<2k chars) reduce VRAM pressure but can fragment context, yielding weaker agreement signals and more uncertainty in DocRagu outputs.
	- Larger chunks (>8–10k chars) improve local coherence but inflate routing tensors, increasing latency and risking OOM on consumer GPUs.
- **Memory overhead:** Each chunk produces capsule activations plus routing coefficients; memory scales roughly linearly with chunk length. The Librarian keeps only chunk metadata and archives the raw document elsewhere to avoid duplicating payload in RAM.
- **Failure handling:** If a chunk triggers routing instability (gradient spike or OOM), the Librarian retries with a finer split (halving size) and records the “hard” region for review.

### 3.4. Sgt. Rock Circadian Rhythm (Idle Detection)
- **Signals watched:** CPU and GPU utilization, GPU VRAM headroom, and recent user input activity.
- **Heuristic:** Idle if CPU < ~30%, GPU < ~20% utilization with >1–2 GB free VRAM, and no user input for N seconds (configurable, e.g., 60–180s). Short bursts of activity pause dispatch; sustained idleness resumes it.
- **Scheduling cadence:** Checks run on a short interval (e.g., every 10–30s). When idle is confirmed, a limited batch of chunks is dispatched, then the system re-checks before sending more—preventing long hogging runs.
- **Backoff:** If resource spikes or user input arrives, Sgt. Rock backs off and retries after a cool-down interval, similar to a light sleep/wake cycle.
- **Why this matters:** Keeps digestion opportunistic and non-intrusive, aligning heavy routing/training with real idle windows.

#### 3.4.1. Resource-Safe Scheduling (Foreground Protection)
- **Two-band thresholds:** A high-priority band for foreground (tight CPU/GPU caps, higher VRAM reserve) and a low-priority band for background. Training only runs in the low band; if metrics rise into the foreground band, background jobs pause immediately.
- **VRAM guardrail:** Reserve a fixed floor (e.g., 1–2 GB) for interactive inference; background batches are sized so estimated activations + optimizer state stay below a VRAM ceiling. If the guardrail is breached, batch size halves or training skips a cycle.
- **Step budgeting:** Background runs in short bursts (e.g., K steps or M seconds), then yields and rechecks resources before continuing.
- **Preemption:** Any new user input triggers an immediate stop of background work; partial progress is safely discarded or checkpointed only if already in a sync-safe point.
- **Staggered dispatch:** Only one heavy job (CapsNet digestion or O1 fine-tune) runs at a time; others queue to prevent overlapping memory spikes.
- **Lightweight mode for constrained devices:** On low-VRAM devices, use smaller chunk sizes, gradient accumulation (micro-batches), and freeze non-critical weights to reduce activation footprint during background steps.

With the core agent design established, we next analyze the communication costs between agents, since these determine how scalable the architecture is.

### 3.5. Agent Communication Overhead
- **Message contents:** Primarily tensor batches (embeddings, capsule outputs) passed between adjacent agents. Typical shapes are `[batch, seq_len, dim]` for embeddings and `[batch, num_caps, cap_dim]` for capsule outputs.
- **Scaling with chunk size:** Overhead grows roughly linearly with chunk length because more tokens produce larger tensors. Larger chunks improve context but increase transfer volume and latency.
- **Scaling with agent count:** With $N$ sequential agents, transfers scale as $O(N)$ per chunk. Adding parallel branches (e.g., specialty capsules) increases fan-out; batching transfers and reusing shared buffers keeps per-branch overhead near-linear.
- **Latency budget:** On a consumer GPU with PCIe, intra-process tensor handoffs are mostly in-memory; the dominant cost is compute, but for large chunks (>8–10k chars) transfer/serialization can add tens of milliseconds per hop. Keeping chunks near ~5k chars keeps hop latency low and routing stable.
- **Mitigations:**
	- Keep tensors on-GPU across adjacent agents to avoid PCIe copies.
	- Use fixed-capacity shared buffers for embeddings/capsules to reduce allocation churn.
	- Cap chunk size and fan-out; prefer staggered dispatch over wide simultaneous branching when VRAM is tight.

### 3.6. Multi-Modal Extensibility (Text + Image/Audio)
- **Front-end adapters:** Add modality-specific scanners (e.g., a vision encoder for images, a small audio encoder for spectrograms) that output embeddings aligned to Roo’s text embedding space via projection heads.
- **Capsule fusion:** Use separate primary capsule sets per modality, then fuse via agreement routing in MajorRoo/DocRagu; shared higher-level capsules capture cross-modal concepts (e.g., caption+image alignment).
- **Chunking for media:** Large images/videos are pre-chunked (tiles/clips) with per-chunk features; Librarian tracks media metadata and pairs text spans with relevant tiles to limit routing load.
- **Routing cost control:** Keep per-modality batch small and stagger media capsules to avoid VRAM spikes; reuse the same idle-time scheduler so heavy vision/audio passes run when resources are free.
- **Evaluation:** Measure cross-modal retrieval/alignment accuracy and latency impact; monitor disagreement rates between text-only vs fused responses to ensure fusion helps rather than hurts.
- **Example:** User sends a photo and a caption. Vision encoder -> image capsules; Roo -> text capsules; MajorRoo fuses them, DocRagu outputs a grounded reply (e.g., verifies caption consistency). If capsule and O1 disagree, both answers are shown for user choice.

## 4. Advantages of Compartmentalization
This architecture preserves the core benefits of CapsNets while solving the efficiency problem:
*   **Resource Politeness:** By processing small chunks during idle times, the system never locks the UI or overheats the GPU. The user perceives a lightweight assistant, while a heavy neural network trains in the background.
*   **Robust Semantic Hierarchy:** Because the "Slow Growth" approach allows for deeper, more expensive routing algorithms (since time is less of a constraint in background tasks), the network develops highly stable part-whole semantic relationships.
*   **Scalability:** New agents (specialized capsules) can be added without retraining the entire monolith. A "Math Capsule" or "Code Capsule" can be plugged into the scheduler's rotation seamlessly.
*   **Resilience:** If a chunk causes a gradient spike or error, it is isolated to one agent's current task, not crashing the whole training run. The Librarian can flag the "indigestible" chunk for review.

## 5. 2025 Upgrades: Agreement-Gated Responses, Memory, and Symbolic Grounding
We implemented several changes to make the architecture more accountable, data-efficient, and user-steerable.

### 5.1. Capsule ↔ O1 Arbitration (Why: safety and clarity)
After DocRagu produces a capsule-based reply, an O1 nano reasoner runs a parallel completion. If they differ, we present both and ask the user to choose. Rationale: expose ambiguity, avoid silent failure modes, and let the user pick the trusted path.

### 5.1.1. Failure Modes and Safeguards (Both Confident, Both Wrong)
- **Observed risk:** Capsule and O1 can agree confidently on a wrong answer (shared blind spots or bad premise). Arbitration alone won’t catch this when outputs align.
- **Safeguards:**
	- Require a disagreement budget: periodically surface a “second look” even on agreements for sensitive intents (e.g., safety, arithmetic, code-gen), optionally sampling a lightweight verifier.
	- Track confidence entropy: if both are high-confidence but the input is out-of-distribution (detected via embedding distance to known domains), down-rank confidence and flag for user confirmation.
	- Memory-based recall: if prior turns on similar topics produced disagreements or corrections, force a dual-presentation even when current answers match.
	- Lightweight checks: for arithmetic/code answers, run quick evaluators or test snippets; for factual claims, route to retrieval or mark as unverified.
- **User-facing behavior:** In high-risk domains, append a short caveat (“agreement detected; verification recommended”) or request a quick confirm.
- **Data feedback:** Log these agreement-but-risky cases to `data/memory.json` for targeted fine-tuning and to improve the verifier heuristics.
- **User-in-the-loop correction:** When a user flags an error, capture their correction as a supervised label, replay it through both capsule and O1 traces, and store it in memory for targeted retraining and confidence calibration.

### 5.2. User-Directed Resolution (Why: agency and alignment)
Commands `choose capsule`, `choose o1`, and `ask more info` let the user explicitly steer resolution. Rationale: keep the human in the loop and gather supervision signals for future tuning.

### 5.3. Persistent Memory of Disagreements (Why: learn from conflict)
Each turn (user input, capsule reply, O1 note, arbitration state) is stored in `data/memory.json`. Rationale: enable recall, spot recurring disagreement patterns, and curate training data to improve the weaker side.

### 5.4. Linguist Symbol Grounding (Why: compositional reuse)
NLP embeddings and capsule vectors are mapped to a symbolic vocabulary via Agent Linguist. Rationale: accumulate reusable concepts that can be referenced across sessions and reduce forgetting.

#### 5.4.1. Method (Symbolic Vocabulary Made Concrete)
- **Mapping approach:** Agent Linguist encodes vectors to discrete symbols using a learned codebook (vector quantization) backed by a lightweight conceptual embedding space. New concepts get `SYM_xxxx` IDs; recurring concepts snap to nearest existing symbols within a distance threshold.
- **Concrete example:** Repeated mentions of detective noir scenes across documents may map to a persistent symbol `SYM_0197`, which then anchors future capsule activations for film-noir style descriptions.
- **Optional alignment:** When available, symbols can be aligned to external knowledge-graph anchors (e.g., Wikidata IDs) via nearest-neighbor search in a shared embedding space; otherwise they remain local symbols.
- **Persistence:** Symbols and counts are stored in `data/linguist_vocab.json`, enabling reuse across sessions and recall during arbitration and summarization.

#### 5.4.2. Evaluation
- **Compression fidelity:** Measure reconstruction error when decoding symbols back to vectors; lower error implies symbols preserve semantic detail.
- **Stability:** Track symbol churn (how often new symbols are created vs re-used) on a rolling window; excessive churn suggests thresholds are too tight or domains are shifting.
- **Downstream utility:** Evaluate whether symbol hints improve agreement rates or reduce latency in DocRagu by seeding known concepts; compare with/without symbol conditioning on dialogue benchmarks.
- **Human check:** Periodically sample high-frequency symbols, show their nearest raw texts, and ask for quick human labeling to catch drift.

### 5.5. Background O1 Training and Scheduling (Why: keep costs bounded)
Sgt. Rock keeps O1 nano training scheduled during idle periods while continuing chunk digestion. Rationale: continuously improve arithmetic/reasoning skill without impacting interactivity.

### 5.6. Future Work: Confidence and Auto-Routing
- Add lightweight confidence estimation for capsule and O1 outputs to auto-select when the gap is large, while still surfacing disagreements for transparency.
- Incorporate user choices from arbitration as supervised signals to fine-tune the weaker side (capsule or O1) and to calibrate confidence scores.
- Summarize `data/memory.json` periodically to reduce bloat and preserve salient disagreement patterns.

### 5.7. Evaluation Plan
- Track disagreement rate (capsule vs O1) and resolution latency (time/user turns to resolve).
- Measure user choice distribution (capsule vs O1) to identify dominant strengths and weaknesses.
- Monitor retention: how often past disagreements reoccur after targeted updates.
- Task benchmarks: arithmetic/code questions (O1-heavy) vs narrative/contextual questions (capsule-heavy) to ensure complementary coverage.

### 5.8. Empirical Comparison: Slow Growth vs Batch Training
- **Convergence time:** Slow growth converges in wall-clock terms only during idle slices; effective compute time per token is similar, but calendar time is longer. Expect ~1.5–3× longer calendar time to reach a target validation loss compared to a fully saturated GPU batch run, depending on idle availability.
- **Final performance:** Given equal total update steps, we expect comparable or slightly better dialogue robustness for slow growth because updates happen with fresher, mixed-context data and less overheating/instability. Batch training reaches the target faster in clock time but risks overfitting recent domains without continual data refresh.
- **Efficiency trade-off:** Slow growth optimizes for user experience (no UI stalls) and thermal/cost budgets. Batch training optimizes for shortest time-to-accuracy but monopolizes hardware. If idle time averages 40–60% of the day, slow growth can match batch performance within roughly 1–2 days of background digestion instead of hours of dedicated training.
- **Next steps:** Instrument both regimes on the same dialogue benchmarks (e.g., blend of arithmetic, code, and narrative tasks) tracking (a) validation loss vs wall-clock, (b) user-perceived latency during training, and (c) disagreement rate capsule↔O1 over time. Report curves to quantify the efficiency/speed trade-off.

### 5.9. Measuring “Subconscious-Like” Learning
- **Retention over time:** Re-test prior disagreement topics after updates; measure recurrence rate and decay curve of repeated mistakes.
-   Example: recurrence rate $r_t = \frac{\text{repeated mistakes at }t}{\text{topics re-tested at }t}$; aim for $r_t \downarrow$ after each targeted update.
- **Incremental user-style adaptation:** Track alignment to a user’s phrasing/format preferences (e.g., brevity, tone, structure) via style similarity scores over rolling windows.
-   Example: cosine style similarity $s = \frac{u \cdot \hat{u}}{\lVert u \rVert \lVert \hat{u} \rVert}$ between current response embedding $u$ and target user-style exemplar $\hat{u}$; target $s \uparrow$ over time.
- **Repeated-error reduction:** Count how often the same factual/code/arithmetic error reappears post-correction; target monotonic decline.
- **Latency under load:** Verify that background digestion does not increase user-facing response latency beyond a fixed budget, indicating stable “sleep-like” processing.
- **Symbol reuse:** Measure reuse rate of Linguist symbols versus churn; higher reuse with stable reconstruction error suggests compositional consolidation.
- **Confidence calibration drift:** Track expected calibration error (ECE) over time to ensure continual learning doesn’t destabilize confidence estimates.
-   Example: $\text{ECE} = \sum_{b=1}^B \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|$ over $B$ bins with $n_b$ samples each; target ECE stability or decline.

### 5.10. Ethical Considerations of Persistent Disagreement Memory
- **Data minimization:** Store only the question, candidate answers, and user choice/correction needed to improve; avoid unrelated personal details.
- **Retention limits:** Apply TTLs or summarization to old disagreement logs; purge raw details once distilled into lessons.
- **User consent and control:** Show what is stored, allow delete/opt-out per user/session, and offer a “forget this thread” control.
- **Bias monitoring:** Check whether disagreements over-index on particular users/groups/topics; rebalance or down-weight to avoid skew in tuning.
- **Security:** Encrypt at rest, restrict access, and avoid sending disagreement traces to external services without explicit consent.
- **Scope of use:** Use disagreement data only for safety/quality improvement, not profiling or targeting.

## 6. Conclusion
The "Slow Growth" architecture demonstrates that Capsule Networks remain viable for chatbots when paired with agent-based decomposition and asynchronous scheduling. By aligning computation to idle windows and augmenting the system with agreement-gated responses, persistent memory, and symbolic grounding, we can deploy hierarchically aware models on standard hardware without sacrificing interactivity. This approach advances toward "Subconscious AI," where the assistant learns continuously in the background, surfaces disagreements transparently, and improves with user-guided resolution.
