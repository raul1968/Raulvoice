Raul Voice Assistant
====================

Overview
--------
Raul is a capsule-first multi-agent assistant with an optional O1 nano model. O1 is disabled by default at runtime in earlier builds; it can be toggled via chat commands.

Quickstart
----------
1. Install deps (global or venv):
   ```bash
   pip install -r agents/Raulnano/requirements.txt
   ```
2. Run Raul GUI:
   ```bash
   python raul.py
   ```
3. O1 toggle via chat:
   - `o1 on` to enable O1 arbitration
   - `o1 off` to disable O1

O1 Training (optional)
----------------------
O1 ships untrained/mismatched by default. To train on your data:
1. Place UTF-8 `.txt` files in `data/text_data/` (3–5k chunks recommended).
2. Rebuild tokenizer (creates `o1_tokenizer.model/.vocab`):
   ```bash
   python agents/Raulnano/test_tokenizer.py
   ```
3. Train O1:
   ```bash
   python agents/Raulnano/train.py --epochs 50 --batch_size 16 --output agents/Raulnano/o1_model.pth
   ```
4. If you ship without the checkpoint, keep `o1_model.pth` out of git (see .gitignore).

Git Hygiene
-----------
- Ignore large/binary artifacts: `agents/Raulnano/o1_model.pth`, `agents/Raulnano/o1_tokenizer.*`, caches.
- `config.json` contains only model params; no secrets expected.

Runtime Notes
-------------
- Time/date intents: ask "what time is it" or "what is the date".
- O1 low-quality outputs are suppressed; capsule reply is used by default.

Known State
-----------
- O1 checkpoint in repo root (if present) may be stale/untrained; retrain per steps above.
- Chunk size for training is ~5k; longer docs should be pre-split.