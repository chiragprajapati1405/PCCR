# Full OfficeBench evaluation (LegoMem-style) — setup & plan

Evaluates PCCR (EM trajectory reuse + ρ-gate + 2-D parallelism) on the **full**
OfficeBench benchmark in its **native Docker** env, all **9 apps**, native
state-based eval. Backbone: Cerebras gpt-oss-120b (9 keys).

## Status
- ✅ Docker (colima) + `officebench` image built; 1-task smoke passes end-to-end
  (`python -m officebench_eval.smoke 1-1 0` → success=True).
- ✅ Split generated: `officebench_eval/split.json` (148 train / 152 test,
  stratified by level: train L1 46/L2 47/L3 55, test L1 47/L2 48/L3 57).
- ⏳ Next: build EM memory bank from train (T0) → main comparison harness (T1–T8).

## One-time setup
```bash
# 1. Docker via colima (headless)
brew install colima docker
colima start --cpu 4 --memory 6 --disk 40

# 2. OfficeBench patches (the repo is gitignored; reapply after a fresh clone)
#   a) container pip is PEP-668 locked -> add --break-system-packages:
#      edit OfficeBench/docker/setup_py.sh:
#        python3 -m pip install --upgrade pip --break-system-packages
#        python3 -m pip install -r requirements.txt --break-system-packages
#   b) Dockerfile COPYs openai_key.txt -> create a placeholder (we use Cerebras):
echo placeholder > OfficeBench/openai_key.txt

# 3. Build the image (Ubuntu+LibreOffice+Tesseract, ~10-20 min, ~5 GB)
docker build -t officebench -f OfficeBench/docker/Dockerfile OfficeBench/

# 4. Host-side python deps (into .venv)
.venv/bin/pip install gymnasium docker fire pytz rich pandas scikit-learn \
  icalendar openpyxl python-docx pytesseract PyPDF2 rpyc google-generativeai \
  tabulate mysql-connector-python PyMuPDF pdf2docx
```

## Run
```bash
source cerebras.env                       # 9 Cerebras keys
python -m officebench_eval.split          # (re)generate split.json
python -m officebench_eval.smoke 1-1 0    # substrate check (one task)
# T0..T8 harness: forthcoming (build memory bank -> compare by level)
```

## Notes / gotchas (all handled in code)
- **DOCKER_HOST**: the python docker SDK needs colima's socket; `smoke.py`
  auto-sets `unix://~/.colima/default/docker.sock` if unset.
- **Backbone branch**: OfficeBench's `LLMPolicy` routes any model name containing
  'gpt' to the OpenAI client. We pass a placeholder branch-name (`local-oss`)
  then swap in `CerebrasLLM(model_name="gpt-oss-120b")`.
- **py3.9**: OfficeBench's `evaluation.py` `main()` uses a py3.12 f-string; we call
  the `utils.evaluate` functions directly instead.

## Test plan (what we measure)
| ID | Test |
|---|---|
| T0 | Build EM bank from 148 train trajectories (successful only) |
| T1 | Main comparison by L1/L2/L3: no-memory vs retrieve-all vs PCCR ρ-gated |
| T2 | Orchestrator vs agent memory ablation |
| T3 | Store-usefulness: retrieval↔success correlation by level/category |
| T4 | ρ-gate θ frontier (success vs procedures retrieved / tokens) |
| T5 | Intra-task parallelism (multi-app L2/L3) |
| T6 | Task-level parallelism (concurrent tasks; 9 keys) |
| T7 | Parallel ≡ sequential equivalence |
| T8 | LegoMem head-to-head (relative gains vs +12–13pp / L3 +14–19pp) |
