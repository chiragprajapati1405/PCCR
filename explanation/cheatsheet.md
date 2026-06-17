# PCCR — ONE-PAGE CHEAT SHEET  (glance during meeting)

## THE FORMULA
        ρ(pattern, store) = U(pattern, store) / C(store)
        consult store  ⟺  ρ ≥ θ
  U = usefulness (0–1)   C = relative access cost   θ = the dial
  Always-read (NOT gated): PM, WM, STM.  Gated: EM, SM, ENT.
  STM hit → short-circuit (skip everything).  Miss → run the ρ-gate.

## COST  C(store)   (relative compute, not seconds)
  ENT = 0.15   → dict lookup by name (instant)
  EM  = 0.55   → FAISS vector search + deserialize traces  (~3.7× costlier)
  SM            (rides inside EM's FAISS index)
  ⚠ EM costs MORE than ENT. Vector search ≫ keyed lookup.

## UTILITY  U(pattern, store)   (hand-set prior, refined by closed loop)
  pattern                       EM     ENT
  single_cal_create             0.60   0.10
  multi_cal_find_and_create     0.78   0.72
  remind_notify                 0.66   0.55
  email_query  (lookup)         0.48   0.66
  email_send                    0.62   0.10
  cal_query                     0.58   0.10
  word_create                   0.62   0.10
  word_query   (doc read)       0.45   0.10
  unknown                       0.70   0.60

## THE KEY REWRITE:  consult ⟺ U ≥ C·θ   (θ sets a utility BAR per store)
  store           bar@θ=1.0   θ=1.2    θ=1.4
  EM (C=0.55)     U≥0.55      U≥0.66   U≥0.77      ← bar rises FAST
  ENT (C=0.15)    U≥0.15      U≥0.18   U≥0.21      ← barely moves
  → Raising θ prunes EM first (high cost), ENT stays. THIS explains the frontier.
    θ=1.0: most patterns consult EM | θ=1.2: only multi/remind/unknown | θ=1.4: ~only multi

## CLOSED LOOP  (how U changes)   lr = 0.05, clamp[0,1]
  consulted & SUCCESS → U += 0.05 (↑)
  consulted & FAILURE → U -= 0.05 (↓)
  Real drift (2-agent online run):
    multi.episodic       0.78 → 0.83  ↑ (helped)
    multi.entity         0.72 → 0.77  ↑
    email_send.episodic  0.62 → 0.57  ↓ (didn't help → self-prunes)

## RESULTS TO QUOTE  (3-agent, 32 held-out, frozen)
  Method        Acc     Consults
  Retrieve-All  10/32   46
  Boolean       10/32   39
  PCCR (θ=1.0)  11/32   27     ← same acc, −41% consults
  PCCR (θ=1.2)  11/32    6     ← same acc, −87% vs retrieve-all, −89% tokens
  PCCR (θ=1.4)   8/32    6     ← overshoot, accuracy drops

  Per-pattern proof: email_query EM 0/7 | email_send ENT 0/10 | word_query EM&ENT 0/3

## ONE-LINERS IF CORNERED
  • Cost basis: relative compute (dict lookup vs vector search); replaceable by logged latency/tokens.
  • Utility origin: reasoned prior per (pattern,store); refined by closed loop; ground via counterfactual (with/without store).
  • Patterns: hand-designed taxonomy from OfficeBench task types + keyword classifier.
  • Who classifies: PatternSTM.classify (keywords), at ingestion — NOT the LLM.
  • Who computes the free hour: the orchestrator LLM, reasoning over both calendars in WM.
  • shared_context: a FIELD inside WM (cross-agent handoff), not a separate memory.
  • learned_rules: implemented in reference pkg; EMPTY in OfficeBench runs (consolidation writes EM/SM only).
  • Why report consults/tokens not wall-clock: FAISS is ms, LLM dominates; consults/tokens = deployment-relevant cost.
  • Scale caveat: 11/32 = proof-of-concept; accuracy wiggle is single-task noise; next step = scale + variance.
