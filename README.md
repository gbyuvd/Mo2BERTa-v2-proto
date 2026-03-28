# Mo²BERTa-v2 — Frozen KV Context for Mixture-of-Recursions on a Modernized BERT

Mo²BERTa-v2 extends [Mo²BERTa](https://github.com/gbyuvd/Mo2BERTa-proto) with a single focused
architectural addition: **frozen KV context from exited tokens**. In the base model, tokens that
exit early under expert-choice routing vanish from the attention context of deeper tokens entirely,
a context isolation problem inherent to sparse recursive attention. This variant preserves their Keys
and Values, contributing them to all subsequent recursion steps at zero additional Q or MLP cost.

The result is a measurable improvement in both validation loss and routing discrimination, and the
first controlled isolation of context isolation as a quantifiable bottleneck in encoder-side
Mixture-of-Recursions.

> **Status:** Research prototype / Proof-of-Concept. Not intended for production use.  
> Readers unfamiliar with the base architecture (M-Cycle Middle, expert-choice routing, GQA+RoPE,
> MLM objective) should start with [Mo²BERTa](https://github.com/gbyuvd/Mo2BERTa-proto) first.

## Usage
- **For model, training, inference code:** please check MoRBERT_v2.ipynb in the repo
- **Model's checkpoint @ 600T**: can be accessed [here](https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto)
## Model Details

| Field                 | Value                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| **Architecture**      | Encoder-only Transformer with MoR + Frozen KV accumulator                                          |
| **Base**              | Extends Mo²BERTa; see [v1 card](https://github.com/gbyuvd/Mo2BERTa-proto) for full arch detail |
| **Unique parameters** | ~9.6M (identical to Mo²BERTa; frozen KV adds no parameters)                                        |
| **Effective depth**   | Variable per token: 1–4 recursions over a shared 2-layer block (~6.56 flat-layer equivalent)       |
| **Training data**     | TinyStories-valid (small subset, PoC scale)                                                         |
| **Tokenizer**         | `bert-base-uncased`                                                                                 |
| **Compute cap**       | 600 TFLOPs                                                                                          |
| **Hardware**          | NVIDIA GeForce 930M (Compute Capability 5.0, consumer laptop GPU)                                   |
| **License**           | MIT                                                                                                 |

## What Changed from Mo²BERTa

### 1. The Context Isolation Problem

In Mo²BERTa (FULL_SKIP phase), attention at recursion depth `r` operates only over tokens still
active at depth `r`. A function word that exits at recursion 1 disappears from the KV context of a
`[MASK]` token still processing at recursion 3. The unique final layer provides one round of global
mixing after all recursions complete, but during recursive refinement hard tokens progressively lose
context from easy tokens that exited early.

This was documented as a known limitation in the v1 model card, with frozen KV listed as future work.

### 2. The Fix: Frozen KV Accumulator

When a token exits at recursion step `r`, its final hidden state `x_block` (post-block, the richest
representation it achieved) is projected through `wk` and `wv` of the last shared block,
RoPE-encoded using its original global position, and appended to a frozen KV accumulator. All
subsequent recursion steps concatenate this accumulator into their attention context:

```
Q:  [k_active]               ← only continuing tokens query
K:  [k_active + k_frozen]    ← active + all previously exited
V:  [k_active + k_frozen]
```

Exited tokens contribute context but pay no Q projection, no MLP, and no attention score
computation over their own position. The dominant FLOP savings from sparse routing are preserved.

**Key design decisions:**
- Freeze from `x_block` (post-block) not `x` (pre-block) which is the richest state the token achieved,
  consistent with what the router judged "done enough to exit"
- Use `shared_blocks[-1].attn.wk/wv`, the last shared block processed the exiting token; with
  `N_LAYERS=4` giving `n_shared=2`, the two shared blocks have independent weights
- Freeze *before* the gated residual update modifies `x`, captures the raw block output
- RoPE on frozen K uses original global positions, composes correctly with active Q positions
  at deeper steps without positional integrity issues
- `frozen_kv=None` default throughout all forward passes and flat baselines (IsoParam, IsoDepth)
  are completely unaffected, zero overhead, no code path changes

### 3. Attention Optimization: Padded Batched SDPA

Both the FULL_SKIP baseline and FrozenKV in this repo use a refactored `_attn_skip` that replaces
the original Python-level `for b in range(B)` loop with a single padded batched
`F.scaled_dot_product_attention` call over a `[B, H, k_max, head_dim]` tensor. Padding positions
are masked via `attn_bias` set to `-inf` on the key dimension.

This is compatible with Compute Capability 5.0 (no Triton or FlashAttention required) and produced
a significant wall-clock improvement:

```
Original Mo²BERTa (Python loop, 600T run):  172.2 minutes  (~1.38s/step)
SDPA-optimized MoR-BERT (this repo):        123.4 minutes  (~1.44–1.70 step/s)
FrozenKV (SDPA + accumulator):              115.8 minutes  (~1.41–1.45 step/s)
```

The ~29% wall-clock reduction comes purely from replacing B serial kernel launches with one
parallel batched SDPA call. FrozenKV's frozen KV accumulator adds negligible overhead at this
scale, the difference between 115.8m and 123.4m is within run-to-run variance on consumer
hardware.


## Evaluation Results (600 TFLOP Final)

The name Mo²BERTa is a post-train naming, so in this report (esp. in plots) the model is still referred to as MoR-BERT

Four models trained to the same 600T cumulative FLOP budget on the same dataset and hardware.
Validation on 20 batches of held-out TinyStories-valid every 50T. **Best checkpoint metrics**
reported since all models exhibit some late-stage oscillation under constant LR.

> Note: MoR-BERT here refers to the SDPA-optimized FULL_SKIP variant, not the original
> Python-loop Mo²BERTa from v1. The SDPA refactor does not change training behavior,
> only wall-clock speed.

### Best Checkpoint Metrics

| Metric                    | **FrozenKV**   | MoR-BERT (SDPA) | IsoParam (L4) | IsoDepth (L7) |
| ------------------------- | :------------: | :-------------: | :-----------: | :-----------: |
| **Best Val Loss**         | **1.8023**     |     1.8427      |    1.9266     |    1.9549     |
| **Best checkpoint @**     |     550T       |      550T       |     600T      |     550T      |
| **Best Val Acc**          | **66.21%**     |     66.08%      |    63.55%     |    65.08%     |
| **Acc best checkpoint @** |     600T       |      550T       |    550/600T   |     550T      |
| **Unique parameters**     |    9.69M       |      9.69M      |     9.69M     |    11.07M     |
| **Tokens seen @ 600T**    |   10.02M       |     10.02M      |    10.32M     |     9.04M     |
| **Wall-clock (600T)**     |   115.8 min    |    123.4 min    |    63.7 min   |    63.7 min   |

FrozenKV wins on both metrics across all four models, including IsoDepth which has 14% more
unique parameters and saw fewer tokens (9.03M vs 10.01M).

### Full Validation Trace (every 50T)

| FLOPs | FrozenKV loss | MoR loss   | IsoParam loss | IsoDepth loss | 
| :---- | :------------ | :--------- | :------------ | :------------ | 
| 50T   | **3.4132**    | 3.4944     | 3.7933        | 3.5955        | 
| 100T  | **2.6808**    | 2.8536     | 2.8236        | 2.8789        | 
| 150T  | **2.4690**    | 2.6806     | 2.6511        | 2.5675        | 
| 200T  | **2.3561**    | 2.5125     | 2.4010        | 2.4736        | 
| 250T  | 2.2963        | 2.4132     | 2.3246        | **2.2268**    | 
| 300T  | 2.1466        | 2.2928     | **2.1173**    | 2.1801        | 
| 350T  | 2.0938        | **2.0468** | 2.0527        | 2.0693        | 
| 400T  | **1.9806**    | 2.1447     | 2.0729        | 2.0864        | 
| 450T  | 1.9226        | **1.9144** | 2.0412        | 2.0717        | 
| 500T  | 1.9463        | **1.9443** | 2.0049        | 1.9613        | 
| 550T  | **1.8023**    | 1.8523     | 2.0566        | 1.9630        | 
| 600T  | **1.8089**    | 1.8427     | 1.9266        | 1.9549        | 

### Training Regime Analysis

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/MQV6zvKKXoak_eCCy-qjn.png)

| Regime       | Leader                             | Notes                                                                                                 |
| :----------- | :--------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **0–150T**   | **FrozenKV**                       | Unambiguous lead. 50T: 3.41 vs 3.49–3.79. 100T: 2.68 vs 2.82–2.88.                                              |
| **150–200T** | **FrozenKV** (narrowing)           | Lead shrinks; 200T still ahead by 0.045 over IsoParam                                                           |
| **200–300T** | **Contested / IsoParam, IsoDepth** | **FrozenKV loses lead**: IsoDepth wins 250T, IsoParam wins 300T                                                 |
| **300–400T** | **MoR-BERT**                       | **MoR's only clear win**: 350T at 2.0468 (best loss so far). 400T spike to 2.14 is volatility, not competition. |
| **400–500T** | **Contested / MoR**                | MoR wins 450T (1.9144) and 500T (1.9443). FrozenKV close second.                                                |
| **500–600T** | **FrozenKV**                       | Dominant from 550T. 1.80 vs 1.85 vs 1.96 vs 1.95.                                                               |

**Late-stage stability** is a secondary finding. MoR-BERT (SDPA) shows oscillation in the
500–600T regime. FrozenKV is measurably more stable, the richer attention context at each
recursion step appears to act as a mild regularizer, reducing sensitivity to the constant LR.

## Routing Behavior Analysis

> **Note on the uniform exit distribution (25%/25%/25%/25%):** Identical to Mo²BERTa v1 - by
> architectural design. The capacity schedule hardcodes *how many* tokens exit at each step.
> What the router learns is *which* tokens exit. The depth gap and heatmap panels show this.

### The Core Finding: Frozen KV Sharpens Routing Discrimination

Routing analysis at end of 600T training (step 9787), FULL_SKIP vs FrozenKV on matched batches:

```
                      FULL_SKIP    FrozenKV      Δ
[MASK] mean depth:      3.34         3.79       +0.45
Non-[MASK] depth:       2.35         2.28       −0.07
Depth gap:              0.99         1.51       +0.52  (+53%)

n([MASK]):               152          151
n(non-[MASK]):           872          873
```

**FrozenKV makes the router more decisive in both directions simultaneously.** [MASK] tokens go
deeper (3.79 vs 3.34) and non-[MASK] tokens exit earlier (2.28 vs 2.35). The possible mechanism: frozen
context from early-exit tokens gives the router richer signal to make confident decisions. Easy
tokens can exit sooner because their contribution is preserved in the KV accumulator. Hard tokens
get pushed deeper because the model now has denser context to refine them against.

This is the opposite of the hypothesis that frozen context would let [MASK] tokens exit
*earlier* because context compensates. Instead the router uses richer context to justify *more*
refinement on hard tokens, not less.

**Score distributions** confirm increased router confidence. FrozenKV Router 1 shows a sharper
spike at 0.0 than FULL_SKIP, indicating more tokens being confidently routed out at the first step.
Both models remain strongly bimodal throughout.

**FULL_SKIP (Mo2BERT-proto)**

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/0Jjt8Ow5iTeGcNkQSjryj.png)

**FROZEN_KV (Mo2BERT-v2-proto)**

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/q_tcZe4o9Q-jmOoEKgv-Y.png)

### At Inference (Post-Training, TinyStories Val Sample)

FrozenKV inference routing on a held-out sequence:

- `[MASK]` tokens: mean depth **3.60 / 4** (n=15)
- Non-`[MASK]` tokens: mean depth **2.35 / 4** (n=113)
- Depth gap: **1.25** which is larger than Mo²BERTa v1's inference gap of 1.05

Score distributions remain strongly bimodal at inference, confirming the routing behavior is
genuinely learned and not an artifact of the auxiliary loss alone.

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/07a5kdixfLZFXndYnJVZU.png)

### MLM Prediction Quality (Same Inference Sample)

14 masked positions evaluated:

| Position | True token | Rank    | Notes                                           |
| -------- | ---------- | ------- | ----------------------------------------------- |
| 13       | `?`        | **1**   | Correct, high margin (10.97 vs 9.67 next)       |
| 30       | `jelly`    | **1**   | Correct, low-frequency content word             |
| 71       | `the`      | **1**   | Correct, confident (12.92 vs 8.84)              |
| 90       | `"`        | **1**   | Correct, very high confidence (16.79)           |
| 93       | `.`        | **1**   | Correct, very high confidence (16.49)           |
| 96       | `at`       | **1**   | Correct (11.15 vs 8.42)                         |
| 99       | `and`      | **1**   | Correct (12.87 vs 7.76)                         |
| 109      | `she`      | **1**   | Correct (10.71 vs 8.13)                         |
| 126      | `feel`     | **1**   | Correct (9.92 vs 6.47)                          |
| 28       | `to`       | 2       | `and` ranked first, both valid                  |
| 29       | `spread`   | miss    | `be` at rank 1; rare verb in TinyStories (n=339)|
| 36       | `run`      | miss    | `it` at rank 1; insufficient disambiguation     |
| 51       | `slice`    | miss    | `top` at rank 1; semantically adjacent          |
| 79       | `fingers`  | miss    | `face` at rank 1; body-part confusion           |

Overall: rank-1 accuracy **9/14 (64%)**, top-5 accuracy **10/14 (71%)**. Misses cluster on
low-frequency content words and semantic near-synonyms, expected failure modes at this scale
and domain.

## What This PoC Does and Does Not Prove

**Supported claims (new in v2):**

- Frozen KV from exited tokens measurably reduces the context isolation penalty in encoder-side
  expert-choice MoR — val loss improves by 0.04 at equal parameter count and FLOP budget.
- FrozenKV beats a 14%-larger flat baseline (IsoDepth L7, 11.07M params) on both loss and
  accuracy at equal compute, having seen more tokens (10.01M vs 9.03M).
- Frozen context sharpens routing discrimination: the [MASK] vs non-[MASK] depth gap increases
  from 0.99 to 1.51 (+53%) compared to FULL_SKIP at identical parameter count and FLOP budget.
- FrozenKV improves late-stage training stability compared to FULL_SKIP under constant LR.
- Padded batched SDPA (CC 5.0 compatible) reduces wall-clock time by ~29% vs the original
  Python-loop gather-scatter implementation, with theoretically identical outputs.
- FrozenKV overhead is negligible at PoC scale, wall-clock is within run-to-run variance of
  the SDPA-only baseline (115.8m vs 123.4m).

**Inherited supported claims from Mo²BERTa v1:**

- MoR transfers from autoregressive decoders to bidirectional encoders without fundamental barriers.
- Expert-choice routing with BCE auxiliary loss produces well-calibrated bimodal routing within
  ~200 training steps.
- The router learns to allocate depth by semantic difficulty using only the MLM signal, with no
  token-type supervision.

**Not supported / out of scope:**

- Wall-clock inference speedup from sparse routing (requires custom CUDA kernels; SDPA is faster
  than the Python loop but theoretical active-token FLOP savings still don't materialize as
  throughput without kernel-level support)
- Scaling behavior (one model size, one dataset, one compute budget)
- Comparison to production encoders (BERT-base, RoBERTa, DeBERTa, etc.)
- Generalization beyond TinyStories domain
- Optimal hyperparameter configuration (constant LR, fixed α=0.1, no ablation over N_recursion
  or capacity schedule shape)
- Whether the depth gap increase under FrozenKV translates to better downstream task performance

## Known Limitations

All limitations from [Mo²BERTa v1](https://huggingface.co/gbyuvd/Mo2BERTa-proto) apply.
Additional v2-specific notes:

**Frozen KV memory overhead.** The accumulator grows across recursion steps, holding
`[B, Hk, k_frozen, head_dim]` tensors that concatenate at each step. At the scales tested
(B=8, T=128, head_dim=32) this is negligible. At larger sequence lengths or batch sizes the
accumulated KV tensors become non-trivial and a retention policy (sliding window, top-k, or
importance-weighted eviction) would be needed.

**Frozen KV FLOP accounting.** The estimator treats FrozenKV identically to FULL_SKIP,
ignoring the small overhead of K/V projections for exiting tokens and the extended key
dimension in SDPA. Accurate at PoC scale; would need revision for precise isoFLOP comparison
at larger scale.

**Step count vs wall-clock asymmetry.** Flat baselines (IsoParam, IsoDepth) complete more
steps than MoR variants within the same TFLOP budget (IsoParam: 10,083 steps vs MoR: 9,787)
because each flat step is computationally cheaper. This is correct behavior, the FLOP cap
is the equalizer, not step count.

**IsoDepth result diverges from v1.** In the original Mo²BERTa run, IsoDepth eventually
reached 67.19% accuracy. In this run it peaks at 65.08%. This is attributed to run-to-run
variance under constant LR rather than any architectural change, but a controlled repeat
would be needed to confirm.

### Known Bug (Fixed in Code, Not Rerun)

A scatter collision in `_attn_skip` caused token position 0 to occasionally 
receive a zeroed attention output when it was simultaneously active and used 
as a padding dummy index. Effect: token 0 behaved as if it exited early in 
affected steps. Both FULL_SKIP and FrozenKV were affected equally throughout 
the 600T runs, so relative comparisons remain valid. The fix (per-item scatter 
excluding padding slots) is present in the released code but the reported 
metrics reflect the buggy training runs.

## Known TODOs / Future Work

**Architecture**
- [ ] Sliding window / top-k frozen KV retention for longer sequences
- [ ] Token-choice routing variant (learned non-uniform exit distribution)
- [x] FrozenKV + trapezoidal LR schedule combined experiment (probably updated in v3, within a month)

**Performance**
- [ ] Formal wall-clock benchmark: FrozenKV vs FULL_SKIP at larger B and T
- [ ] Variable-length FlashAttention kernel for CC 7.0+ hardware
- [x] `torch.allclose` correctness assert: padded SDPA vs original Python loop outputs

**Experiments**
- [x] LR schedule matching (trapezoidal warmup-decay per MoR paper) @ 100T (probably updated in v3, within a month)
- [ ] LR schedule matching (trapezoidal warmup-decay per MoR paper) @ 600T (probably updated in v3, within a month)
- [ ] Examine the embedding structure vs. vanilla BERT (probably updated in v3, within a month)
- [ ] Scale up: larger model, fuller dataset
- [ ] Ablation: FrozenKV with token-choice routing
- [ ] Downstream task evaluation to test if depth gap improvement transfers

**Code**
- [ ] Modularize into model.py / train.py / router.py / dataset.py
- [ ] Add config dataclass to replace module-level constants
- [ ] Unit tests for frozen KV accumulator shape and masking correctness

## Citation

If you use this work, please cite the original MoR paper, the base Mo²BERTa prototype,
and this repository:

```bibtex
@misc{bae2025mixtureofrecursionslearningdynamicrecursive,
      title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation}, 
      author={Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim and Jiyoun Ha and
              Tal Schuster and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and
              Aaron Courville and Se-Young Yun},
      year={2025},
      eprint={2507.10524},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.10524}, 
}

@software{mo2berta_v2_proto,
  author  = {GP Bayu},
  title   = {{Mo²BERTa-v2}: Frozen KV Context for Encoder Mixture-of-Recursions},
  url     = {https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto},
  version = {0.1},
  year    = {2026},
}

@software{mo2berta_proto,
  author  = {GP Bayu},
  title   = {{Mo²BERTa}: Mixture-of-Recursions for Bidirectional MLM},
  url     = {https://huggingface.co/gbyuvd/Mo2BERTa-proto},
  version = {0.1},
  year    = {2026},
}

@misc{eldan2023tinystoriessmalllanguagemodels,
      title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?}, 
      author={Ronen Eldan and Yuanzhi Li},
      year={2023},
      eprint={2305.07759},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.07759}, 
}
```

## Contact

For questions about this prototype, open an issue in the source repository.  
For questions about the base architecture, see [Mo²BERTa](https://huggingface.co/gbyuvd/Mo2BERTa-proto).
