# Training findings: does the policy use the instruction?

Setup for both experiments: 320 expert demos (`--task stack`, 2 belt + 1 table bystander, 64 px, front + wrist cameras),
GRU belief, regression head, three text encoders (`bow`, `scratch`, frozen `pretrained` MiniLM), identical
hyper-parameters, CPU only. Metric: **language gap** = validation action error with another episode's instruction minus
the error with the correct one (0 = the text is ignored).

| experiment | steps | val action L1 (bow / scratch / pretrained) | language gap |
|---|---|---|---|
| 1. original generator | 1200 | 0.046 / 0.046 / 0.051 | +0.0008 / +0.0013 / +0.0010 |
| 2. roles placed at random belt positions, aux head also locates the pick object | 1500 | 0.054 / 0.050 / 0.051 | +0.0002 / +0.0007 / +0.0001 |

| 3. as 2, plus image-space grounding loss (`--ground-w 1`) | 1500 | 0.052 / 0.049 / 0.054 | +0.0020 / +0.0022 / -0.0002 |

Further probes (experiments 1 and 2): the action error on the first 12 steps of an episode (the arm has not moved, only the
text says which object is meant), and the auxiliary target-localisation error with right vs wrong text over the first
8 frames. **All gaps are within +-0.006: none of the models used the instruction.**

## Experiment 3: grounding supervision
The grounding head (which image patch contains the object the instruction names) reaches 0.69-0.73 top-1 accuracy over the
4x4 patch grid, but accuracy with a *wrong* instruction is the same (0.67-0.74, fluctuating around the right-text value with
no consistent sign), for all three encoders. So the heads learn *where objects usually are for each role* (pick objects sit
on the table, supports on the belt, plus position priors), not *which object the words describe*. The wiring is correct
(unit tests show the text changes the belief); it simply has not been learned.

Likely reason (hypothesis, not yet tested): **too few distinct (scene, instruction) pairs**. 288 training episodes are 288
instructions, and learning "red", "second from the conveyor", "larger" from 288 examples at 64 px is a tiny compositional
problem; the number of *frames* is large but they share the same instruction. Grounding needs thousands of distinct pairs,
and those do not need trajectories.

## What this does and does not show
* All encoders learn the motion structure (phase accuracy ~96 %, vacuum ~95 %) and localise targets far better than the
  mean (aux error 0.50 vs 1.19 on the first frames), so the pipeline trains.
* **Experiment 1 found a data leak:** the support object was nearly always the most downstream belt item, making the
  instruction redundant. Fixing it (roles placed first at random inside their feasible window) removed that cue (support
  rank 35 / 49 / 16 % instead of ~always first) but did **not** make the models read the text in 1500 steps.
* Therefore the comparison between text encoders is **inconclusive**: with a zero gap, differences in action error are noise.
  It says nothing against the pretrained encoder; the models simply never learned to ground language at this scale.
* Not tested: more steps, a GPU-scale batch/data size, a bigger vision backbone, the flow head, and closed-loop success
  (the policies were not good enough to be worth rolling out).

## Hypotheses and next steps (in order of expected payoff)
1. **Grounding pre-training on static scenes (next).** Generate tens of thousands of scenes + instructions (reset + render only,
   no expert rollout), train the language-conditioned grounding heads on them, then fine-tune on demonstrations. Success
   criterion: ground accuracy with the right text clearly above wrong-text accuracy. (Direct image-space grounding
   supervision, experiment 3, is already implemented.)
2. **Weight the loss on decision frames** (start of approach / transfer, where the goal is not yet implied by the arm's own
   motion). Teacher-forced action loss is dominated by frames where the proprioceptive velocity already reveals the goal.
3. **More data and steps on the GPU** (`make train`), larger images, and a pretrained vision backbone.
4. Re-run the probes above (`lang_gap` is logged during training) and only then compare encoders and heads.
