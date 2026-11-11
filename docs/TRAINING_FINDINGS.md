# Training findings: does the policy use the instruction?

Setup for both experiments: 320 expert demos (`--task stack`, 2 belt + 1 table bystander, 64 px, front + wrist cameras),
GRU belief, regression head, three text encoders (`bow`, `scratch`, frozen `pretrained` MiniLM), identical
hyper-parameters, CPU only. Metric: **language gap** = validation action error with another episode's instruction minus
the error with the correct one (0 = the text is ignored).

| experiment | steps | val action L1 (bow / scratch / pretrained) | language gap |
|---|---|---|---|
| 1. original generator | 1200 | 0.046 / 0.046 / 0.051 | +0.0008 / +0.0013 / +0.0010 |
| 2. roles placed at random belt positions, aux head also locates the pick object | 1500 | 0.054 / 0.050 / 0.051 | +0.0002 / +0.0007 / +0.0001 |

Further probes (experiment 1 and 2): the action error on the first 12 steps of an episode (the arm has not moved, only the
text says which object is meant), and the auxiliary target-localisation error with right vs wrong text over the first
8 frames. **All gaps are within +-0.006: none of the models used the instruction.**

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
1. **Direct grounding supervision in image space.** Add a head that predicts, over the patch tokens, which patch contains the
   object named by the instruction (projected object centroids are available in the simulator). This is referring-expression
   comprehension and gives the text a gradient that does not depend on the action loss. Compare the language gap with/without.
2. **Weight the loss on decision frames** (start of approach / transfer, where the goal is not yet implied by the arm's own
   motion). Teacher-forced action loss is dominated by frames where the proprioceptive velocity already reveals the goal.
3. **More data and steps on the GPU** (`make train`), larger images, and a pretrained vision backbone.
4. Re-run the probes above (`lang_gap` is logged during training) and only then compare encoders and heads.
