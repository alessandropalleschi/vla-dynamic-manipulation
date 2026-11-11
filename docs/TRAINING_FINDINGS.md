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

| 4. grounding **pre-training on 30k static scenes**, then (see below) fine-tune on demos | 4000 (pre-train) | n/a (grounding only) | n/a |

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

## Experiment 4: grounding pre-training on static scenes (confirms the data-scale explanation)
30,000 scenes (`vla_mujoco.data.static_scenes`, ~30 ms each, no rollout), 64 px, 4x4 patch grid, 4000 steps, batch 64.
Metric: top-1 patch accuracy for the object the instruction names (pick and support), validation scenes only.

| encoder | right text | wrong text | gap | @ step 1000 | pick / support |
|---|---|---|---|---|---|
| bow | 0.754 | 0.273 | +0.481 | 0.637 | 0.736 / 0.779 |
| scratch | 0.754 | 0.261 | +0.493 | 0.629 | 0.736 / 0.779 |
| pretrained (MiniLM) | 0.761 | 0.259 | +0.502 | 0.709 | 0.738 / 0.793 |

Location-prior baseline 0.154, chance 0.063. With thousands of distinct (scene, instruction) pairs all three encoders learn to
ground language (gap about +0.5), whereas 288 demo episodes gave no gap. So the earlier failure was data scale, not the encoders
or the architecture.

* **Encoder comparison:** the final accuracies differ by less than the noise (std. error ~0.008 on this validation set).
  The pretrained encoder reaches high accuracy faster (0.709 vs ~0.63 at step 1000), which is the only difference we trust.
  Expect a larger advantage with more diverse phrasing (paraphrases) than the template language used here.
* **Plateau at ~0.75:** probably the 4x4 patch grid at 64 px (objects share patches, label sits near cell borders); the
  resolution experiment (finer grid / larger images) is the next test.
* **Probe caveat:** "wrong text" is another scene's instruction, which often differs in obvious attributes; it shows the
  text is *used*, not that the finest distinctions (e.g. same-colour objects told apart only by ordinal position) are resolved.
  Per-attribute breakdowns (colour / category / size / ordinal) would be the stricter test.

## Experiments 5-6: does grounding reach the actions? (fine-tune on demos from the pre-trained grounding model)
Same 320 demos / hyper-parameters as experiment 3 (1500 steps), initialised from the experiment-4 checkpoints (`--init`).
`ground_acc` is on demo frames; the "first 12 steps" probe is the action error at the start of each episode (the arm has not
moved, so only the text says which object is meant); gap = wrong-text error minus right-text error, 32 validation episodes.

| model | demo ground_acc (right / wrong) | val action gap | first-12-steps action gap |
|---|---|---|---|
| no pre-training (exp. 3), pretrained text | 0.687 / 0.684 | -0.0002 | -0.0006 |
| grounding pre-trained, **no slots**, pretrained text | 0.750 / 0.591 | +0.0010 | +0.0017 |
| grounding pre-trained, **slots**, pretrained text | 0.749 / 0.612 | +0.0011 (+0.0042 @1000) | **+0.0197** |
| no pre-training (exp. 3), bow | 0.693 / 0.678 | +0.0020 | -0.0019 |
| grounding pre-trained, no slots, bow | 0.781 / 0.666 | +0.0017 | -0.0052 |
| grounding pre-trained, slots, bow | 0.741 / 0.639 | +0.0039 (+0.0057 @1000) | -0.0013 |

* **Grounding transfers:** after pre-training, grounding on demo frames depends on the text (gap +0.12 to +0.16 vs ~0 without it).
* **But the actions barely use it.** Without slots the action gap is zero even at the decision frames: the grounding head was a side
  branch the belief did not read. With grounded object slots (pick/support patch distributions pool patch features and expected
  coordinates into the belief) the gaps grow (validation +0.004-0.006 at step 1000, first-12-steps +0.020 for the pretrained
  text encoder), so the pathway helps, **but the effect is small** (the error is ~0.30) and appeared for only one of two encoders.
  With 32 validation episodes and no per-episode confidence intervals this should be read as "promising, not established".
* **Action quality got worse at the decision frames after pre-training** (0.30-0.36 vs 0.21): the pre-trained models had
  no action/proprioception training before fine-tuning, so equal fine-tuning steps are not an equal footing.
* The teacher-forced action loss is dominated by frames where the arm's own motion already implies the goal, so it gives
  language little gradient.

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
1. **Grounding pre-training on static scenes (done, experiment 4); fine-tuning on demos is running.** Generate tens of thousands of scenes + instructions (reset + render only,
   no expert rollout), train the language-conditioned grounding heads on them, then fine-tune on demonstrations. Success
   criterion: ground accuracy with the right text clearly above wrong-text accuracy. (Direct image-space grounding
   supervision, experiment 3, is already implemented.)
2. **Weight the loss on decision frames (next; the experiments above point here)** (start of approach / transfer, where the goal is not yet implied by the arm's own
   motion). Teacher-forced action loss is dominated by frames where the proprioceptive velocity already reveals the goal.
3. **More data and steps on the GPU** (`make train`), larger images, and a pretrained vision backbone.
4. Re-run the probes above (`lang_gap` is logged during training) and only then compare encoders and heads.
