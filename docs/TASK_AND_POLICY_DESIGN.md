# Task and policy design: multi-object, multi-step tasks under partial observability

Status: **build step 1 is implemented** (entity pool, generic pick/place expert, predicate checker, referring-expression
language, four task kinds incl. a 2-step tower); steps 2-3 (per-cup suction, bacon strip, belief + flow-matching policy)
are still design only.
Target tasks: stack on a moving item; *sandwich* (take a slice, place it on the half on the conveyor);
*bacon sandwich* (bacon strip onto the bottom slice, then close with the top slice).

## 1. Task model

* **Entities, not slots.** A compiled pool of free bodies (about 12) is activated per episode. Each entity has a
  category, attributes, pose, state and a *grasp spec* (suction patches, mass, flatness). A belt spawner emits items.
* **Goals are predicates over entity ids** (`on`, `between`, `covers`, `in`, `unmoved`, `stable(t)`), arranged in a
  **DAG of steps with preconditions**. One generic checker replaces the hardcoded `stack_status()`.
  Bacon sandwich: `on(bacon, bottom)` and `covers(bottom, bacon, >= 70 % of length)` -> `on(top, bottom)` with
  `between(bacon, bottom, top)` -> `stable(0.5 s)` and `unmoved(others)`.
* **Language is generated from the task graph and the scene graph**: a subgoal sentence per DAG node
  ("put the bacon strip on the bottom slice", "now close the sandwich"), a full-task instruction, and referring
  expressions that are checked for uniqueness (attributes, category/function, spatial and ordinal-on-belt relations,
  exclusion). LLM paraphrases are added at data-generation time; text is encoded by a frozen pretrained encoder.
  Paired sampling (same scene, different instruction, different action) prevents attribute shortcuts.
* **Skills the expert composes**: `Pick(entity)` and `Place(entity, support)`, reusing the current grasp, intercept and
  track-and-descend code. Place height comes from a ray-cast support query (highest surface under the xy), so layers and
  the lumpy bacon-on-bread surface work. A planner schedules which belt item to catch, with recovery on failed seals.

### Modelling the bacon
* **Articulated strip**: 6-10 rigid segments joined by passive hinges with stiffness and damping, randomised
  (length, stiffness, initial curvature). Captures sag, drape and swing while staying fast and stable in contact.
  MuJoCo cable/flex bodies are the higher-fidelity fallback; a rigid slab is a baseline for ablations.
* **Per-cup suction with leakage** (needed for a 3-4 cm wide strip): each of the 4 cups seals from a short ray-cast;
  holding force scales with sealed area, it fails (strip slips off) under high acceleration, and channels A/B can be
  selected. The grasp is attached to the segment under the cups, so the rest of the strip hangs naturally.
  This upgrade is a prerequisite for the bacon task and also makes all tasks more realistic.

## 2. Policy: belief-conditioned flow-matching action expert

```
cameras (front, wrist, side) -> frozen ViT tokens --+
instruction / current subgoal -> frozen text enc ---+--> belief core: h_t = f(h_{t-1}, obs_t, a_{t-1})
proprio, last action, dt ---------------------------+        |
                                                             +--> aux heads (training only, no privileged input at test)
                                                             |      entity set: pos/vel mixture density, visibility
                                                             |      task progress: predicate truth values
                                                             |      dynamics: belt speed, friction, bacon stiffness
                                                             v
                          flow-matching DiT action expert (cross-attends to h_t + vision tokens)
                                -> action chunk H=16 (0.8 s) of [dx,dy,dz,dyaw] (continuous)
                                   + vacuum per channel (Bernoulli head)
```

**Decisions and reasons**
* **Flow matching over diffusion.** Same expressivity for multimodal actions (e.g. where to intercept, which slice
  first), but a simpler loss and 5-10 Euler steps at inference, which fits 20 Hz closed-loop control of moving objects.
  Autoregressive action tokens are rejected: weaker on precise, continuous high-rate tracking.
* **Hidden state: a recurrent belief core (GRU or SSM) with probabilistic auxiliary heads**, not an RSSM at first.
  Heteroscedastic / mixture heads give a *belief distribution* over occluded entities without the KL-balancing
  difficulty of a stochastic latent. The RSSM (deterministic h plus discrete z) stays as an ablation, and becomes the
  default only if calibration under multimodal futures (belt speed jumps) is poor.
* **What the belief must hold** (the real sources of partial observability here): objects hidden by the arm or an
  occluder; the bacon once the top slice closes (progress cannot be read from the image, so predicate truth values are
  an explicit output); items that leave the field of view; latent dynamics (belt speed, friction, stiffness).
* **Hierarchy**: a ~1 Hz subgoal channel (same model with a text head, or an external VLM) feeds the 20 Hz policy.
  The expert logs the subgoal text for every step, so no extra labelling is needed.
* **Vacuum is a discrete head**, not flowed: timing is sharp (grip 0.35 s, release 0.2 s delays).
* **Execution**: execute the first 4-8 actions of each chunk and re-plan with overlap (real-time chunking /
  inpainting of the committed prefix) so replanning does not jitter.

**Training recipe**
1. Expert data with action noise plus recovery segments, failed-grasp and re-grasp episodes, domain randomisation.
2. Perception + belief pretraining on the privileged auxiliary targets.
3. Joint behaviour cloning (flow-matching loss + aux + progress), truncated BPTT over 32-64 steps with burn-in, and
   **observation dropout** (random camera/time-span masking) so the policy must use its belief.
4. DAgger: roll out the policy, relabel visited states with the expert.
5. Optional RL fine-tuning of the flow head for catch timing.

**Evaluation** (success per instruction category, plus): occlusion length 0 / 0.5 / 1 / 2 s; belief NLL and
calibration on hidden entities; progress-prediction accuracy after the sandwich closes; held-out attribute/relation/
category combinations; latency. Ablations: no memory vs frame window vs GRU vs RSSM; regression vs flow; with/without
aux heads; with/without observation dropout.

## 3. Build order
1. **(done)** Entity pool, belt spawner, predicate checker, generic `Pick`/`Place` with support query, referring-expression
   generator (current stacking task kept as a special case).
2. Per-cup suction with leakage, articulated bacon, thin bread slabs; sandwich and bacon-sandwich tasks; subgoal logging.
3. Belief + flow-matching policy (pretrained encoders), then DAgger and the evaluation suite above.
