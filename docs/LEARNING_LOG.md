# Learning log

Running notes. Kept because the mistakes are the point of the exercise, and because a
lab notebook you only write in when things go well is a marketing document.

---

## Why this repo exists

I build hybrid bioprocess models professionally — mechanistic core, data-driven layer,
state estimation, control on top. The modelling is familiar ground.

What is *not* automatic for me, and what I wanted to drill deliberately, is the
production-ML discipline that surrounds it: Flyte, MLflow, Optuna, model registries,
packaging, strict typing, CI. In my day job that machinery grew up inside a platform I
built myself, which means I know the *problems* it solves intimately and the *standard
tools* less well than I would like.

So: pick a domain I have opinions about, and use it as a vehicle for practising the
tooling. Deliberate practice, with a bit of biology to keep it honest.

---

## Entry 1 — Where to put the ML

First real decision, and I think the most consequential one in the repo.

The tempting move is to let a network learn `dX/dt` directly, or to learn residuals on
each state. It is flexible, it is fashionable, and it is how you end up with a model that
produces lactate from nowhere.

Instead: the learned component corrects **one scalar**, the specific growth rate, as a
bounded multiplier.

```
mu_eff = mu_mech(S, L) x correction(features)
```

Consequences I did not fully anticipate when I chose it:

- Mass balances become **structurally** safe. The ML layer routes through `mu`, and
  everything downstream of `mu` is stoichiometry I wrote by hand. The network can re-time
  the biology; it cannot break conservation. This is much stronger than checking
  conservation afterwards and hoping.
- The learned object becomes **arguable**. A correction multiplier over time is a curve a
  scientist can look at and reject. That conversation is worth more than a few points of
  RMSE, and you simply cannot have it with a black box emitting derivatives.
- It constrains what the ML *can* fix. If the mechanism is wrong about stoichiometry
  rather than kinetics, this architecture cannot save it. That is a real limitation and I
  would rather have it explicit than discover it later.

Narrow seams. Boring, and correct.

---

## Entry 2 — The solver hung, and it was my fault

Default correction was a gradient-boosted tree. First hybrid simulation: >100 s, no
result. Assumed I had an infinite loop somewhere.

I did not. A tree ensemble is piecewise-constant, so the ODE right-hand side became
**discontinuous** at every split boundary. LSODA is adaptive; it detected the
discontinuities and duly shrank its step size to nothing, chasing edges that were not
physical features but artefacts of my model class.

Smooth MLP: **0.71 s.** Same problem, same tolerances, same everything.

The lesson I actually want to keep: **when a learned component lives inside a numerical
integrator, the smoothness of the model class is a functional requirement.** It sits
alongside accuracy, not below it. And no held-out metric will ever surface it — a tree
correction scores *fine* on a test set. You only find out at integration time, and only if
you understand the solver well enough to interpret what it is telling you.

`tree_estimator` is still in the codebase, documented as a warning. Deleting it would have
deleted the lesson.

---

## Entry 3 — My own guardrail caught me

Wrote `constraints.py` to catch the ML layer misbehaving. On the very first full test run,
it failed — on the **mechanistic model**, before any ML was involved at all.

Glucose reached −0.89 mM.

Cause: cells kept paying maintenance costs out of a glucose pool that had already hit zero.
The growth term shuts off correctly as S → 0 (Monod handles it); the maintenance term had
no such gate. So it kept consuming a substrate that was not there.

The model had been running "fine" for a day. Plausible curves. Nothing alarming.

Two things I took from this:

1. It is a small, humbling, perfect demonstration of the repo's own thesis, arriving
   uninvited. **No accuracy metric would have caught this.** The trajectory looked
   entirely reasonable.
2. The fix had to be **smooth** (`S / (Km + S)`), not a hard `if S > 0` switch — because a
   hard switch reintroduces exactly the discontinuity from Entry 2. The lessons compound,
   which is the nice thing about learning them in the same codebase.

---

## Entry 4 — I spent an afternoon blaming the wrong thing

The hybrid model would not beat the mechanistic baseline. nRMSE 0.0275 vs 0.0225.

What I did, in order, and it is not a flattering list:

1. Assumed the label was too noisy. Added Savitzky–Golay smoothing before differentiating.
   *(This was, in fairness, a genuine and necessary fix — see below.)*
2. Assumed the model was overfitting. Swept regularisation. Found that `alpha=1.0`
   "worked" — then realised it "worked" by shrinking the correction toward 1.0, i.e. by
   quietly becoming the mechanistic model again. A hollow win, and `NullCorrection` exists
   precisely to make that kind of self-deception visible.
3. Assumed the label filter was wrong. Changed clipping to dropping. Got *worse*, because
   my new filter deleted the late-culture rows where the decay signal actually lived. I had
   removed the thing I was trying to learn.
4. Finally, and far too late, **looked at the data.**

The plant–model gap was **3.8%**. The assay noise was **3%**. Signal-to-noise ≈ 1.3.

There was nothing to learn. My synthetic plant's unmodelled effect only kicked in after
~168 h, by which point the culture was substrate-limited and growth was small anyway, so
the effect barely moved the trajectory. The mechanistic model was accidentally almost
right, and I had spent an afternoon interrogating a model for failing to extract a signal
I had never put into the data.

Moved the decay onset earlier and increased its severity. Hybrid immediately: **+24% nRMSE,
+20% titre error, zero violations.**

**Diagnose the dataset before you diagnose the model.** I know this. I say it to other
people. I did not do it, because the model was the interesting thing and the data was the
boring thing, and that is exactly the bias that makes the mistake so common.

### The sub-lesson, which is nearly as good

The training label is `mu_obs / mu_mech`, where `mu_obs` is estimated by finite-differencing
cell density. **Differentiation is a high-pass filter.** 3% multiplicative assay noise on
Xv, differenced at 6-hour spacing, becomes a substantial fraction of `mu_max` — so the label
is largely noise, the model dutifully learns the noise, and then *injects it back into the
ODE*, where the integrator faithfully propagates it.

Smoothing `log(Xv)` before differentiating was the difference between a working hybrid model
and a broken one. It is one line. It is also exactly what any process engineer does by eye
before reading a slope off a plot, which is a nice reminder that the domain instinct and the
numerical method are the same insight wearing different clothes.

---

## Entry 5 — Making the gate real

It is easy to write a constraint checker and then never wire it to anything with teeth. A
guardrail that has never stopped anything is not a guardrail; it is a comment.

So admissibility gained teeth in four places:

- **`EvaluationReport.passed`** is `constraints_ok AND accurate`. Not accuracy with a
  constraint footnote.
- **`test_scientific_constraints.py`** injects a deliberately rogue correction (3x growth
  rate) and asserts it is *caught*. Negative tests. A validation framework that has only
  ever been shown passing cases is untested.
- **The Flyte `validation_gate` task** raises, failing the DAG, so an inadmissible model
  cannot reach the registry no matter how good its metrics are.
- **The Optuna sweep prunes** inadmissible trials rather than penalising them. This one is
  subtle and I nearly got it wrong: if you put the violation in the objective as a penalty
  term, the optimiser will simply discover the exchange rate — the point at which a small
  mass-balance violation buys a large RMSE gain — and take that trade every time. It is not
  misbehaving. It is doing exactly what you asked. Constraints must be **vetoes**, not
  **prices**.

---

## Still to do

- [ ] Swap the sklearn MLP for a Torch module behind the same `CorrectionModel` Protocol —
      the interface was designed for this, so it should be a new file and nothing else. Good
      test of whether the abstraction was real or decorative.
- [ ] Ray Tune as an alternative to the Optuna sweep, for the distributed-execution practice.
- [ ] A proper model registry stage (MLflow Model Registry), with the validation gate as the
      promotion criterion.
- [ ] Train the correction on **rollout** rather than one-step targets. Right now it learns
      from observed states but is deployed on its own predicted states — a classic
      distribution shift, and I suspect it is leaving accuracy on the table.
- [ ] Uncertainty on the correction. A bounded multiplier with no error bar is a confident
      claim, and I am not sure it has earned that.
