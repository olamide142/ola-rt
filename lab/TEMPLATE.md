# {{TITLE}}

- **Status:** planned
- **Track:** <!-- A runtime · B batching · C scheduling · D memory · E adaptive · F cluster · G kernels · R rust -->
- **Research question:** <!-- RQ1–RQ4 from notes/thesis.md, or "learning" -->
- **Experiment:** `{{EXPERIMENT}}` (parameters in `config.json`)
- **Started:** {{DATE}} at revision `{{REVISION}}`
- **Entry:** `{{ENTRY}}`

<!-- Fill Question, Hypothesis, and Prediction BEFORE you look at any
results. Writing the prediction first is what turns a run into a test. -->

## Question

<!-- One sentence. "Does X change Y under Z?" -->

## Hypothesis

<!-- Falsifiable: direction and rough size. "SJF cuts p95 completion by at
least 20% against FIFO on mixed_short_long, because…" State why: the model,
theory, or earlier entry it comes from. -->

## Prediction

<!-- Numbers you expect, before running. Also: what result would make you
abandon the hypothesis? -->

## Protocol

<!-- What varies, what is fixed, seeds, train/held-out split, clock (virtual
or host). Anything you changed in code, with the commit. If you chose any
setting by looking at results, say which seeds you looked at and which
seeds you report. -->

## Results

<!-- Paste the summary from runs/NNN.txt and the key rows of
`lab/lab compare`. Give means with 95% intervals, not single numbers. -->

## Interpretation

<!-- Did it support the hypothesis? Alternative explanations? Threats to
validity: cost model, trace realism, too few seeds, tuning on test seeds,
search space that excludes the baseline. -->

## What I learned

<!-- Systems, mathematics, and Rust. What surprised you? Which code did you
read, and what did you not understand yet? -->

## Follow-ups

<!-- Next entries this suggests, and code changes it needs. Link entries
as notebook/<entry>/notes.md. -->
