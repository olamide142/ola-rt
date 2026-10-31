# OlaRT lab

This is where experiments become findings. Every experiment you run for a
question gets an **entry**: a dated directory holding exactly what ran, the
versioned reports it produced, and your notes. Entries accumulate into the
evidence behind papers, and the act of predicting, running, and explaining
each one is how you learn the system.

- `lab/lab`: the tool (`make lab-*` wraps it).
- `TEMPLATE.md`: the notes every entry starts from.
- `CURRICULUM.md`: graded experiments by track, each tied to code to read,
  mathematics to learn, a Rust exercise, and open research questions.
- `MODELS.md`: which models load, what fits in memory, and how to fetch
  them (`lab/lab fetch org/repo`).
- `notebook/`: the entries. `INDEX.md` lists them (`make lab-list`).

## The loop

1. **Pick a question.** From `CURRICULUM.md`, a follow-up of an earlier
   entry, or `notes/thesis.md`. One question per entry.
2. **Create the entry**, which also runs it once:

   ```bash
   make lab-new NAME=sjf-vs-fifo-mixed ID=scheduler-comparison \
       SET="trace=mixed_short_long seeds=10 requests=200"
   ```

   Model-backed experiments add `MODEL=/path/to/artifact` (and
   `BACKEND=wgpu` or `cuda`); kernel studies add `FEATURES=cubecl-cpu` or
   `FEATURES=wgpu`. The tool builds a release `ola` once per feature set,
   each in its own `ola-rt/target/lab-…` directory: the first build of a
   new set compiles the whole GPU stack and can take ten minutes or more;
   later runs reuse it.
3. **Before reading the summary, write the Question, Hypothesis, and
   Prediction** in `notes.md`. A prediction written afterwards teaches you
   nothing. (`make lab-new` prints the summary; look away, or write the
   prediction before creating the entry.)
4. **Read the results**: `runs/001.txt` is the summary;
   `reports/*.json` is the full report with raw samples and 95% intervals.
   For traceable experiments, `traces/run-001.jsonl` has every scheduling
   decision (see "Reading traces" in `../TESTING_GUIDE.md`).
5. **Replicate or vary.** `make lab-run ENTRY=lab/notebook/<entry>` runs the
   same configuration again and adds a report; for virtual-time studies
   this checks determinism, and for host-time studies it measures noise.
   To change a parameter, make a new entry: one configuration per entry
   keeps comparisons honest.
6. **Compare** two reports: `make lab-compare ENTRY=…` (first versus latest
   run of an entry) or `lab/lab compare a.json b.json` (across entries). A
   metric counts as improved or regressed only when the 95% intervals
   separate and the change exceeds 5%.
7. **Write it up**: Results, Interpretation, What I learned, Follow-ups.
   Set Status to `done`, `inconclusive`, or `abandoned`. Negative results
   are findings; keep them.
8. **Commit the entry** (`notes.md`, `config.json`, `args`, `reports/`,
   `runs/`, `runs.log`). Traces are ignored by git; regenerate them with
   `make lab-run`.

Each run records the git revision in `runs.log`, with `+dirty` if
`ola-rt/` had uncommitted changes. Commit code changes before the runs you
intend to cite, so every result maps to a revision.

## Rules that keep results honest

These are the mistakes this project has already made at least once.

- **The search space must contain the baseline's behaviour.** The first
  `adaptive-policy-study` searched token budgets of 16–32 while FIFO's
  unbounded batching was best; its "adaptive" policy was 2.4× worse than
  FIFO and looked like a policy result. Before comparing a tuned policy
  with a baseline, check the baseline is reachable inside the grid.
- **Never choose settings on the seeds you report.** If you pick a
  parameter by looking at seeds 1–5, report seeds 6–10 (raise `seeds`)
  and say so in Protocol.
- **Train on one trace, evaluate on held-out traces.** The studies use seed
  0 to select and seeds 1… to evaluate; keep that split in your own
  experiments.
- **Know your clock.** Virtual-time results are deterministic and reproduce
  anywhere, but only measure what the `CostModel` says; host-time results
  are real but noisy. Host-time studies need release builds (the lab tool
  uses them), warm-ups, at least five repetitions, and an idle machine.
- **Ask whether the effect can exist.** Online adaptation cannot beat the
  best fixed setting when the best setting never changes. Before testing
  "adaptive beats fixed", show the optimum moves.
- **Intervals, not single numbers.** Report the mean and its 95% interval;
  with 5 seeds, overlapping intervals usually mean "not shown".
- **One change at a time.** If a code change and a parameter change land
  in the same comparison, you cannot attribute the effect.

## From entries to papers

A paper is a sequence of entries around one question: a characterization
(what happens, across workloads), a mechanism (why, with traces), an
intervention (a new policy), and robustness checks (other seeds, traces,
cost models, hardware). Track that sequence through Follow-ups, and use the
Track and Research question fields to find related entries in `INDEX.md`.
When an entry's result is surprising, the next entry should try to break
it.
