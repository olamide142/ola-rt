Done. I expanded the calendar to **39 hours 45 minutes of deliberate study/research per week**, excluding your Olareport work and walks. The recurring structure still runs through **31 December 2026**.

I added four recurring blocks: **Advanced Mathematics — Core Study II** (Mon/Wed/Fri 09:30–11:00), **AI Engineering — Foundations & Systems** (Tue/Thu 09:30–11:00), **Applied Math / OR — Problem Solving** (Mon/Wed/Fri 13:30–14:30), and **AI Engineering — Build / Benchmark** (Tue/Thu 13:30–14:30).

That puts your weekly allocation at roughly:

| Area | Hours/week |
|---|---:|
| Mathematics + OR + modelling | ~22.75 |
| AI engineering + Ola-RT | ~13 |
| PhD coursework | 2 |
| Research-paper study | 1 |
| Weekly synthesis | 1 |
| **Total** | **39.75** |

The important thing now is that every one of those hours has a destination. These are the goals I would use.

# 1. Algebra and precalculus

Your immediate goal is **mathematical manipulation fluency**.

The prerequisite sequence we established begins with algebra before calculus, linear algebra, multivariable calculus, numerical methods and optimization. :chatgpt-content-reference{index="1"}

You should become comfortable with fractions, powers, roots, logs, exponentials, inequalities, functions, inverse functions, summations, basic vectors/matrices and trigonometry. :chatgpt-content-reference{index="2"}

Your exit criterion is that equations such as

\[
Ae^{kt}=B
\]

or

\[
\frac{x-a}{x+b}=c
\]

can be rearranged without disrupting your train of thought.

You should also become fluent with notation:

\[
\sum,\quad \prod,\quad \forall,\quad \exists,\quad \in,\quad \subseteq.
\]

That becomes important later in both mathematics and Theory of Computation.

---

# 2. Single-variable calculus

The goal is not merely knowing differentiation rules.

You should understand **change, approximation and accumulation**.

Master:

\[
\lim_{x\to a}f(x)
\]

\[
f'(x)
=
\lim_{h\to0}
\frac{f(x+h)-f(x)}{h}
\]

and

\[
\int_a^b f(x)\,dx.
\]

You should be able to derive and use product, quotient and chain rules; differentiate exponential, logarithmic and trigonometric functions; understand Taylor expansions; solve maxima/minima problems; and perform basic integration.

The most important mental interpretation for your future work is:

\[
\frac{df}{dx}
=
\text{sensitivity of output to input}.
\]

That eventually becomes optimization.

Your exit criterion: given a new function, you should be able to analyze its shape, derivatives, extrema and local approximation without depending heavily on software.

---

# 3. Linear algebra

This should eventually become one of your strongest subjects.

Master:

\[
Ax=b
\]

at several levels: algebraic, geometric, numerical and computational.

Goals include vectors and matrices, linear systems, Gaussian elimination, rank, basis, linear independence, column/null spaces, inner products, norms, orthogonality, projections, eigenvalues/eigenvectors and matrix decompositions.

Then understand:

\[
A=LU,
\qquad
A=QR,
\qquad
A=U\Sigma V^T.
\]

Your earlier preparation plan specifically identifies these topics as foundational for ML, numerical analysis and optimization. :chatgpt-content-reference{index="3"}

Your exit criterion is much stronger than “I can multiply matrices.”

You should be able to look at

\[
Y=XW
\]

and reason about dimensions, transformations, rank, conditioning, memory requirements and computational complexity.

---

# 4. Multivariable calculus

This is the direct mathematical bridge into optimization and machine learning.

Master partial derivatives:

\[
\frac{\partial f}{\partial x_i},
\]

the gradient:

\[
\nabla f(x),
\]

the Jacobian:

\[
J_f(x),
\]

and Hessian:

\[
H_f(x).
\]

Also learn directional derivatives, Taylor approximations in several variables, critical points and constrained extrema.

The payoff is being able to understand equations such as

\[
x_{k+1}=x_k-\alpha\nabla f(x_k)
\]

and

\[
x_{k+1}
=
x_k-H(x_k)^{-1}\nabla f(x_k)
\]

rather than memorizing them. :chatgpt-content-reference{index="4"}

Your exit criterion: if I give you an inference-performance function such as

\[
L=L(b,m,p,k),
\]

you should naturally think about sensitivity with respect to batch size \(b\), memory \(m\), precision \(p\), and KV-cache allocation \(k\).

---

# 5. Probability and statistics

Introduce this progressively rather than letting it displace the foundational math immediately.

You eventually need probability spaces, conditional probability, Bayes' theorem, random variables, expectation, variance, covariance, common distributions, sampling, estimation, confidence intervals, hypothesis testing and regression.

Then move toward:

\[
X_t
\]

as a stochastic process.

Later:

- Markov processes,
- Poisson processes,
- Monte Carlo,
- stochastic approximation,
- queueing,
- time series.

This matters because actual inference workloads are stochastic rather than fixed.

Arrival times, sequence lengths, GPU utilization and latency are random variables.

Your exit criterion is being able to model uncertainty rather than merely average it away.

---

# 6. Numerical analysis

Here your mathematics becomes computational mathematics.

Scott's text progresses through numerical algorithms, nonlinear equations, linear systems, vector spaces, iterative methods, conjugate gradients, interpolation, approximation, eigenvalue algorithms and differential equations. :chatgpt-content-reference{index="5"} :chatgpt-content-reference{index="6"}

Core goals:

Root finding:

\[
f(x)=0
\]

using bisection, fixed-point iteration and Newton methods.

Linear systems:

\[
Ax=b
\]

using direct and iterative methods.

Approximation:

\[
\hat f(x)\approx f(x)
\]

using interpolation and least squares.

Numerical differentiation/integration.

Eigenvalue computation.

ODE solvers.

Most importantly, understand the difference between:

\[
\boxed{\text{accuracy}},\qquad
\boxed{\text{stability}},\qquad
\boxed{\text{conditioning}}.
\]

Scott explicitly treats accuracy, stability and finite-precision effects as central concerns of numerical algorithms. :chatgpt-content-reference{index="7"}

Your exit criterion: whenever a program produces a number, your instinct becomes:

> How accurate is it, how stable was the algorithm, and how sensitive was the original problem?

---

# 7. Floating-point computation

This deserves its own goal because of your inference direction.

Understand:

\[
FP64,\ FP32,\ TF32,\ FP16,\ BF16,\ FP8
\]

along with machine epsilon, rounding, overflow, underflow and cancellation.

Understand why

\[
(a+b)+c
\]

need not numerically equal

\[
a+(b+c).
\]

Your earlier mathematics plan explicitly calls these out as important for ML systems. :chatgpt-content-reference{index="8"}

Then connect this to:

- mixed precision,
- quantization,
- accumulation precision,
- numerical reproducibility,
- tensor cores,
- model accuracy.

Exit criterion: you should be able to explain why changing datatype can simultaneously change speed, memory consumption and numerical behavior.

---

# 8. Optimization / Operations Research

This becomes a major pillar of your PhD identity.

Start with:

\[
\min_x f(x).
\]

Master gradient descent, line search, Newton, quasi-Newton/BFGS, conjugate gradient, trust-region methods and convexity.

Nocedal and Wright devote the early part of the text to line searches, trust regions, conjugate gradients, Newton methods, derivatives and quasi-Newton algorithms. :chatgpt-content-reference{index="9"} :chatgpt-content-reference{index="10"}

Then constrained problems:

\[
\min_x f(x)
\]

subject to

\[
g_i(x)\le0,
\qquad
h_j(x)=0.
\]

Master:

- Lagrange multipliers,
- KKT conditions,
- LP,
- duality,
- QP,
- nonlinear programming,
- integer programming,
- mixed-integer optimization.

Then later:

- stochastic optimization,
- robust optimization,
- dynamic programming,
- approximate dynamic programming,
- multiobjective optimization.

Your exit criterion is:

> Given a real decision problem, I can identify decision variables, objective functions and constraints, determine the mathematical class of the problem, and choose a reasonable solution method.

---

# 9. Mathematical modelling

This is a separate skill from solving equations.

Banerjee's modelling process explicitly moves through defining the problem, assumptions, diagrams, selecting equations, solving and interpreting the result. :chatgpt-content-reference{index="11"}

Your goal is to become capable of moving from:

> “This GPU server is overloaded.”

to:

\[
\lambda(t)=\text{request arrival rate}
\]

\[
\mu_i=\text{service rate of GPU }i
\]

\[
x_{ij}=\text{request allocation decision}
\]

and eventually to an optimization/control problem.

You should practice discrete models, ODEs, stochastic models, queues, state-space models, parameter estimation and validation.

Exit criterion:

\[
\boxed{
\text{Reality}
\rightarrow
\text{Assumptions}
\rightarrow
\text{Variables}
\rightarrow
\text{Equations}
\rightarrow
\text{Predictions}
}
\]

becomes natural.

---

# 10. Theory of Computation

For your course, aim beyond passing examinations.

Master:

- formal languages,
- grammars,
- regular languages,
- finite automata,
- pumping lemma,
- context-free languages,
- pushdown automata,
- Turing machines,
- decidability,
- reductions,
- computability,
- complexity foundations.

The larger goal is learning how computer scientists reason formally about:

\[
\boxed{\text{what can be computed}}
\]

and

\[
\boxed{\text{what cannot be computed efficiently or at all}}.
\]

Exit criterion: you can write clean proofs involving constructions, closure properties, contradiction and reductions.

---

# 11. Computer architecture and GPU systems

Treat this as the physical reality underneath AI software.

Goals:

CPU fundamentals, caches, memory hierarchy, pipelines, SIMD/vectorization, threads, synchronization, NUMA and performance counters.

Then GPU architecture:

- SMs,
- warps/wavefronts,
- occupancy,
- global/shared/register memory,
- memory coalescing,
- tensor cores,
- kernel launches,
- bandwidth versus compute limits.

Understand arithmetic intensity and eventually the Roofline model.

Then connect architecture to:

\[
GEMM,
\quad
attention,
\quad
softmax,
\quad
normalization.
\]

Exit criterion: when a kernel is slow, you can reason whether the problem is compute, memory, synchronization, occupancy, launch overhead or algorithmic structure.

---

# 12. AI engineering

This area should make you capable of taking a model from data to production.

Master the complete pipeline:

\[
\boxed{
\text{Data}
\rightarrow
\text{Model}
\rightarrow
\text{Training}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Inference}
\rightarrow
\text{Serving}
\rightarrow
\text{Monitoring}
}
\]

Topics include tensors, autograd, optimizers, datasets/batching, training loops, checkpoints, evaluation, model architectures, PyTorch/Burn, transformers, tokenization, embeddings, attention, fine-tuning, LoRA, quantization and multimodal models.

Then engineering:

- experiment tracking,
- reproducibility,
- model packaging,
- APIs,
- containers,
- Kubernetes,
- observability,
- distributed training/inference,
- failure recovery.

Your exit criterion is not:

> “I understand transformers.”

It is:

> “I can train, evaluate, optimize, serve, profile and operate one.”

---

# 13. Transformer and LLM fundamentals

You should be able to derive and implement a small transformer.

Understand:

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

and

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
\]

Understand:

- embeddings,
- positional representations,
- multi-head attention,
- MLP blocks,
- residual connections,
- normalization,
- autoregressive generation,
- prefill versus decode,
- sampling.

Then calculate:

- parameter counts,
- FLOPs,
- memory,
- KV-cache size.

Exit criterion: you can explain an LLM from the matrix operations upward rather than treating it as a black-box API.

---

# 14. AI inference systems

This becomes your research specialization.

Master:

- continuous batching,
- KV-cache management,
- PagedAttention,
- FlashAttention,
- prefix caching,
- speculative decoding,
- quantization,
- tensor/pipeline/expert parallelism,
- model placement,
- request routing,
- admission control,
- prefill/decode disaggregation,
- scheduling,
- kernel selection,
- autoscaling.

Metrics must become second nature:

\[
TTFT
\]

\[
TPOT
\]

\[
P50/P95/P99
\]

\[
\text{tokens/sec}
\]

\[
\text{requests/sec}
\]

\[
\text{GPU utilization}
\]

\[
\text{Joules/token}.
\]

Exit criterion:

> You can explain why an inference engine behaves the way it does under changing workloads and experimentally identify the bottleneck.

---

# 15. Ola-RT

Ola-RT should gradually become the place where all these subjects collide.

Your near-term milestones should be:

\[
\text{model loader}
\rightarrow
\text{execution graph}
\rightarrow
\text{tensor/kernels}
\rightarrow
\text{KV cache}
\rightarrow
\text{batch scheduler}
\rightarrow
\text{metrics}
\rightarrow
\text{policy layer}.
\]

Then formalize runtime state:

\[
s_t=
(
q_t,
m_t,
u_t,
\lambda_t,
\ell_t,\ldots
)
\]

where these might represent queue length, available memory, utilization, arrival rate and latency.

Define decisions:

\[
a_t=
(
b_t,
r_t,
p_t,
k_t,\ldots
)
\]

for batching, routing, precision, kernel selection, etc.

Then your thesis question emerges naturally:

\[
\boxed{
\pi:s_t\rightarrow a_t
}
\]

What should the inference execution policy be?

And why should it remain static?

---

# 16. Research methodology

Your goal is to stop merely **consuming papers** and begin interrogating them.

For every paper determine:

\[
\text{Problem}
\rightarrow
\text{Assumptions}
\rightarrow
\text{Method}
\rightarrow
\text{Evidence}
\rightarrow
\text{Limitations}.
\]

Then ask:

> Which assumption can I relax?

> What happens under a different workload?

> Is the comparison fair?

> What mathematical problem is hiding underneath this heuristic?

> Can I reproduce the result?

Exit criterion: one paper regularly produces multiple research questions.

---

# 17. Scientific experimentation and benchmarking

Become extremely disciplined here.

Every experiment should have:

- hypothesis,
- baseline,
- independent variables,
- dependent variables,
- controls,
- repetitions,
- confidence/variance reporting,
- hardware/software configuration,
- reproducible scripts.

For inference experiments, record things such as GPU model, precision, batch size, sequence length, concurrency, request distribution, memory use and latency distributions.

Exit criterion:

> Someone else could reproduce your experiment from your repository.

---

# 18. AI + energy optimization

This should remain your long-term application track rather than distracting from the fundamentals now.

Eventually model:

\[
E=\text{energy consumed}
\]

\[
P_t=\text{power}
\]

\[
C_t=\text{electricity/carbon cost}
\]

alongside AI workload state.

Then formulate things such as:

\[
\min
\quad
\alpha L+
\beta E+
\gamma C
\]

subject to SLA and capacity constraints.

Problems worth eventually attacking include GPU power capping, carbon-aware scheduling, datacenter workload shifting, thermal-aware inference, inference placement, energy-price-aware scheduling and joint latency-energy optimization.

That gives you the bridge from **Ola-RT research → AI infrastructure → AI/energy optimization company**.

---

## The December 31 target

By the end of this first calendar cycle, I would want you to be able to demonstrate something concrete in every major stream:

\[
\boxed{
\begin{array}{ll}
\text{Mathematics} & \text{derive and solve}\\
\text{Numerical} & \text{implement and analyze}\\
\text{OR} & \text{formulate and optimize}\\
\text{Modelling} & \text{translate reality into equations}\\
\text{AI engineering} & \text{build and deploy}\\
\text{Architecture} & \text{reason about hardware performance}\\
\text{Inference} & \text{benchmark and explain}\\
\text{Ola-RT} & \text{implement runtime components}\\
\text{Research} & \text{identify and test novel questions}
\end{array}}
\]

The **40-hour week isn't supposed to produce 40 hours of reading**. By design, a large fraction of it is problem solving, derivation, implementation, benchmarking and research. That is what will make the hours compound rather than merely accumulate.
