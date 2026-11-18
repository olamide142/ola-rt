# Paper 1 — Formalizing Inference Execution Policies for Generative AI Systems

## Abstract

Modern generative AI inference engines employ increasingly sophisticated mechanisms for request scheduling, continuous batching, memory management, model parallelism, cache allocation, and execution control. However, these mechanisms are typically implemented as engine-specific heuristics, making inference policies difficult to describe, compare, analyze, reproduce, or optimize independently of their implementations. This paper introduces a formal representation of an **inference execution policy** as a decision-making system operating over observable runtime state, available execution actions, system constraints, objectives, and state transitions. The framework decomposes inference execution into policy dimensions including admission control, batching, request scheduling, KV-cache management, resource placement, parallelism, and execution configuration. Existing policies from representative inference engines are expressed using the proposed abstraction to demonstrate its generality. We further define policy equivalence, policy composition, and measurable execution objectives such as time-to-first-token, inter-token latency, throughput, memory utilization, cost, and service-level objective violations. The resulting formulation separates *what execution decisions should be made* from *how the underlying runtime implements them*, establishing a common mathematical and systems foundation for the analysis and optimization of inference execution policies.

---

# Paper 2 — Optimal Inference Execution Policies under Resource and Service-Level Constraints

## Abstract

Inference engines must simultaneously allocate limited compute and memory resources while satisfying competing objectives such as latency, throughput, cost, and service-level objectives. Existing systems predominantly rely on manually designed scheduling and resource-management heuristics whose distance from optimal execution is generally unknown. Building upon a formal representation of inference execution policies, this paper formulates generative AI inference as a constrained optimization problem. Decision variables capture request admission, batch composition, execution ordering, memory allocation, and resource assignment, while constraints represent GPU memory capacity, compute availability, request dependencies, and latency requirements. We develop offline optimization formulations for inference execution under known workloads and investigate exact and approximate solution methods for obtaining optimal or near-optimal schedules. These solutions provide an **optimization oracle** against which practical inference policies can be evaluated. We introduce measures of policy optimality gap and characterize conditions under which commonly used scheduling heuristics diverge from optimal execution. Experiments across heterogeneous request lengths, arrival patterns, memory pressures, and service-level requirements quantify the trade-offs between latency, throughput, utilization, and cost. The work establishes mathematical optimization as both a mechanism for constructing inference policies and a principled benchmark for evaluating runtime scheduling decisions.

---

# Paper 3 — Online Inference Execution under Stochastic Workloads

## Abstract

Offline optimization of inference execution assumes knowledge of future request arrivals, sequence lengths, and resource demands, whereas production inference systems must make decisions online under substantial uncertainty. This paper studies inference execution as a **stochastic online resource-allocation and scheduling problem**. Requests arrive dynamically with heterogeneous prompt lengths, unknown generation lengths, priorities, and latency requirements, while the runtime must continuously determine admission, batching, scheduling, and memory-allocation decisions. We formulate the inference server as a stochastic decision system incorporating prefill and decode phases, continuous batching, finite accelerator memory, and dynamically evolving KV-cache demand. We develop online execution policies using queueing-theoretic and stochastic optimization techniques and analyze their stability, latency, throughput, and service-level characteristics. Their performance is compared against offline optimization oracles through optimality-gap and regret-based measures. Particular attention is given to tail latency and workload regimes in which throughput-oriented policies induce queue instability or service-level degradation. The resulting methods provide inference runtimes with principled mechanisms for making execution decisions without assuming knowledge of future workloads and establish a foundation for adaptive execution in changing production environments.

---

# Paper 4 — Adaptive Inference Execution under Non-Stationary Workloads

## Abstract

Production AI workloads are not stationary: request arrival rates, prompt lengths, generation lengths, application classes, service-level requirements, and resource availability can change substantially over time. Consequently, an execution policy optimized for one workload regime may become inefficient when the operating environment changes. This paper formulates generative AI inference as a **non-stationary online optimization problem** in which both workload characteristics and optimal execution policies evolve over time. We introduce mechanisms for detecting and responding to workload shifts and develop adaptive execution policies that dynamically modify batching, admission, scheduling, cache allocation, and resource-management decisions. Performance is characterized using dynamic regret, which measures the difference between the runtime's decisions and a sequence of time-varying optimal policies. We additionally investigate receding-horizon and model-predictive-control formulations in which short-term workload forecasts are incorporated into runtime optimization. Experiments evaluate abrupt workload changes, gradual distribution drift, mixed interactive and batch traffic, resource contention, and changing service-level objectives. The results characterize the conditions under which adaptation improves inference efficiency and quantify the cost of delayed or excessive policy adaptation. This work establishes inference execution as a dynamic control problem rather than a fixed scheduling problem.

---

# Paper 5 — Learning Inference Execution Policies from Optimization Oracles

## Abstract

Mathematical optimization can produce high-quality inference execution decisions but may be too computationally expensive to invoke at the timescales required by production inference runtimes. This paper investigates **learning to optimize inference execution**, using solutions generated by optimization algorithms as supervision for low-latency learned policies. Runtime states describing queue composition, sequence characteristics, accelerator utilization, KV-cache pressure, and service-level requirements are mapped to execution decisions including batch composition, scheduling, admission, and resource allocation. Offline optimization and model-predictive-control procedures generate expert decisions from which predictive execution policies are learned. We compare imitation-based policy learning, direct objective learning, contextual decision models, and reinforcement-learning approaches against heuristic, online-optimization, and oracle baselines. Particular attention is given to generalization across previously unseen workload distributions and hardware configurations, decision overhead, constraint satisfaction, and robustness under distribution shift. We further investigate hybrid architectures in which learned models predict workload characteristics or candidate decisions while mathematical optimization enforces feasibility. The results demonstrate how optimization and machine learning can be combined to produce inference policies that retain much of the quality of computationally expensive optimization while operating within the microsecond- to millisecond-scale decision budgets of practical inference runtimes.

---

# Paper 6 — Policy-Driven Inference Optimization across Heterogeneous and Distributed AI Systems

## Abstract

Large-scale generative AI inference increasingly spans heterogeneous accelerators, multiple model replicas, distributed KV caches, alternative model variants, and geographically or topologically distributed compute resources. In such environments, execution decisions extend beyond scheduling requests on a single accelerator to include model placement, request routing, parallelism configuration, cache placement, resource allocation, and communication-aware execution. This paper extends the inference execution policy framework to **distributed and heterogeneous inference systems**. We formulate joint routing, placement, scheduling, and execution configuration as a multi-objective resource-allocation problem over accelerators with differing compute capacity, memory, communication costs, energy characteristics, and monetary cost. The framework supports dynamic selection among model replicas, quantization levels, tensor and pipeline parallel configurations, and heterogeneous hardware resources while enforcing latency, memory, and service-level constraints. We develop hierarchical optimization policies that separate slower-timescale placement and provisioning decisions from fine-grained runtime scheduling decisions. Evaluation under heterogeneous workloads and accelerator configurations examines the Pareto frontier between latency, throughput, cost, energy consumption, and resource utilization. The resulting system demonstrates how a common policy abstraction can coordinate execution decisions across the full inference stack, extending policy optimization from individual inference engines to distributed AI-serving infrastructure.


## EXTRAS
Your strongest papers will come from the angle your thesis already takes: execution policies under changing workloads, treated formally and optimized adaptively. The basic       
   │  mechanisms (paged KV, continuous batching, chunked prefill, disaggregation, prefix caching, migration) are all published. Your contribution comes from formalizing them, from     
   │  non-stationary workloads, from what optimization can guarantee, and from where models disagree with real hardware.                                                                
   │                                                                                                                                                                                    
   │  Below are the directions OlaRT can support, grouped by your research questions. Ready means it works today with lab entries; build means some Rust first.                         
   │                                                                                                                                                                                    
   │  RQ2: how static policies behave under shifting load (measurement papers)                                                                                                          
   │                                                                                                                                                                                    
   │  1. When does the best policy change? (ready to start, then build)                                                                                                                 
   │  For each workload regime (prompt/output mix, arrival rate, burstiness, prefix sharing) and each cost model, find the best fixed setting: scheduler, token budget, chunk size,     
   │  admission. Map where the winner flips.                                                                                                                                            
   │  - This is the prerequisite for every adaptive paper, and nobody can argue with it.                                                                                                
   │  - Today's result already shows the hard part: under the default cost model, "large batches" wins everywhere. The finding is which conditions break that.                          
   │  - Builds on: scheduler-comparison, adaptive-policy-study, plus new trace generators (curriculum E3).                                                                              
   │                                                                                                                                                                                    
   │  2. The scheduling trade-off curve between tail latency and starvation. (ready)                                                                                                    
   │  SJF-style policies cut p95 but starve long requests. Measure the full curve across loads and workload mixes, then design a policy that sits on it (ageing, bounded slowdown).     
   │  - Related work exists (Sarathi-Serve, fairness schedulers like VTC), but a systematic curve with a formal starvation bound is still open.                                         
   │                                                                                                                                                                                    
   │  3. Fragmentation versus prefix reuse across page sizes. (ready)                                                                                                                   
───│  kv-page-size already measures both. Derive the expected waste analytically, show where theory and measurement disagree, and find the best page size per workload.                 
1 ○│  - A clean workshop paper, especially if you add real-model measurements.                                                                                                          
2 ○│                                                                                                                                                                                    
3 ○│  RQ1: formalizing execution policies                                                                                                                                               
   │                                                                                                                                                                                    
   │  4. A formal model of an execution policy. (ready)                                                                                                                                 
   │  OlaRT is built on exactly this structure: snapshot → policy → validated plan → observation. Write the state, actions and constraints down as a constrained MDP or control problem,
   │  then show that baselines (FIFO, SJF, watermark admission, chunked prefill) are special cases.                                                                                     
   │  - Your notes/fomalism.md is already the draft; the runtime is the evidence that the formalism is executable.                                                                      
   │  - Venue: a workshop first, then the formal core in an OR or performance venue.                                                                                                    
   │                                                                                                                                                                                    
   │  5. Joint versus separate decisions (thesis H2). (build)                                                                                                                           
   │  Does choosing admission, batching and preemption together beat optimizing each alone? Formulate it as a joint MILP over the existing parameters, compare it with coordinate-wise  
   │  tuning, and measure the gap across workloads.                                                                                                                                     
   │  - This is the "joint" story your thesis flags as potentially the most important.

RQ3: adaptive optimization (method papers)                                                                                                                                        
   │                                                                                                                                                                                    
   │  6. Online optimization by simulated replay, and when it pays. (started)                                                                                                           
   │  ReplayOptimizer exists, and the first result is a careful negative one. The paper needs:                                                                                          
   │  - regimes where the optimum moves (from direction 1);                                                                                                                             
   │  - comparison against a bandit and against an offline choice made with hindsight;                                                                                                  
   │  - regret measured over time, and its cost in CPU.                                                                                                                                 
   │                                                                                                                                                                                    
   │  Negative and conditional results make strong papers when they are well characterized.                                                                                             
   │                                                                                                                                                                                    
   │  7. How accurate must the cost model be? (ready, small build)                                                                                                                      
   │  Run the optimizer with a deliberately wrong cost model while a different one actually runs. How much error can adaptation tolerate before it does harm? Then repeat with models   
   │  calibrated on CPU against WGPU.                                                                                                                                                   
   │  - Nobody characterizes this well, and every simulator-guided system depends on it.                                                                                                
   │                                                                                                                                                                                    
   │  8. Stable adaptation. (ready)                                                                                                                                                     
   │  Today's runs showed small replay windows drifting off a good setting on noise. Study the switching margin, window size and rate limits formally: oscillation, recovery time, and  
   │  conditions for not switching wrongly.                                                                                                                                             
   │  - This is control theory applied to policy switching, and fits your mathematics plan well.                                                                                        
   │                                                                                                                                                                                    
───│  9. Online optimization with service-level guarantees. (build)                                                                                                                     
1 ○│  Minimize cost subject to latency targets (AdaptiveConstraints plus a constrained objective), with feasibility or regret bounds.                                                   
2 ○│  - This is the most "PhD-shaped" method paper in the list.                                                                                                                         
3 ○│                                                                                                                                                                                    
   │  RQ4: learned policies                                                                                                                                                             
   │                                                                                                                                                                                    
   │  10. Do learned policies transfer? (big build)                                                                                                                                     
   │  Train on one workload or cost model, test on another (thesis H3). The simulator and seeded traces give you unlimited training data. The scientific contribution is the transfer   
   │  protocol and its failure modes, not the reinforcement-learning algorithm.                                                                                                         
   │                                                                                                                                                                                    
   │  Cluster inference                                                                                                                                                                 
   │                                                                                                                                                                                    
   │  11. Routing under heterogeneity and imperfect information. (ready)                                                                                                                
   │  - Workers of unequal capacity (workers=8,4,2,2), stale load summaries, network delay.                                                                                             
   │  - Compare power-of-d choices, KV-headroom routing and prefix affinity.                                                                                                            
   │  - Known theory (Mitzenmacher's power of two choices) gives you analytical predictions to test against.

12. When to transfer KV state and when to recompute. (build)                                                                                                                      
   │  OlaRT now ships KV snapshots as binary frames and has a transfer-or-recompute cost model. Measure the break-even point against link bandwidth, sequence length and model size, on 
   │  real links.                                                                                                                                                                       
   │  - Llumnix and Mooncake raise this question but don't map it cleanly.                                                                                                              
   │                                                                                                                                                                                    
   │  13. Online cluster weights. (build)                                                                                                                                               
   │  Carry the replay idea up a level, using run_cluster_trial, and compare it with the fixed weights chosen offline. This gives your hierarchical-optimization story its second level.
   │                                                                                                                                                                                    
   │  Methodology papers                                                                                                                                                                
   │                                                                                                                                                                                    
   │  14. Simulation versus reality. (needs a model and hardware)                                                                                                                       
   │  How well does the virtual-time simulator, calibrated by cost-calibration, predict real latency distributions, and where does it fail?                                             
   │  - This paper validates everything else you publish from the simulator, and reviewers will ask for it.                                                                             
   │                                                                                                                                                                                    
   │  15. Kernels and occupancy. (needs a working GPU)                                                                                                                                  
   │  Paged-attention occupancy against page size and batch shape, with a roofline analysis, using the Nsight workflow already built.                                                   
───│                                                                                                                                                                                    
1 ○│  How to sequence them                                                                                                                                                              
2 ○│                                                                                                                                                                                    
3 ○│  1. First six months: workshop papers from what's ready (directions 1, 3, 11), each a series of lab entries. Workshops such as EuroMLSys and the MLSys workshops welcome           
   │     characterization work.                                                                                                                                                         
   │  2. Thesis core: directions 4 (formalism), 6 and 7 (adaptive optimization and cost-model fidelity), 9 (guarantees). Aim at MLSys or EuroSys for the systems side, and at           
   │     performance or OR venues (SIGMETRICS, IFIP Performance, operations-research journals) for the formal side.                                                                     
   │  3. Validation throughout: direction 14. Systems venues will want real-hardware numbers, so getting a working GPU and a real model checkpoint is the most important practical step.
   │     Right now neither is available, and nearly everything above runs only in simulation.                                                                                           
   │                                                                                                                                                                                    
   │  A realistic target is a few strong papers, each backed by a deep series of lab entries, rather than hundreds of thin ones. The lab's Follow-ups field is how one entry grows into 
   │  a paper.                                                                                                                                                                          
   │                                                                                                                                                                                    
   │  To start, direction 1 is the foundation the adaptive papers depend on, and it starts with curriculum E3. I can also turn this list into a page in lab/, so it sits next to the    
   │  curriculum.                                                       

