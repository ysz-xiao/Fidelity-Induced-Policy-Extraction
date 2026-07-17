# Fidelity-Induced Policy Extraction (FIPE)

Official implementation of the paper **"Extracting Fidelity Rules From Trained Deep Reinforcement Learning in the StarCraft Multi-Agent Challenge"**.

## Overview

Existing Interpretable Policy Extraction (IPE) methods often produce **inconsistent explanations**: the extracted rule-based policy disagrees with the original DRL agent's decisions, undermining trust and interpretability.

**FIPE** addresses this by introducing a **fidelity measurement** directly into the reinforcement learning feedback loop. The key ideas are:

1. **Fidelity-Induced Sampling** — During experience collection, the agent's policy is a mixture of the teacher and the current student: $$\pi_i = \beta \pi^* + (1 - \beta) \hat{\pi}_i$$ where $\beta$ controls the teacher-student mix rate. Experiences are weighted by action consistency between student and teacher, prioritizing states where they disagree.

2. **Policy Competition** — A library of candidate policies (max size $M$) is maintained. After each iteration, all policies are evaluated on a small batch and ranked by $$\text{metric} = \lambda \cdot \text{win\_rate} + (1 - \lambda) \cdot \text{fidelity}$$ Underperformers are pruned, keeping only the best $M$ policies.

3. **Iterative Refinement** — Over successive rounds, the student policy improves its fidelity to the teacher while maintaining task performance.

## Paper

- **Title**: Extracting Fidelity Rules From Trained Deep Reinforcement Learning in the StarCraft Multi-Agent Challenge
- **Domain**: Explainable AI (XAI) / Interpretable Policy Extraction
- **Environment**: [SMAC](https://github.com/oxwhirl/smac) (StarCraft II) + control tasks (Predator-Prey, CartPole, MountainCar, LunarLander, FlappyBird, Taxi Problem)

### Baselines

The codebase includes implementations of two baseline IPE methods:
- **DAGGER** (Dataset Aggregation) — iteratively collects new data under the current policy and retrains
- **VIPER** (Verifiable Innovative Policy Extraction via Reinforcement) — uses a probabilistic sampling strategy to prioritize states where the student is most uncertain

### Interpretable Student Models

| Model | Description | Configurations |
|-------|-------------|----------------|
| DT (Gini) | Decision tree with Gini impurity | max_depth ∈ {3, 5, 10, 12, 24} |
| DT (Entropy) | Decision tree with information gain | — |
| GBDT | Gradient boosting decision tree | — |
| KNN (Ball) | K-nearest neighbors (Ball Tree) | n ∈ {1, 10, 100} |
| KNN (Brute) | K-nearest neighbors (Brute Force) | n ∈ {1, 10, 100} |
| SVM (SVC) | Support vector classifier | probability=True |
| SVM (LinearSVC) | Linear support vector classifier | probability=True |
| GP | Gaussian process classifier | — |

## Repository Structure

```
├── our_FIPE.py               # FIPE algorithm (fidelity-induced sampling + policy competition)
├── VIPER_DAGGER.py           # DAGGER, VIPER, and FIPE baselines
├── 2s_vs_1sc_DAG_VIP.py      # DAGGER/VIPER/FIPE variants targeting 2s_vs_1sc
├── Store_EXPs.py             # Experience collection from a trained teacher
├── runner.py                 # RL training/evaluation runner
│
├── agent/                    # MARL agent implementations
├── common/                   # Shared utilities
│   ├── arguments.py          # Hyperparameter definitions
│   ├── replay_buffer.py      # Experience replay buffer
│   ├── rollout.py            # Rollout worker (including fidelity-induced rollout)
│   └── utils.py              # TD-lambda target computation, etc.
│
├── network/                  # Neural network architectures
│   ├── qmix_net.py / vdn_net.py / qtran_net.py
│   ├── coma_critic.py / maven_net.py
│   └── commnet.py / g2anet.py
│
├── policy/                   # MARL policy implementations
│   ├── iql.py / qmix.py / vdn.py / coma.py
│   ├── qtran_base.py / qtran_alt.py / maven.py
│   └── central_v.py / reinforce.py
│
├── smac/                     # SMAC environment (StarCraft II maps included)
│
└── _YSZ_/                    # Custom library
    ├── agent/                # Interpretable model implementations (DT, KNN, SVM, GP, GBDT...)
    ├── env/                  # Additional environments (CartPole, FlappyBird, PredatorPrey, etc.)
    ├── interact/             # Experience management
    ├── test/                 # Evaluation utilities (action similarity, model comparison)
    ├── trainsform/           # Dimensionality reduction (t-SNE)
    ├── visualization/        # Experiment visualization
    ├── xai/                  # Policy imitation methods
    └── _base_/               # Version metadata
```

## Requirements

```bash
pip install torch numpy matplotlib scikit-learn seaborn pandas
pip install smac
pip install pysc2
```

You also need StarCraft II installed with SMAC map packs. See the [SMAC guide](https://github.com/oxwhirl/smac) for installation instructions.

## Usage

All arguments are defined in `common/arguments.py`. Key parameters include `--map` (SMAC scenario), `--alg` (teacher MARL algorithm), and `--load_model` / `--evaluate` (evaluation mode).

### 1. Train a Teacher MARL Model

```bash
python runner.py --map=3m --alg=qmix
```

### 2. Collect Experiences

```bash
python Store_EXPs.py --map=3m --alg=qmix
```

### 3. Run FIPE (Fidelity-Induced Policy Extraction)

```bash
python our_FIPE.py --map=3m --alg=qmix --evaluate=True --load_model=True
```

To change the student model, modify `xai_model_name` in `XAI_main()` inside the script.

### 4. Run Baselines (DAGGER / VIPER)

```bash
python VIPER_DAGGER.py --map=3m --alg=qmix --evaluate=True --load_model=True
```

Uncomment the desired trainer in `XAI_main()`:
- `DAGGER_trainer(...)` — DAGGER baseline
- `VIPER_trainer(...)` — VIPER baseline
- `OUR_trainer(...)` — FIPE (fidelity-induced sampling)

### 5. Map-Specific Variants

```bash
python 2s_vs_1sc_DAG_VIP.py --map=2s_vs_1sc --alg=qmix --evaluate=True --load_model=True
```

## Evaluation Metrics

- **Win Rate** — proportion of episodes won
- **Fidelity** — proportion of actions where the student agrees with the teacher
- **Reward** — cumulative episode return

During policy competition, the combined metric is: $$\text{score} = \lambda \cdot \text{win\_rate} + (1 - \lambda) \cdot \text{fidelity}$$

## SMAC Maps Tested

The paper evaluates on four SMAC maps:
- `3m` — 3 marines vs 3 marines (symmetric, small-scale)
- `2s_vs_1sc` — 2 stalkers vs 1 spine crawler (asymmetric, error-sensitive)
- `8m` — 8 marines vs 8 marines (high-dimensional)
- `3s_vs_5z` — 3 stalkers vs 5 zealots (hard asymmetric, tests generalization)

Non-SMAC control tasks include Predator-Prey, CartPole, MountainCar, LunarLander, FlappyBird, and Taxi Problem.


### Related Work
- [DAGGER: Dataset Aggregation](https://www.cs.cmu.edu/~sross1/publications/Ross-AISTATS11.pdf) — Ross et al., AISTATS 2011
- [VIPER: Verifiable Innovative Policy Extraction via Reinforcement](https://arxiv.org/abs/1902.10146) — Bastani et al., ICML 2018
- [SMAC: StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac) — Samvelyan et al., 2019
- [PyMARL](https://github.com/oxwhirl/pymarl) — Value-based MARL framework

## License

This project is built upon open-source MARL benchmarks and is intended for research purposes.
