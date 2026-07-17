# Fidelity-Induced Policy Extraction (FIPE)

Official code for extracting interpretable policies from trained multi-agent reinforcement learning (MARL) models, built on the [SMAC](https://github.com/oxwhirl/smac) benchmark.

## Abstract

FIPE investigates how to distill policy knowledge from complex MARL models (QMIX, VDN, COMA, etc.) into interpretable ML models such as decision trees, KNN, SVM, GBDT, and Gaussian processes — balancing **fidelity** (how closely the extracted policy matches the original) against **task performance** (win rate).

Key components:
1. Train state-of-the-art MARL algorithms on StarCraft II micromanagement scenarios
2. Collect interaction experiences from the teacher model
3. Train interpretable surrogate policies via behavioral cloning
4. Optimize the fidelity–performance trade-off through model selection and ensembling
5. Support iterative distillation with DAgger-style data aggregation

## Environment

- **SMAC** (StarCraft Multi-Agent Challenge) — decentralized micromanagement scenarios in StarCraft II
- Supported maps: `3m`, `8m`, `2s3z`, `2s_vs_1sc`, `5m_vs_6m`, and more (see `smac/env/starcraft2/maps/smac_maps.py`)

## Algorithms

### Teacher MARL Policies
- [IQL](https://arxiv.org/abs/1511.08779)
- [VDN](https://arxiv.org/abs/1706.05296)
- [QMIX](https://arxiv.org/abs/1803.11485)
- [COMA](https://arxiv.org/abs/1705.08926)
- [QTRAN](https://arxiv.org/abs/1905.05408) (base / alt)
- [MAVEN](https://arxiv.org/abs/1910.07483)
- [CommNet](https://arxiv.org/abs/1605.07736)
- [G2ANet](https://arxiv.org/abs/1911.10715)
- Central-V / REINFORCE (external training algorithms for CommNet/G2ANet)

### Interpretable Surrogate Models
| Model | File | Description |
|-------|------|-------------|
| DT (Gini) | `_YSZ_/agent/DT_Gini.py` | Decision tree with Gini impurity, max_depth ∈ {3,5,10,12,24} |
| DT (Entropy) | `_YSZ_/agent/DT_Entropy.py` | Decision tree with information gain |
| GBDT | `_YSZ_/agent/DT_GBDT.py` | Gradient boosting decision tree |
| KNN (Ball) | `_YSZ_/agent/KNN_Ball.py` | K-nearest neighbors (Ball Tree), n ∈ {1,10,100} |
| KNN (Brute) | `_YSZ_/agent/KNN_Brute.py` | K-nearest neighbors (Brute Force), n ∈ {1,10,100} |
| SVM (SVC) | `_YSZ_/agent/SVM_SVC.py` | Support vector classifier with probability estimates |
| SVM (Linear) | `_YSZ_/agent/SVM_LinearSVC.py` | Linear support vector classifier with probability estimates |
| GP | `_YSZ_/agent/GP.py` | Gaussian process classifier |

## Repository Structure

```
├── our_FIPE.py               # Main entry: Fidelity-Induced Policy Extraction
├── VIPER_DAGGER.py           # VIPER + DAgger iterative distillation
├── 2s_vs_1sc_DAG_VIP.py      # DAG + VIP variant (targeting specific maps)
├── Store_EXPs.py             # Experience collection and storage
├── runner.py                 # Unified training and evaluation runner
│
├── agent/                    # MARL agent definitions
├── common/                   # Shared components
│   ├── arguments.py          # Hyperparameters for all algorithms
│   ├── replay_buffer.py      # Experience replay buffer
│   ├── rollout.py            # Rollout worker
│   └── utils.py              # Utility functions (td-lambda targets, etc.)
│
├── network/                  # Neural network architectures
│   ├── qmix_net.py / vdn_net.py / qtran_net.py
│   ├── coma_critic.py / maven_net.py
│   └── commnet.py / g2anet.py
│
├── policy/                   # Policy implementations
│   ├── iql.py / qmix.py / vdn.py / coma.py
│   ├── qtran_base.py / qtran_alt.py / maven.py
│   └── central_v.py / reinforce.py
│
├── smac/                     # SMAC environment (including StarCraft II maps)
│
└── _YSZ_/                    # Custom utility library
    ├── agent/                # Interpretable model implementations
    ├── env/                  # Additional test environments (CartPole, FlappyBird, etc.)
    ├── interact/             # Interaction experience management
    ├── test/                 # Evaluation tools (action similarity, model comparison)
    ├── trainsform/           # Data transformation and dimensionality reduction (t-SNE)
    ├── visualization/        # Experiment visualization
    ├── xai/                  # XAI policy imitation methods
    └── _base_/               # Version metadata
```

## Quick Start

### 1. Requirements

```bash
pip install torch numpy matplotlib scikit-learn seaborn pandas
pip install smac
pip install pysc2
```

You also need StarCraft II installed with SMAC map packs. See the [SMAC installation guide](https://github.com/oxwhirl/smac) for details.

### 2. Experience Collection

Collect interaction experiences from the teacher model:

```bash
python Store_EXPs.py --map=3m --alg=qmix
```

Key arguments (all defined in `common/arguments.py`):
- `--map`: SMAC map name
- `--alg`: Teacher MARL algorithm
- `--n_steps`: Total time steps (default: 2,000,000)
- `--load_model`: Load a pretrained model
- `--evaluate`: Evaluation-only mode

### 3. FIPE: Policy Extraction

Extract interpretable policies from a trained teacher:

```bash
python our_FIPE.py --map=3m --alg=qmix --evaluate=True --load_model=True
```

### 4. VIPER + DAgger Iterative Distillation

Iteratively expand the experience pool and refine the interpretable model:

```bash
python VIPER_DAGGER.py --map=2s_vs_1sc --alg=qmix --evaluate=True --load_model=True
```

### 5. DAG + VIP Variant

```bash
python 2s_vs_1sc_DAG_VIP.py --map=2s_vs_1sc --alg=qmix --evaluate=True --load_model=True
```

## Evaluation Metrics

- **Win Rate**: Task completion rate
- **Reward**: Cumulative episode reward
- **Fidelity**: Action agreement between student and teacher policies
- **Model Selection**: In iterative settings, the top-K models are retained by `win_rate + fidelity` score

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{...,
  title={...},
  author={...},
  year={...}
}
```

### MARL Foundations
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [VDN: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [COMA: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [MAVEN: Multi-Agent Variational Exploration](https://arxiv.org/abs/1910.07483)

### Interpretable / Verifiable Policy Extraction
- [VIPER: Verifiable Innovative Policy Extraction via Reinforcement](https://arxiv.org/abs/1902.10146)
- [DAgger: Dataset Aggregation](https://www.cs.cmu.edu/~sross1/publications/Ross-AISTATS11.pdf)

### Infrastructure
- [SMAC: StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac)
- [PyMARL](https://github.com/oxwhirl/pymarl)

## License

This project is built upon the open-source StarCraft MARL benchmark codebase. It is intended for research purposes only.
