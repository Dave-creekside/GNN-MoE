# GNN-MoE: An Evolving Architecture for Dense Expert Collaboration

[![Status](https://img.shields.io/badge/Status-Research%20&%20Development-blueviolet)](https://shields.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

This repository chronicles the evolution of a novel language model architecture, starting from a Graph-Coupled Mixture of Experts (GNN-MoE) and advancing to a highly sophisticated Adaptive Orthogonal Hypergraph-Coupled MoE (HGNN-MoE). The core innovation is a departure from traditional sparse MoE models, instead proposing a dense architecture where all experts process data and collaborate via a learned communication topology.

---

## 📖 Table of Contents

- [Architectural Vision](#-architectural-vision)
- [Project Evolution](#-project-evolution)
  - [Phase 1: GNN-MoE](#phase-1-gnn-moe---foundations)
  - [Phase 2: HGNN-MoE](#phase-2-hgnn-moe---higher-order-interactions)
  - [Phase 3: Adaptive Orthogonality](#phase-3-adaptive-orthogonality---enforcing-specialization)
- [Key Concepts](#-key-concepts)
  - [Dense vs. Sparse MoE](#dense-vs-sparse-moe)
  - [Expert Coupling: GNN vs. HGNN](#expert-coupling-gnn-vs-hgnn)
  - [Expert Specialization via Orthogonality](#expert-specialization-via-orthogonality)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Installation](#1-installation)
  - [Training the Model](#2-training-the-model)
  - [Running Inference](#3-running-inference)
- [Demonstrations](#-demonstrations)

---

## 🔭 Architectural Vision

The fundamental hypothesis of this project is that complex reasoning requires not just specialized experts, but active collaboration between them. Unlike sparse MoE architectures that route tokens to a small subset of experts, our dense approach activates all experts and enables them to communicate through a graph-based network. This allows the model to learn rich, dynamic patterns of information sharing, analogous to a team of human experts collaborating on a problem.

This vision has evolved through three major phases, each building upon the last to create a more powerful, efficient, and intelligent system.

## 🚀 Project Evolution

This project has progressed through three distinct phases, each introducing a significant architectural enhancement.

### **Phase 1: GNN-MoE - Foundations**

- **Directory:** [`gnn_MoE/`](./gnn_MoE/)
- **Core Idea:** Replace the sparse router of a traditional MoE with a dense architecture where experts are treated as nodes in a graph. A Graph Neural Network (GNN) acts as a "coupler," allowing experts to exchange information at each layer of the transformer.
- **Key Features:**
  - All experts are active for all tokens.
  - A learnable adjacency matrix allows the model to determine the optimal communication patterns between experts.
  - Established the foundational codebase for configuration, training, and analysis.

### **Phase 2: HGNN-MoE - Higher-Order Interactions**

- **Directory:** [`hgnn_MoE/`](./hgnn_MoE/)
- **Core Idea:** Evolve from GNNs to Hypergraph Neural Networks (HGNNs) to model more complex, higher-order relationships. While a GNN edge connects two experts, a hyperedge can connect an arbitrary number of experts, allowing for the modeling of multi-expert coalitions.
- **Key Features:**
  - **Richer Interactions:** Captures group-level dynamics, not just pairwise communication.
  - **Improved Scalability:** Offers potential for better memory efficiency as the number of experts grows.
  - **Static Hyperedges:** Initial implementation used static strategies like "all-pairs" or "all-triplets" to define hyperedges.

### **Phase 3: Adaptive Orthogonality - Enforcing Specialization**

- **Directory:** [`orthogon/`](./orthogon/)
- **Core Idea:** To prevent experts from learning redundant information, introduce an orthogonality constraint to their weight matrices. This encourages each expert to learn a unique "axis" in the feature space. The system was further enhanced with an adaptive controller that dynamically tunes the strength of this constraint.
- **Key Features:**
  - **Orthogonality Loss:** A soft constraint added to the main training objective that penalizes non-orthogonal expert weights.
  - **Emergent Specialization:** Experts are forced to become non-redundant, increasing the model's parameter efficiency.
  - **Adaptive Controller:** An intelligent system (`AdaptiveWeightOrthogonalityController`) that monitors expert specialization in real-time and adjusts the orthogonality constraint strength to achieve a target level of specialization, eliminating manual tuning and improving training stability.

## 🧠 Key Concepts

### Dense vs. Sparse MoE

- **Sparse MoE (Traditional):** An input token is routed to a small number of "chosen" experts (e.g., top 2). Experts work in isolation.
- **Dense MoE (This Project):** An input token is processed by *all* experts. The experts then communicate and share information through a graph coupling layer before their outputs are combined.

### Expert Coupling: GNN vs. HGNN

- **GNN Coupler:** Models pairwise relationships. Each edge in the graph connects two experts, representing a direct line of communication.
- **HGNN Coupler:** Models group relationships. Each hyperedge can connect multiple experts, representing a collaborative discussion or a shared context.

### Expert Specialization via Orthogonality

To ensure that our densely-activated experts don't all learn the same thing, we enforce mathematical orthogonality on their weight matrices.

- **Concept:** If two vectors are orthogonal, they are at a 90-degree angle, representing independent directions in space. By encouraging expert weights to be orthogonal, we push them to learn unique, non-overlapping features.
- **Adaptive Control:** The final architecture uses an intelligent controller to apply this constraint dynamically. It starts with a gentle push and adjusts the strength based on how specialized the experts are becoming, ensuring stable and effective training.

## 📁 Repository Structure

The project is organized chronologically by its major phases:

```
.
├── gnn_MoE/              # Phase 1: The original GNN-Coupled MoE.
├── hgnn_MoE/             # Phase 2: Evolution to Hypergraph-Coupled MoE.
│   └── build-log/        # Detailed logs of the development process.
└── orthogon/             # Phase 3: Introduction of Orthogonality.
    ├── project-knowledge/  # High-level documentation on the concepts.
    └── adaptive-orthogonal/ # The latest, most advanced implementation.
        ├── run_gnn_moe.py   # Main script for training runs.
        ├── gnn_moe_architecture.py # Defines all model components.
        └── ...
```

**Key Files:**
- `training_hgnn_moe.ipynb`: A Jupyter notebook for configuring and running training experiments and hyperparameter sweeps.
- `inference_hgnn_moe.ipynb`: A Jupyter notebook for loading a trained model and generating text.

## 🚀 Getting Started

### 1. Installation

The project requires Python 3.9+ and the dependencies listed in `requirements.txt`. It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch Geometric dependencies
# (Adjust for your specific CUDA version if using GPU)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

### 2. Training the Model

The easiest way to train the model is by using the `training_hgnn_moe.ipynb` notebook.

1.  **Open the Notebook:** Launch Jupyter and open `training_hgnn_moe.ipynb`.
2.  **Configure Your Run:** In the first cell, modify the `TrainingConfig` object to set your desired hyperparameters (e.g., `num_experts`, `learning_rate`).
3.  **Execute the Cells:** Run the cells in order to start the training process. The notebook also includes cells for running hyperparameter sweeps.

### 3. Running Inference

Once you have a trained model checkpoint, you can use the `inference_hgnn_moe.ipynb` notebook to generate text.

1.  **Open the Notebook:** Launch Jupyter and open `inference_hgnn_moe.ipynb`.
2.  **Set Paths:** In the second cell, update the `CHECKPOINT_PATH` and `CONFIG_PATH` variables to point to your trained model and its configuration file.
3.  **Generate Text:** In the third cell, enter your desired prompt and run the cell to see the model's output.

## 🧪 Demonstrations

The `training_hgnn_moe.ipynb` notebook contains a "Demonstration Commands" section that provides pre-configured setups to run and observe the behavior of each major architectural phase:

1.  **Standard GNN-MoE:** A baseline run with the original GNN coupler.
2.  **HGNN-MoE with Static Orthogonality:** Demonstrates the hypergraph coupler combined with a fixed-strength orthogonality loss.
3.  **HGNN-MoE with Adaptive Orthogonality:** Showcases the most advanced version of the architecture with the intelligent adaptive controller.

You can uncomment and run these demonstrations to see the different models in action.

## 📊 Performance Highlights

- **99.7% Expert Specialization** achieved with adaptive orthogonality
- **Cross-platform compatibility** (M3 MacBook, Linux with RTX 4070, Colab A100)
- **Intelligent adaptation** eliminates manual hyperparameter tuning
- **Stable training** with emergency intervention systems

## 🔬 Research Applications

This architecture serves as a foundation for exploring:
- **Hierarchical expert organization**
- **Dynamic hyperedge formation**
- **Multi-scale expert interactions**
- **Efficient scaling to larger expert counts**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see the individual README files in each phase directory for specific development guidelines.

## 📞 Contact

For questions about the architecture or implementation details, please refer to the extensive documentation in the `orthogon/project-knowledge/` directory or the build logs in `hgnn_MoE/build-log/`.

---

**GNN-MoE: Where expert collaboration meets intelligent specialization.** ✨
