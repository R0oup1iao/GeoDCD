# GeoDCD: Geometric Dynamic Causal Discovery

This project implements the **GeoDCD** model. It is a dynamic causal discovery framework designed for large-scale nonlinear time-series systems.

## 🌟 Key Features

GeoDCD aims to address the limitations of traditional causal discovery methods in handling large-scale, dynamically changing systems. Its core innovations include:

1.  **Geometric-Aware**: Explicitly incorporates the physical coordinates/spatial information (`coords`) of nodes into causal modeling, utilizing **Geometric Pooling** to capture local topological structures.
2.  **Dynamic Mechanism**: Capable of not only discovering static structures but also inferring the dynamic changes of causal relationships over time (Time-varying Causal Strength).
3.  **Scalability via Grouping**: Decomposes large-scale causal graphs into local subgraphs through Hierarchical Geometric Clustering, effectively solving the computational bottleneck of Large-Scale causal discovery.

## 📂 Directory Structure

```text
.
├── data/                       # Data storage directory
├── results/                    # Experiment outputs (model weights, visualization images, GIFs)
├── scripts/                    # Execution scripts
│   ├── generate_data.py        # Standard synthetic data generation (Lorenz96/NC8/TVSEM)
│   ├── generate_cluster_data.py # [GeoDCD Specific] Generate Lorenz data with spatial cluster structures
│   └── ...
├── src/                        # Core code
│   ├── run.py                  # Training entry point
│   ├── evaluate.py             # Inference and visualization entry point
│   ├── model.py                # GeoDCD model definition (including GeometricPooler)
│   ├── dataloader.py           # Data loading
│   └── visualize.py            # Dynamic causal graph visualization
└── requirements.txt            # Dependencies
````

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1\. Data Generation

The advantages of GeoDCD are most evident on data with spatial structures. We recommend generating Cluster-Lorenz data for testing:

```bash
# Generate 4 spatial clusters, 32 nodes per cluster (Total N=128)
python scripts/generate_cluster_data.py --num_groups 4 --nodes_per_group 32 --num_replicas 1
```

### 2\. Training

Use `src/run.py` to start training. Define the hierarchical grouping structure via the `--hierarchy` parameter.

```bash
# Example: For a system with N=128, use two-level geometric compression (128 -> 32 -> 8)
accelerate launch src/run.py \
    --dataset cluster_lorenz \
    --N 128 \
    --hierarchy 32 8 \
    --epochs 100 \
    --batch_size 64 \
    --project_name "GeoDCD-Exp"
```

| Parameter | Description |
| :--- | :--- |
| `--hierarchy` | **Core Parameter**. Defines the bottom-up grouping structure. For example, `32 8` means the first layer clusters nodes into 32 Patches, and the second layer clusters them into 8. |
| `--lambda_l1` | Structural sparsity regularization coefficient, promoting the discovery of sparse causal graphs. |
| `--num_bases` | Number of bases for causal graph decomposition, used to reduce GPU memory usage. |

### 3\. Evaluation & Visualization

Use `src/evaluate.py` to perform inference and generate static structural plots and dynamic evolution GIFs.

```bash
accelerate launch src/evaluate.py \
    --dataset cluster_lorenz \
    --model_path ./results/cluster_lorenz/YYYYMMDD_HHMMSS/model.pth \
    --N 128 \
    --hierarchy 32 8
```

**Outputs (located in `results/`):**

  * `result_static.png`: Visualizes the **Geometric Windows** learned by the model and the static causal skeleton.
  * `causal_evolution.gif`: Dynamically displays the changes in causal strength over time, verifying the model's **Dynamic** capability.
  * `est_dynamic.npy`: The inferred complete dynamic causal matrix.

## 📝 Citation

If you use GeoDCD in your research, please cite our work:

```
```