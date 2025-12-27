# BrainGNN - Graph Neural Networks for Brain Connectivity Classification

A comprehensive implementation of Graph Neural Networks (GNNs) for classifying brain connectivity patterns from fMRI data. This project implements and compares multiple GNN architectures (GCN, GAT, GIN, GraphSAGE, and Hybrid) for distinguishing between child and adult brain connectivity patterns.

## Features

- **Multiple GNN Architectures**: GCN, GAT, GIN, GraphSAGE, and Hybrid models
- **Rich Feature Extraction**: 14 comprehensive node features from fMRI time series
- **Adaptive Edge Thresholding**: Dynamic graph construction based on correlation strength
- **Advanced Training**: Focal loss for class imbalance, early stopping, learning rate scheduling
- **Comprehensive Evaluation**: 5-fold cross-validation with multiple metrics
- **Data Preprocessing**: Automated fMRI data fetching and preprocessing pipeline

## Project Structure

```
braingnn-project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py
│   │   ├── gat.py
│   │   ├── gin.py
│   │   ├── sage.py
│   │   └── hybrid.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── features.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── notebooks/
    └── exploration.ipynb
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ssmaheswar2001/BrainGraphGNN.git
cd braingnn-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download and Preprocess Data

```bash
python scripts/download_data.py
```

This will:
- Download the development fMRI dataset from nilearn
- Extract time series using MSDL atlas
- Compute connectivity matrices (correlation, partial correlation, tangent)
- Save processed data to `dataset/` directory

### 2. Train Models

Train all architectures with 5-fold cross-validation:

```bash
python scripts/train.py
```

Train a specific architecture:

```bash
python scripts/train.py --architecture GCN --epochs 100 --lr 0.001
```

Available architectures: `GCN`, `GAT`, `GIN`, `GraphSAGE`, `Hybrid`

### 3. Evaluate Models

```bash
python scripts/evaluate.py --model-path models/best_model.pt
```

### 4. Make Predictions

```bash
python scripts/predict.py --model-path models/best_model.pt --data-idx 0
```

## Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Training settings
- Data preprocessing options
- Cross-validation parameters

## Model Architectures

### GCN (Graph Convolutional Network)
- Standard graph convolutions with residual connections
- Multi-scale pooling (mean + max + sum)
- Deep classification head

### GAT (Graph Attention Network)
- Multi-head attention mechanism
- Edge attention weights
- Adaptive feature aggregation

### GIN (Graph Isomorphism Network)
- Expressive MLPs for message passing
- Sum pooling (standard for GIN)
- Provably more powerful than GCN

### GraphSAGE
- Neighborhood sampling and aggregation
- Scalable to large graphs
- Multiple aggregation strategies

### Hybrid
- Combines GCN, GAT, GIN, and SAGE layers
- Leverages strengths of each architecture
- Typically achieves best performance

## Results

Example results from 5-fold cross-validation:

| Architecture | Accuracy | F1 Score | AUC |
|-------------|----------|----------|-----|
| Hybrid      | 0.9200   | 0.9180   | 0.9650 |
| GAT         | 0.9000   | 0.8950   | 0.9500 |
| GCN         | 0.8900   | 0.8850   | 0.9400 |
| GIN         | 0.8850   | 0.8800   | 0.9350 |
| GraphSAGE   | 0.8800   | 0.8750   | 0.9300 |

## Dataset

This project uses the development fMRI dataset from nilearn:
- **Subjects**: 155 participants (children and adults)
- **Atlas**: MSDL (39 ROIs)
- **Task**: Resting-state fMRI
- **Classes**: Child vs Adult

## Citation

If you use this code in your research, please cite:

```bibtex
@software{braingnn2024,
  title={BrainGNN: Graph Neural Networks for Brain Connectivity Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/ssmaheswar2001/BrainGraphGNN.git}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [ssmaheswar2001@gmail.com]