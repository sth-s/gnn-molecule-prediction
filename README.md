# gnn-molecule-prediction
Molecular toxicity prediction (Tox21) using GCN/GIN and PyTorch Geometric

## Description
This project implements a pipeline for predicting molecular toxicity from the Tox21 dataset using Graph Neural Networks (GCN/GIN) based on PyTorch Geometric.

## Structure
- `notebooks/` — Jupyter Notebook for Colab with steps from setup to final training and evaluation.
- `src/` — (optional) module with model implementation, data loading and processing functions.
- `data/` — (optional) dataset files.
- `requirements.txt` — list of dependencies.
- `.gitignore` — files and folders that should not be committed.

## Installation and Usage

### Option 1: Using Google Colab
1. Clone the repository:
   ```bash
   git clone <URL>
   cd gnn-molecule-prediction
   ```

2. Open notebooks/gcn_tox21.ipynb in Google Colab:
   - File → Open notebook → GitHub → Enter repository URL

3. Run the cells to install dependencies and train the model.

### Option 2: Using Conda (Recommended for Local Development)
1. Clone the repository:
   ```bash
   git clone <URL>
   cd gnn-molecule-prediction
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate gnn-tox21
   ```

3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```

4. Open notebooks/gcn_tox21.ipynb and run the cells.
