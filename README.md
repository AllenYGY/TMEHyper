# TMEHyper

**Spatial Perturbation Prediction via Hypergraph-Regularized Autoencoder**

A deep learning framework for predicting perturbation responses in spatial transcriptomics by modeling tumor microenvironment (TME) interactions through hypergraph structures.

## Installation

```bash
pip install torch numpy scipy scikit-learn pandas scanpy anndata tqdm
```

Optional scGPT integration:
```bash
pip install scgpt
# Download pretrained model to scGPT/save/scGPT_human/
```

## Usage

```bash
# Basic training
python -m sp_hyperrae.main --data your_data.h5ad

# Full model with all innovations
python -m sp_hyperrae.main --data your_data.h5ad --use_spectral --use_contrast

# Simplified mode (no hypergraph)
python -m sp_hyperrae.main --data your_data.h5ad --simple
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Input h5ad file | Required |
| `--epochs` | Training epochs | 50 |
| `--batch_size` | Batch size | 64 |
| `--lr` | Learning rate | 1e-4 |
| `--use_spectral` | Enable spectral conv (A1) | False |
| `--use_contrast` | Enable contrastive (B1-B3) | False |

### Data Format

AnnData (`.h5ad`) with:

- `adata.X`: Expression matrix
- `adata.obs['cell_type']`: Cell types
- `adata.obs['perturbation']`: Perturbation labels
- `adata.obsm['spatial']`: Coordinates (N x 2)

## Model Outputs

| Output | Dimension | Description |
|--------|-----------|-------------|
| `delta_x_pred` | [batch, G] | Predicted expression changes |
| `z_cell` | [batch, 512] | Cell embeddings (scGPT) |
| `z_tme` | [batch, 32] | TME embeddings |
| `z_tme_post` | [batch, 32] | Post-perturbation TME |
| `h_low` | [batch, 16] | Low-frequency features |
| `h_high` | [batch, 16] | High-frequency features |

## Project Structure

```txt
sp_hyperrae/
├── model.py          # Main SPHyperRAE model
├── spectral_conv.py  # Spectral hypergraph conv (A1)
├── contrastive.py    # Hyperedge contrastive (B1-B3)
├── decoder.py        # RAE decoder + TME predictor (C1)
├── hypergraph.py     # Hyperedge construction
├── losses.py         # Loss functions
├── data.py           # Data loading
├── train.py          # Training loop
└── main.py           # Entry point
```
---

## Data Source

- GSE193460 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460>
- GSE245582 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245582>
- GSE
  - <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221321>
  - <https://download.brainimagelibrary.org/0c/bd/0cbd479c521afff9/extras/tumors/>

## Acknowledgments

We thank the developers of the following projects:

- **[scGPT](https://github.com/bowang-lab/scGPT)**: Pre-trained single-cell language model
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Scanpy](https://scanpy.readthedocs.io/)**: Single-cell analysis toolkit
- **[AnnData](https://anndata.readthedocs.io/)**: Annotated data matrices
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning utilities

---
