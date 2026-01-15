# GLEAM links single-cell 3D genome and cellular electrophysiology with calibrated uncertainty

GLEAM is a graph-based method for integrating multi-modal single-cell data
and predicting missing modalities, with a focus on single-cell multi-omics
(e.g., scRNA-seq, scATAC-seq, scHi-C).

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/keleslab/gleam.git
cd gleam
conda create -n gleam python==3.13
conda activate gleam
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirement.txt
pip install torch_geometric==2.7.0 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install pyg_lib==0.5.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install -e .

```

## Example

Currently, this tool is under construction as I intend to unify all codes so they can all use Python instead of R. If you want to try it now, we provide basic functions for this and feel free to use the ipynb file in the "examples" folder.



