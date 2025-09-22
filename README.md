# MADCrowner

The official Repo for Arxiv 2025 paper MADCrowner:Margin Aware Dental Crown Design with TemplateDeformation and Refinement:

- `train_crown_deformer_final.py`
- `inference.py`
- `run.sh`
- `test.sh`
- `fdi_template/`
- `mydataset/`
- `models/`

## Setup

First setup pytorch3d https://miropsota.github.io/torch_packages_builder/pytorch3d/

Then setup the requirments
```bash
pip install -r requirements.txt
```

If using multiple GPUs, configure Accelerate first:

```bash
accelerate config
```

## Training

```bash
bash run.sh
```

## Inference / Testing

```bash
bash test.sh
```

Update paths and checkpoints inside the scripts as needed.
