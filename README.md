# MADCrowner

This folder contains a curated subset of DCrownFormer for training and inference:

- `train_crown_deformer_final.py`
- `inference.py`
- `run.sh`
- `test.sh`
- `fdi_template/`
- `mydataset/`
- `models/`

## Setup

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
