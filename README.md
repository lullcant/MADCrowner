# MADCrowner:Margin Aware Dental Crown Design with TemplateDeformation and Refinement  <img src="./Assets/Madcrowner.jpg" alt="icon" width="48" height="48" style="vertical-align:-4px;margin-right:6px;">
The official Repo for MADCrowner:Margin Aware Dental Crown Design with Template Deformation and Refinement:

![Main Figure](./Assets/graphical_abs.png)
## Repo Structure
- `train_crown_deformer.py`
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

## Contact Information
Since the paper is under review, please contact 1155230127@link.cuhk.edu.hk or mcncaa219040@gmail.com if you want the complete code.

