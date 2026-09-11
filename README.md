# SACOM
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
PyTorch implementation of **SACOM** for source-attribution-constrained
multimodal completion under modality missingness and observation degradation.
## Required Runtime Files

```text
SACOM/
├── main.py
├── sacom_model.py
├── sacom_trainer.py
├── reliability_projection.py
├── data_loader.py
├── metrics.py
├── utils.py
├── requirements.txt
└── configs/
    └── default.yaml
```

## Installation
python -m pip install -r requirements.txt

## Run the Code

Train and test:

```bash
python main.py \
  --mode all \
  --config configs/default.yaml \
  --data_path ./data/sacom_dataset.npz \
  --run_dir ./outputs/sacom
```

Train only:
```bash
python main.py \
  --mode train \
  --config configs/default.yaml \
  --data_path ./data/sacom_dataset.npz \
  --run_dir ./outputs/sacom
```
Test only:

```bash
python main.py \
  --mode test \
  --config configs/default.yaml \
  --data_path ./data/sacom_dataset.npz \
  --checkpoint ./outputs/sacom/checkpoints/best.pt \
  --run_dir ./outputs/sacom
```

