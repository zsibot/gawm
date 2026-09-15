# GAWM

GAWM (Geometry-Aware World Model) is the official implementation accompanying [Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning](https://arxiv.org/abs/2609.03565). It learns geometry-aware, action-conditioned visual dynamics through physical-state prediction and inverse dynamics. At inference time, GAWM uses CEM to optimize action sequences based on costs in the learned latent space.

## Highlights

- Geometry-aware representations learned through physical-state alignment and inverse dynamics.
- Lightweight JEPA-style latent world model for visual robotic planning.
- GAWM-Enhanced Cube checkpoint reaching 96% success over 50 evaluation episodes.

## Installation

GAWM uses Python 3.10, a CUDA-capable GPU, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zsibot/gawm.git
cd gawm
uv sync --frozen
```

Configure the data location and HDF5 runtime:

```bash
export STABLEWM_HOME="$HOME/.stable-wm"
export HDF5_PLUGIN_PATH="$(uv run python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')"
```

## Data and checkpoints

Download and decompress the datasets from the official [LeWM collection](https://huggingface.co/collections/quentinll/lewm):

- [Cube](https://huggingface.co/datasets/quentinll/lewm-cube)
- [PushT](https://huggingface.co/datasets/quentinll/lewm-pusht)
- [Reacher](https://huggingface.co/datasets/quentinll/lewm-reacher)
- [TwoRoom](https://huggingface.co/datasets/quentinll/lewm-tworooms)

Download the pretrained models from the [GAWM checkpoint repository](https://huggingface.co/kikihuang/GA-WM), then use the following layout:

```text
$STABLEWM_HOME/
├── ogbench/
│   └── cube_single_expert.h5
├── pusht_expert_train.h5
├── reacher.h5
├── tworoom.h5
└── models/
    ├── cube/gawm_cube_object.ckpt
    ├── cube_enhanced/gawm_cube_bcinit_object.ckpt
    ├── pusht/gawm_pusht_object.ckpt
    ├── reacher/gawm_reacher_object.ckpt
    └── tworoom/gawm_tworoom_object.ckpt
```

## Training

The following commands train the four task models for 10 epochs on one GPU. Checkpoints are saved under `$STABLEWM_HOME/runs/`.

```bash
# Cube
CUDA_VISIBLE_DEVICES=0 uv run python train.py data=ogb \
  subdir=runs/cube output_model_name=gawm_cube trainer.devices=1

# PushT
CUDA_VISIBLE_DEVICES=0 uv run python train.py data=pusht \
  subdir=runs/pusht output_model_name=gawm_pusht trainer.devices=1

# Reacher
CUDA_VISIBLE_DEVICES=0 uv run python train.py data=dmc \
  subdir=runs/reacher output_model_name=gawm_reacher trainer.devices=1

# TwoRoom
CUDA_VISIBLE_DEVICES=0 uv run python train.py data=tworoom \
  subdir=runs/tworoom output_model_name=gawm_tworoom trainer.devices=1
```

## Evaluation

```bash
# Cube
CUDA_VISIBLE_DEVICES=0 uv run python eval.py --config-name=cube \
  policy=models/cube/gawm_cube

# GAWM-Enhanced Cube
CUDA_VISIBLE_DEVICES=0 uv run python eval.py --config-name=cube \
  policy=models/cube_enhanced/gawm_cube_bcinit \
  +bc_init=true +smooth_weight=1.0

# PushT
CUDA_VISIBLE_DEVICES=0 uv run python eval.py --config-name=pusht \
  policy=models/pusht/gawm_pusht

# Reacher
CUDA_VISIBLE_DEVICES=0 uv run python eval.py --config-name=reacher \
  policy=models/reacher/gawm_reacher \
  eval.dataset_name=reacher

# TwoRoom
CUDA_VISIBLE_DEVICES=0 uv run python eval.py --config-name=tworoom \
  policy=models/tworoom/gawm_tworoom
```

| Checkpoint | Success rate |
|---|---:|
| Cube | 88% |
| GAWM-Enhanced Cube | 96% |
| PushT | 98% |
| Reacher | 86% |
| TwoRoom | 100% |


## Citation

```bibtex
@article{liu2026physicallygroundedjepa,
  title={Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning},
  author={Liu, Muyuan and Huang, Yue and Liang, Zheng and Gao, Xiang},
  journal={arXiv preprint arXiv:2609.03565},
  year={2026},
  url={https://arxiv.org/abs/2609.03565}
}
```

## Acknowledgments

- [LeWM](https://github.com/lucas-maes/le-wm)
- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining)

## License

Released under the [MIT License](LICENSE).
