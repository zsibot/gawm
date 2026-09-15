import os
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from data_adapters import build_data_pipeline
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, StateProbe
from utils import ModelObjectCallBack


def _validate_state_indices(values, name, state_dim):
    """Validate indices into the final, transformed state-alignment target."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer indices")
    try:
        indices = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integer indices") from error
    for index in indices:
        if type(index) is not int:
            raise ValueError(f"{name} entries must be integers, got {index!r}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate indices: {indices}")
    invalid = [index for index in indices if index < 0 or index >= state_dim]
    if invalid:
        raise ValueError(
            f"{name} contains invalid indices {invalid} for state_align_dim={state_dim}"
        )
    return indices


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    if "state_align" in batch:
        batch["state_align"] = torch.nan_to_num(batch["state_align"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds : n_preds + ctx_len]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    if lambd>0.0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    else:
        output["loss"] = output["pred_loss"]

    state_align_cfg = cfg.loss.get("state_align")
    if state_align_cfg is not None and float(state_align_cfg.get("weight",0.0))>0.0:
        state_target = output["state_align"]

        state_dim = state_target.size(-1)
        supervised_indices = _validate_state_indices(
            state_align_cfg.get("supervised_indices"),
            "state_align.supervised_indices",
            state_dim,
        )
        if not supervised_indices:
            raise ValueError(
                "state_align.supervised_indices must be a non-empty list"
            )
        velocity_indices = _validate_state_indices(
            state_align_cfg.get("velocity_indices", []),
            "state_align.velocity_indices",
            state_dim,
        )

        prev_emb = torch.cat([emb[:, :1], emb[:, :-1]], dim=1)
        pair = torch.cat([prev_emb, emb], dim=-1)

        # Select the physical target dimensions used by the loss. Velocity
        # dimensions are skipped only for the dummy t=0 pair [z0, z0], which
        # has no temporal evidence.
        state_loss_mask = torch.zeros_like(state_target, dtype=torch.bool)

        supervised_idx = torch.as_tensor(
            supervised_indices, device=emb.device, dtype=torch.long
        )
        state_loss_mask[..., supervised_idx] = True

        if velocity_indices:
            velocity_idx = torch.as_tensor(
                velocity_indices,
                device=emb.device,
                dtype=torch.long,
            )
            state_loss_mask[:, 0, velocity_idx] = False

        decoded_state = self.model.decode_state(pair)

        state_err = (decoded_state - state_target).pow(2)
        masked_state_err = state_err.masked_fill(~state_loss_mask, 0.0)

        num_supervised_terms = state_loss_mask.sum().to(state_err.dtype)
        state_loss = masked_state_err.sum() / num_supervised_terms

        output["decoded_state"] = decoded_state
        output["state_emb_loss"] = state_loss
        output["loss"] += state_align_cfg.weight * state_loss

    # Inverse Dynamics Model loss: from (emb_t, emb_{t+1}) recover action_t.
    # Complements state_align ("geometric decodability") with
    # "action / controllability decodability" of the latent.
    idm_cfg = cfg.loss.get("idm")
    if (
        idm_cfg is not None
        and float(idm_cfg.get("weight", 0.0)) > 0.0
    ):
        T_total = emb.size(1)
        if T_total >= 2:
            # pairs over the full window: (B, T-1, 2*D)
            pair = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)
            B_, Tm1, D2 = pair.shape
            pred_act = self.model.idm_head(pair.reshape(B_ * Tm1, D2))
            pred_act = pred_act.view(B_, Tm1, -1)
            tgt_act = batch["action"][:, : Tm1].to(pred_act.dtype)
            # action tensor is (B, T, effective_act_dim) where
            # effective_act_dim = frameskip * action_dim (matches Embedder input)
            output["idm_loss"] = F.mse_loss(pred_act, tgt_act)
            output["loss"] = output["loss"] + float(idm_cfg.weight) * output["idm_loss"]
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    state_align_cfg = cfg.loss.get("state_align")

    data_pipeline = build_data_pipeline(
        dataset,
        data_config=cfg.data,
        img_size=cfg.img_size,
    )

    with open_dict(cfg):
        cfg.wm.action_dim = data_pipeline.action_dim
        cfg.wm.state_align_dim = data_pipeline.state_alignment.output_dim

    dataset.transform = data_pipeline.transform

    split_generator = torch.Generator().manual_seed(cfg.split_seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=split_generator,
    )

    train_generator = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    validation_rows = []
    for dataset_index in val_set.indices:
        episode, start_step = dataset.clip_indices[dataset_index]
        validation_rows.append(dataset.offsets[episode] + start_step)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    # Optional state-alignment head
    if state_align_cfg is not None and float(state_align_cfg.get("weight",0.0))>0.0:
        world_model.state_probe = StateProbe(
            input_dim=embed_dim * 2,
            hidden_dim=int(state_align_cfg.get("hidden_dim",256)),
            output_dim=cfg.wm.state_align_dim,
        )
        world_model.decode_state_on_encode = False
        world_model.state_probe_uses_pair = True
        world_model.state_probe_embed_dim = embed_dim


    # Optional inverse-dynamics head
    idm_cfg = cfg.loss.get("idm")
    if idm_cfg is not None and float(idm_cfg.get("weight", 0.0)) > 0.0:
        world_model.idm_head = MLP(
            input_dim=embed_dim * 2,
            output_dim=effective_act_dim,
            hidden_dim=int(idm_cfg.get("hidden_dim", 512)),
            norm_fn=torch.nn.LayerNorm,
        )


    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "validation_rows.npy", validation_rows)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

#    object_dump_callback = ModelObjectCallBack(
#        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
#    )

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=cfg.trainer.max_epochs,
    )


    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
       # enable_checkpointing=True,
        enable_checkpointing=False,
    )

    # Optional: warm-start from a previously-trained `_weights.ckpt` without
    # restoring optimizer / scheduler / epoch state. This enables short
    # fine-tunes (e.g. +1 epoch on a new loss) that start from a baseline.
    init_from = cfg.get("init_from", None)
    if init_from:
        init_path = Path(os.path.expanduser(str(init_from)))
        print(f"[init_from] Loading weights from {init_path}")
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = world_model.load_state_dict(sd, strict=False)
        print(f"[init_from] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  first missing: {missing[:5]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:5]}")


    resume_from = cfg.get("resume_from", None) #None

    if resume_from:
     resume_from = Path(os.path.expanduser(str(resume_from)))
     print(f"[resume_from] Resuming full trainer state from {resume_from}")

    manager = spt.Manager(
     trainer=trainer,
     module=world_model,
     data=data_module,
     seed=cfg.seed,
     ckpt_path=resume_from,
    )

    manager()


    return


if __name__ == "__main__":
    run()
