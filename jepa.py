"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        state_probe=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.state_probe = state_probe

    def decode_state(self, emb):
        if self.state_probe is None:
            return None
        if (
            getattr(self, "state_probe_uses_pair", False)
            and emb.size(-1) == getattr(self, "state_probe_embed_dim", emb.size(-1))
        ):
            emb = torch.cat([emb, emb], dim=-1)
        return self.state_probe(emb)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if getattr(self, "decode_state_on_encode", True):
            decoded_state = self.decode_state(info["emb"])
            if decoded_state is not None:
                info["decoded_state"] = decoded_state

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # Check if we should use state-space planning via state_probe
        if getattr(self, "use_state_planning", False) and getattr(self, "state_probe", None) is not None:
            pred_state = self.decode_state(pred_emb[..., -1:, :])
            goal_state = self.decode_state(goal_emb[..., -1:, :].detach())
            cost = F.mse_loss(
                pred_state,
                goal_state,
                reduction="none",
            ).sum(dim=tuple(range(2, pred_state.ndim)))  # (B, S)
        else:
            # return last-step cost per action candidate (latent space)
            cost = F.mse_loss(
                pred_emb[..., -1:, :],
                goal_emb[..., -1:, :].detach(),
                reduction="none",
            ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        # Optional action-smoothness penalty (CEM-side regularizer).
        # action_candidates: (B, S, H, action_block * raw_action_dim).
        # Reshape to per-physical-step actions and penalize squared first
        # differences across time. Encourages CEM to prefer smooth motions
        # similar to expert demonstrations.
        smooth_w = float(getattr(self, "smooth_weight", 0.0))
        if smooth_w > 0.0:
            block = int(getattr(self, "action_block", 1))
            B, S, H, AD = action_candidates.shape
            raw_dim = AD // block
            a_phys = action_candidates.reshape(B, S, H * block, raw_dim)
            da = a_phys[..., 1:, :] - a_phys[..., :-1, :]
            smooth_cost = da.pow(2).mean(dim=(-2, -1))  # (B, S)
            cost = cost + smooth_w * smooth_cost

        return cost

    @torch.inference_mode()
    def get_bc_plan(self, info_dict: dict, horizon: int, history_size: int = 3):
        """Goal-conditioned BC head produces an open-loop action plan of length
        `horizon` by autoregressively decoding actions and rolling latents
        forward through the predictor. Returns (n_envs, horizon, eff_act_dim).
        Requires `self.bc_head` to be set.
        """
        assert getattr(self, "bc_head", None) is not None, "bc_head not set"
        device = next(self.parameters()).device

        pixels = info_dict["pixels"].to(device).float()
        goal = info_dict["goal"].to(device).float()

        # encode goal (use last frame embedding as goal vector)
        goal_enc = self.encode({"pixels": goal})
        goal_emb = goal_enc["emb"][:, -1]  # (B, D)

        # encode current obs history
        cur_enc = self.encode({"pixels": pixels})
        emb = cur_enc["emb"]  # (B, T_h, D)

        plans = []
        HS = history_size
        for _ in range(horizon):
            bc_in = torch.cat([emb[:, -1], goal_emb], dim=-1)
            a = self.bc_head(bc_in)  # (B, eff_act_dim)
            plans.append(a)

            # advance latent: feed last HS frames of emb through predictor
            emb_in = emb[:, -HS:]
            T_in = emb_in.size(1)
            # past actions unknown — pad with zeros, current action at last slot
            act_pad = torch.zeros(a.size(0), max(T_in - 1, 0), a.size(-1), device=device, dtype=a.dtype)
            act_seq = torch.cat([act_pad, a.unsqueeze(1)], dim=1)
            act_emb = self.action_encoder(act_seq)
            nxt = self.predict(emb_in, act_emb)[:, -1:]
            emb = torch.cat([emb, nxt], dim=1)

        return torch.stack(plans, dim=1).cpu()
