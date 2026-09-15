import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class StateProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x.float())


class ActionFlowMatchingHead(nn.Module):
    """Temporal action expert with Transformer denoiser for flow matching."""

    def __init__(
        self,
        emb_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        goal_conditioned: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.goal_conditioned = bool(goal_conditioned)

        in_dim = action_dim + emb_dim * (2 if goal_conditioned else 1) + 1
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        # Start from near-zero velocity prediction for stable early training.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _sinusoidal_pos_emb(length: int, dim: int, device, dtype):
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
        half = max(dim // 2, 1)
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device, dtype=dtype))
            * torch.arange(half, device=device, dtype=dtype)
            / max(half - 1, 1)
        )
        args = pos * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.size(1) < dim:
            emb = F.pad(emb, (0, dim - emb.size(1)))
        return emb[:, :dim].unsqueeze(0)  # (1, T, D)

    def _predict_velocity_seq(
        self,
        a_t: torch.Tensor,           # (B, T, A)
        emb_t: torch.Tensor,         # (B, T, D)
        t: torch.Tensor,             # (B, T, 1)
        goal_emb: torch.Tensor | None = None,  # (B, T_g, D)
    ) -> torch.Tensor:
        if self.goal_conditioned:
            if goal_emb is None:
                goal_emb = torch.zeros_like(emb_t[:, :1]).expand_as(emb_t)
            elif goal_emb.size(1) != emb_t.size(1):
                goal_emb = goal_emb.expand(-1, emb_t.size(1), -1)
            token = torch.cat([a_t, emb_t, goal_emb, t], dim=-1)
        else:
            token = torch.cat([a_t, emb_t, t], dim=-1)
        h = self.in_proj(token)
        h = h + self.t_proj(t)
        h = h + self._sinusoidal_pos_emb(h.size(1), h.size(2), h.device, h.dtype)
        # Causal mask: position i may only attend to positions <= i. This keeps
        # train/eval consistent — at train time the head sees a sequence of
        # T_model frames; at eval time it sees T=1 (the current observation),
        # which equals training position 0 under the causal mask.
        T = h.size(1)
        if T > 1:
            causal_mask = torch.triu(
                torch.ones(T, T, device=h.device, dtype=torch.bool),
                diagonal=1,
            )
            h = self.transformer(h, mask=causal_mask, is_causal=True)
        else:
            h = self.transformer(h)
        h = self.out_norm(h)
        return self.out_proj(h)

    @staticmethod
    def loss(
        action_head: "ActionFlowMatchingHead",
        emb: torch.Tensor,       # (B, T, D)
        action: torch.Tensor,    # (B, T, A)
        valid: torch.Tensor,     # (B, T)
        goal_emb: torch.Tensor | None = None,  # (B, T_g, D)
        num_time_samples: int = 1,
    ) -> torch.Tensor:
        """Standard flow matching objective on action trajectories."""
        B, T, D = emb.shape
        A = action.size(-1)
        K = max(int(num_time_samples), 1)

        emb_k = emb.unsqueeze(1).expand(B, K, T, D).reshape(B * K, T, D)
        action_k = action.unsqueeze(1).expand(B, K, T, A).reshape(B * K, T, A)
        valid_k = valid.unsqueeze(1).expand(B, K, T).reshape(B * K, T)

        goal_emb_k = None
        if goal_emb is not None:
            Tg = goal_emb.size(1)
            goal_emb_k = goal_emb.unsqueeze(1).expand(B, K, Tg, D).reshape(B * K, Tg, D)

        x0 = torch.randn_like(action_k)
        t_scalar = torch.rand(B * K, 1, 1, device=action.device, dtype=action.dtype)
        t = t_scalar.expand(B * K, T, 1)

        a_tau = (1.0 - t) * x0 + t * action_k
        v_target = (action_k - x0).detach()

        v_pred = action_head._predict_velocity_seq(a_tau, emb_k, t, goal_emb=goal_emb_k)

        per_token = (v_pred - v_target).pow(2).mean(dim=-1)
        weighted = per_token * valid_k.float()
        return weighted.sum() / valid_k.float().sum().clamp(min=1)

    @torch.no_grad()
    def sample(
        self,
        emb: torch.Tensor,  # (B, T, D)
        goal_emb: torch.Tensor | None = None,  # (B, T_g, D)
        num_flow_steps: int = 16,
        noise_scale: float = 1.0,
        clamp_value: float = None,
    ) -> torch.Tensor:
        """Inference-time action sampling (no grad)."""
        return self._sample_impl(
            emb,
            goal_emb=goal_emb,
            num_flow_steps=num_flow_steps,
            noise_scale=noise_scale,
            clamp_value=clamp_value,
        )

    def sample_train(
        self,
        emb: torch.Tensor,  # (B, T, D)
        goal_emb: torch.Tensor | None = None,  # (B, T_g, D)
        num_flow_steps: int = 16,
        noise_scale: float = 1.0,
        clamp_value: float = None,
    ) -> torch.Tensor:
        """Training-time differentiable action sampling."""
        return self._sample_impl(
            emb,
            goal_emb=goal_emb,
            num_flow_steps=num_flow_steps,
            noise_scale=noise_scale,
            clamp_value=clamp_value,
        )

    def _sample_impl(
        self,
        emb: torch.Tensor,
        num_flow_steps: int,
        noise_scale: float,
        clamp_value: float | None,
        goal_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the learned velocity field with Euler steps."""
        B, T, D = emb.shape
        A = self.action_dim

        steps = max(int(num_flow_steps), 1)
        dt = 1.0 / float(steps)

        action = noise_scale * torch.randn(B, T, A, device=emb.device, dtype=emb.dtype)

        for i in range(steps):
            t_scalar = (i + 0.5) / float(steps)
            t = torch.full((B, T, 1), t_scalar, device=emb.device, dtype=emb.dtype)
            # Detach action between Euler steps: avoids BPTT through the full
            # integration chain. Gradient flows only through the current step's
            # velocity prediction to the action head parameters.
            v = self._predict_velocity_seq(action.detach(), emb, t, goal_emb=goal_emb)
            action = action.detach() + dt * v

        if clamp_value is not None:
            action = action.clamp(min=-float(clamp_value), max=float(clamp_value))

        return action


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
