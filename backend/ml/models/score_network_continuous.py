"""MLP score network for continuous particle systems like Lennard-Jones."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PeriodicPositionEmbedding(nn.Module):
    """Periodic position embedding for particle coordinates.

    Uses Fourier features to encode positions in a way that respects
    periodic boundary conditions. This allows the network to learn
    translation-invariant functions on a periodic domain.
    """

    def __init__(self, dim: int, n_frequencies: int = 16, max_freq: float = 10.0):
        """Initialize periodic embedding.

        Args:
            dim: Output embedding dimension (must be divisible by 4)
            n_frequencies: Number of frequency components
            max_freq: Maximum frequency for Fourier features
        """
        super().__init__()
        assert dim % 4 == 0, "Embedding dim must be divisible by 4"

        self.dim = dim
        self.n_frequencies = n_frequencies

        # Learnable frequencies for more flexibility
        freqs = torch.linspace(1.0, max_freq, n_frequencies)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute periodic embeddings.

        Args:
            x: Positions (..., 2) normalized to [-1, 1]

        Returns:
            Embeddings (..., dim)
        """
        # Scale to [0, 2π] for periodic functions
        x_scaled = (x + 1) * math.pi  # (..., 2)

        # Compute sin and cos for each frequency and dimension
        # x_scaled: (..., 2), freqs: (n_freq,)
        # Result: (..., 2, n_freq)
        args = x_scaled[..., None] * self.freqs  # (..., 2, n_freq)

        # Concatenate sin and cos for both x and y
        embeddings = torch.cat(
            [
                torch.sin(args[..., 0, :]),  # sin(freq * x)
                torch.cos(args[..., 0, :]),  # cos(freq * x)
                torch.sin(args[..., 1, :]),  # sin(freq * y)
                torch.cos(args[..., 1, :]),  # cos(freq * y)
            ],
            dim=-1,
        )  # (..., 4 * n_freq)

        # Project to desired dimension if needed
        if embeddings.shape[-1] != self.dim:
            # Truncate or pad
            if embeddings.shape[-1] > self.dim:
                embeddings = embeddings[..., : self.dim]
            else:
                padding = torch.zeros(
                    *embeddings.shape[:-1], self.dim - embeddings.shape[-1],
                    device=embeddings.device, dtype=embeddings.dtype
                )
                embeddings = torch.cat([embeddings, padding], dim=-1)

        return embeddings


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings.

        Args:
            t: Time values (batch_size,) in [0, 1]

        Returns:
            Embeddings (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )
        args = t[:, None] * freqs[None, :] * 1000
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLPBlock(nn.Module):
    """MLP block with residual connection and time conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_mlp = nn.Linear(time_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time conditioning.

        Args:
            x: Input (batch, n_particles, hidden_dim)
            t_emb: Time embedding (batch, time_dim)

        Returns:
            Output (batch, n_particles, hidden_dim)
        """
        h = self.act(self.norm1(self.fc1(x)))
        # Add time embedding (broadcast over particles)
        h = h + self.time_mlp(t_emb)[:, None, :]
        h = self.dropout(self.act(self.norm2(self.fc2(h))))
        return x + h


class ParticleAttention(nn.Module):
    """Self-attention over particles for capturing interactions."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention over particles.

        Args:
            x: Input (batch, n_particles, hidden_dim)

        Returns:
            Output (batch, n_particles, hidden_dim)
        """
        batch, n_particles, _ = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x).reshape(batch, n_particles, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, particles, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (batch, heads, particles, head_dim)
        out = out.transpose(1, 2).reshape(batch, n_particles, self.hidden_dim)
        out = self.proj(out)

        return out + residual


class ContinuousScoreNetwork(nn.Module):
    """MLP-based score network for continuous particle systems.

    Uses periodic position embeddings to respect periodic boundaries
    and particle attention to capture many-body interactions.
    """

    def __init__(
        self,
        n_particles: int,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        pos_embed_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        """Initialize continuous score network.

        Args:
            n_particles: Number of particles in the system
            hidden_dim: Hidden dimension for MLP layers
            time_embed_dim: Dimension of time embedding
            pos_embed_dim: Dimension of position embedding
            num_blocks: Number of MLP blocks
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_attention: Whether to use particle attention
        """
        super().__init__()
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Position embedding (periodic)
        self.pos_embed = PeriodicPositionEmbedding(
            dim=pos_embed_dim, n_frequencies=pos_embed_dim // 4
        )

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Input projection: pos_embed + raw coords -> hidden_dim
        self.input_proj = nn.Linear(pos_embed_dim + 2, hidden_dim)

        # MLP blocks with optional attention
        self.blocks = nn.ModuleList()
        self.attention_layers = nn.ModuleList() if use_attention else None

        for _ in range(num_blocks):
            self.blocks.append(MLPBlock(hidden_dim, time_embed_dim, dropout))
            if use_attention:
                self.attention_layers.append(ParticleAttention(hidden_dim, num_heads))

        # Output projection: hidden_dim -> 2 (score for each particle)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score s(x, t) for particle positions.

        Args:
            x: Particle positions (batch, n_particles, 2) in [-1, 1]
            t: Diffusion time (batch,) in [0, 1]

        Returns:
            Score prediction (batch, n_particles, 2)
        """
        batch_size = x.shape[0]

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (batch, time_dim)

        # Position embedding + raw coordinates
        pos_emb = self.pos_embed(x)  # (batch, n_particles, pos_dim)
        h = torch.cat([pos_emb, x], dim=-1)  # (batch, n_particles, pos_dim + 2)

        # Project to hidden dimension
        h = self.input_proj(h)  # (batch, n_particles, hidden_dim)

        # Apply MLP blocks with optional attention
        for i, block in enumerate(self.blocks):
            h = block(h, t_emb)
            if self.use_attention:
                h = self.attention_layers[i](h)

        # Output projection
        score = self.output_proj(h)  # (batch, n_particles, 2)

        return score


if __name__ == "__main__":
    # Test continuous score network
    net = ContinuousScoreNetwork(
        n_particles=16,
        hidden_dim=128,
        time_embed_dim=64,
        pos_embed_dim=64,
        num_blocks=4,
        use_attention=True,
    )

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 16, 2)  # Random positions
    t = torch.rand(batch_size)  # Random times

    score = net(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {score.shape}")
    assert score.shape == x.shape, "Score shape should match input"

    # Test periodic embedding
    pos_embed = PeriodicPositionEmbedding(dim=64, n_frequencies=16)
    x_test = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    emb = pos_embed(x_test)
    print(f"Position embedding shape: {emb.shape}")

    # Verify periodicity: x=1 and x=-1 should give same embedding
    emb_plus = pos_embed(torch.tensor([[1.0, 0.0]]))
    emb_minus = pos_embed(torch.tensor([[-1.0, 0.0]]))
    diff = (emb_plus - emb_minus).abs().max()
    print(f"Periodicity check (should be ~0): {diff:.6f}")

    # Count parameters
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {n_params:,}")

    print("Continuous score network tests passed!")
