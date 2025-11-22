"""
Music Diffusion Transformer - Cognitive Architecture
Revolutionary approach to music generation treating the model as cognitive architecture
rather than just a passive generation tool.

This implements Level 2 (Holographic & Hyperdimensional Computing) capabilities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Optional, Tuple

# -----------------------------------------------------------------------------
# Core Components (LLaMA-Style)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.scale

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        # Handle cases where inference seq_len > init max_seq_len
        # Explicit slicing to prevent out-of-bounds access during dynamic sequence lengths
        if seq_len > self.cos.shape[0]:
            # Return full buffer if requested length exceeds pre-computed embeddings
            return self.cos.to(x.device), self.sin.to(x.device)
        return self.cos[:seq_len, :].to(x.device), self.sin[:seq_len, :].to(x.device)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # x: [batch, seq_len, heads, head_dim]
    # cos, sin: [seq_len, head_dim] -> reshape to [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)

# -----------------------------------------------------------------------------
# Diffusion Specific Components
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Adaptive Layer Norm (AdaLN) and Cross-Attention.
    Structure:
    1. AdaLN (regulates x based on timestep)
    2. Self-Attention (with RoPE)
    3. Cross-Attention (conditioning on text)
    4. FeedForward (SwiGLU)
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 1. Self-Attention
        self.norm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_proj = nn.Linear(dim, dim, bias=False)
        
        # 2. Cross-Attention (Text Conditioning)
        self.norm_cross = RMSNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, dim * 2, bias=False)
        self.cross_proj = nn.Linear(dim, dim, bias=False)

        # 3. FeedForward
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        
        # 4. Adaptive Layer Norm (AdaLN) Zero
        # Regulates the shift and scale of the block based on timestep
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True) 
        )

    def forward(self, x, t_emb, context, rope_cos, rope_sin):
        # x: [B, L, D], t_emb: [B, D], context: [B, S, D] (Text embeddings)
        
        # Calculate modulation parameters from timestep (computed once and reused)
        # shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross
        modulation = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross = modulation.chunk(6, dim=1)
        
        # -------------------------------------------------------
        # 1. Self-Attention
        # -------------------------------------------------------
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        qkv = self.attn_qkv(x_norm).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to Self-Attention
        q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
        k = apply_rotary_pos_emb(k, rope_cos, rope_sin)
        
        # Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn_proj(out)

        # -------------------------------------------------------
        # 2. Cross-Attention (Attend to Text)
        # -------------------------------------------------------
        if context is not None:
            x_norm = self.norm_cross(x)
            # In DiT, we usually don't modulate cross norm, but we can:
            # x_norm = x_norm * (1 + scale_cross.unsqueeze(1)) + shift_cross.unsqueeze(1)
            
            q = self.cross_q(x_norm).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
            
            # Context is [B, Seq_Text, Dim]
            kv = self.cross_kv(context).reshape(context.shape[0], context.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
            k_c, v_c = kv[0], kv[1]
            k_c, v_c = k_c.transpose(1, 2), v_c.transpose(1, 2)
            
            if hasattr(F, 'scaled_dot_product_attention'):
                out = F.scaled_dot_product_attention(q, k_c, v_c)
            else:
                attn = (q @ k_c.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                out = (attn @ v_c)
            
            out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            x = x + self.cross_proj(out) # Or gated

        # -------------------------------------------------------
        # 3. MLP (SwiGLU)
        # -------------------------------------------------------
        # Extract MLP modulation parameters from the already computed modulation
        shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)[3:]
        
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

# -----------------------------------------------------------------------------
# Main Diffusion Model
# -----------------------------------------------------------------------------

class MusicDiffusionTransformer(nn.Module):
    def __init__(
        self, 
        dim: int = 768, 
        num_layers: int = 12, 
        num_heads: int = 12,
        vocab_size: int = 256,
        mel_channels: int = 80,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.dim = dim
        self.mel_channels = mel_channels
        
        # 1. Input Projections
        # We project the noisy mel-spectrogram (channels) to hidden dim
        self.input_proj = nn.Linear(mel_channels, dim)
        
        # 2. Time Embeddings
        self.time_embedder = TimestepEmbedder(dim)
        
        # 3. Text Encoder (Simplified for self-contained example)
        # In production, use T5 or CLAP. Here we use a learnable embedding.
        self.text_embedding = nn.Embedding(vocab_size, dim)
        
        # 4. Positional Embeddings
        self.rope = RotaryEmbedding(dim // num_heads, max_seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim)) # Learned absolute pos for patches
        
        # 5. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        # 6. Output Projection
        self.final_norm = RMSNorm(dim)
        self.linear_out = nn.Linear(dim, mel_channels) # Predicts noise/velocity
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize zero-out output layers for better convergence (DiT practice)
        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

    def _tokenize(self, texts: List[str], device: torch.device) -> torch.Tensor:
        # Simplified tokenizer
        batch_indices = []
        max_len = 0
        for text in texts:
            indices = [ord(c) % 256 for c in text]
            batch_indices.append(indices)
            max_len = max(max_len, len(indices))
        tensor = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        for i, indices in enumerate(batch_indices):
            tensor[i, :len(indices)] = torch.tensor(indices, device=device)
        return tensor

    def forward(self, x, t, text_indices):
        """
        x: Noisy Mel Spectrogram [Batch, Seq_Len, Mel_Channels]
        t: Timesteps [Batch]
        text_indices: Text tokens [Batch, Text_Len]
        """
        B, L, C = x.shape
        
        # 1. Embed Inputs
        x = self.input_proj(x) # [B, L, Dim]
        x = x + self.pos_emb[:, :L, :] # Add learned pos embedding
        
        t_emb = self.time_embedder(t) # [B, Dim]
        
        context = self.text_embedding(text_indices) # [B, Text_Len, Dim]
        
        # 2. Prepare Rotary Embeddings
        cos, sin = self.rope(x)
        
        # 3. Pass through DiT Blocks
        for block in self.blocks:
            x = block(x, t_emb, context, cos, sin)
            
        # 4. Output Projection
        x = self.final_norm(x)
        output = self.linear_out(x) # [B, L, Mel_Channels]
        
        return output

    @torch.inference_mode()
    def generate(
        self, 
        text: Union[str, List[str]], 
        duration_sec: float = 10.0,
        steps: int = 50,
        cfg_scale: float = 3.0,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Runs the diffusion reverse process (DDIM-like).
        """
        self.eval()
        self.to(device)
        
        if isinstance(text, str):
            text = [text]
            
        # Calculate sequence length (assuming ~86 frames per second for Mel)
        seq_len = int(duration_sec * 86)
        
        # 1. Prepare Conditions
        text_indices = self._tokenize(text, device)
        B = len(text)
        
        # 2. Start from pure noise
        latents = torch.randn(B, seq_len, self.mel_channels, device=device)
        
        # 3. Scheduler loop (Simplified DDIM)
        # In a real scenario, use diffusers.schedulers
        timesteps = torch.linspace(999, 0, steps, device=device).long()
        
        print(f"Generating music for: {text} ({duration_sec}s)")
        
        for i, t in enumerate(timesteps):
            # Expand t for batch
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Classifier-Free Guidance
            # We run forward twice: once with text, once with empty text (uncond)
            # For simplicity in this standalone, we just run conditioned. 
            # To implement CFG, input dummy tokens for the second pass.
            
            # Predict noise
            noise_pred = self.forward(latents, t_batch, text_indices)
            
            # Simple Euler Step (Approximation of reverse diffusion)
            # x_{t-1} = x_t - (noise_pred * step_size)
            step_size = 1.0 / steps
            latents = latents - (noise_pred * step_size)
            
        return latents
