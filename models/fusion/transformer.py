"""
Cross-Modal Transformer Block implementation for ACMFT.

This module contains the implementation of transformer-based components
for cross-modal fusion in the ACMFT architecture, including self-attention,
cross-attention, and feed-forward networks.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention is All You Need".
    Supports both self-attention and cross-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor (batch_size, tgt_len, embed_dim)
            key: Key tensor (batch_size, src_len, embed_dim)
            value: Value tensor (batch_size, src_len, embed_dim)
            attn_mask: Attention mask to prevent attending to certain positions
            key_padding_mask: Mask for padded elements in the key sequence

        Returns:
            output: Attention output (batch_size, tgt_len, embed_dim)
            attention_weights: Attention weights for visualization
        """
        # Handle empty tensor case
        if query.size(0) == 0 or key.size(0) == 0 or value.size(0) == 0:
            # Return empty tensors with correct shape
            empty_output = torch.zeros((0, query.size(1), self.embed_dim), device=query.device)
            empty_weights = torch.zeros((0, self.num_heads, query.size(1), key.size(1)), device=query.device)
            return empty_output, empty_weights
            
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # Linear projections and reshape
        q = self.q_proj(query).reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch_size, src_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, tgt_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, tgt_len, src_len)

        # Apply masks if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        if key_padding_mask is not None:
            # Expand key_padding_mask to match scores dimensions
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(expanded_mask, -1e9)

        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, tgt_len, head_dim)

        # Transpose and reshape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)

        # Return output and attention weights (for visualization)
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Feed Forward Network as used in transformer blocks.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            output: Transformed tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    A standard Transformer encoder layer with self-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for transformer encoder layer with self-attention.

        Args:
            src: Source tensor (batch_size, src_len, d_model)
            src_mask: Attention mask to prevent attending to certain positions
            src_key_padding_mask: Mask for padded elements in the source sequence

        Returns:
            output: Transformed tensor (batch_size, src_len, d_model)
            attention_weights: Self-attention weights for visualization
        """
        # Self-attention block
        src2, attention_weights = self.self_attn(
            self.norm1(src), self.norm1(src), self.norm1(src), attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        # Feed-forward block
        src2 = self.feed_forward(self.norm2(src))
        src = src + self.dropout2(src2)

        return src, attention_weights


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for attending between different modalities.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # Target modality
        key_value: torch.Tensor,  # Source modality
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention layer.

        Args:
            query: Query modality tensor (batch_size, tgt_len, d_model)
            key_value: Key-value modality tensor (batch_size, src_len, d_model)
            attn_mask: Attention mask to prevent attending to certain positions
            key_padding_mask: Mask for padded elements in the key sequence

        Returns:
            output: Enhanced query tensor (batch_size, tgt_len, d_model)
            attention_weights: Cross-attention weights for visualization
        """
        # Normalize inputs
        query_norm = self.norm1(query)
        key_value_norm = self.norm2(key_value)

        # Cross-attention
        attn_output, attention_weights = self.cross_attn(
            query_norm, key_value_norm, key_value_norm, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Residual connection
        enhanced = query + self.dropout(attn_output)

        return enhanced, attention_weights


class CrossModalTransformerBlock(nn.Module):
    """
    A complete cross-modal transformer block that implements the ACMFT fusion architecture.
    This block processes three modalities (visual, audio, HR) through self-attention and
    cross-attention mechanisms to enable information exchange between modalities.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        # Self-attention layers for each modality
        self.self_attn_visual = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_audio = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_hr = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        # Cross-attention layers for visual modality
        self.cross_attn_v2a = CrossAttentionLayer(d_model, nhead, dropout)  # Visual attending to Audio
        self.cross_attn_v2h = CrossAttentionLayer(d_model, nhead, dropout)  # Visual attending to HR

        # Cross-attention layers for audio modality
        self.cross_attn_a2v = CrossAttentionLayer(d_model, nhead, dropout)  # Audio attending to Visual
        self.cross_attn_a2h = CrossAttentionLayer(d_model, nhead, dropout)  # Audio attending to HR

        # Cross-attention layers for HR modality
        self.cross_attn_h2v = CrossAttentionLayer(d_model, nhead, dropout)  # HR attending to Visual
        self.cross_attn_h2a = CrossAttentionLayer(d_model, nhead, dropout)  # HR attending to Audio

        # Final feed-forward and normalization for each modality
        self.ffn_visual = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.ffn_audio = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.ffn_hr = FeedForward(d_model, dim_feedforward, dropout, activation)

        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_audio = nn.LayerNorm(d_model)
        self.norm_hr = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the cross-modal transformer block.

        Args:
            visual: Visual features (batch_size, visual_len, d_model)
            audio: Audio features (batch_size, audio_len, d_model)
            hr: HR features (batch_size, hr_len, d_model)

        Returns:
            enhanced_visual: Enhanced visual features
            enhanced_audio: Enhanced audio features
            enhanced_hr: Enhanced HR features
        """
        # Handle the case when any modality is empty
        if visual.size(0) == 0:
            # If visual is empty, return empty tensors for all modalities
            return visual, audio, hr
        
        # Check batch sizes and ensure they match
        batch_size = visual.size(0)
        
        # Step 1: Self-attention for each modality
        visual_self, _ = self.self_attn_visual(visual)
        
        # Only apply audio processing if it has the same batch size (i.e., not empty)
        if audio.size(0) == batch_size:
            audio_self, _ = self.self_attn_audio(audio)
        else:
            # Use a copy of the original tensor for empty audio
            audio_self = audio
            
        # Only apply HR processing if it has the same batch size (i.e., not empty)
        if hr.size(0) == batch_size:
            hr_self, _ = self.self_attn_hr(hr)
        else:
            # Use a copy of the original tensor for empty HR
            hr_self = hr

        # Step 2: Cross-attention between modalities - handle cases where modalities may be empty

        # Visual modality attending to others
        if audio_self.size(0) == batch_size:
            visual_from_audio, _ = self.cross_attn_v2a(visual_self, audio_self)
        else:
            # If audio is empty, no cross-attention, just use visual_self
            visual_from_audio = visual_self
            
        if hr_self.size(0) == batch_size:
            visual_from_hr, _ = self.cross_attn_v2h(visual_self, hr_self)
        else:
            # If HR is empty, no cross-attention, just use visual_self
            visual_from_hr = visual_self

        # Audio modality attending to others - only if audio is not empty
        if audio_self.size(0) == batch_size:
            audio_from_visual, _ = self.cross_attn_a2v(audio_self, visual_self)
            
            if hr_self.size(0) == batch_size:
                audio_from_hr, _ = self.cross_attn_a2h(audio_self, hr_self)
            else:
                # If HR is empty, no cross-attention, just use audio_self
                audio_from_hr = audio_self
        else:
            # If audio is empty, just pass through
            audio_from_visual = audio_self
            audio_from_hr = audio_self

        # HR modality attending to others - only if HR is not empty
        if hr_self.size(0) == batch_size:
            hr_from_visual, _ = self.cross_attn_h2v(hr_self, visual_self)
            
            if audio_self.size(0) == batch_size:
                hr_from_audio, _ = self.cross_attn_h2a(hr_self, audio_self)
            else:
                # If audio is empty, no cross-attention, just use hr_self
                hr_from_audio = hr_self
        else:
            # If HR is empty, just pass through
            hr_from_visual = hr_self
            hr_from_audio = hr_self

        # Step 3: Aggregate cross-modal information for each modality
        # Adjust divisors based on available modalities
        
        # Visual modality aggregation
        divisor_v = 1  # Start with 1 for visual_self
        if audio_self.size(0) == batch_size: divisor_v += 1
        if hr_self.size(0) == batch_size: divisor_v += 1
        visual_enhanced = (visual_self + visual_from_audio + visual_from_hr) / divisor_v
        
        # Audio modality aggregation (only if audio is present)
        if audio_self.size(0) == batch_size:
            divisor_a = 1  # Start with 1 for audio_self
            if visual_self.size(0) == batch_size: divisor_a += 1
            if hr_self.size(0) == batch_size: divisor_a += 1
            audio_enhanced = (audio_self + audio_from_visual + audio_from_hr) / divisor_a
        else:
            audio_enhanced = audio_self
            
        # HR modality aggregation (only if HR is present)
        if hr_self.size(0) == batch_size:
            divisor_h = 1  # Start with 1 for hr_self
            if visual_self.size(0) == batch_size: divisor_h += 1
            if audio_self.size(0) == batch_size: divisor_h += 1
            hr_enhanced = (hr_self + hr_from_visual + hr_from_audio) / divisor_h
        else:
            hr_enhanced = hr_self

        # Step 4: Final feed-forward and normalization (only for non-empty modalities)
        visual_output = visual_enhanced + self.dropout(self.ffn_visual(self.norm_visual(visual_enhanced)))
        
        if audio_enhanced.size(0) == batch_size:
            audio_output = audio_enhanced + self.dropout(self.ffn_audio(self.norm_audio(audio_enhanced)))
        else:
            audio_output = audio_enhanced
            
        if hr_enhanced.size(0) == batch_size:
            hr_output = hr_enhanced + self.dropout(self.ffn_hr(self.norm_hr(hr_enhanced)))
        else:
            hr_output = hr_enhanced

        return visual_output, audio_output, hr_output
