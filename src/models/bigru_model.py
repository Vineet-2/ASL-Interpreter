"""
PyTorch BiGRU with Attention Pooling Sign Recognition Classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


from src.config import FEATURE_DIM, NUM_CLASSES


class AttentionPooling(nn.Module):
    """
    Self-attention pooling layer to dynamically aggregate temporal hidden states.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, hidden_dim)
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_dim)
        return context, weights.squeeze(-1)


class BiGRUSignClassifier(nn.Module):
    """
    2-Layer Bidirectional GRU with Attention Pooling for Sign Language Recognition.
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Spatial input projection & normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 2 * hidden_dim because bidirectional
        gru_out_dim = hidden_dim * 2
        
        # Attention Pooling
        self.attention = AttentionPooling(gru_out_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional boolean mask (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
            attn_weights: (batch_size, seq_len)
        """
        # Normalization and initial projection
        norm_x = self.input_norm(x)
        proj_x = self.input_proj(norm_x)
        
        # BiGRU temporal processing
        gru_out, _ = self.gru(proj_x)  # (batch_size, seq_len, 2 * hidden_dim)
        
        # Attention Pooling
        context, attn_weights = self.attention(gru_out, mask)  # (batch_size, 2 * hidden_dim)
        
        # Logits
        logits = self.classifier(context)  # (batch_size, num_classes)
        
        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning (class_indices, confidences).
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confs, indices = torch.max(probs, dim=-1)
        return indices, confs
