"""Embedding layer implementations."""

import numpy as np
from typing import Optional, Literal

from llm_engineering_fundamentals.positional.encodings import (
    sinusoidal_positional_encoding,
    learned_positional_embedding,
)


class TokenEmbedding:
    """
    Token Embedding Layer.
    
    Maps discrete token IDs to continuous dense vectors.
    
    This is essentially a lookup table: embedding_table[token_id]
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_std: float = 0.02,
        padding_idx: Optional[int] = None,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            init_std: Standard deviation for initialization
            padding_idx: If specified, this token ID will have zero embedding
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Initialize embedding table
        # Shape: [vocab_size, d_model]
        self.weight = np.random.normal(0, init_std, (vocab_size, d_model))
        
        # Set padding token to zero if specified
        if padding_idx is not None:
            self.weight[padding_idx] = 0.0
    
    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Simple lookup
        embeddings = self.weight[token_ids]
        
        return embeddings
    
    def count_parameters(self) -> int:
        """Count parameters in embedding table."""
        return self.vocab_size * self.d_model


class CombinedEmbedding:
    """
    Combined Token + Positional Embedding Layer.
    
    This is the complete input embedding layer used in transformers.
    Combines token embeddings with positional information.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        positional_type: Literal["sinusoidal", "learned", "none"] = "learned",
        dropout: float = 0.1,
        scale_embeddings: bool = False,
        padding_idx: Optional[int] = None,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            positional_type: Type of positional embedding
                           - "sinusoidal": Fixed sinusoidal (original Transformer)
                           - "learned": Learned positional embeddings (GPT/BERT)
                           - "none": No positional embeddings (use RoPE/ALiBi in attention)
            dropout: Dropout rate on embeddings
            scale_embeddings: Whether to scale embeddings by sqrt(d_model)
            padding_idx: Padding token ID
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_type = positional_type
        self.dropout_rate = dropout
        self.scale_embeddings = scale_embeddings
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size, d_model, padding_idx=padding_idx
        )
        
        # Positional embedding
        if positional_type == "sinusoidal":
            # Pre-compute sinusoidal encodings
            self.positional_encoding = sinusoidal_positional_encoding(
                max_seq_len, d_model
            )
        elif positional_type == "learned":
            # Learned positional embeddings
            self.positional_encoding = learned_positional_embedding(
                max_seq_len, d_model
            )
        else:
            self.positional_encoding = None
    
    def __call__(
        self,
        token_ids: np.ndarray,
        positions: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> np.ndarray:
        """
        Compute combined embeddings.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            positions: Position indices of shape (batch_size, seq_len).
                      If None, uses range(seq_len)
            training: Whether in training mode (for dropout)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding(token_ids)
        
        # Scale if requested (original Transformer does this)
        if self.scale_embeddings:
            token_embeds = token_embeds * np.sqrt(self.d_model)
        
        # Add positional embeddings
        if self.positional_encoding is not None:
            if positions is None:
                # Default positions: 0, 1, 2, ..., seq_len-1
                positions = np.arange(seq_len)[None, :]  # (1, seq_len)
                positions = np.broadcast_to(positions, (batch_size, seq_len))
            
            # Lookup positional embeddings
            pos_embeds = self.positional_encoding[positions]
            
            # Combine
            embeddings = token_embeds + pos_embeds
        else:
            embeddings = token_embeds
        
        # Apply dropout
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=embeddings.shape
            )
            embeddings = embeddings * mask / (1 - self.dropout_rate)
        
        return embeddings
    
    def count_parameters(self) -> dict[str, int]:
        """Count parameters."""
        params = {
            "token_embedding": self.token_embedding.count_parameters(),
        }
        
        if self.positional_type == "learned" and self.positional_encoding is not None:
            params["positional_embedding"] = self.max_seq_len * self.d_model
        else:
            params["positional_embedding"] = 0  # Sinusoidal has no params
        
        params["total"] = sum(params.values())
        
        return params


class TiedEmbedding:
    """
    Tied Embedding + Output Projection.
    
    Shares weights between input embedding and output projection layer.
    This reduces parameters and often improves performance.
    
    Used in: GPT-2, BERT, T5, LLaMA, most modern LLMs
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        positional_type: Literal["sinusoidal", "learned", "none"] = "learned",
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            positional_type: Type of positional embedding
            dropout: Dropout rate
        """
        # Combined embedding layer
        self.embedding = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            positional_type=positional_type,
            dropout=dropout,
        )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def embed(
        self,
        token_ids: np.ndarray,
        positions: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> np.ndarray:
        """
        Embed tokens (input embedding).
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            positions: Position indices
            training: Whether in training mode
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(token_ids, positions, training)
    
    def project_to_vocab(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Project hidden states to vocabulary (output projection).
        
        Uses the TRANSPOSED embedding weights:
            logits = hidden @ embedding_weight.T
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, d_model)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Use transposed embedding weights
        embedding_weight = self.embedding.token_embedding.weight
        
        # Matrix multiplication
        logits = hidden_states @ embedding_weight.T
        
        return logits
    
    def count_parameters(self) -> dict[str, int]:
        """Count parameters (tied weights counted once)."""
        params = self.embedding.count_parameters()
        
        # Note: Output projection shares weights, so no additional parameters
        params["note"] = "Output projection shares embedding weights (tied)"
        
        return params


def compare_tied_vs_untied(vocab_size: int, d_model: int, max_seq_len: int) -> dict:
    """
    Compare parameter counts for tied vs untied embeddings.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        
    Returns:
        Dictionary with parameter comparison
    """
    # Tied embedding
    tied_params = vocab_size * d_model + max_seq_len * d_model
    
    # Untied embedding (separate output projection)
    untied_params = vocab_size * d_model + max_seq_len * d_model + vocab_size * d_model
    
    savings = untied_params - tied_params
    savings_pct = (savings / untied_params) * 100
    
    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "max_seq_len": max_seq_len,
        "tied_parameters": tied_params,
        "untied_parameters": untied_params,
        "savings": savings,
        "savings_percentage": savings_pct,
    }

