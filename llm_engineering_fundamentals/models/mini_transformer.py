"""Complete Mini Transformer model (GPT-style)."""

import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass

from llm_engineering_fundamentals.embeddings.layers import TiedEmbedding
from llm_engineering_fundamentals.transformer.block import StackedTransformer
from llm_engineering_fundamentals.decoding.strategies import greedy_decode, beam_search
from llm_engineering_fundamentals.sampling.methods import sample_next_token


@dataclass
class MiniTransformerConfig:
    """Configuration for Mini Transformer."""
    
    # Model architecture
    vocab_size: int = 10000
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024  # 4x d_model
    
    # Positional encoding
    max_seq_len: int = 512
    positional_type: Literal["sinusoidal", "learned", "none"] = "learned"
    
    # Regularization
    dropout: float = 0.1
    
    # Training
    norm_first: bool = True  # Pre-LN (modern)
    
    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2


class MiniTransformer:
    """
    Complete Mini Transformer (GPT-style).
    
    A decoder-only transformer for autoregressive language generation.
    Combines all components built in previous projects.
    """
    
    def __init__(self, config: MiniTransformerConfig):
        """
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Embedding layer (tied input/output)
        self.embedding = TiedEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            positional_type=config.positional_type,
            dropout=config.dropout,
        )
        
        # Transformer layers
        self.transformer = StackedTransformer(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            norm_first=config.norm_first,
        )
    
    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            token_ids: Token IDs of shape (batch, seq_len)
            training: Whether in training mode
            
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens
        hidden = self.embedding.embed(token_ids, training=training)
        
        # Create causal mask (prevent attending to future)
        mask = self._create_causal_mask(seq_len)
        
        # Apply transformer layers
        hidden = self.transformer(hidden, mask=mask, training=training)
        
        # Project to vocabulary
        logits = self.embedding.project_to_vocab(hidden)
        
        return logits
    
    def __call__(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        """Convenience method for forward pass."""
        return self.forward(token_ids, training=training)
    
    def generate_greedy(
        self,
        prompt_ids: np.ndarray,
        max_length: int = 50,
    ) -> np.ndarray:
        """
        Generate text using greedy decoding.
        
        Args:
            prompt_ids: Prompt token IDs of shape (1, prompt_len)
            max_length: Maximum generation length
            
        Returns:
            Generated sequence of shape (1, final_len)
        """
        def model_fn(tokens: np.ndarray) -> np.ndarray:
            return self.forward(tokens, training=False)
        
        return greedy_decode(
            model_fn=model_fn,
            initial_tokens=prompt_ids,
            max_length=max_length,
            eos_token_id=self.config.eos_token_id,
        )
    
    def generate_beam_search(
        self,
        prompt_ids: np.ndarray,
        num_beams: int = 5,
        max_length: int = 50,
        length_penalty: float = 0.6,
    ) -> np.ndarray:
        """
        Generate text using beam search.
        
        Args:
            prompt_ids: Prompt token IDs of shape (1, prompt_len)
            num_beams: Number of beams
            max_length: Maximum generation length
            length_penalty: Length penalty
            
        Returns:
            Generated sequence
        """
        def model_fn(tokens: np.ndarray) -> np.ndarray:
            return self.forward(tokens, training=False)
        
        return beam_search(
            model_fn=model_fn,
            initial_tokens=prompt_ids,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=length_penalty,
            eos_token_id=self.config.eos_token_id,
        )
    
    def generate_sample(
        self,
        prompt_ids: np.ndarray,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> np.ndarray:
        """
        Generate text using sampling.
        
        Args:
            prompt_ids: Prompt token IDs of shape (1, prompt_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated sequence of shape (1, final_len)
        """
        sequence = prompt_ids[0].tolist()
        
        for _ in range(max_length - len(sequence)):
            # Get logits
            seq_array = np.array([sequence])
            logits = self.forward(seq_array, training=False)
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_tokens=np.array(sequence) if repetition_penalty != 1.0 else None,
            )
            
            sequence.append(next_token)
            
            # Stop if EOS
            if next_token == self.config.eos_token_id:
                break
        
        return np.array([sequence])
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask for autoregressive generation.
        
        Mask[i, j] = True if token i can attend to token j.
        Causal: can only attend to past (lower triangle).
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Mask of shape (seq_len, seq_len)
        """
        # Lower triangular matrix (including diagonal)
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        return mask
    
    def count_parameters(self) -> dict:
        """Count model parameters."""
        embedding_params = self.embedding.count_parameters()
        transformer_params = self.transformer.count_parameters()
        
        return {
            "embedding": embedding_params["total"],
            "transformer": transformer_params["total_parameters"],
            "total": embedding_params["total"] + transformer_params["total_parameters"],
        }


def create_mini_gpt(
    vocab_size: int = 10000,
    d_model: int = 256,
    num_layers: int = 6,
) -> MiniTransformer:
    """
    Create a Mini-GPT style model with sensible defaults.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of layers
        
    Returns:
        MiniTransformer model
    """
    config = MiniTransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=max(d_model // 64, 1),  # 64 dims per head
        d_ff=4 * d_model,  # Standard 4x expansion
        positional_type="learned",
        dropout=0.1,
        norm_first=True,  # Pre-LN (modern)
    )
    
    return MiniTransformer(config)

