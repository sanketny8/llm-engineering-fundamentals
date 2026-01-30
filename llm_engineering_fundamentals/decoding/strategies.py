"""Decoding strategies for text generation."""

import numpy as np
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class DecodingConfig:
    """Configuration for decoding."""
    max_length: int = 50
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    
    # Sampling (used in Project 09)
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def greedy_decode(
    model_fn: Callable[[np.ndarray], np.ndarray],
    initial_tokens: np.ndarray,
    max_length: int = 50,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Greedy decoding: always pick the most likely next token.
    
    Args:
        model_fn: Function that takes token IDs and returns logits
                 Shape: (batch, seq_len) -> (batch, seq_len, vocab_size)
        initial_tokens: Starting tokens of shape (batch, seq_len)
        max_length: Maximum sequence length
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Generated sequences of shape (batch, final_seq_len)
    """
    batch_size, initial_len = initial_tokens.shape
    
    # Current sequence
    sequence = initial_tokens.copy()
    
    # Track which sequences are finished
    finished = np.zeros(batch_size, dtype=bool)
    
    for _ in range(max_length - initial_len):
        # Get model predictions
        logits = model_fn(sequence)  # (batch, seq_len, vocab_size)
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Greedy: take argmax
        next_tokens = np.argmax(next_token_logits, axis=-1)  # (batch,)
        
        # For finished sequences, use pad token (or keep repeating EOS)
        if eos_token_id is not None:
            next_tokens = np.where(finished, eos_token_id, next_tokens)
        
        # Append to sequence
        sequence = np.concatenate([sequence, next_tokens[:, None]], axis=1)
        
        # Update finished status
        if eos_token_id is not None:
            finished = finished | (next_tokens == eos_token_id)
            
            # If all sequences finished, stop
            if np.all(finished):
                break
    
    return sequence


class BeamSearcher:
    """
    Beam search decoder.
    
    Maintains multiple hypotheses (beams) and selects the best one.
    """
    
    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ):
        """
        Args:
            model_fn: Function that takes tokens and returns logits
            num_beams: Number of beams to maintain
            length_penalty: Length penalty exponent (alpha)
            early_stopping: Whether to stop when num_beams sequences finish
        """
        self.model_fn = model_fn
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
    
    def search(
        self,
        initial_tokens: np.ndarray,
        max_length: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform beam search.
        
        Args:
            initial_tokens: Starting tokens of shape (1, seq_len)
            max_length: Maximum sequence length
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Tuple of (best_sequence, best_score)
        """
        batch_size, initial_len = initial_tokens.shape
        if batch_size != 1:
            raise ValueError("Beam search only supports batch_size=1")
        
        # Initialize beams: (sequence, score, finished)
        # Start with single beam
        beams = [(initial_tokens[0].tolist(), 0.0, False)]
        finished_beams = []
        
        for step in range(max_length - initial_len):
            candidates = []
            
            for seq, score, finished in beams:
                if finished:
                    # Keep finished beams as-is
                    candidates.append((seq, score, True))
                    continue
                
                # Get model predictions for this beam
                seq_array = np.array([seq])  # (1, seq_len)
                logits = self.model_fn(seq_array)  # (1, seq_len, vocab_size)
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Get log probabilities
                log_probs = self._log_softmax(next_token_logits)
                
                # Get top-k tokens for this beam
                top_k_indices = np.argsort(log_probs)[-self.num_beams:]
                
                for token_id in top_k_indices:
                    new_seq = seq + [token_id]
                    new_score = score + log_probs[token_id]
                    
                    # Check if finished
                    is_finished = (eos_token_id is not None and token_id == eos_token_id)
                    
                    candidates.append((new_seq, new_score, is_finished))
            
            # Sort candidates by normalized score
            candidates.sort(key=lambda x: self._normalize_score(x[1], len(x[0])), reverse=True)
            
            # Keep top num_beams
            beams = candidates[:self.num_beams]
            
            # Move finished beams to finished_beams
            active_beams = []
            for seq, score, finished in beams:
                if finished:
                    finished_beams.append((seq, score))
                else:
                    active_beams.append((seq, score, finished))
            
            # If early stopping and enough beams finished, stop
            if self.early_stopping and len(finished_beams) >= self.num_beams:
                break
            
            # If all beams finished, stop
            if not active_beams:
                break
            
            beams = active_beams
        
        # Collect all beams (finished + active)
        all_beams = finished_beams + [(seq, score) for seq, score, _ in beams]
        
        # Sort by normalized score and return best
        all_beams.sort(key=lambda x: self._normalize_score(x[1], len(x[0])), reverse=True)
        
        best_seq, best_score = all_beams[0]
        return np.array(best_seq), best_score
    
    def _log_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute log softmax."""
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        return logits - np.log(np.sum(exp_logits))
    
    def _normalize_score(self, score: float, length: int) -> float:
        """Apply length normalization."""
        if self.length_penalty == 0.0:
            return score
        return score / (length ** self.length_penalty)


def beam_search(
    model_fn: Callable[[np.ndarray], np.ndarray],
    initial_tokens: np.ndarray,
    num_beams: int = 5,
    max_length: int = 50,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Beam search decoding (convenience function).
    
    Args:
        model_fn: Function that takes tokens and returns logits
        initial_tokens: Starting tokens of shape (1, seq_len)
        num_beams: Number of beams
        max_length: Maximum sequence length
        length_penalty: Length penalty exponent
        early_stopping: Whether to stop early
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Best sequence of shape (seq_len,)
    """
    searcher = BeamSearcher(
        model_fn=model_fn,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
    )
    
    best_seq, best_score = searcher.search(
        initial_tokens=initial_tokens,
        max_length=max_length,
        eos_token_id=eos_token_id,
    )
    
    return best_seq


def compare_decoding_strategies(
    model_fn: Callable[[np.ndarray], np.ndarray],
    initial_tokens: np.ndarray,
    max_length: int = 20,
    eos_token_id: Optional[int] = None,
) -> dict:
    """
    Compare different decoding strategies.
    
    Args:
        model_fn: Model function
        initial_tokens: Starting tokens
        max_length: Maximum length
        eos_token_id: EOS token ID
        
    Returns:
        Dictionary with results from each strategy
    """
    results = {}
    
    # Greedy
    greedy_output = greedy_decode(
        model_fn, initial_tokens, max_length, eos_token_id
    )
    results["greedy"] = greedy_output[0]
    
    # Beam search with different beam widths
    for num_beams in [1, 3, 5]:
        beam_output = beam_search(
            model_fn, initial_tokens, num_beams, max_length,
            length_penalty=0.6, eos_token_id=eos_token_id
        )
        results[f"beam_{num_beams}"] = beam_output
    
    return results

