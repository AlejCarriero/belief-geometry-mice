"""Bayes-optimal belief states (the forward algorithm) and the optimal next-token loss.

Shapes and alignment follow src/hmm.py:

    tokens:   (n_sequences, sequence_length), integers
    beliefs:  (n_sequences, sequence_length + 1, n_states)

    beliefs[:, t] = P(hidden_states[:, t] | tokens[:, :t])

So beliefs[:, t] is the belief about the hidden state just before token t is
emitted, given every token before t. beliefs[:, 0] is the starting belief, and
beliefs[:, t + 1] is the belief after seeing tokens 0 to t.

When pairing with the transformer later: the residual stream at position t has
seen tokens 0 to t, so it pairs with beliefs[:, t + 1].
"""

import numpy as np

from src.hmm import HiddenMarkovModel


def compute_belief_states(
    hmm: HiddenMarkovModel,
    tokens: np.ndarray,
    initial_belief: np.ndarray | None = None,
) -> np.ndarray:
    """Run the forward algorithm: the Bayes-optimal belief over hidden states at every position.

    Update rule (Shai et al. 2024, Section 2.2), for a row vector of beliefs b_t:

        b_{t+1} = b_t @ T(x_t) / (b_t @ T(x_t) @ ones)

    where T(x_t) is hmm.token_transition_matrices[x_t] and x_t = tokens[n, t].
    The denominator is P(token t | tokens before t), a single number per sequence.

    Requirements (each one has a test in tests/test_beliefs.py):

    * Normalise at every step, not once at the end. Unnormalised products shrink
      by a constant factor per token and underflow to zero after a few hundred tokens.
    * Each sequence is processed independently of the others in the batch.
    * If some token has probability zero given the tokens before it (the sequence
      is impossible under the model), raise ValueError instead of dividing by zero.
    * A sequence_length of 0 is allowed: return only the starting beliefs.

    Looping over sequences is fine. Vectorising across sequences is optional.

    Args:
        hmm: The process that generated the tokens.
        tokens: Integer array of shape (n_sequences, sequence_length). For a
            single sequence, pass tokens[np.newaxis, :].
        initial_belief: Probability vector of shape (n_states,): the belief before
            any token. Defaults to hmm.stationary_distribution().

    Returns:
        Float64 array of shape (n_sequences, sequence_length + 1, n_states), with
        beliefs[:, 0] equal to the initial belief and beliefs[:, t + 1] the belief
        after seeing tokens 0 to t. Every beliefs[n, t] sums to 1.

    Raises:
        ValueError: If any sequence has probability zero under the model.
    """
    # Milestone 2: your implementation goes here.
    raise NotImplementedError("Milestone 2: write the forward algorithm here.")


def next_token_distributions(hmm: HiddenMarkovModel, beliefs: np.ndarray) -> np.ndarray:
    """Return the Bayes-optimal distribution of the next token for each belief.

    P(next token = x | belief b) = sum over states i of b[i] * P(next token = x | state i).
    That is a belief-weighted average of each state's token probabilities, so it is
    a single matrix product.

    Args:
        hmm: The process.
        beliefs: Array whose last axis has length n_states, for example the
            (n_sequences, sequence_length + 1, n_states) output of compute_belief_states.

    Returns:
        Array of the same shape except that the last axis has length n_tokens.
    """
    return beliefs @ hmm.token_probabilities_given_state()


def bayes_optimal_token_losses(
    hmm: HiddenMarkovModel, tokens: np.ndarray, beliefs: np.ndarray
) -> np.ndarray:
    """Negative log probability that the Bayes-optimal predictor gives each actual token.

    losses[n, t] = -log P(tokens[n, t] | tokens[n, :t]), computed from beliefs[n, t].

    Averaging over sequences, losses.mean(axis=0), estimates the conditional entropy
    H(X_t | X_<t): the lowest expected cross-entropy any model can reach at position t.
    Summing over positions gives -log P(whole sequence), by the chain rule.

    The last belief, beliefs[:, sequence_length], is not used: it would predict a
    token that is not in the data.

    Args:
        hmm: The process that generated the tokens.
        tokens: Integer array of shape (n_sequences, sequence_length).
        beliefs: Array of shape (n_sequences, sequence_length + 1, n_states),
            aligned as described at the top of this module.

    Returns:
        Array of shape (n_sequences, sequence_length), in nats (natural log), to
        match PyTorch's cross-entropy loss.
    """
    n_sequences, sequence_length = tokens.shape
    expected_belief_shape = (n_sequences, sequence_length + 1, hmm.n_states)
    if beliefs.shape != expected_belief_shape:
        # The most likely cause is an off-by-one in how beliefs were aligned with tokens.
        raise ValueError(
            f"Expected beliefs of shape {expected_belief_shape}, got {beliefs.shape}."
        )

    # beliefs[:, t] predicts token t, for t = 0 .. sequence_length - 1.
    predicted_distributions = next_token_distributions(hmm, beliefs[:, :sequence_length])

    # Pick out, for every (sequence, position), the probability of the token that occurred.
    sequence_index = np.arange(n_sequences)[:, np.newaxis]
    position_index = np.arange(sequence_length)[np.newaxis, :]
    probability_of_actual_token = predicted_distributions[sequence_index, position_index, tokens]

    return -np.log(probability_of_actual_token)
