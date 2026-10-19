"""Hidden Markov models in token-labelled form, and sampling from them.

Representation
--------------
A model with S hidden states and X tokens is stored as an array of shape
(X, S, S), one S x S matrix per token:

    token_transition_matrices[x, i, j] = P(emit token x and move to state j | in state i)

This is the form used by Shai et al. (2024), Section 2.1. The only constraint is
that, for each current state i, the S * X numbers in row i across all token
matrices form a probability distribution:

    sum over x and j of token_transition_matrices[x, i, j] = 1.

Summing over tokens gives the ordinary state transition matrix,
T[i, j] = P(move to j | in i).

The textbook HMM (a transition matrix plus an emission matrix) is a special case.
See `hmm_from_transition_and_emission`.
"""

from typing import NamedTuple, Sequence

import numpy as np

# Tolerance for "sums to 1" checks. Probabilities typed with a few decimal places
# (as in the paper) sum to 1 to within about 1e-15, so 1e-9 only catches real errors.
PROBABILITY_SUM_TOLERANCE = 1e-9


class HiddenMarkovModel:
    """A stationary hidden Markov model in token-labelled form.

    Attributes:
        token_transition_matrices: Read-only array of shape (n_tokens, n_states, n_states).
            Entry [x, i, j] is P(emit x and move to j | in state i).
        token_names: One display name per token, used in plots.
        n_tokens: Number of distinct tokens.
        n_states: Number of hidden states.
    """

    def __init__(
        self,
        token_transition_matrices: np.ndarray,
        token_names: Sequence[str] | None = None,
    ) -> None:
        # np.array (not np.asarray) makes a copy, so later changes to the caller's
        # array cannot silently change this model.
        matrices = np.array(token_transition_matrices, dtype=np.float64)
        _validate_token_transition_matrices(matrices)
        matrices.setflags(write=False)

        self.token_transition_matrices = matrices
        self.n_tokens = matrices.shape[0]
        self.n_states = matrices.shape[1]

        if token_names is None:
            token_names = [str(token_index) for token_index in range(self.n_tokens)]
        if len(token_names) != self.n_tokens:
            raise ValueError(
                f"Got {len(token_names)} token names for {self.n_tokens} tokens."
            )
        self.token_names = tuple(token_names)

    def state_transition_matrix(self) -> np.ndarray:
        """Return T with T[i, j] = P(move to j | in i), ignoring which token is emitted."""
        return self.token_transition_matrices.sum(axis=0)

    def token_probabilities_given_state(self) -> np.ndarray:
        """Return an (n_states, n_tokens) array with [i, x] = P(next token is x | in state i).

        This sums the token-labelled matrices over the destination state j.
        """
        # Sum over j (last axis) gives shape (n_tokens, n_states); transpose to (n_states, n_tokens).
        return self.token_transition_matrices.sum(axis=2).T

    def stationary_distribution(self) -> np.ndarray:
        """Return the stationary distribution pi over hidden states, with pi T = pi.

        We solve the linear system directly rather than running the chain forward:
            (T transposed - I) pi = 0     (pi is unchanged by one step)
            sum of pi = 1                 (pi is a probability vector)
        Stacking these gives n_states + 1 equations in n_states unknowns, which
        least squares solves exactly when a unique solution exists.

        Raises:
            ValueError: If the stationary distribution is not unique (the chain is
                reducible), or the solution is not a valid probability vector.
        """
        transition_matrix = self.state_transition_matrix()
        identity = np.eye(self.n_states)
        fixed_point_equations = transition_matrix.T - identity

        # For an irreducible chain the fixed point equations have rank n_states - 1:
        # exactly one direction (pi itself) is left free, and normalisation pins it down.
        # A lower rank means several separate stationary distributions exist.
        if np.linalg.matrix_rank(fixed_point_equations) < self.n_states - 1:
            raise ValueError("Stationary distribution is not unique: the chain is reducible.")

        normalisation_equation = np.ones((1, self.n_states))
        system_matrix = np.vstack([fixed_point_equations, normalisation_equation])
        right_hand_side = np.zeros(self.n_states + 1)
        right_hand_side[-1] = 1.0

        stationary, *_ = np.linalg.lstsq(system_matrix, right_hand_side, rcond=None)

        # Least squares can return tiny negative values like -1e-17 for states with
        # zero stationary probability. Clip those, but fail on anything larger.
        if np.any(stationary < -PROBABILITY_SUM_TOLERANCE):
            raise ValueError(f"Stationary solution has negative entries: {stationary}")
        stationary = np.clip(stationary, 0.0, None)
        return stationary / stationary.sum()

    def stationary_token_distribution(self) -> np.ndarray:
        """Return P(token = x) for a single token emitted from the stationary distribution.

        P(x) = sum over i of pi[i] * P(next token is x | in state i).
        """
        return self.stationary_distribution() @ self.token_probabilities_given_state()


def _validate_token_transition_matrices(matrices: np.ndarray) -> None:
    """Raise ValueError if `matrices` is not a valid (n_tokens, n_states, n_states) array."""
    if matrices.ndim != 3:
        raise ValueError(
            f"Expected shape (n_tokens, n_states, n_states), got {matrices.shape}."
        )
    _, n_rows, n_columns = matrices.shape
    if n_rows != n_columns:
        raise ValueError(f"Each token matrix must be square, got {n_rows} x {n_columns}.")
    if np.any(matrices < 0):
        raise ValueError("Probabilities must be non-negative.")

    # For each current state i: total probability over every (token, next state) pair.
    row_totals = matrices.sum(axis=(0, 2))
    if not np.allclose(row_totals, 1.0, atol=PROBABILITY_SUM_TOLERANCE, rtol=0.0):
        raise ValueError(
            "For each current state, probabilities over all (token, next state) pairs "
            f"must sum to 1. Got row totals {row_totals}."
        )


def hmm_from_transition_and_emission(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    token_names: Sequence[str] | None = None,
) -> HiddenMarkovModel:
    """Build a token-labelled HMM from a textbook "move, then emit" description.

    At each step the hidden state first moves from i to j with probability
    transition_matrix[i, j], and the new state j then emits token x with
    probability emission_matrix[j, x]. So

        token_transition_matrices[x, i, j] = transition_matrix[i, j] * emission_matrix[j, x].

    Args:
        transition_matrix: Shape (n_states, n_states). Rows sum to 1.
        emission_matrix: Shape (n_states, n_tokens). Rows sum to 1.
        token_names: Optional display names, one per token.

    Returns:
        The equivalent HiddenMarkovModel.
    """
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    emission_matrix = np.asarray(emission_matrix, dtype=np.float64)
    n_states = transition_matrix.shape[0]
    n_tokens = emission_matrix.shape[1]

    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(f"Transition matrix must be square, got {transition_matrix.shape}.")
    if emission_matrix.shape[0] != n_states:
        raise ValueError(
            f"Emission matrix needs one row per state ({n_states}), "
            f"got {emission_matrix.shape[0]}."
        )

    token_transition_matrices = np.empty((n_tokens, n_states, n_states))
    for token in range(n_tokens):
        # Scale column j of the transition matrix by P(new state j emits this token).
        probability_new_state_emits_token = emission_matrix[:, token]
        token_transition_matrices[token] = (
            transition_matrix * probability_new_state_emits_token[np.newaxis, :]
        )

    return HiddenMarkovModel(token_transition_matrices, token_names)


def make_mess3() -> HiddenMarkovModel:
    """Return the Mess3 process exactly as printed by Shai et al. (2024).

    Source: Shai et al., arXiv:2405.15943 version 3, Appendix A.3. The numbers
    below are copied digit for digit from the three printed matrices T(A), T(B) and
    T(C). The paper does not print the parameters that generate them; see
    `make_mess3_from_parameters` and the test that checks the two agree.
    """
    matrix_for_token_a = [
        [0.765, 0.00375, 0.00375],
        [0.0425, 0.0675, 0.00375],
        [0.0425, 0.00375, 0.0675],
    ]
    matrix_for_token_b = [
        [0.0675, 0.0425, 0.00375],
        [0.00375, 0.765, 0.00375],
        [0.00375, 0.0425, 0.0675],
    ]
    matrix_for_token_c = [
        [0.0675, 0.00375, 0.0425],
        [0.00375, 0.0675, 0.0425],
        [0.00375, 0.00375, 0.765],
    ]
    token_transition_matrices = np.array(
        [matrix_for_token_a, matrix_for_token_b, matrix_for_token_c]
    )
    return HiddenMarkovModel(token_transition_matrices, token_names=("A", "B", "C"))


def make_mess3_from_parameters(
    jump_probability: float, emission_accuracy: float
) -> HiddenMarkovModel:
    """Build Mess3 from its two parameters.

    Source: Marzen and Crutchfield (2017), "Nearly maximally predictive features and
    their dimensions", Phys. Rev. E 95, 051301(R); arXiv:1702.08565,
    Supplementary Materials, Appendix A. They write the matrices in terms of x and
    alpha, with beta = (1 - alpha) / 2 and y = 1 - 2x, for example
        T(0) = [[alpha*y, beta*x, beta*x], [alpha*x, beta*y, beta*x], [alpha*x, beta*x, beta*y]].
    Each entry factorises as P(move i to j) * P(state j emits the token), so we
    build it through `hmm_from_transition_and_emission`, which is easier to read:

    * The hidden state stays put with probability 1 - 2x and jumps to each of the
      other two states with probability x (x is `jump_probability`).
    * The new state emits "its own" token with probability alpha and each of the
      other two tokens with probability (1 - alpha) / 2 (alpha is `emission_accuracy`).

    Args:
        jump_probability: x in the source, between 0 and 0.5.
        emission_accuracy: alpha in the source, between 0 and 1.
    """
    if not 0.0 <= jump_probability <= 0.5:
        raise ValueError(f"jump_probability must be in [0, 0.5], got {jump_probability}.")
    if not 0.0 <= emission_accuracy <= 1.0:
        raise ValueError(f"emission_accuracy must be in [0, 1], got {emission_accuracy}.")

    n_states = 3
    transition_matrix = np.full((n_states, n_states), jump_probability)
    np.fill_diagonal(transition_matrix, 1.0 - 2.0 * jump_probability)

    off_target_emission = (1.0 - emission_accuracy) / 2.0
    emission_matrix = np.full((n_states, n_states), off_target_emission)
    np.fill_diagonal(emission_matrix, emission_accuracy)

    return hmm_from_transition_and_emission(
        transition_matrix, emission_matrix, token_names=("A", "B", "C")
    )


class SampledSequences(NamedTuple):
    """Token sequences together with the hidden states that produced them.

    Attributes:
        tokens: Integer array of shape (n_sequences, sequence_length).
        hidden_states: Integer array of shape (n_sequences, sequence_length + 1).
            hidden_states[:, t] is the state just before token t is emitted, and
            hidden_states[:, t + 1] is the state that step moved to. The extra
            column is the state after the final token.
    """

    tokens: np.ndarray
    hidden_states: np.ndarray


def sample_sequences(
    hmm: HiddenMarkovModel,
    n_sequences: int,
    sequence_length: int,
    rng: np.random.Generator,
    initial_state_distribution: np.ndarray | None = None,
) -> SampledSequences:
    """Sample independent token sequences from `hmm`.

    Each sequence starts from a hidden state drawn from `initial_state_distribution`
    (the stationary distribution by default, as in Shai et al. 2024, Appendix A.6).
    All sequences are advanced together, one position at a time.

    Args:
        hmm: The process to sample from.
        n_sequences: Number of independent sequences.
        sequence_length: Number of tokens per sequence.
        rng: NumPy random generator (see src/seeding.py).
        initial_state_distribution: Probability vector over hidden states.

    Returns:
        SampledSequences with the tokens and the hidden states.
    """
    if initial_state_distribution is None:
        initial_state_distribution = hmm.stationary_distribution()
    initial_state_distribution = np.asarray(initial_state_distribution, dtype=np.float64)
    if initial_state_distribution.shape != (hmm.n_states,):
        raise ValueError(
            f"Initial distribution must have shape ({hmm.n_states},), "
            f"got {initial_state_distribution.shape}."
        )

    cumulative_table = _cumulative_joint_table(hmm.token_transition_matrices)

    tokens = np.empty((n_sequences, sequence_length), dtype=np.int64)
    hidden_states = np.empty((n_sequences, sequence_length + 1), dtype=np.int64)
    hidden_states[:, 0] = rng.choice(hmm.n_states, size=n_sequences, p=initial_state_distribution)

    for position in range(sequence_length):
        token, next_state = _sample_token_and_next_state(
            cumulative_table, hidden_states[:, position], hmm.n_states, rng
        )
        tokens[:, position] = token
        hidden_states[:, position + 1] = next_state

    return SampledSequences(tokens=tokens, hidden_states=hidden_states)


def _cumulative_joint_table(token_transition_matrices: np.ndarray) -> np.ndarray:
    """Rearrange the matrices into one cumulative distribution per current state.

    Returns an array of shape (n_states, n_tokens * n_states). Row i is the
    cumulative sum of P(token x, next state j | state i), with the outcome (x, j)
    stored at column x * n_states + j.
    """
    n_tokens, n_states, _ = token_transition_matrices.shape

    # Axis order (x, i, j) -> (i, x, j), then merge (x, j) into one axis per row i.
    per_current_state = np.transpose(token_transition_matrices, (1, 0, 2))
    joint_table = per_current_state.reshape(n_states, n_tokens * n_states)

    cumulative_table = np.cumsum(joint_table, axis=1)
    # Rounding can leave the last cumulative value just below 1. Setting it to
    # exactly 1 guarantees every uniform draw in [0, 1) falls inside some outcome.
    cumulative_table[:, -1] = 1.0
    return cumulative_table


def _sample_token_and_next_state(
    cumulative_table: np.ndarray,
    current_states: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one (token, next state) pair for each sequence by inverse transform sampling.

    For a uniform draw u, the chosen outcome is the first column whose cumulative
    probability exceeds u. Counting the columns whose cumulative value is <= u gives
    that column's index directly. Outcomes with zero probability have the same
    cumulative value as the column before them, so they can never be chosen.
    """
    cumulative_rows = cumulative_table[current_states]
    uniform_draws = rng.random(size=(len(current_states), 1))
    outcome_index = np.sum(uniform_draws >= cumulative_rows, axis=1)

    # Undo the flattening used in _cumulative_joint_table: column = x * n_states + j.
    token = outcome_index // n_states
    next_state = outcome_index % n_states
    return token, next_state
