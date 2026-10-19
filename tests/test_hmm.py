"""Tests for src/hmm.py: model definitions, stationary statistics and sampling.

Mess3 is symmetric, so its stationary distribution is uniform and each token has
frequency 1/3. A broken sampler could easily reproduce those numbers, so most
tests also run on the two hand-checkable processes in tests/example_processes.py.
"""

from typing import Callable

import numpy as np
import pytest

from src.hmm import (
    HiddenMarkovModel,
    hmm_from_transition_and_emission,
    make_mess3,
    make_mess3_from_parameters,
    sample_sequences,
)
from tests.example_processes import make_asymmetric_two_state_hmm, make_golden_mean_hmm

ALL_TEST_PROCESSES: dict[str, Callable[[], HiddenMarkovModel]] = {
    "mess3": make_mess3,
    "asymmetric_two_state": make_asymmetric_two_state_hmm,
    "golden_mean": make_golden_mean_hmm,
}


def analytic_bigram_distribution(hmm: HiddenMarkovModel) -> np.ndarray:
    """P(token x then token y) at stationarity: pi @ T(x) @ T(y) @ ones.

    Returns an (n_tokens, n_tokens) array indexed [x, y].
    """
    stationary = hmm.stationary_distribution()
    ones = np.ones(hmm.n_states)
    bigram = np.zeros((hmm.n_tokens, hmm.n_tokens))
    for first_token in range(hmm.n_tokens):
        for second_token in range(hmm.n_tokens):
            bigram[first_token, second_token] = (
                stationary
                @ hmm.token_transition_matrices[first_token]
                @ hmm.token_transition_matrices[second_token]
                @ ones
            )
    return bigram


def empirical_unigram_distribution(tokens: np.ndarray, n_tokens: int) -> np.ndarray:
    counts = np.bincount(tokens.ravel(), minlength=n_tokens)
    return counts / counts.sum()


def empirical_bigram_distribution(tokens: np.ndarray, n_tokens: int) -> np.ndarray:
    """Frequencies of adjacent token pairs within each sequence, indexed [x, y]."""
    first_tokens = tokens[:, :-1].ravel()
    second_tokens = tokens[:, 1:].ravel()
    pair_codes = first_tokens * n_tokens + second_tokens
    counts = np.bincount(pair_codes, minlength=n_tokens * n_tokens)
    return (counts / counts.sum()).reshape(n_tokens, n_tokens)


# Sample size and tolerance for the frequency tests. 20,000 sequences of length 10
# give 200,000 tokens. Tokens within a sequence are correlated, so the usual
# independent-draw formula underestimates the standard error. We measured it
# directly instead: across 100 seeds, the largest standard error of any unigram or
# bigram frequency was 0.0024 (Mess3), 0.0017 (asymmetric) and 0.0013 (golden mean).
# A tolerance of 0.01 is therefore at least 4 standard errors. The seed is fixed,
# so the outcome is deterministic.
N_SEQUENCES_FOR_FREQUENCIES = 20_000
SEQUENCE_LENGTH_FOR_FREQUENCIES = 10
FREQUENCY_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_array_that_is_not_three_dimensional() -> None:
    with pytest.raises(ValueError, match="shape"):
        HiddenMarkovModel(np.ones((2, 2)) / 2)


def test_rejects_non_square_token_matrices() -> None:
    with pytest.raises(ValueError, match="square"):
        HiddenMarkovModel(np.ones((2, 2, 3)) / 6)


def test_rejects_negative_probabilities() -> None:
    matrices = np.array([[[1.2, -0.2], [0.5, 0.5]]])
    with pytest.raises(ValueError, match="non-negative"):
        HiddenMarkovModel(matrices)


def test_rejects_rows_that_do_not_sum_to_one() -> None:
    matrices = np.array([[[0.5, 0.4], [0.5, 0.5]]])  # first row sums to 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        HiddenMarkovModel(matrices)


def test_rejects_wrong_number_of_token_names() -> None:
    with pytest.raises(ValueError, match="token names"):
        HiddenMarkovModel(make_golden_mean_hmm().token_transition_matrices, token_names=("a",))


def test_model_keeps_its_own_copy_of_the_matrices() -> None:
    matrices = make_golden_mean_hmm().token_transition_matrices.copy()
    hmm = HiddenMarkovModel(matrices)
    matrices[0, 0, 0] = 999.0
    assert hmm.token_transition_matrices[0, 0, 0] == 0.5


# ---------------------------------------------------------------------------
# Mess3 definition
# ---------------------------------------------------------------------------


def test_mess3_has_three_states_and_three_tokens() -> None:
    mess3 = make_mess3()
    assert mess3.n_states == 3
    assert mess3.n_tokens == 3
    assert mess3.token_names == ("A", "B", "C")


def test_mess3_rows_sum_to_one_over_tokens_and_next_states() -> None:
    row_totals = make_mess3().token_transition_matrices.sum(axis=(0, 2))
    np.testing.assert_allclose(row_totals, np.ones(3), atol=1e-12)


def test_mess3_printed_matrices_match_marzen_crutchfield_parametrisation() -> None:
    # Checks the matrices typed in from Shai et al. (Appendix A.3) against an
    # independent construction from the parametric form in Marzen and Crutchfield
    # (2017). A typo in any of the 27 numbers would make this fail. The values
    # x = 0.05 and alpha = 0.85 are our inference: the paper prints only the matrices.
    printed = make_mess3().token_transition_matrices
    from_parameters = make_mess3_from_parameters(
        jump_probability=0.05, emission_accuracy=0.85
    ).token_transition_matrices
    np.testing.assert_allclose(printed, from_parameters, atol=1e-12)


def test_mess3_hidden_state_stays_with_probability_0_9() -> None:
    expected = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    np.testing.assert_allclose(make_mess3().state_transition_matrix(), expected, atol=1e-12)


def test_move_then_emit_construction_matches_hand_calculation() -> None:
    transition_matrix = np.array([[0.9, 0.1], [0.3, 0.7]])
    emission_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    hmm = hmm_from_transition_and_emission(transition_matrix, emission_matrix)
    # Emit token 0 while moving from state 0 to state 1:
    # P(move 0 -> 1) * P(state 1 emits 0) = 0.1 * 0.2.
    assert hmm.token_transition_matrices[0, 0, 1] == pytest.approx(0.02)
    # Emit token 1 while staying in state 1: 0.7 * 0.8.
    assert hmm.token_transition_matrices[1, 1, 1] == pytest.approx(0.56)


# ---------------------------------------------------------------------------
# Stationary distribution and analytic token statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process_name", ALL_TEST_PROCESSES)
def test_stationary_distribution_is_a_fixed_point_probability_vector(process_name: str) -> None:
    hmm = ALL_TEST_PROCESSES[process_name]()
    stationary = hmm.stationary_distribution()

    assert stationary.shape == (hmm.n_states,)
    assert np.all(stationary >= 0)
    assert stationary.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(stationary @ hmm.state_transition_matrix(), stationary, atol=1e-12)


def test_asymmetric_stationary_distribution_matches_hand_calculation() -> None:
    stationary = make_asymmetric_two_state_hmm().stationary_distribution()
    np.testing.assert_allclose(stationary, [0.75, 0.25], atol=1e-12)


def test_golden_mean_stationary_distribution_matches_hand_calculation() -> None:
    stationary = make_golden_mean_hmm().stationary_distribution()
    np.testing.assert_allclose(stationary, [2 / 3, 1 / 3], atol=1e-12)


def test_mess3_stationary_distribution_is_uniform() -> None:
    np.testing.assert_allclose(make_mess3().stationary_distribution(), np.ones(3) / 3, atol=1e-12)


def test_stationary_distribution_rejects_reducible_chain() -> None:
    # Two states that never communicate: any mixture of them is stationary.
    matrices = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="not unique"):
        HiddenMarkovModel(matrices).stationary_distribution()


def test_asymmetric_token_distribution_matches_hand_calculation() -> None:
    # P(token 0) = 0.75 * 0.8 + 0.25 * 0.2 = 0.65
    token_distribution = make_asymmetric_two_state_hmm().stationary_token_distribution()
    np.testing.assert_allclose(token_distribution, [0.65, 0.35], atol=1e-12)


def test_analytic_bigram_helper_matches_hand_calculation() -> None:
    # Checks the test helper itself before the sampler is compared against it.
    # By hand: pi @ T(0) = (0.555, 0.095), pi @ T(1) = (0.195, 0.155), and the rows
    # of T(0) and T(1) sum to (0.8, 0.2) and (0.2, 0.8). For example
    # P(0, 0) = 0.555 * 0.8 + 0.095 * 0.2 = 0.463.
    expected = np.array([[0.463, 0.187], [0.187, 0.163]])
    np.testing.assert_allclose(
        analytic_bigram_distribution(make_asymmetric_two_state_hmm()), expected, atol=1e-12
    )


def test_asymmetric_tokens_are_correlated() -> None:
    # If adjacent tokens were independent, P(0, 0) would be 0.65 ** 2 = 0.4225.
    # The bigram test below is only informative because this differs.
    bigram = analytic_bigram_distribution(make_asymmetric_two_state_hmm())
    assert abs(bigram[0, 0] - 0.65**2) > 0.03


# ---------------------------------------------------------------------------
# Sampling: shapes, ranges and reproducibility
# ---------------------------------------------------------------------------


def test_sample_shapes_and_dtypes() -> None:
    samples = sample_sequences(make_mess3(), n_sequences=7, sequence_length=10, rng=np.random.default_rng(0))
    assert samples.tokens.shape == (7, 10)
    assert samples.hidden_states.shape == (7, 11)
    assert samples.tokens.dtype == np.int64
    assert samples.hidden_states.dtype == np.int64


@pytest.mark.parametrize("process_name", ALL_TEST_PROCESSES)
def test_sampled_values_are_in_range(process_name: str) -> None:
    hmm = ALL_TEST_PROCESSES[process_name]()
    samples = sample_sequences(hmm, n_sequences=500, sequence_length=20, rng=np.random.default_rng(0))
    assert samples.tokens.min() >= 0
    assert samples.tokens.max() < hmm.n_tokens
    assert samples.hidden_states.min() >= 0
    assert samples.hidden_states.max() < hmm.n_states


def test_sampling_is_reproducible_with_the_same_seed() -> None:
    first = sample_sequences(make_mess3(), 50, 10, rng=np.random.default_rng(42))
    second = sample_sequences(make_mess3(), 50, 10, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(first.tokens, second.tokens)
    np.testing.assert_array_equal(first.hidden_states, second.hidden_states)


def test_initial_state_follows_given_initial_distribution() -> None:
    samples = sample_sequences(
        make_mess3(),
        n_sequences=200,
        sequence_length=5,
        rng=np.random.default_rng(0),
        initial_state_distribution=np.array([0.0, 0.0, 1.0]),
    )
    assert np.all(samples.hidden_states[:, 0] == 2)


# ---------------------------------------------------------------------------
# Sampling: agreement with the model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("process_name", ALL_TEST_PROCESSES)
def test_every_sampled_step_has_nonzero_probability(process_name: str) -> None:
    # Catches indexing bugs that produce (state, token, next state) triples the
    # model forbids. Most informative for the golden mean process, which has zeros.
    hmm = ALL_TEST_PROCESSES[process_name]()
    samples = sample_sequences(hmm, n_sequences=2_000, sequence_length=20, rng=np.random.default_rng(1))

    current_states = samples.hidden_states[:, :-1]
    next_states = samples.hidden_states[:, 1:]
    step_probabilities = hmm.token_transition_matrices[samples.tokens, current_states, next_states]
    assert np.all(step_probabilities > 0)


def test_golden_mean_never_emits_two_ones_in_a_row() -> None:
    samples = sample_sequences(make_golden_mean_hmm(), 2_000, 20, rng=np.random.default_rng(2))
    consecutive_ones = (samples.tokens[:, :-1] == 1) & (samples.tokens[:, 1:] == 1)
    assert not consecutive_ones.any()
    # And 1s do occur, so the test above is not passing trivially.
    assert (samples.tokens == 1).any()


@pytest.mark.parametrize("process_name", ALL_TEST_PROCESSES)
def test_empirical_token_frequencies_match_analytic(process_name: str) -> None:
    hmm = ALL_TEST_PROCESSES[process_name]()
    samples = sample_sequences(
        hmm, N_SEQUENCES_FOR_FREQUENCIES, SEQUENCE_LENGTH_FOR_FREQUENCIES, rng=np.random.default_rng(3)
    )
    empirical = empirical_unigram_distribution(samples.tokens, hmm.n_tokens)
    np.testing.assert_allclose(empirical, hmm.stationary_token_distribution(), atol=FREQUENCY_TOLERANCE)


@pytest.mark.parametrize("process_name", ALL_TEST_PROCESSES)
def test_empirical_bigram_frequencies_match_analytic(process_name: str) -> None:
    hmm = ALL_TEST_PROCESSES[process_name]()
    samples = sample_sequences(
        hmm, N_SEQUENCES_FOR_FREQUENCIES, SEQUENCE_LENGTH_FOR_FREQUENCIES, rng=np.random.default_rng(4)
    )
    empirical = empirical_bigram_distribution(samples.tokens, hmm.n_tokens)
    np.testing.assert_allclose(empirical, analytic_bigram_distribution(hmm), atol=FREQUENCY_TOLERANCE)


def test_mess3_sampled_steps_match_token_transition_matrices() -> None:
    # The most direct check of the sampler: for each current state i, the observed
    # frequency of each (token x, next state j) pair should equal the matrix entry
    # T(x)[i, j]. Given the current state, each step is an independent draw, so the
    # binomial standard error sqrt(p * (1 - p) / n) is valid here. We allow 4 of them
    # per entry. A single fixed tolerance would not work: one loose enough for the
    # 0.765 entries would miss a mistake in the 0.00375 entries.
    hmm = make_mess3()
    samples = sample_sequences(
        hmm, N_SEQUENCES_FOR_FREQUENCIES, SEQUENCE_LENGTH_FOR_FREQUENCIES, rng=np.random.default_rng(5)
    )
    current_states = samples.hidden_states[:, :-1].ravel()
    next_states = samples.hidden_states[:, 1:].ravel()
    tokens = samples.tokens.ravel()

    for current_state in range(hmm.n_states):
        from_this_state = current_states == current_state
        n_steps_from_this_state = from_this_state.sum()
        counts = np.zeros((hmm.n_tokens, hmm.n_states))
        np.add.at(counts, (tokens[from_this_state], next_states[from_this_state]), 1)
        empirical = counts / n_steps_from_this_state

        expected = hmm.token_transition_matrices[:, current_state, :]
        standard_error = np.sqrt(expected * (1 - expected) / n_steps_from_this_state)
        deviation_in_standard_errors = np.abs(empirical - expected) / standard_error
        assert np.all(deviation_in_standard_errors < 4), (
            f"State {current_state}: largest deviation is "
            f"{deviation_in_standard_errors.max():.1f} standard errors."
        )
