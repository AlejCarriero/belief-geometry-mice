"""Tests for src/beliefs.py.

Two groups:

* test_forward_*: tests for compute_belief_states, which you write. They fail with
  NotImplementedError until then. Run only these with:
      python -m pytest -k forward
* test_optimal_loss_* and test_reference_*: tests for the loss functions and for the
  brute-force reference below. These use the brute-force beliefs, not your
  function, so they pass already.

The reference computes beliefs by a completely different route: it lists every
possible hidden path, multiplies the probabilities along each path, and adds up
the paths that end in each state. It is far too slow for real use (the number of
paths grows as n_states ** length) but it is easy to check by eye, which is what
makes it a good reference.
"""

import itertools

import numpy as np
import pytest

from src.beliefs import (
    bayes_optimal_token_losses,
    compute_belief_states,
    next_token_distributions,
)
from src.hmm import HiddenMarkovModel, make_mess3, sample_sequences
from tests.example_processes import make_asymmetric_two_state_hmm, make_golden_mean_hmm

SMALL_TEST_PROCESSES = {
    "mess3": make_mess3,
    "asymmetric_two_state": make_asymmetric_two_state_hmm,
    "golden_mean": make_golden_mean_hmm,
}


# ---------------------------------------------------------------------------
# Brute-force reference: sum over every hidden path
# ---------------------------------------------------------------------------


def path_probability(
    hmm: HiddenMarkovModel,
    token_sequence: tuple[int, ...],
    hidden_path: tuple[int, ...],
    initial_belief: np.ndarray,
) -> float:
    """P(hidden path and tokens) = initial[s_0] * T(x_0)[s_0, s_1] * T(x_1)[s_1, s_2] * ...

    `hidden_path` has one more entry than `token_sequence`.
    """
    probability = initial_belief[hidden_path[0]]
    for step, token in enumerate(token_sequence):
        probability *= hmm.token_transition_matrices[token, hidden_path[step], hidden_path[step + 1]]
    return probability


def brute_force_beliefs(
    hmm: HiddenMarkovModel, token_sequence: tuple[int, ...], initial_belief: np.ndarray
) -> np.ndarray:
    """Beliefs for one sequence, shape (len(token_sequence) + 1, n_states).

    Row t is P(s_t = j | tokens 0 .. t-1): add up the probability of every hidden path
    s_0 .. s_t that ends in state j, then normalise.
    """
    sequence_length = len(token_sequence)
    beliefs = np.zeros((sequence_length + 1, hmm.n_states))
    for prefix_length in range(sequence_length + 1):
        prefix = token_sequence[:prefix_length]
        unnormalised = np.zeros(hmm.n_states)
        for hidden_path in itertools.product(range(hmm.n_states), repeat=prefix_length + 1):
            final_state = hidden_path[-1]
            unnormalised[final_state] += path_probability(hmm, prefix, hidden_path, initial_belief)
        beliefs[prefix_length] = unnormalised / unnormalised.sum()
    return beliefs


def brute_force_sequence_probability(
    hmm: HiddenMarkovModel, token_sequence: tuple[int, ...], initial_belief: np.ndarray
) -> float:
    """P(token sequence) = sum over every hidden path of P(path and tokens)."""
    return sum(
        path_probability(hmm, token_sequence, hidden_path, initial_belief)
        for hidden_path in itertools.product(range(hmm.n_states), repeat=len(token_sequence) + 1)
    )


def all_possible_sequences(
    hmm: HiddenMarkovModel, sequence_length: int, initial_belief: np.ndarray
) -> list[tuple[int, ...]]:
    """Every token sequence of this length that has non-zero probability."""
    return [
        token_sequence
        for token_sequence in itertools.product(range(hmm.n_tokens), repeat=sequence_length)
        if brute_force_sequence_probability(hmm, token_sequence, initial_belief) > 0
    ]


# ---------------------------------------------------------------------------
# Checks on the reference itself (hand calculations)
# ---------------------------------------------------------------------------


def test_reference_mess3_after_one_token_matches_hand_calculation() -> None:
    # From the uniform belief, one token A: the belief is proportional to the column
    # sums of T(A) divided by 3. Column sums are 0.85, 0.075, 0.075 (each state's
    # chance of emitting A after arriving), which already sum to 1.
    mess3 = make_mess3()
    beliefs = brute_force_beliefs(mess3, (0,), mess3.stationary_distribution())
    np.testing.assert_allclose(beliefs[1], [0.85, 0.075, 0.075], atol=1e-12)


def test_reference_golden_mean_token_reveals_state() -> None:
    # In the golden mean process a 1 always leads to state 1 and a 0 to state 0.
    golden_mean = make_golden_mean_hmm()
    beliefs = brute_force_beliefs(golden_mean, (0, 1, 0), golden_mean.stationary_distribution())
    np.testing.assert_allclose(beliefs[1:], [[1, 0], [0, 1], [1, 0]], atol=1e-12)


# ---------------------------------------------------------------------------
# Your forward algorithm: basic contract
# ---------------------------------------------------------------------------


def test_forward_output_shape_and_dtype() -> None:
    mess3 = make_mess3()
    tokens = np.zeros((4, 7), dtype=np.int64)
    beliefs = compute_belief_states(mess3, tokens)
    assert beliefs.shape == (4, 8, 3)
    assert beliefs.dtype == np.float64


def test_forward_first_belief_is_stationary_by_default() -> None:
    mess3 = make_mess3()
    beliefs = compute_belief_states(mess3, np.zeros((2, 3), dtype=np.int64))
    np.testing.assert_allclose(beliefs[:, 0], np.tile(np.ones(3) / 3, (2, 1)), atol=1e-12)


def test_forward_first_belief_is_initial_belief_when_given() -> None:
    mess3 = make_mess3()
    initial_belief = np.array([0.5, 0.3, 0.2])
    beliefs = compute_belief_states(mess3, np.zeros((2, 3), dtype=np.int64), initial_belief)
    np.testing.assert_allclose(beliefs[:, 0], np.tile(initial_belief, (2, 1)), atol=1e-12)


def test_forward_every_belief_is_a_probability_vector() -> None:
    mess3 = make_mess3()
    samples = sample_sequences(mess3, n_sequences=500, sequence_length=10, rng=np.random.default_rng(0))
    beliefs = compute_belief_states(mess3, samples.tokens)
    assert np.all(beliefs >= 0)
    np.testing.assert_allclose(beliefs.sum(axis=2), 1.0, atol=1e-12)


def test_forward_empty_sequences_return_only_the_initial_belief() -> None:
    mess3 = make_mess3()
    beliefs = compute_belief_states(mess3, np.zeros((3, 0), dtype=np.int64))
    assert beliefs.shape == (3, 1, 3)
    np.testing.assert_allclose(beliefs[:, 0], np.tile(np.ones(3) / 3, (3, 1)), atol=1e-12)


# ---------------------------------------------------------------------------
# Your forward algorithm: correctness
# ---------------------------------------------------------------------------


def test_forward_mess3_after_one_token_matches_hand_calculation() -> None:
    mess3 = make_mess3()
    beliefs = compute_belief_states(mess3, np.array([[0]]))  # token A
    np.testing.assert_allclose(beliefs[0, 1], [0.85, 0.075, 0.075], atol=1e-12)


@pytest.mark.parametrize("process_name", SMALL_TEST_PROCESSES)
def test_forward_matches_brute_force_for_every_short_sequence(process_name: str) -> None:
    # Every possible sequence of length 4 (81 for Mess3, fewer for the others).
    hmm = SMALL_TEST_PROCESSES[process_name]()
    initial_belief = hmm.stationary_distribution()
    for token_sequence in all_possible_sequences(hmm, 4, initial_belief):
        expected = brute_force_beliefs(hmm, token_sequence, initial_belief)
        actual = compute_belief_states(hmm, np.array([token_sequence]))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-12, err_msg=f"tokens {token_sequence}")


def test_forward_matches_brute_force_with_a_non_stationary_start() -> None:
    hmm = make_asymmetric_two_state_hmm()
    initial_belief = np.array([0.1, 0.9])
    for token_sequence in all_possible_sequences(hmm, 4, initial_belief):
        expected = brute_force_beliefs(hmm, token_sequence, initial_belief)
        actual = compute_belief_states(hmm, np.array([token_sequence]), initial_belief)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-12, err_msg=f"tokens {token_sequence}")


def test_forward_handles_a_batch_of_different_sequences() -> None:
    # The same answers whether sequences are processed together or one at a time.
    # Catches, for example, normalising across the batch instead of per sequence.
    mess3 = make_mess3()
    samples = sample_sequences(mess3, n_sequences=6, sequence_length=8, rng=np.random.default_rng(1))
    together = compute_belief_states(mess3, samples.tokens)
    for sequence_index in range(6):
        alone = compute_belief_states(mess3, samples.tokens[sequence_index : sequence_index + 1])
        np.testing.assert_allclose(together[sequence_index], alone[0], atol=1e-12)


def test_forward_golden_mean_belief_equals_true_state_after_first_token() -> None:
    # Ties your alignment convention to the sampler's: beliefs[:, t] must describe
    # hidden_states[:, t]. In the golden mean process every token reveals the state.
    golden_mean = make_golden_mean_hmm()
    samples = sample_sequences(golden_mean, 200, 10, rng=np.random.default_rng(2))
    beliefs = compute_belief_states(golden_mean, samples.tokens)
    true_state_one_hot = np.eye(2)[samples.hidden_states[:, 1:]]
    np.testing.assert_allclose(beliefs[:, 1:], true_state_one_hot, atol=1e-12)


# Sampling noise sets the tolerance. With a correct forward algorithm, over 20
# different seeds, the worst bin deviation was 0.0061 on average and 0.0086 at most.
# Bins with fewer than 5,000 points are skipped because they are noisier.
MIN_POINTS_PER_CALIBRATION_BIN = 5_000
CALIBRATION_TOLERANCE = 0.015


def test_forward_beliefs_are_calibrated_against_sampled_hidden_states() -> None:
    # If beliefs are the true posterior, then whenever the belief gives state 0 a
    # probability of about p, the hidden state really is 0 about a fraction p of the
    # time. We group beliefs into 10 bins and compare.
    mess3 = make_mess3()
    samples = sample_sequences(mess3, n_sequences=50_000, sequence_length=10, rng=np.random.default_rng(3))
    beliefs = compute_belief_states(mess3, samples.tokens)

    belief_in_state_0 = beliefs[:, :, 0].ravel()
    is_in_state_0 = (samples.hidden_states == 0).ravel()
    bin_edges = np.linspace(0, 1, 11)
    bin_of_each_point = np.digitize(belief_in_state_0, bin_edges[1:-1])

    for bin_index in range(10):
        in_bin = bin_of_each_point == bin_index
        if in_bin.sum() < MIN_POINTS_PER_CALIBRATION_BIN:
            continue
        mean_belief = belief_in_state_0[in_bin].mean()
        observed_fraction = is_in_state_0[in_bin].mean()
        assert abs(observed_fraction - mean_belief) < CALIBRATION_TOLERANCE, (
            f"Bin {bin_index}: mean belief {mean_belief:.3f}, observed {observed_fraction:.3f}"
        )


def test_forward_long_sequence_does_not_underflow() -> None:
    # Each Mess3 token has probability around 1/3, so the probability of a whole
    # 2,000-token sequence is around 3 ** -2000, far below the smallest float64.
    # Only per-step normalisation keeps the beliefs finite.
    mess3 = make_mess3()
    samples = sample_sequences(mess3, n_sequences=2, sequence_length=2_000, rng=np.random.default_rng(4))
    beliefs = compute_belief_states(mess3, samples.tokens)
    assert np.all(np.isfinite(beliefs))
    np.testing.assert_allclose(beliefs.sum(axis=2), 1.0, atol=1e-9)


def test_forward_raises_on_an_impossible_sequence() -> None:
    # The golden mean process never emits 1 then 1.
    with pytest.raises(ValueError):
        compute_belief_states(make_golden_mean_hmm(), np.array([[0, 1, 1, 0]]))


# ---------------------------------------------------------------------------
# Optimal next-token loss (uses the brute-force beliefs, so passes already)
# ---------------------------------------------------------------------------


def test_optimal_loss_next_token_distributions_sum_to_one() -> None:
    mess3 = make_mess3()
    beliefs = np.array([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]])
    distributions = next_token_distributions(mess3, beliefs)
    assert distributions.shape == (2, 3)
    np.testing.assert_allclose(distributions.sum(axis=1), 1.0, atol=1e-12)


def test_optimal_loss_stationary_belief_predicts_stationary_tokens() -> None:
    hmm = make_asymmetric_two_state_hmm()
    prediction = next_token_distributions(hmm, hmm.stationary_distribution())
    np.testing.assert_allclose(prediction, [0.65, 0.35], atol=1e-12)


def test_optimal_loss_mess3_first_position_is_log_3() -> None:
    # With no context every Mess3 token has probability exactly 1/3.
    mess3 = make_mess3()
    initial_belief = mess3.stationary_distribution()
    token_sequences = [(0, 1), (2, 2), (1, 0)]
    tokens = np.array(token_sequences)
    beliefs = np.array([brute_force_beliefs(mess3, seq, initial_belief) for seq in token_sequences])
    losses = bayes_optimal_token_losses(mess3, tokens, beliefs)
    np.testing.assert_allclose(losses[:, 0], np.log(3), atol=1e-12)


@pytest.mark.parametrize("process_name", SMALL_TEST_PROCESSES)
def test_optimal_loss_sums_to_minus_log_sequence_probability(process_name: str) -> None:
    # Chain rule: P(x_0 .. x_3) = P(x_0) P(x_1 | x_0) P(x_2 | x_0, x_1) P(x_3 | ...).
    hmm = SMALL_TEST_PROCESSES[process_name]()
    initial_belief = hmm.stationary_distribution()
    for token_sequence in all_possible_sequences(hmm, 4, initial_belief):
        beliefs = brute_force_beliefs(hmm, token_sequence, initial_belief)[np.newaxis]
        losses = bayes_optimal_token_losses(hmm, np.array([token_sequence]), beliefs)
        expected = -np.log(brute_force_sequence_probability(hmm, token_sequence, initial_belief))
        assert losses.sum() == pytest.approx(expected, abs=1e-12)


def exact_expected_loss_by_position(hmm: HiddenMarkovModel, sequence_length: int) -> np.ndarray:
    """Exact H(X_t | X_<t) for each position t, by weighting every sequence by its probability."""
    initial_belief = hmm.stationary_distribution()
    expected_loss = np.zeros(sequence_length)
    for token_sequence in all_possible_sequences(hmm, sequence_length, initial_belief):
        probability = brute_force_sequence_probability(hmm, token_sequence, initial_belief)
        beliefs = brute_force_beliefs(hmm, token_sequence, initial_belief)[np.newaxis]
        losses = bayes_optimal_token_losses(hmm, np.array([token_sequence]), beliefs)[0]
        expected_loss += probability * losses
    return expected_loss


def test_optimal_loss_golden_mean_matches_known_entropies() -> None:
    # Position 0: entropy of the token distribution (2/3, 1/3).
    # Later positions: the state is known after one token. In state 0 (probability
    # 2/3) the next token is a fair coin, worth log 2; in state 1 it is certain.
    # So every later position has expected loss (2/3) log 2.
    expected_loss = exact_expected_loss_by_position(make_golden_mean_hmm(), sequence_length=5)
    first_token_entropy = -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)
    assert expected_loss[0] == pytest.approx(first_token_entropy, abs=1e-12)
    np.testing.assert_allclose(expected_loss[1:], (2 / 3) * np.log(2), atol=1e-12)


def test_optimal_loss_mess3_never_increases_with_more_context() -> None:
    # For a stationary process, conditioning on more of the past can only reduce
    # the expected uncertainty about the next token: H(X_t | X_<t) never increases with t.
    expected_loss = exact_expected_loss_by_position(make_mess3(), sequence_length=5)
    assert expected_loss[0] == pytest.approx(np.log(3), abs=1e-12)
    assert np.all(np.diff(expected_loss) <= 1e-12)
    # And it does fall: context helps, so the beliefs carry real information.
    assert expected_loss[-1] < expected_loss[0] - 0.05


def test_optimal_loss_rejects_misaligned_beliefs() -> None:
    mess3 = make_mess3()
    tokens = np.zeros((2, 5), dtype=np.int64)
    beliefs_missing_initial = np.full((2, 5, 3), 1 / 3)
    with pytest.raises(ValueError, match="shape"):
        bayes_optimal_token_losses(mess3, tokens, beliefs_missing_initial)
