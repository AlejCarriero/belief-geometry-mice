"""Small processes whose answers can be worked out by hand, shared by several test files.

Mess3 is symmetric, so its stationary distribution is uniform and each token has
frequency 1/3. Bugs can hide behind that symmetry, so tests also use these two:

* An asymmetric two-state model with stationary distribution (0.75, 0.25).
* The golden mean process, which can never emit two 1s in a row.
"""

import numpy as np

from src.hmm import HiddenMarkovModel


def make_asymmetric_two_state_hmm() -> HiddenMarkovModel:
    """Two states (0, 1), two tokens (0, 1).

    Underlying state transitions T = [[0.9, 0.1], [0.3, 0.7]].
    State 0 emits token 0 with probability 0.8; state 1 emits token 0 with probability 0.2.
    The token is chosen by the state being left (so each row of T is scaled by the
    emission probability of that row's state).

    Hand calculation of the stationary distribution: balance between the two states
    needs pi_0 * 0.1 = pi_1 * 0.3, so pi_0 = 3 * pi_1 and pi = (0.75, 0.25).
    """
    matrix_for_token_0 = [[0.72, 0.08], [0.06, 0.14]]  # 0.8 * row 0 of T, 0.2 * row 1
    matrix_for_token_1 = [[0.18, 0.02], [0.24, 0.56]]  # 0.2 * row 0 of T, 0.8 * row 1
    return HiddenMarkovModel(np.array([matrix_for_token_0, matrix_for_token_1]))


def make_golden_mean_hmm() -> HiddenMarkovModel:
    """Golden mean process: a 1 is always followed by a 0.

    From state 0: emit 0 and stay (probability 0.5) or emit 1 and go to state 1 (0.5).
    From state 1: emit 0 and go back to state 0 (probability 1).
    Stationary distribution: pi_0 * 0.5 = pi_1, so pi = (2/3, 1/3).
    """
    matrix_for_token_0 = [[0.5, 0.0], [1.0, 0.0]]
    matrix_for_token_1 = [[0.0, 0.5], [0.0, 0.0]]
    return HiddenMarkovModel(np.array([matrix_for_token_0, matrix_for_token_1]))
