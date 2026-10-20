# Project notes

These notes sit alongside the code. For every milestone they explain three things:

1. **The intuition:** what problem the step solves and why it is needed.
2. **The logical development:** the order in which you would build it if starting from nothing, and why each piece follows from the one before.
3. **The files:** what each script and each function does, and what the tests guard against.

Milestones 0, 1 and 2 are written up from the code as it stands. Milestones 3 to 6 are a plan: they describe what we intend to build and the decisions still to make, and they will be rewritten as each milestone is finished.

Paper details (numbers, architecture, training settings) are not repeated here. They live in [docs/paper_notes.md](docs/paper_notes.md) with section references.

## Contents

* [The big picture](#the-big-picture)
* [Conventions used everywhere](#conventions-used-everywhere)
* [How to run things](#how-to-run-things)
* [Milestone 0: setup](#milestone-0-setup)
* [Milestone 1: the data-generating process](#milestone-1-the-data-generating-process)
* [Milestone 2: belief states and the optimal loss](#milestone-2-belief-states-and-the-optimal-loss)
* [Milestones 3 to 6: the plan](#milestones-3-to-6-the-plan)
* [Glossary](#glossary)
* [Questions to be ready for](#questions-to-be-ready-for)

---

## The big picture

**The question.** If a transformer is trained only to predict the next token of sequences produced by a hidden process, does it end up representing, inside its activations, what an ideal observer would believe about that hidden process?

**Why "belief".** The model never sees the hidden state. The best any observer can do is keep a probability distribution over the hidden states, updated by Bayes' rule each time a token arrives. That distribution is the belief. With 3 hidden states it is 3 numbers that sum to 1, so every belief is a point inside a triangle (the probability simplex). For the Mess3 process, the set of all beliefs reachable from real token sequences forms a fractal inside that triangle.

**The claim we replicate (Shai et al. 2024).** A linear map from the transformer's residual stream to the belief recovers this fractal. In other words, the geometry of the ideal observer's beliefs is present, linearly, inside the network.

**The pipeline.**

```
hidden Markov process
        |
        |  sample (src/hmm.py)
        v
token sequences  ------------------------------->  true beliefs
        |              forward algorithm               |
        |              (src/beliefs.py)                |
        v                                              |
tiny transformer, trained only on next-token           |
prediction (src/model.py, src/train.py)                |
        |                                              |
        |  record residual stream activations          |
        v                                              v
activations  -->  linear regression  -->  predicted beliefs  ==  compare
                  (src/probe.py)
```

**Why it matters for AI safety.** If simple linear tools can read a model's internal picture of the hidden state of its world, the same approach might one day read other hidden variables, such as goal-like states, and show when a model's internal state drifts away from what its designers intended. The extension in Milestone 5 is a first, small step towards hidden states that change over time.

---

## Conventions used everywhere

Most bugs in this kind of project are off-by-one errors in how arrays line up. These conventions are fixed across every file.

| Name | Shape | Meaning |
|---|---|---|
| `token_transition_matrices` | (X, S, S) | Entry `[x, i, j]` is P(emit token x and move to state j, given the current state is i) |
| `tokens` | (N, L) | N sequences of L tokens, as integers |
| `hidden_states` | (N, L + 1) | `hidden_states[:, t]` is the state just before token t is emitted |
| `beliefs` | (N, L + 1, S) | `beliefs[:, t]` = P(`hidden_states[:, t]` given `tokens[:, :t]`) |

Here X is the number of tokens, S the number of hidden states, N the number of sequences and L the sequence length.

**The timeline.** Read this picture until it feels obvious. Almost every later step depends on it.

```
hidden state:   s0 --x0--> s1 --x1--> s2 --x2--> s3
belief index:   b[0]       b[1]       b[2]       b[3]
tokens used:    none       x0         x0, x1     x0, x1, x2
```

* `b[0]` is the starting belief, before any token.
* `b[t + 1]` is the belief after seeing tokens 0 to t.
* `b[t]` is what you use to predict token t.
* A transformer at position t has seen tokens 0 to t, so its activations pair with `b[t + 1]`.

**Other conventions.**

* Beliefs are row vectors and are multiplied on the left: `b @ T`.
* Losses are in nats (natural logarithm), matching PyTorch's cross-entropy.
* Randomness: NumPy draws come from an explicit `numpy.random.Generator` passed into each function; PyTorch is seeded once. See [src/seeding.py](src/seeding.py).

---

## How to run things

```bash
python -m pip install -r requirements.txt
python -m pytest                                   # everything
python -m pytest tests/test_beliefs.py -k forward  # only the forward algorithm tests
python -m pytest -x                                # stop at the first failure
```

---

## Milestone 0: setup

### Intuition

Before any science, make two things true. First, a fresh copy of the project should fail loudly and early if something is missing, rather than halfway through a training run. Second, every random choice should be repeatable, so a surprising result can be reproduced and inspected.

### Logical development

1. Pin the packages, so the code runs against the same library versions that produced the results.
2. Decide how randomness flows through the code before writing any random code. Retrofitting this later is painful.
3. Write tests that check the environment itself, so problems show up as a named failing test rather than a confusing error.
4. Record every detail taken from the paper in one place, with section numbers, so nothing is quoted from memory.

### Files

* **[requirements.txt](requirements.txt):** exact versions of NumPy, PyTorch, matplotlib, scikit-learn and pytest. Exact rather than minimum versions, because those are the only versions actually tested.
* **[pytest.ini](pytest.ini):** tells pytest that tests live in `tests/` and puts the repository root on Python's import path, so tests can write `from src.hmm import ...`.
* **[.gitignore](.gitignore):** keeps caches out of the repository.
* **[src/seeding.py](src/seeding.py):** one function, `set_global_seed(seed)`. It seeds PyTorch's global generator (needed because layer initialisation draws from it) and returns a fresh NumPy `Generator`. NumPy's global random state is deliberately avoided: when every random function receives its generator as an argument, the function signature tells you it is random, and two parts of the code cannot silently disturb each other's random stream.
* **[tests/test_setup.py](tests/test_setup.py):** 10 tests. Python is 3.10 or newer; each package imports (one test per package, so a missing one is named); PyTorch can do a forward and backward pass on the CPU; matplotlib can save a figure without a screen; the same seed gives the same draws and different seeds give different draws.
* **[docs/paper_notes.md](docs/paper_notes.md):** every fact taken from the paper, with its section, plus the choices we make where the paper says nothing.

---

## Milestone 1: the data-generating process

### Intuition

We need a process whose hidden state we control completely, so we know the right answer to "what should an ideal observer believe?". A hidden Markov model gives exactly that: a hidden state that wanders according to fixed probabilities, and tokens that carry noisy evidence about it.

**The token-labelled representation.** The paper describes a process with one matrix per token:

$$T^{(x)}_{ij} = P(\text{emit } x \text{ and move to } j \mid \text{in } i)$$

This single object holds both the transitions and the emissions. It also covers both common textbook forms as special cases:

* **Move, then emit from the new state:** $T^{(x)}_{ij} = A_{ij} B_{jx}$. Mess3 is of this form.
* **Emit from the current state, then move:** $T^{(x)}_{ij} = B_{ix} A_{ij}$. The asymmetric test process is of this form.

Its biggest advantage shows up in Milestone 2: updating a belief after a token becomes a single matrix product.

**Mess3 in words.** Three hidden states, each with its own favourite token.

* The hidden state stays where it is with probability 0.9 and jumps to each of the other two with probability 0.05.
* The state it lands in emits its own token with probability 0.85 and each of the other two with probability 0.075.

Two properties make it interesting. Tokens are only weak evidence (a favourite token appears 85% of the time, not always), and states are sticky, so evidence builds up over several tokens. The observer is never certain, and its beliefs land on an ever-growing set of points that forms a fractal.

### Logical development

1. **Choose the representation:** an array of shape (X, S, S). Everything else is computed from it.
2. **Say what makes it valid, and check that first.** All entries are non-negative, and for each current state i the probabilities over every (token, next state) pair sum to 1. Checking this in the constructor means every later function can trust its input.
3. **Derive simpler quantities by summing.** Summing over tokens gives the ordinary state transition matrix T. Summing over the next state gives each state's token probabilities.
4. **Find the stationary distribution**, the long-run fraction of time spent in each state. It satisfies $\pi T = \pi$. We solve this as a set of linear equations rather than simulating for a long time, which is exact and fast.
5. **Work out what the token statistics should be,** before sampling anything:
   * single tokens: $P(x) = \pi\, T^{(x)} \mathbf{1}$
   * pairs: $P(x, y) = \pi\, T^{(x)} T^{(y)} \mathbf{1}$, where $\mathbf{1}$ is a column of ones that sums over the final state.
6. **Write the sampler.** From the current state, draw the token and the next state together, because in general they are not independent.
7. **Test against things you can verify by hand,** then statistically, then by breaking the code on purpose to confirm the tests notice.

### Walkthrough of [src/hmm.py](src/hmm.py)

| Function or class | What it does | Why it exists |
|---|---|---|
| `HiddenMarkovModel.__init__` | Copies the matrices, validates them, makes the copy read-only, stores the sizes and token names | A model that cannot be changed after creation cannot be corrupted by accident |
| `state_transition_matrix` | Sums over tokens: P(move to j given in i) | Needed for the stationary distribution and for tests |
| `token_probabilities_given_state` | Sums over the next state: P(next token is x given in i), shape (S, X) | Turns a belief into a next-token prediction (Milestone 2) |
| `stationary_distribution` | Solves $\pi T = \pi$ with $\sum \pi = 1$ | Starting belief and starting state for every sequence |
| `stationary_token_distribution` | $\pi$ times the token probabilities per state | The analytical target for token frequencies |
| `_validate_token_transition_matrices` | Raises a clear error for wrong shapes, negative numbers or rows that do not sum to 1 | Fail early with a readable message |
| `hmm_from_transition_and_emission` | Builds the token-labelled matrices from a textbook transition matrix A and emission matrix B (move, then emit) | Easier to define new processes this way, including the mouse-like one in Milestone 5 |
| `make_mess3` | Mess3 with the 27 numbers copied from the paper (Appendix A.3) | The paper is the source of truth |
| `make_mess3_from_parameters` | Mess3 from its two parameters (jump probability x, emission accuracy alpha), following Marzen and Crutchfield (2017) | An independent construction, used to check the 27 typed numbers |
| `SampledSequences` | A named pair: `tokens` and `hidden_states` | Keeps the hidden states, which later analyses need (for example, true switch times in Milestone 5) |
| `sample_sequences` | Draws N independent sequences of length L, all advanced together one position at a time | The data source for training and probing |
| `_cumulative_joint_table` | For each current state, lays out the S times X outcomes (token, next state) in one row and takes the running sum | Prepares inverse transform sampling |
| `_sample_token_and_next_state` | Draws a uniform number per sequence, finds which outcome it falls into, and decodes that into a token and a next state | One step of sampling |

**Two details worth understanding.**

* **The stationary solve.** The equations $(T^\top - I)\pi = 0$ only fix the direction of $\pi$, not its scale. Adding one more equation, $\sum_i \pi_i = 1$, pins it down. If the chain splits into parts that never communicate, the first set of equations has rank below S minus 1, the answer is not unique, and the function raises an error instead of returning one arbitrary answer.
* **Inverse transform sampling.** Lay the outcome probabilities end to end along the interval from 0 to 1. A uniform random number lands in exactly one segment, and segment lengths are the probabilities. Counting how many running-sum values are at most the draw gives the segment index. Outcomes with probability zero have zero length and can never be chosen. The last running-sum value is set to exactly 1, so rounding can never leave a draw outside every segment.

### Tests: [tests/example_processes.py](tests/example_processes.py) and [tests/test_hmm.py](tests/test_hmm.py)

**Why two extra processes.** Mess3 is symmetric: its stationary distribution is uniform and every token appears a third of the time. A broken sampler can easily reproduce those numbers. So most tests also use:

* **An asymmetric two-state process.** Stationary distribution (0.75, 0.25), P(token 0) = 0.65, and P(0 then 0) = 0.463. That last number differs from 0.65 squared (0.4225), so adjacent tokens are correlated and the pair test is informative.
* **The golden mean process.** It can never emit two 1s in a row, and every token reveals the hidden state. A sampler with an indexing bug would produce forbidden pairs.

**What the 38 tests cover.**

| Group | Guards against |
|---|---|
| Input validation | Silently accepting an invalid process |
| Mess3 definition | Typos in the 27 numbers (checked against the parametric form), wrong stay probability |
| Stationary distribution | Wrong solve; checked against hand calculations for both small processes |
| Analytical statistics | Wrong formulas; the pair formula is itself checked against a hand calculation |
| Sampling basics | Wrong shapes, values out of range, irreproducible draws, ignoring the initial distribution |
| Sampling against the model | Forbidden transitions, wrong token frequencies, wrong pair frequencies, wrong joint (token, next state) frequencies |

**How we know the tests work.** The sampler was broken on purpose in a copy of the code:

* Swapping token and next state: caught by 4 tests. The Mess3 single-token test did not catch it, because of the symmetry; the asymmetric process did.
* Using the wrong axis order when flattening: caught by 4 tests.
* Changing `>=` to `>`: not caught, correctly, because for continuous random draws the two behave identically.

**Tolerances are measured, not guessed.** Over 100 seeds, the largest standard error of any token or pair frequency was 0.0024, so the tolerance of 0.01 is about 4 standard errors. The joint (token, next state) test uses 4 binomial standard errors per entry, which can still catch a mistake in the smallest entries (0.00375).

### Things to be able to explain

* Why the token-labelled form is more general than a transition matrix plus an emission matrix.
* Why Mess3 counts as an ordinary HMM, and in which order (move, then emit).
* Why the stationary solve stacks S plus 1 equations, and what rank S minus 1 means.
* Why a symmetric test process is weak, with the swap bug as the example.

---

## Milestone 2: belief states and the optimal loss

### Intuition

**The question.** Having seen tokens $x_0, \dots, x_{t-1}$, how likely is each hidden state now?

**One step of Bayes' rule.** Suppose you already hold the belief $b_t$. When token $x_t$ arrives, ask: what is the joint probability of this token and each next state?

$$P(x_t, s_{t+1} = j \mid x_{<t}) = \sum_i b_t(i)\, T^{(x_t)}_{ij} = \big(b_t\, T^{(x_t)}\big)_j$$

Each term is "probability I was in state i" times "probability that state i emits this token and moves to j". Condition on the token you actually saw by dividing by the total:

$$b_{t+1} = \frac{b_t\, T^{(x_t)}}{b_t\, T^{(x_t)}\, \mathbf{1}}, \qquad b_0 = \pi$$

**This is the forward algorithm you already know.** For a move-then-emit process, $b\,T^{(x)}$ means: push the belief through the transition matrix, then weight each state by how likely it was to emit $x$. The token-labelled form does both in one product.

**Why normalise at every step.** Without it, the numbers shrink by about a factor of 3 per token for Mess3 and underflow to exactly zero after a few hundred tokens. Normalising keeps every belief a proper probability vector.

**The normaliser is valuable in its own right.** The denominator $b_t T^{(x_t)} \mathbf{1}$ is $P(x_t \mid x_{<t})$: the probability the ideal observer gave to the token that actually came. That leads straight to the best achievable loss.

**The optimal loss.** Any predictor q has cross-entropy $\mathbb{E}[-\log q(x_t \mid x_{<t})]$. The smallest possible value comes from predicting with the true conditional distribution, and it equals the conditional entropy $H(X_t \mid X_{<t})$. We estimate it by averaging $-\log P(x_t \mid x_{<t})$ over sampled sequences, position by position. No model can beat it on average, which makes it the target for training in Milestone 3.

**Where the fractal comes from.** Each token applies one of three maps to the belief: multiply by $T^{(x)}$ and renormalise. Starting from the centre of the triangle:

* After one token there are 3 possible beliefs. After token A it is (0.85, 0.075, 0.075), near corner A.
* After two tokens there are 9, after three 27, and in general $3^t$ after t tokens.

Each map pulls beliefs towards one corner, and applying them over and over builds a self-similar pattern, like an iterated function system.

### Logical development of your function, `compute_belief_states`

This is the order that keeps each step checkable. It describes the reasoning, not the code.

1. **On paper first.** Start from the uniform belief and apply token A. Confirm you get (0.85, 0.075, 0.075). If you can do this by hand, you understand the update.
2. **One belief, one token.** Write the update for a single belief vector and check it reproduces the paper calculation.
3. **One sequence.** Repeat the update over the tokens of one sequence, storing the starting belief at index 0 and each new belief at the next index. This is where the timeline picture above matters.
4. **Impossible sequences.** Decide what happens when the normaliser is zero (the model says this sequence cannot occur): raise `ValueError` rather than divide by zero.
5. **Many sequences.** Loop over the sequences in the batch. Vectorising across sequences is optional and best left until everything passes.
6. **Run the tests in a sensible order:** shape and starting-belief tests first, then the hand calculation, then the brute-force comparison, then the rest.

### Walkthrough of [src/beliefs.py](src/beliefs.py)

| Function | What it does | Status |
|---|---|---|
| `compute_belief_states` | The forward algorithm: beliefs of shape (N, L + 1, S) | **Yours to write.** The docstring states the update rule, shapes and requirements |
| `next_token_distributions` | Belief times each state's token probabilities: the ideal observer's prediction for the next token | Written |
| `bayes_optimal_token_losses` | For every sequence and position, minus log of the probability the ideal observer gave the actual token. Shape (N, L) | Written. Raises an error if the belief array has the wrong shape, which almost always means an off-by-one |

Useful facts about the losses:

* Averaging over sequences, `losses.mean(axis=0)`, estimates $H(X_t \mid X_{<t})$ at each position.
* Summing over positions gives minus log of the probability of the whole sequence. That is the chain rule, $P(x_0, \dots, x_{L-1}) = \prod_t P(x_t \mid x_{<t})$, and one of the tests checks exactly this.

**Exact optimal loss for Mess3**, from listing every possible sequence:

| Position | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Loss (nats) | 1.0986 | 0.8577 | 0.8134 | 0.7981 | 0.7947 | 0.7938 |

Position 0 is exactly ln 3, because with no context every token is equally likely. The loss falls as context accumulates and levels off at about 0.79 nats.

### Tests: [tests/test_beliefs.py](tests/test_beliefs.py)

**The brute-force reference.** At the top of the file, beliefs are computed by a completely different route: list every possible hidden path, multiply the probabilities along it, and add up the paths ending in each state. The number of paths grows as S to the power of the length, so it is hopeless for real use, but every line can be checked by eye. That is what makes it a trustworthy reference.

**The 15 tests for your function** (run with `-k forward`):

| Tests | Catch |
|---|---|
| Output shape, starting belief (default and given), probabilities summing to 1, empty sequences | Breaking the agreed shapes and conventions |
| Token A from the uniform belief gives (0.85, 0.075, 0.075) | A wrong update rule, on a case you can check by hand |
| Brute force over every hidden path, for all length-4 sequences of three processes, plus a non-stationary start | Wrong maths |
| A batch of sequences together versus one at a time | Normalising across the batch instead of within each sequence |
| Golden mean belief equals the true sampled state | Your alignment not matching the sampler's |
| Calibration against 50,000 sampled sequences | Beliefs that are not the true posterior |
| A 2,000-token sequence | Normalising only at the end, which underflows |
| An impossible golden mean sequence | Dividing by zero without raising an error |

**The calibration test in plain words.** Collect every moment where the belief gave state 0 a probability of about 0.7. If the beliefs are right, the hidden state really was 0 about 70% of the time. The test checks this in 10 bins. With a correct implementation, over 20 seeds, the worst bin was off by 0.0086 at most, so the tolerance is 0.015.

**The 11 other tests** check the loss functions and the reference itself, using brute-force beliefs, so they already pass:

* position 0 is exactly ln 3 for Mess3
* the losses sum to minus log of the sequence probability
* the golden mean matches its known entropies: 0.6365 nats at position 0 and two thirds of ln 2 (0.4621) afterwards
* the Mess3 loss never rises with more context

**How we know the tests work.** A correct version was written outside the repository and passes all 26 tests, both vectorised and one sequence at a time. Seven typical mistakes each fail at least one test:

* transposed matrices
* forgetting the transition step
* normalising across the batch
* normalising only at the end
* starting from uniform instead of the stationary distribution
* an off-by-one in the alignment
* no error on impossible sequences

### Things to be able to explain

* The derivation of the update rule from Bayes' rule, in one line.
* Why normalising every step is necessary, and what the normaliser means.
* Why the optimal loss is a conditional entropy, and why no model can beat it on average.
* Why there are $3^t$ possible beliefs after t tokens, and how that becomes a fractal.

---

## Milestones 3 to 6: the plan

Nothing in this section has been built yet. It records the intended design, the reasoning behind it, and the decisions still open, so the plan can be checked before code is written.

### Milestone 3: train a tiny transformer (planned)

**Files:** `src/model.py`, `src/train.py`.

**Intuition.** The transformer only ever sees tokens and is only ever asked to predict the next one. If it learns to predict as well as the ideal observer, the information in the belief must be somewhere inside it. Milestone 3 trains it to that point; Milestone 4 looks for the belief.

**The anatomy, as planned** (sizes from the paper, Appendix A.6):

* **Embedding.** Each of the 3 tokens becomes a 64-number vector, and each of the 10 positions adds its own 64-number vector. The sum is the starting residual stream.
* **Four blocks.** Each block lets every position read from earlier positions (one attention head, head dimension 8, causal mask) and then transforms each position on its own (an MLP with 256 hidden units and ReLU). Each block adds its output back onto the residual stream rather than replacing it.
* **Output.** A final layer norm and a linear map from 64 numbers to 3 logits, one per possible next token.

**The residual stream** is that running 64-number vector at each position. Every block reads from it and writes back into it. We will record it at 5 points: after the embedding, and after each of the 4 blocks.

**Decisions to make.**

* Positional embedding type, layer norm placement and whether to use a start token. The paper does not say; whatever we choose goes into [docs/paper_notes.md](docs/paper_notes.md).
* Plain SGD as in the paper, or Adam. We will time a few hundred steps on the CPU first.
* Target alignment. The likely choice is to sample 11 tokens per sequence, feed tokens 0 to 9 and predict tokens 1 to 10.

**What to compare against.** At its first position the model has already seen one token, so its best possible loss there is the optimal loss for predicting token 1 (0.8577 nats), not ln 3. Each model position t compares with the optimal value for token t + 1.

**Sanity checks.**

* The loss should approach the optimal curve but never beat it on average. Beating it means something is wrong, usually future tokens leaking through a broken causal mask.
* If the loss stalls well above optimal, stop and debug before Milestone 4. Probing a model that has not learned the task tells us nothing.

### Milestone 4: find the belief geometry (planned; you write the regression)

**Files:** `src/probe.py` (collect activations and fit the regression), `src/plotting.py` (triangle plots).

**Intuition.** If some fixed affine map (a matrix times the activation, plus a constant) turns the residual stream into the true belief, the belief is linearly readable from the network. "Linear" is the important word: a flexible nonlinear decoder can extract almost anything from almost any representation, but a linear one only finds what is laid out plainly.

**Logical development.**

1. **Collect.** Run held-out sequences through the trained model. Store the residual stream at every position and at each of the 5 recording points, paired with `beliefs[:, t + 1]`.
2. **Split by sequence, not by row.** Rows from the same sequence share their beginnings, so splitting rows at random would leak information between training and test sets.
3. **Fit (your function).** Find the matrix W and offset c that minimise the squared error between W times the activation plus c and the true belief, on the training split. This is ordinary least squares with an intercept.
4. **Evaluate on the held-out split:** R squared and mean squared error, per recording point, for all points concatenated, and per position.
5. **Controls.**
   * **An untrained model.** Its activations are still a fixed function of the input, and at early positions there are very few distinct inputs, so even random features can fit surprisingly well. The control measures how much of the fit comes from training.
   * **Shuffled beliefs.** Pair activations with beliefs from other rows. Held-out R squared should collapse to about zero; anything else reveals overfitting.
6. **Plot.** Map each belief $(b_0, b_1, b_2)$ to the point $(b_1 + b_2/2,\ \tfrac{\sqrt 3}{2} b_2)$, which places the three states at the corners of an equilateral triangle. Show true and predicted beliefs side by side, colouring each point by its true belief (as red, green and blue), so a correctly decoded point has the same colour in both panels.

**A built-in check for your regression.** Every true belief sums to 1. With an intercept, least-squares predictions then also sum to exactly 1, whatever the activations. If yours do not, something is wrong.

**A caveat to state honestly.** For Mess3, the next-token distribution is an invertible linear function of the belief (3 states, 3 tokens, full-rank token matrix). So any model that predicts well carries the belief somewhere. Finding it linearly is still not automatic, because the residual stream reaches log probabilities through a layer norm, but the paper's stronger claim of geometry beyond next-token prediction rests on a different process (RRXOR) that we are not replicating. Probing for the next-token distribution as well as the belief makes this visible.

### Milestone 5: a mouse-like extension (planned)

**Intuition.** Mice in a tactile task switched between exploiting, exploring and nesting within trials, and nesting became more likely later in a session. The toy version asks whether the residual stream still tracks the ideal observer when the hidden dynamics drift with time.

**Design, as agreed so far.**

* 3 hidden states with long dwell times (sticky).
* The probability of entering the nesting-like state rises with position in the sequence, so the transition matrix changes with time.
* Probably only 2 tokens. With fewer tokens than states, the belief holds strictly more information than the next-token prediction, so finding it says more than it does for Mess3.
* Emission probabilities must overlap between states. Otherwise a single token reveals the state and there is no interesting geometry.
* Context of 64 tokens, with one whole session fitting inside one context, so position means time in session.

**Development.**

1. Build the process with `hmm_from_transition_and_emission`, once per position.
2. Extend the forward algorithm so step t uses the matrices for position t. The update rule is unchanged; only the matrix changes.
3. Compute a second, **time-blind** belief using the time-averaged transition matrix. It is the reference observer for question 3.
4. Train the same transformer and repeat the Milestone 4 analysis.

**The three questions and how we plan to measure them.**

1. **Is the drifting belief still linearly encoded?** Held-out R squared, per position.
2. **How quickly does the decoded belief update at a switch?** Align windows of data on true state switches and average. Compare the decoded belief with the ideal observer's belief, not with the true state, because no observer can know about a switch before the evidence arrives.
3. **Has the model absorbed the time-in-session prior?** The ideal observer's belief already includes that prior, and position can be read straight from the positional embedding, so a rising decoded nesting belief proves little on its own. The cleaner test asks whether the residual stream fits the time-aware belief better than the time-blind one, especially when the recent tokens are identical.

### Milestone 6: write-up (planned)

A short README for a busy researcher:

* the question
* what was replicated and what is new, kept clearly separate
* the method in plain language
* 2 to 4 figures
* results, including anything that failed
* limitations
* next steps, linked to the broader interest in reading goal-like hidden states from model internals

---

## Glossary

* **Hidden Markov model (HMM):** a hidden state that changes at random according to fixed probabilities, plus visible tokens whose probabilities depend on the hidden state.
* **Token-labelled transition matrix:** one matrix per token; entry (i, j) is the probability of emitting that token and moving from state i to state j.
* **Stationary distribution:** the long-run fraction of time spent in each state; unchanged by one step of the chain.
* **Belief (mixed state in the paper):** the probability distribution over hidden states given the tokens seen so far.
* **Belief simplex:** the triangle (for 3 states) containing every possible belief.
* **Mixed-state presentation:** the paper's name for the set of all reachable beliefs together with the token-driven moves between them.
* **Forward algorithm (filtering):** the recursive Bayesian update that computes beliefs one token at a time.
* **Bayes-optimal predictor:** predicts the next token using the true belief; no predictor does better on average.
* **Cross-entropy:** the average of minus log of the probability a predictor gave to what actually happened. Lower is better.
* **Nats:** units of information when using natural logarithms (1 nat is about 1.44 bits).
* **Conditional entropy, $H(X_t \mid X_{<t})$:** the unavoidable uncertainty about token t given everything before it; the best achievable cross-entropy at position t.
* **Entropy rate:** the value the conditional entropy settles to with long context (about 0.79 nats for Mess3).
* **Residual stream:** the running vector at each position that every transformer block reads from and adds to.
* **Linear probe:** a linear (here affine) map trained to read a quantity out of activations.
* **R squared:** the fraction of the variance in the target that the predictions explain; 1 is perfect, 0 is no better than predicting the mean.
* **Barycentric coordinates:** a way to place a probability vector inside a triangle, with each corner standing for one state.

---

## Questions to be ready for

**Why is a belief a point in a triangle?** It is 3 non-negative numbers summing to 1. That set is a triangle, with each corner meaning certainty about one state.

**Why does the Mess3 belief set look like a fractal?** Each token applies one of three maps that pull the belief towards one corner. Repeating three such maps in every possible order produces a self-similar set, as in an iterated function system.

**Is Mess3 an ordinary HMM?** Yes. The state moves first (stays with probability 0.9), then the new state emits its own token with probability 0.85. It does not factorise in the other order (emit, then move).

**Why compare the model with the ideal observer and not with the true hidden state?** The true state is not knowable from the tokens. The ideal observer is the best anyone can do with the same information, so it is the fair benchmark.

**If the probe finds the belief, does the model use it?** Not necessarily. Decodable does not mean used. Showing use needs causal tests, such as editing the activations along the decoded direction and checking that predictions change as expected. That is a natural next step, not part of this project.

**Why is Mess3 a weak test of "more than next-token prediction"?** With 3 states and 3 tokens, the next-token distribution and the belief are linked by an invertible linear map, so a good predictor must carry the belief anyway. A process with fewer tokens than states, as planned for Milestone 5, makes the test stronger.

**Why split training and test data by sequence?** Sequences share prefixes, and positions within a sequence are correlated. Splitting rows at random lets near-duplicates sit on both sides and inflates held-out scores.

**Why bother with an untrained-model control?** Random networks still produce activations that are fixed functions of the input, and a linear map can sometimes decode a lot from them. The control shows how much of the result is due to learning.
