# Details taken from the paper

Source: Shai, Marzen, Teixeira, Gietelink Oldenziel and Riechers, "Transformers represent belief state geometry in their residual stream", arXiv:2405.15943, version 3 (4 February 2025). Section and appendix numbers below refer to that version.

Everything in the first five sections was read from the paper. Things the paper does not state are listed at the end, together with the choice this project makes.

## Mess3 process (Section 3.1 and Appendix A.3)

* 3 hidden states and 3 tokens (A, B, C).
* The paper cites the process to Marzen and Crutchfield (2017), "Nearly maximally predictive features and their dimensions", Phys. Rev. E 95(5), 051301(R).
* Convention (Section 2.1): entry (i, j) of the matrix for token x is the joint probability of emitting x and moving to state j, given that the current state is i.
* Matrices as printed in Appendix A.3:

| | T(A) | T(B) | T(C) |
|---|---|---|---|
| row 1 | 0.765, 0.00375, 0.00375 | 0.0675, 0.0425, 0.00375 | 0.0675, 0.00375, 0.0425 |
| row 2 | 0.0425, 0.0675, 0.00375 | 0.00375, 0.765, 0.00375 | 0.00375, 0.0675, 0.0425 |
| row 3 | 0.0425, 0.00375, 0.0675 | 0.00375, 0.0425, 0.0675 | 0.00375, 0.00375, 0.765 |

* The paper prints these numbers but not the parameters that generate them. The code uses the printed matrices directly, so the paper stays the source of truth.

### Where the parameters come from (Marzen and Crutchfield 2017)

* Source: arXiv:1702.08565 version 1, Supplementary Materials, Appendix A. The matrices are written in terms of x and alpha, with beta = (1 - alpha) / 2 and y = 1 - 2x. For example, the matrix for token 0 has first row (alpha y, beta x, beta x).
* That paper uses x = 0.15, alpha = 0.6 and x = 0.05, alpha = 0.6. Neither matches Shai et al.
* Setting x = 0.05 and alpha = 0.85 reproduces all 27 printed numbers in Shai et al. This is our inference, and the test `test_mess3_printed_matrices_match_marzen_crutchfield_parametrisation` checks it.
* Every entry factorises as P(move from i to j) times P(state j emits the token). So Mess3 is an ordinary HMM in which the state moves first and then the new state emits:
  * The hidden state stays with probability 1 - 2x = 0.9 and jumps to each other state with probability x = 0.05.
  * The new state emits its own token with probability alpha = 0.85 and each other token with probability 0.075.

## Belief states (Section 2.2)

* An optimal observer updates its belief (a probability vector over hidden states) after seeing token x by multiplying the row vector of beliefs by T(x) and renormalising so the entries sum to 1.
* For a stationary process the starting belief is the stationary distribution.

## Transformer (Appendix A.6)

* Context window 10.
* 4 layers, 1 attention head per layer.
* Model dimension 64, head dimension 8.
* MLP dimension 256, ReLU activation.
* Causal attention masking.
* Layer normalisation is applied.

## Training (Appendix A.6)

* Plain SGD, learning rate 0.01, no weight decay, batch size 64.
* Every batch is 64 freshly generated sequences, each starting from a hidden state drawn from the stationary distribution.
* The paper reports "1,000,000 epochs". Because every batch is new data there is no fixed dataset to pass over, so we read this as 1,000,000 gradient steps (64 million sequences). This is our interpretation.
* The caption of Figure S2 says validation loss is shown relative to the theoretical optimum (1.0 means optimal). The text does not give a final number.

## Probing (Sections 2.3, 3.1, 3.2 and Appendix A.5)

* Affine regression from activations to beliefs, fitted by ordinary least squares over all input sequences.
* For Mess3 the paper probes the final layer of the residual stream, before the final layer norm and the unembedding, at all context positions.
* Cross validation: only 20% of input and activation pairs were used to fit the regression.
* Concatenating the residual stream across all layers is the analysis used for the RRXOR process (Section 3.2), where the belief geometry is not visible in the final layer alone.
* Reported metric: mean squared error averaged across input sequences (Figure 6).

## Not stated in the paper, and our choices

These will be settled in Milestone 3 and recorded here.

* Type of positional embedding.
* Whether a beginning of sequence token is used.
* Where layer norm sits (before or after each sublayer).
* Whether the loss averages over every position in the context window.
