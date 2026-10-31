# Attentive Chess: What elo can a transformer model hit?

A transformer model written from scratch to play Chess ♟️

This repository holds a spirited recreation of the following paper: [Grandmaster-Level Chess Without Search (Ruoss et al. 2024)](https://arxiv.org/abs/2402.04494). Further, I referenced the associated [google-deepmind/searchless_chess](https://github.com/google-deepmind/searchless_chess) repository, and utilized their provided dataset.

To practice the skill of implementing models from research papers, there is **absolutely zero AI output** in this repository. Everything is my own.


## Dataset

The training dataset comes from [Google Deepmind's searchless-chess](https://github.com/google-deepmind/searchless_chess) repository, using their Action-Value approach (of the three predictors) sourced from Lichess/Stockfish16. For now, I'm only using their 1.2 GB partial training data and test data.

**FEN record format:**
1. piece placement date ranks 8...1 (/ separated)
	white: PNBRQK
	black: pnbrqk
	empty square: 1-8 (# of adjacent empty sq)
2. active color [w|b]
3. castling: [-KQkq]
4. en passant: square or -
5. halfmove clock: fifty move rule
6. number of fullmoves

79 token input:
- first 77: FEN serialized
- move ID
- bucket K#
- shift by 1, so [0] ... [move id]

## Model

token IDs:          [B, 79]



token embeddings:   [B, 79, D]
position embeddings:   [79, D]
combined:           [B, 79, D]

## Paper notes: *Grandmaster-Level Chess Without Search* (bit of an info dump, sorry)

- released feb 2024
- 270m transformer supervised learning
- lichess blitz 2895
- hypothesis: *is it possible to use supervised learning to obtain a chess policy that generalizes well and thus leads to strong play without explicit search?*
- lichess action-values (board states labeled my win%?), stockfish 16 state-values derive win%
- dataset
    - downloaded 10M games from lichess
    - numerical model
	    - $s$ = all board states
	    - $V^{SF}(s)$ = state-value estimation for each state, win percentage
		    - from Stockfish 16, time limit 50ms unbounded depth/lvl
		    - sigmoid? from lichess, $[0,100] \%$ 
		- $Q^{SF}(s,a)$ = action-value estimation for legal moves $\forall a \in \mathcal{A}_{legal}(s)$ 
			- also time limit 50ms unbounded depth/lvl
		- best action $= a^{\mathrm{SF}}(s) = \underset{a \in \mathcal{A}_{\mathrm{legal}}(s)}{\arg\max} Q^{\mathrm{SF}}(s,a).$
	- largest training set based on 10M games
		- each game is a sequence of multiple states
		- $\approx$ 530M state-value estimates 
			- so each game provided around 53 states
		- 15.32B action-value estimates
			- scored potential "action" i.e. moves
		- test dataset: 1k games from a different month, has 14% overlap with training set (popular positions)
- binning
	- divide winrate bins into $K=128$ bins
		- $[0,100] \rightarrow [0, 128]$ "128 probabilities"
- model
	- decoder-only transformer, largest is 270M params
	- parametrize a discrete probability distribution 
	- model outputs log probabilities
	- context size
		- 79 for action-value prediction
		- 78 for state-value prediction and behavioral cloning
	- output size is $K$ (number of bins)
	- learned positional encodings as length of input sequence is constant
- tokenization
	- state $s$ encoded as FEN string, 77 chars, where ASCII code of each char is one token
	- FEN string contains:
		- description of all pieces on board
		- whose turn it is
		- castling availability
		- potential en passant target
		- half-move clock
		- full-move counter
- training protocol
	- trained by minimizing cross-entropy loss (log-loss)
	- mini-batch based SGD using adam
	- 10M steps
	- target labels
		- state/action-value prediction: bin indices
		- behavioral cloning: action indices
- prediction
	- **(AV) action-value prediction** (i'll focus on this)
		- input: chess position $s$, legal move $a$ 
		- target: bin $\#$ 
	- (SV) state-value prediction
		- input: chess position $s$ 
		- target: bin $\#$ 
	- (BC) behavioral cloning



## References

- https://arxiv.org/pdf/2402.04494v1
- https://github.com/google-deepmind/searchless_chess
