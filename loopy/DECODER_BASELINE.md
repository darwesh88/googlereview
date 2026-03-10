# Decoder Baseline

This is the first output-side decoding experiment.

## Goal

Compare two ways of turning concept codes back into normal words:

1. fixed canonical decode
2. context-aware alias decode

## Why this was next

The first input-side ambiguity gate was too blunt.
Canonical input rewriting remained the better baseline.

So the next higher-value problem became output recovery:

- can we keep the training win from canonical concept normalization
- while recovering more natural wording like `user` vs `customer`

## Files

- [surface_decoder.py](C:/Users/adarw/Desktop/googlereview/loopy/surface_decoder.py)
- [evaluate_surface_decoder.py](C:/Users/adarw/Desktop/googlereview/loopy/evaluate_surface_decoder.py)

## What the decoder does

- learns alias preferences from aligned original text
- uses nearby rewritten context tokens to choose among aliases
- falls back to the most common alias for each concept
- includes a later conservative refinement with copy-first and fallback controls

## First baseline result

Observed evaluation:

- examples: `76`
- canonical alias accuracy: `0.2895`
- contextual alias accuracy: `0.3158`

Interpretation:

- overall, local-context alias recovery is better than always decoding to the canonical form
- the gain is small, but real
- this made the output-side decoder the right place to keep exploring

## Conservative refinement result

Observed evaluation:

- canonical alias accuracy: `0.2895`
- refined contextual alias accuracy: `0.3158`
- copy used: `0`
- fallback to canonical: `52`

Interpretation:

- the refinement changed which concepts were helped or hurt
- but it did not improve the aggregate score beyond the first decoder baseline
- copy-first did not activate on this evaluation split

## Current takeaway

- context-aware decoding is still directionally correct
- the first contextual decoder is the better current baseline
- more hand-tuned threshold work on synthetic data is probably low value from here
- the next serious test should be a real-corpus evaluation or a stronger learned alias decoder
