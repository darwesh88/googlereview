# Concept Middleware

This is the new direction for Loopy.

## Why this replaces the codec-first path

The discrete codec learned rough phonetic patterns, but it did not reconstruct exact text well enough.
That is too weak for the first proof.

## New idea

Keep normal text.
Replace only selected concepts with stable concept IDs before training.

Example:

- `tree` -> `<n2>`
- `house` -> `<n3>`

At inference time:

1. user text is converted into mixed text plus concept IDs
2. the model works on that mixed representation
3. concept IDs are converted back into readable text

## Rules

- Do not replace everything.
- Replace only known concepts.
- Keep the mapping reversible.
- Keep the rest of the sentence as normal text.

## First experiment

Measure whether concept rewriting makes the corpus:

- more consistent
- easier to model in a narrow domain
- less noisy for repeated entities and actions

## Goal

Test a mixed symbolic layer before going back to learned latent symbols.
