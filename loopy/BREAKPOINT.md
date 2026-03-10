# Breakpoint

Loopy has pivoted.

## Current state

- v1 symbolic middleware is archived as a completed research branch
- v2 semantic binary codec is now the active direction
- toy local tests have passed at 1-byte, 2-byte, and 4-byte settings
- multiple real local corpus runs have passed strongly
- the best current CPU baseline is the 20-epoch real-corpus run
- Google Colab is now the right next scale-up step before H100 rentals

## What v1 taught us

- narrow synthetic domains gave positive results
- the first broad real corpus did not
- shallow concept replacement is too weak for the bigger goal

## What v2 is trying to do

- bytes in
- grouped binary patches internally
- exact reconstruction
- compression-aware training
- later, prior modeling over the binary stream

## Most recent result

The 20-epoch real-corpus CPU run is the current best baseline.

- byte accuracy: `0.9849`
- estimated bpb: `1.8326`
- raw capacity bpb: `6.0`
- reconstruction remained highly faithful on real noisy text

Interpretation:

- the representation keeps improving with more training on real data
- fragile detail errors still remain, but the overall signal is strong
- CPU validation is sufficient; the next meaningful step is medium-scale GPU validation

## Next resume step

Prepare the first Google Colab-scale experiment plan using the current best CPU baseline.
