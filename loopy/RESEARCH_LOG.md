# Loopy Research Log

## Current direction

Use a mixed symbolic layer.

- Keep normal text.
- Replace selected concepts with reversible IDs like `<n4>`.
- Train a small token LM on plain text and concept-rewritten text.
- Compare loss, perplexity, and sample generations.

## Why we changed direction

The codec-first path did run end to end, but it did not reconstruct exact text well enough.
It learned rough sound and shape, not faithful text.

## Experiments tried

### 1. Byte-level VQ codec smoke test

Command:

`python -m loopy.train_codec --data-path loopy/example_corpus.txt --epochs 1 --batch-size 4`

Result:

- training loop worked
- checkpoint saved
- compression looked high because of fixed chunking
- reconstruction was mostly garbage

Metrics:

- val loss: `4.9120`
- recon loss: `4.2279`
- codebook perplexity: `3.23`
- compression: `7.43x`

Takeaway:

Pipeline was alive, but the model was undertrained and the codebook was barely used.

### 2. Smaller codec overfit attempt, chunk size 8

Command:

`python -m loopy.train_codec --data-path loopy/example_corpus.txt --epochs 50 --batch-size 4 --codebook-size 64 --d-model 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4`

Result:

- losses improved
- reconstruction still bad
- codebook still weak

Best metrics:

- val loss: `3.2983`
- recon loss: `3.1449`
- codebook perplexity: `4.87`
- compression: `7.43x`

Takeaway:

The fixed `8-byte -> 1 code` bottleneck was too aggressive.

### 3. Smaller codec overfit attempt, chunk size 4

Command:

`python -m loopy.train_codec --data-path loopy/example_corpus.txt --epochs 60 --batch-size 4 --codebook-size 64 --d-model 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --chunk-size 4`

Result:

- better than chunk size 8
- output looked closer to English
- still failed exact reconstruction

Best metrics:

- val loss: `2.9781`
- recon loss: `2.7612`
- codebook perplexity: `8.52`
- compression: `3.90x`

Takeaway:

Less compression helped, but the architecture was still too lossy.

### 4. Simpler decoder codec attempt

Command:

`python -m loopy.train_codec --data-path loopy/example_corpus.txt --epochs 80 --batch-size 4 --codebook-size 64 --d-model 128 --encoder-layers 2 --decoder-layers 2 --num-heads 4 --chunk-size 4 --dropout 0.0 --weight-decay 0.0`

Result:

- training became unstable
- codebook collapsed toward one code
- reconstruction failed badly

Takeaway:

The architecture change exposed instability instead of solving the core issue.

### 5. Chunk-local codec with full overfit mode

Command:

`python -m loopy.train_codec --data-path loopy/example_corpus.txt --epochs 120 --batch-size 4 --codebook-size 128 --d-model 128 --encoder-layers 2 --decoder-layers 2 --chunk-size 4 --dropout 0.0 --weight-decay 0.0 --learning-rate 0.0001 --overfit-all`

Best metrics:

- val loss: `2.6144`
- recon loss: `1.2834`
- vq loss: `1.3310`
- codebook perplexity: `15.09`
- compression: `3.90x`

Sample reconstruction:

- source: `A small robot woke up before sunrise and looked at the empty street.`
- reconstruction: `d saall ook e thke we thsoke bussoke ard looked an tiette thoat iett`

Takeaway:

This was the strongest codec result so far, but it still failed the real test.
The codec learned approximate local patterns, not exact reversible text.

## Decision

Pause codec-first work.

Do not build a latent LM on top of a broken codec.

## New direction

Use reversible concept middleware.

Core idea:

- keep normal text
- rewrite selected concepts into stable IDs like `<n4>`
- decode them back after generation

Why this is better right now:

- cheaper to test
- easier to reason about
- reversible
- gives a strong baseline against learned symbolic ideas

## Concept middleware result

Command:

`python -m loopy.rewrite_corpus --input loopy/example_corpus.txt --output loopy/runs/concept_example.txt --lexicon loopy/concepts.sample.json`

Result:

- total replacements: `14`
- concepts used: `12`
- example rewrite: `A small <n4> woke up before sunrise and looked at the empty <n6>.`

Takeaway:

The middleware works.
The rewritten corpus stays readable.

## Token LM comparison result

Commands:

`python -m loopy.train_token_lm --data-path loopy/example_corpus.txt --output-dir loopy/runs/lm_plain --epochs 120 --batch-size 4 --dropout 0.0 --weight-decay 0.0 --learning-rate 0.001 --overfit-all`

`python -m loopy.train_token_lm --data-path loopy/runs/concept_example.txt --output-dir loopy/runs/lm_concept --epochs 120 --batch-size 4 --dropout 0.0 --weight-decay 0.0 --learning-rate 0.001 --overfit-all`

Results:

- plain loss: `0.2041658640`
- concept loss: `0.2041313648`
- plain perplexity: `1.2265016`
- concept perplexity: `1.2264593`
- plain vocab: `197`
- concept vocab: `196`

Sample generations:

- plain: `A small robot woke up before sunrise and looked at the empty street.`
- concept: `A small <n4> woke up before sunrise and looked at the empty <n6>.`

Takeaway:

Yes, the concept version is numerically better, but only by a tiny amount.
On this toy corpus, the difference is too small to treat as meaningful evidence.
It shows the rewrite does not hurt modeling, but it does not yet prove the concept layer is useful.

## New decision rule

Do not judge this idea on tiny generic text.
Test it next on a corpus where concept normalization matters more:

- many repeated entities
- synonyms for the same thing
- domain terms
- slightly noisy wording

That is the only setup where this idea has a fair chance to show real value.

## New experiment scaffold

Added a first fair test for the middleware path:

- [concepts.support.json](C:/Users/adarw/Desktop/googlereview/loopy/concepts.support.json)
- [make_domain_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/make_domain_corpus.py)
- [DOMAIN_EXPERIMENT.md](C:/Users/adarw/Desktop/googlereview/loopy/DOMAIN_EXPERIMENT.md)

This experiment uses a synthetic support corpus with:

- repeated concepts
- many alias forms
- multi-word phrases
- enough scale to use a real train/validation split

Decision rule for this stage:

- if the concept run shows a visible validation win, keep pushing this direction
- if the concept run is still flat, the middleware idea is weaker than it sounds and should be reconsidered

## Fairness bug found in the first domain run

The first domain comparison was invalid.

What happened:

- the rewritten concept corpus collapsed many different sentences into the same symbolic sentence
- `load_text_samples()` was deduping repeated lines
- plain corpus kept `301` unique lines
- concept corpus dropped to only `82` unique lines
- that is why the concept run was much faster and much worse

Observed bad comparison:

- plain train/val: `271 / 30`
- concept train/val: `74 / 8`

Fix applied:

- `train_token_lm.py` now loads samples with `dedupe=False`
- repeated lines are preserved for LM experiments
- this keeps the plain and concept runs on the same sample count

Conclusion:

Do not use the first domain metrics as evidence against the concept idea.
Rerun the LM comparison after the loader fix.

## Fair rerun result on the support-domain corpus

After fixing the dedupe bug, the domain comparison became fair:

- plain train/val: `288 / 32`
- concept train/val: `288 / 32`

Results:

- plain loss: `0.5442`
- concept loss: `0.2409`
- plain perplexity: `1.7233`
- concept perplexity: `1.2723`
- plain vocab: `230`
- concept vocab: `185`

Takeaway:

This is the first strong positive result in the project.
The symbolic middle layer made the corpus much easier to model in the narrow domain setting.

## New next step

Do not stop at the favorable synthetic result.

Two follow-ups were added:

- [inspect_run.py](C:/Users/adarw/Desktop/googlereview/loopy/inspect_run.py) to decode concept generations back to normal English
- [make_noisy_support_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/make_noisy_support_corpus.py) for a harder partial-coverage corpus
- [REALISM_EXPERIMENT.md](C:/Users/adarw/Desktop/googlereview/loopy/REALISM_EXPERIMENT.md) with exact commands

Decision rule now:

- if the concept win survives the noisy corpus, the middleware path is genuinely promising
- if it collapses, the current gain was too dependent on a controlled setup

## Realism experiment result

The noisy partial-coverage corpus was the harder follow-up test.

Observed fair comparison:

- plain train/val: `324 / 36`
- concept train/val: `324 / 36`

Results:

- plain loss: `0.6243`
- concept loss: `0.5076`
- plain perplexity: `1.8669`
- concept perplexity: `1.6614`
- plain vocab: `227`
- concept vocab: `190`

Decoded generation check:

- plain output stayed readable
- concept output also stayed readable after decode
- concept output canonicalized wording like `user` -> `customer` and `admin panel` -> `dashboard`

Takeaway:

The gain survived the harder corpus, but it shrank.
That makes the result more credible.

Current interpretation:

- the symbolic middle layer is meaningful
- the remaining challenge is preserving nuance while keeping the training advantage
- the next real test must use a real corpus, not another synthetic one

## Design note added before the break

A new split in the architecture is now explicit:

1. ambiguity-aware encoding
2. context-aware decoding

This came from the realization that the current system solves training consistency, but it still flattens wording too much.

Current intended direction:

- input middleware should skip uncertain rewrites instead of forcing every lexical match into a concept code
- output middleware should use nearby context to recover the most natural surface word from a concept code
- simple context windows or a context tree are the first planned baseline for output recovery

## Focused research memo added

Added [AMBIGUITY_RESEARCH.md](C:/Users/adarw/Desktop/googlereview/loopy/AMBIGUITY_RESEARCH.md) to capture the literature around the next two bottlenecks:

- ambiguity-aware encoding
- context-aware decoding

Main conclusion from the memo:

- Loopy should research these issues now, not later
- the most relevant adjacent fields are word sense disambiguation, entity linking, lexical substitution, surface realization, copy mechanisms, and context-tree modeling with side information
- the best next build should be a confidence-gated encoder plus a context-aware alias decoder

## Ambiguity baseline scaffold added

Moved from research into the first build step for ambiguity-aware encoding.

Added:

- [concept_policy.py](C:/Users/adarw/Desktop/googlereview/loopy/concept_policy.py)
- [rewrite_corpus_contextual.py](C:/Users/adarw/Desktop/googlereview/loopy/rewrite_corpus_contextual.py)
- [concept_policy.support.json](C:/Users/adarw/Desktop/googlereview/loopy/concept_policy.support.json)
- [AMBIGUITY_BASELINE.md](C:/Users/adarw/Desktop/googlereview/loopy/AMBIGUITY_BASELINE.md)

Purpose:

- skip questionable alias rewrites instead of forcing every match into a concept code
- compare canonical concept rewriting vs context-gated concept rewriting on the noisy corpus

## Ambiguity baseline result

The first context-gated input rewrite baseline underperformed the canonical concept rewrite on the noisy corpus.

Observed fair comparison:

- canonical concept loss: `0.5076`
- gated concept loss: `0.5219`
- canonical concept perplexity: `1.6614`
- gated concept perplexity: `1.6852`
- canonical vocab: `190`
- gated vocab: `198`

Rewrite report highlights:

- canonical replacements: `860`
- gated replacements: `778`
- gated skips: `82`

Top skip reasons:

- `bill:missing_window` -> `28`
- `console:missing_window` -> `18`
- `case:missing_window` -> `9`
- `error:missing_window` -> `7`
- `plan:missing_window` -> `6`

Interpretation:

The first rule-based gate was too blunt.
It skipped some rewrites that were still useful for normalization, so the concept representation became slightly less effective.

Example:

- source: `One buyer changed plan twice in one day and the receipt page still reflected the old tier.`
- canonical rewrite: `One buyer changed <n3> twice in one day and the receipt page still reflected the old <n3>.`
- gated rewrite: `One buyer changed plan twice in one day and the receipt page still reflected the old <n3>.`

Current conclusion:

- the idea of ambiguity handling is still valid
- this first manual policy is not good enough
- input-side ambiguity should not be solved with brittle local rules alone
- the stronger next move is output-side context-aware decoding, while keeping canonical input rewriting as the better baseline for now

## Decoder baseline scaffold added

Moved to the next priority after the input-side ambiguity gate underperformed.

Added:

- [surface_decoder.py](C:/Users/adarw/Desktop/googlereview/loopy/surface_decoder.py)
- [evaluate_surface_decoder.py](C:/Users/adarw/Desktop/googlereview/loopy/evaluate_surface_decoder.py)
- [DECODER_BASELINE.md](C:/Users/adarw/Desktop/googlereview/loopy/DECODER_BASELINE.md)

Purpose:

- compare fixed canonical decode vs context-aware alias recovery
- measure exact alias recovery on aligned corpora
- settle whether the next useful gain comes from output-side recovery rather than more input-side rules

## Decoder baseline result

The first context-aware alias decoder produced a small overall win over fixed canonical decode.

Observed evaluation:

- examples: `76`
- canonical alias accuracy: `0.2895`
- contextual alias accuracy: `0.3158`

Interpretation:

- overall, local-context alias recovery is better than always decoding to the canonical form
- the gain is small, but real
- this supports moving forward on the output side rather than returning to brittle input rules

Important nuance:

The improvement is uneven across concepts.

Concepts that improved clearly:

- `n11` api key: `0.1667` -> `0.5000`
- `n12` sync job: `0.0000` -> `0.6000`
- `n3` subscription: `0.2500` -> `0.5000`
- `n5` refund: `0.2000` -> `0.4000`
- `n8` error message: `0.1111` -> `0.2222`

Concepts that got worse:

- `n10` workspace: `0.4000` -> `0.2000`
- `n13` integration: `0.5556` -> `0.3333`
- `n4` invoice: `0.7500` -> `0.0000`
- `n7` dashboard: `0.2500` -> `0.1250`

Current conclusion:

- context-aware decoding is directionally correct
- the current simple decoder is too aggressive for some concepts
- next refinement should be concept-specific confidence control and copy-first behavior

## Refined decoder result

The conservative decoder refinement did not improve the overall score.

Observed evaluation:

- canonical alias accuracy: `0.2895`
- refined contextual alias accuracy: `0.3158`
- examples: `76`
- copy used: `0`
- fallback to canonical: `52`

Interpretation:

- the refinement protected some concepts from over-switching
- but it did not improve the aggregate alias recovery over the first decoder baseline
- the copy-first logic did not activate on this evaluation split

Per-concept outcome:

- `n10` workspace improved from `0.2` to `0.4`
- `n13` integration improved from `0.3333` to `0.5556`
- `n4` invoice improved from `0.0` to `0.25`

But:

- `n11` api key dropped from `0.5` to `0.1667`
- `n5` refund dropped from `0.4` to `0.2`
- `n7` dashboard dropped from `0.125` to `0.0`

Current conclusion:

- context-aware decoding is still the right direction
- but this hand-tuned refinement did not beat the first decoder baseline
- the next serious step should be a real-corpus evaluation or a stronger learned decoder, not more hand-tuned thresholds on synthetic data

## Real-corpus scaffold added

Moved the project from "talking about real data" to an actual reusable pipeline.

Added:

- [prepare_real_corpus.py](C:/Users/adarw/Desktop/googlereview/loopy/prepare_real_corpus.py)
- [concepts.real.template.json](C:/Users/adarw/Desktop/googlereview/loopy/concepts.real.template.json)
- [REAL_CORPUS_RUNBOOK.md](C:/Users/adarw/Desktop/googlereview/loopy/REAL_CORPUS_RUNBOOK.md)
- [data/README.md](C:/Users/adarw/Desktop/googlereview/loopy/data/README.md)

Purpose:

- ingest raw `.txt` or `.jsonl` domain text
- normalize and filter it into one sample per line
- optionally redact obvious sensitive patterns before experiments
- give the next real-data comparison one standard path instead of ad hoc commands

Current decision:

- the next Loopy stage is the first real-corpus run
- keep canonical concept rewriting as the input baseline
- keep the first contextual decoder as the output baseline
- only move to a stronger learned decoder if the real-data win survives

## Real Twitter corpus prepared

Selected dataset:

- public Twitter customer-support slice prepared from `MohammadOthman/mo-customer-support-tweets-945k`
- local prepared corpus: `loopy/data/real/twitter_support_5k.txt`

Preparation result:

- kept before dedupe: `4675`
- kept after dedupe: `4674`
- dropped as too long: `268`
- dropped for too few tokens: `24`
- average chars: `184.95`

Interpretation:

- this is large enough for the first real Loopy test
- the corpus is messy and broad, which makes it a better pressure test than the synthetic support sets
- a conservative starter lexicon was added in [concepts.twitter.real.json](C:/Users/adarw/Desktop/googlereview/loopy/concepts.twitter.real.json)

Current next step:

- rewrite the prepared corpus with the starter Twitter lexicon
- inspect the rewrite report
- then run plain vs concept LM training on the real corpus

## First real-corpus rewrite inspection

The first rewrite on `twitter_support_5k` produced enough signal to inspect before training.

Observed rewrite stats:

- total replacements: `4637`
- concepts used: `18`
- strongest concept: `n16` direct message with `822` replacements

Important false positives found during inspection:

- `plays in order` was rewritten to `<n3>` order
- `i know my password` was rewritten to `<n15>` password reset
- `Direct Message via Twitter` became `<n16> <n16>` because two aliases matched in sequence

Decision:

Tighten the starter Twitter lexicon before the first LM run.

Changes applied to [concepts.twitter.real.json](C:/Users/adarw/Desktop/googlereview/loopy/concepts.twitter.real.json):

- removed bare `order` from `n3`
- removed bare `password` and `passwords` from `n15`
- removed `via twitter` from `n16`

Interpretation:

- this real-corpus inspection step was necessary
- the right strategy on real data is conservative precision first, not maximum rewrite coverage
- rerun the rewrite before starting plain vs concept LM training

## Second real-corpus lexicon tightening

The first tightened rewrite still had one clear false positive:

- `plays in order` was still rewritten to `<n3>` because the canonical form itself is always matched by the middleware

Fix applied:

- changed `n3` canonical from `order` to `purchase order`
- kept `order` only as an alias

Reason:

- the middleware always includes the canonical form in matching
- so a broad canonical like `order` is too risky on real support text
- a more specific canonical keeps the internal concept but reduces accidental surface collisions

## Third real-corpus lexicon tightening

The previous `n3` fix was insufficient.

Reason:

- changing the canonical phrase from `order` to `purchase order` did not help by itself
- `order` was still present as an alias, so the middleware still rewrote bare `order`

Fix applied:

- removed bare `order` from `n3` aliases
- kept safer purchase-related forms like `orders`, `purchase`, `purchases`, `booking`, and `bookings`

Current rule:

- for real data, avoid single-word aliases that also appear often in non-domain generic language

## First real-corpus LM result

The first real-corpus comparison used the prepared Twitter support slice.

Observed fair comparison:

- plain train/val: `4207 / 467`
- concept train/val: `4207 / 467`

Best metrics:

- plain loss: `4.4437`
- concept loss: `4.4667`
- plain perplexity: `85.09`
- concept perplexity: `87.07`
- plain vocab: `11678`
- concept vocab: `11606`

Training dynamics:

- plain model peaked around epoch `9`
- concept model peaked around epoch `7`
- both runs overfit afterward

Interpretation:

This is the first real negative result for the concept middleware path.
The concept version did not beat plain text on the broad Twitter support corpus.

What this likely means:

- the real corpus is much broader and noisier than the synthetic support sets
- the current lexicon is too shallow relative to the dataset's semantic variety
- the vocabulary reduction was tiny, so normalization did not simplify the task enough
- conservative rewriting avoided serious damage, but it also limited upside

Current conclusion:

- Loopy is still promising in narrow controlled domains
- it is not yet validated on broad real-world support corpora
- the next move should not be "add more random concepts"
- the better move is to narrow the real corpus to one domain or one company slice and test again

## Loopy v2 pivot

The project has now pivoted from Loopy v1 to Loopy v2.

Reason:

- v1 was a useful probe, but it stayed too shallow
- narrow synthetic domains gave positive results
- the first broad real corpus did not
- the deeper goal is not synonym rewriting, it is a new internal representation

New active direction:

- build a semantic binary codec
- raw bytes in
- grouped learned bits internally
- exact reconstruction out
- local smoke tests first
- H100 rentals only after local validation

New files added for v2:

- [V2_ARCHITECTURE.md](C:/Users/adarw/Desktop/googlereview/loopy/V2_ARCHITECTURE.md)
- [V2_LOCAL_RUNBOOK.md](C:/Users/adarw/Desktop/googlereview/loopy/V2_LOCAL_RUNBOOK.md)
- [v2_config.py](C:/Users/adarw/Desktop/googlereview/loopy/v2_config.py)
- [binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/binary_codec_v2.py)
- [train_binary_codec_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_binary_codec_v2.py)

Current decision:

- Loopy v1 is archived as research context
- Loopy v2 is the active experiment path

## First v2 smoke result

The first semantic binary codec smoke test did run end to end, but it failed the real objective.

Observed result:

- loss: `3.0286`
- recon loss: `3.0047`
- estimated bpb: `2.24`
- bit density: `0.507`
- best epoch: `12`
- reconstruction was effectively blank except for a few repeated `e` characters

Interpretation:

- the training loop is alive
- the binary bottleneck is active
- the current setup was still too compressed to expect exact reconstruction

Most important lesson:

- the original smoke test used `20` bits for each `16`-byte patch
- that is only `1.25` raw bits per byte before any entropy coding or prior model
- that is too aggressive for a first reconstruction test

Current decision:

- stage v2 properly
- first prove high-capacity exact reconstruction
- only then reduce the bit budget and test actual compression pressure

## High-capacity patch overfit result

The first high-capacity patch run improved substantially over the earlier blank-output smoke test, but it still did not reconstruct well enough.

Observed result:

- loss: `1.1641`
- recon loss: `1.1261`
- align loss: `0.0380`
- raw capacity bpb: `8.0`
- reconstruction was noisy and partly readable, but still far from exact

Interpretation:

- the architecture is learning local character structure
- the binary bottleneck is no longer totally degenerate
- but 4-byte patch reconstruction is still too hard for the current setup

Current decision:

- add byte accuracy as a core metric
- step back to a simpler single-byte sanity test
- only return to multi-byte patch experiments after the byte-level test works

## Byte-level sanity result

The single-byte v2 sanity test was the first strong positive result for the semantic binary codec path.

Observed result:

- loss: `0.1402`
- recon loss: `0.0993`
- byte accuracy: `0.9923`
- bit density: `0.511`
- best epoch: `30`

Interpretation:

- the architecture is not dead
- the binary bottleneck can carry enough information for byte-level reconstruction
- the remaining difficulty is in multi-byte patch composition, not in the basic binary codec path itself

Current decision:

- treat stage 0 as passed
- fix metric/output bugs from the sanity test
- move to the next diagnostic: 2-byte patch reconstruction
## Byte-level sanity result

The single-byte semantic binary codec test was the first clear positive result for Loopy v2.

Observed result:

- loss: `0.1402`
- recon loss: `0.0993`
- byte accuracy: `0.9923`
- bit density: `0.5114`
- raw capacity bpb: `16.0`
- best epoch: `30`

Sample reconstruction:

- source sentence was recovered correctly
- the trailing repeated `h` characters came from decoding padded positions in the preview, not from failure on the real bytes

Fixes applied after this run:

- preview decoding now respects `patch_mask`, so padded positions are not rendered into the sample output
- estimated bits-per-byte now masks inactive patches correctly instead of charging entropy for padded regions

Interpretation:

- the architecture is alive at byte level
- the bottleneck can carry enough information for near-exact reconstruction of active bytes
- the next real question is whether the same path can compose multiple bytes into a single learned patch

Current decision:

- treat stage 0 as passed
- move next to a `patch_size=2` high-capacity diagnostic
- do not go back to GPU rental or real-corpus v2 runs yet
## First 2-byte patch diagnostic

The first `patch_size=2` run showed that the codec can begin composing bytes into a patch, but the current patch encoder was not preserving enough ordered information.

Observed result:

- loss: `0.5617`
- recon loss: `0.5253`
- align loss: `0.0364`
- estimated bpb: `12.6846`
- byte accuracy: `0.8355`
- bit density: `0.5057`
- raw capacity bpb: `16.0`
- best epoch: `30`

Sample reconstruction:

- source: `A small robot woke up before sunrise and looked at the empty street.`
- reconstruction: `A smal lrrbo touke u  befoer sunbise and looked at thee mpty street.`

Interpretation:

- this is materially better than the failed 4-byte run
- the model is clearly learning local byte-pair structure
- but it is still too inaccurate for a true reconstruction pass

Likely cause:

- the old patch encoder mean-pooled encoded byte positions inside each patch
- that discards too much order information once `patch_size > 1`

Fix applied after this run:

- the patch encoder now preserves ordered within-patch features by flattening masked encoded positions before projection into the latent representation

Current decision:

- do not move to `patch_size=4` yet
- rerun the `patch_size=2` diagnostic after the encoder fix
- only if that improves clearly should larger patches be attempted
## Revised 2-byte patch result

After changing the patch encoder to preserve ordered within-patch byte features, the 2-byte patch diagnostic became the first strong multi-byte reconstruction result for Loopy v2.

Observed result:

- loss: `0.1090`
- recon loss: `0.0474`
- align loss: `0.0616`
- estimated bpb: `10.9119`
- byte accuracy: `0.9986`
- bit density: `0.5042`
- raw capacity bpb: `16.0`
- best epoch: `40`

Sample reconstruction:

- source: `A small robot woke up before sunrise and looked at the empty street.`
- reconstruction: exact

Interpretation:

- the order-preserving encoder fix was the right architectural change
- the codec can now reconstruct exact text through a multi-byte binary bottleneck on the toy corpus
- the main blocker is no longer basic patch composition at `patch_size=2`

Important limitation:

- this is still a high-capacity sanity test, not a real compression win
- `32` bits are being used for each `2`-byte patch, which is `16.0` raw bits per byte
- the measured effective rate is still above raw text, so this result proves recoverability, not compression advantage

Current decision:

- treat 2-byte patch reconstruction as passed
- move next to the 4-byte high-capacity test using the revised ordered encoder
- only after 4-byte reconstruction works should the bit budget be tightened
## Revised 4-byte patch result

The revised `patch_size=4` high-capacity run is the strongest v2 result so far.

Observed result:

- loss: `0.2165`
- recon loss: `0.1327`
- align loss: `0.0837`
- estimated bpb: `5.1867`
- byte accuracy: `0.9860`
- bit density: `0.5083`
- raw capacity bpb: `8.0`
- best epoch: `40`

Sample reconstruction:

- source: `A small robot woke up before sunrise and looked at the empty street.`
- reconstruction: exact

Interpretation:

- the codec can now reconstruct exact text with 4-byte patches through the binary bottleneck on the toy corpus
- this is the first result that suggests real compression headroom, not just recoverability
- the measured effective rate proxy is now below raw text (`5.19` estimated bpb vs `8.0` raw capacity bpb and `8` raw text bits per byte)

Important limitation:

- this is still an overfit toy-corpus result
- `estimated_bpb` is an entropy proxy over the learned bits, not yet a real arithmetic-coded end-to-end compressed file size
- the result proves the representation has become compressible, but not yet that it generalizes or beats strong baselines on real text

Current decision:

- treat 4-byte reconstruction as passed
- move next to the moderate-compression test at `24` bits per 4-byte patch (`6.0` raw capacity bpb)
- if that also reconstructs well, v2 becomes strong enough for a first larger local corpus test
## 4-byte moderate-capacity result

The `patch_size=4` moderate-capacity run tightened the bit budget from `32` bits per patch to `24` bits per patch and still preserved almost all of the sentence.

Observed result:

- loss: `0.3300`
- recon loss: `0.2471`
- align loss: `0.0829`
- estimated bpb: `3.6890`
- byte accuracy: `0.9645`
- bit density: `0.5174`
- raw capacity bpb: `6.0`
- best epoch: `40`

Sample reconstruction:

- source: `A small robot woke up before sunrise and looked at the empty street.`
- reconstruction: `A small robot woke up before sknrise and looked at the empty street.`

Interpretation:

- this is not exact reconstruction, so the tighter budget is starting to bite
- however, the result is still very strong for a toy overfit compression step
- most of the sentence survives intact, and the error pattern is now small and localized rather than globally broken

Why this matters:

- the architecture held up when the budget was reduced from `8.0` raw capacity bpb to `6.0`
- the estimated entropy rate stayed far below raw text on the toy corpus
- the learned representation still appears structured and compressible instead of collapsing

Important limitation:

- this is still a toy overfit result
- `estimated_bpb` remains an entropy proxy, not a final compressed file size
- the next informative step is no longer more toy tuning, but a first local real-corpus check with the same settings

Current decision:

- treat the moderate-capacity result as strong enough to justify a first real local run
- keep `patch_size=4` and `bit-groups 6,6,6,6`
- do not rent H100s yet
## First real local v2 corpus result

The first real local corpus run is the strongest overall result in the project so far.

Observed result on `twitter_support_5k`:

- loss: `0.2130`
- recon loss: `0.0677`
- align loss: `0.1452`
- estimated bpb: `2.2744`
- byte accuracy: `0.9821`
- bit density: `0.4752`
- raw capacity bpb: `6.0`
- train / val samples: `4207 / 467`
- best epoch: `9`
- average epoch seconds: `35.49`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. Now ,30 and still waiting.... Agent: Sorry Sam, did you receive your order? Cerin`

Interpretation:

- the codec is stable on real noisy text, not just on the toy corpus
- reconstruction quality is very high overall
- the remaining weak spots are concentrated in fragile surface details like numbers and names
- the learned bit representation still looks highly compressible on real text according to the entropy proxy

Why this matters:

- this is the first result that makes Loopy v2 look like a real research direction rather than just a toy architecture
- the earlier v1 middleware path failed on broad real Twitter support text
- v2 now succeeds on the same real corpus family in the sense that it can reconstruct faithfully through a compressed binary bottleneck

Important limitations:

- `estimated_bpb` is still an entropy proxy, not a true end-to-end arithmetic-coded file size
- this is not yet a benchmark against gzip, zstd, BPE, or other neural codecs
- no explicit rate objective was used yet (`rate_weight = 0`), so compression is currently emerging indirectly rather than being directly optimized
- names, numbers, and rare details are still the fragile part of the reconstruction

Current decision:

- treat this as the first serious real-data validation of Loopy v2
- the next work should focus on quantifying real compression and protecting fragile details
- H100 rentals are now much more justifiable, but a couple of sharp local follow-ups still make sense first
## Small rate-aware real-corpus result

The first rate-aware follow-up on the real Twitter support corpus stayed strong.

Observed result on `twitter_support_5k` with `rate_weight=0.001`:

- loss: `0.2167`
- recon loss: `0.0755`
- rate loss: `0.000417`
- align loss: `0.1408`
- estimated bpb: `2.3733`
- byte accuracy: `0.9800`
- bit density: `0.4890`
- raw capacity bpb: `6.0`
- train / val samples: `4207 / 467`
- best epoch: `10`
- average epoch seconds: `40.96`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. Sow 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri?`

Interpretation:

- adding a small explicit rate penalty did not break the codec on real noisy text
- fidelity stayed very high overall
- the main remaining errors are still small fragile surface details rather than global corruption
- compared with the previous no-rate real run, the result is roughly similar overall, which is a good sign: compression pressure can be introduced without immediate collapse

Current conclusion:

- Loopy v2 is now validated on real text under both zero-rate and small-rate settings
- this is strong enough to justify the next phase on a medium GPU platform like Google Colab before H100 rentals
- the next work should focus on two things:
  1. medium-scale runs and ablations off CPU
  2. measurement of true compressed size, not just entropy proxy
## 20-epoch real-corpus baseline result

A longer 20-epoch run on the real Twitter support corpus improved the baseline further.

Observed result on `twitter_support_5k` with `patch_size=4`, `bit-groups 6,6,6,6`, `rate_weight=0.0`:

- loss: `0.1646`
- recon loss: `0.0596`
- align loss: `0.1049`
- estimated bpb: `1.8326`
- byte accuracy: `0.9849`
- bit density: `0.4840`
- raw capacity bpb: `6.0`
- train / val samples: `4207 / 467`
- best epoch: `20`
- average epoch seconds: `33.39`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Pam, did you beceive your order? Ceris`

Interpretation:

- the longer run improved the aggregate metrics over the 10-epoch baseline
- the representation became even more compressible according to the entropy proxy
- fragile surface-detail errors remain, but the model stays highly faithful overall on real noisy text

Current conclusion:

- CPU validation has done its job
- Loopy v2 now has a strong real-corpus baseline
- the next phase should be medium-scale GPU validation on Google Colab rather than more CPU-only extension runs
## First Colab GPU baseline result

The first Colab GPU baseline reproduced the CPU behavior cleanly and gave the first prototype true bitstream measurement.

Observed training result on Colab GPU:

- loss: `0.2156`
- recon loss: `0.0510`
- align loss: `0.1646`
- estimated bpb: `1.6132`
- byte accuracy: `0.9861`
- bit density: `0.4921`
- raw capacity bpb: `6.0`
- best epoch: `19`
- average epoch seconds: `4.66`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. No! 330 and still waiting.... Agent: Sorry Sam, did you receive your order? Cerio`

Interpretation of training:

- the GPU baseline reproduced the strong CPU behavior
- training became dramatically faster than CPU while keeping high fidelity
- the remaining errors are still concentrated in fragile details like numbers, punctuation, and names

Observed prototype bitstream measurement:

- hard bitstream bpb: `6.0111`
- zlib-compressed learned bitstream bpb: `4.5060`
- gzip-compressed learned bitstream bpb: `4.5061`
- zlib-compressed raw text bpb: `3.0611`
- gzip-compressed raw text bpb: `3.0613`

Interpretation of compression:

- this is the first important reality check for Loopy v2
- the learned representation is highly predictable to the model (`estimated_bpb` is low)
- but the actual packed hard-bitstream plus simple standard compression is still worse than compressing the raw text directly with zlib/gzip
- in other words, the representation is promising for modeling, but the current codec is not yet a competitive real compressor

What this means:

- Loopy v2 is validated as a learned representation / modeling direction
- Loopy v2 is not yet validated as a practical text compressor
- the next work should focus on turning model-side predictability into real bitstream savings

Current decision:

- keep Colab as the active phase
- next run should be a rate-aware Colab follow-up plus repeated bitstream measurement
- do not move to H100 yet
- do not claim compression superiority yet

## Colab CPU rerun of the baseline

After the notebook/auth cleanup, the baseline was rerun in Colab on CPU.

Observed training result:

- loss: `0.2128`
- recon loss: `0.0503`
- align loss: `0.1625`
- estimated bpb: `1.5684`
- byte accuracy: `0.9876`
- bit density: `0.5097`
- raw capacity bpb: `6.0`
- best epoch: `20`
- average epoch seconds: `45.63`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. How 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Cerin`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.4418`
- gzip-compressed learned bitstream bpb: `4.4420`
- zlib-compressed raw text bpb: `3.0611`
- gzip-compressed raw text bpb: `3.0613`

Interpretation:

- this improved the previous baseline bitstream slightly
- the modeling result stayed very strong
- the learned bitstream still does not beat raw-text compression

## Local rate-pressure follow-up (`rate_weight=0.01`)

Observed training result:

- loss: `0.2476`
- recon loss: `0.0787`
- rate loss: `0.00386`
- align loss: `0.1651`
- estimated bpb: `2.1823`
- byte accuracy: `0.9799`
- bit density: `0.4746`
- raw capacity bpb: `6.0`
- best epoch: `8`
- average epoch seconds: `56.06`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot o! ,m. Now F10 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceris`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.3861`
- gzip-compressed learned bitstream bpb: `4.3862`
- zlib-compressed raw text bpb: `3.0611`
- gzip-compressed raw text bpb: `3.0613`

Interpretation:

- this is the first clear sign that rate pressure can improve the real packed bitstream
- the improvement is modest but real relative to the current baseline
- the tradeoff is also real: fidelity dropped and fragile detail errors increased

Current conclusion:

- Loopy v2 has crossed into a real compression/fidelity frontier problem
- the architecture is validated strongly enough to continue
- the next clean step is a controlled sweep between `rate_weight=0.0` and `0.01`

## Local intermediate rate point (`rate_weight=0.003`)

Observed training result:

- loss: `0.2208`
- recon loss: `0.0730`
- rate loss: `0.00121`
- align loss: `0.1466`
- estimated bpb: `2.2935`
- byte accuracy: `0.9810`
- bit density: `0.4820`
- raw capacity bpb: `6.0`
- best epoch: `9`
- average epoch seconds: `54.70`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of em. Now 330 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceris`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.3997`
- gzip-compressed learned bitstream bpb: `4.3998`
- zlib-compressed raw text bpb: `3.0611`
- gzip-compressed raw text bpb: `3.0613`

Interpretation:

- this is currently the best tested middle tradeoff
- it improves packed learned-bitstream size over the fidelity baseline
- it preserves slightly more fidelity than `rate_weight=0.01`

## Colab GPU intermediate rate point (`rate_weight=0.005`)

Observed training result:

- loss: `0.2285`
- recon loss: `0.0734`
- rate loss: `0.00208`
- align loss: `0.1530`
- estimated bpb: `2.3675`
- byte accuracy: `0.9805`
- bit density: `0.4768`
- raw capacity bpb: `6.0`
- best epoch: `8`
- average epoch seconds: `10.61`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. Oow h30 and still waiting.... Agent: Sorry Sam, did you receive your order? Cerin`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.5231`
- gzip-compressed learned bitstream bpb: `4.5233`
- zlib-compressed raw text bpb: `3.0611`
- gzip-compressed raw text bpb: `3.0613`

Interpretation:

- this point is not useful
- it is worse than the fidelity baseline and worse than the stronger local `0.01` rate run
- it should be treated as a rejected rate setting

Updated conclusion:

- the useful region now appears to be below `0.005`
- `0.003` is the best compromise tested so far
- the next clean sweep should probe tighter values such as `0.002` and `0.0025`

## Colab GPU intermediate rate point (`rate_weight=0.002`)

Observed training result:

- loss: `0.2181`
- recon loss: `0.0761`
- rate loss: `0.00088`
- align loss: `0.1411`
- estimated bpb: `2.5042`
- byte accuracy: `0.9803`
- bit density: `0.4958`
- raw capacity bpb: `6.0`
- best epoch: `8`
- average epoch seconds: `9.14`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of 7m. Gow W30 and still waiting.... Agent: Sorry Sam, did you receive your orderd Cerid`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.5546`
- gzip-compressed learned bitstream bpb: `4.5548`

Interpretation:

- this point is clearly worse than `0.003`
- it should be rejected

## Colab GPU intermediate rate point (`rate_weight=0.0025`)

Observed training result:

- loss: `0.2125`
- recon loss: `0.0631`
- rate loss: `0.00102`
- align loss: `0.1484`
- estimated bpb: `2.3094`
- byte accuracy: `0.9840`
- bit density: `0.4847`
- raw capacity bpb: `6.0`
- best epoch: `8`
- average epoch seconds: `9.28`

Sample reconstruction:

- source: `Customer: delivery slot of 7m. Now 930 and still waiting.... Agent: Sorry Sam, did you receive your order? Ceri`
- reconstruction: `Customer: delivery slot of Lm. Now L30 and still waiting.... Agent: Sorry Sam, did you receive your orderd Cerin`

Observed prototype bitstream measurement:

- zlib-compressed learned bitstream bpb: `4.5508`
- gzip-compressed learned bitstream bpb: `4.5510`

Interpretation:

- fidelity improved over `0.002`
- but packed learned-bitstream size stayed poor
- this point is also worse than `0.003`

Final conclusion for this sweep:

- `0.003` is the best tested compromise
- `0.01` is the best packed-bitstream point so far
- the tiny neighborhood sweep around `0.003` did not produce a better point
- the next work should move away from simple rate tuning and toward either better packing or downstream LM usefulness

## Grouped packing follow-up

Grouped bitstream measurement was added to [measure_bitstream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/measure_bitstream_v2.py) and tested on the two most relevant checkpoints.

Observed result on the best fidelity baseline (`v2_twitter_local_20ep`):

- flat zlib learned-bitstream bpb: `4.4178`
- grouped zlib learned-bitstream bpb: `5.6482`

Observed result on the best compromise point (`v2_twitter_rate_003`):

- flat zlib learned-bitstream bpb: `4.3997`
- grouped zlib learned-bitstream bpb: `5.5823`

Interpretation:

- grouped packing is clearly worse than flat packing with the current bit layout
- mixing the groups together is not the main reason the stored bitstream is weak
- the next branch should move away from packing tweaks and toward downstream usefulness

## Downstream export + smoke test

Added [export_stream_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/export_stream_v2.py) and [DOWNSTREAM_LM_PLAN.md](C:/Users/adarw/Desktop/googlereview/loopy/DOWNSTREAM_LM_PLAN.md).

The exporter writes:

- `group_stream.txt`
- `raw_byte_stream.txt`

Both use angle-bracket tokens so [train_token_lm.py](C:/Users/adarw/Desktop/googlereview/loopy/train_token_lm.py) can consume them directly.

Smoke-test result:

- grouped stream LM, 1 epoch:
  - val loss: `2.5831`
  - perplexity: `13.24`
  - vocab size: `261`
- raw-byte stream LM, 1 epoch:
  - val loss: `1.6024`
  - perplexity: `4.97`
  - vocab size: `163`

Interpretation:

- the downstream tooling path works
- but this group-token LM is not yet the right decisive experiment
- it expands each patch back into four tokens, so it removes the sequence-length advantage that makes Loopy v2 interesting
- the correct next downstream branch is a patch-level prior model

## Patch-level prior scaffold + smoke test

Added [train_patch_prior_v2.py](C:/Users/adarw/Desktop/googlereview/loopy/train_patch_prior_v2.py).

This trains a patch-level prior in two modes:

- `learned`: predict next learned patch bits
- `raw`: predict next raw byte patch

The key metric is validation **bits per byte**, so both branches can be compared in the same unit.

One-epoch CPU smoke test:

- learned mode:
  - val loss: `0.6238`
  - val accuracy: `0.5550`
  - val bpb: `5.4109`
  - epoch seconds: `10.86`
- raw mode:
  - val loss: `3.0383`
  - val accuracy: `0.2246`
  - val bpb: `4.3833`
  - epoch seconds: `16.57`

Interpretation:

- the patch-level downstream comparison path is now real and runnable
- after 1 epoch, the learned stream is not yet beating the raw patch baseline in bpb
- this is not decisive yet; the branch needs a longer run, preferably on Colab GPU

## Patch-level prior 5-epoch Colab comparison

Observed result:

- learned mode:
  - loss: `0.5921`
  - accuracy: `0.5777`
  - bpb: `5.1364`
  - average epoch seconds: `0.99`
- raw mode:
  - loss: `2.5640`
  - accuracy: `0.3240`
  - bpb: `3.6991`
  - average epoch seconds: `1.09`

Interpretation:

- this is the first real downstream-usefulness result
- the learned stream still loses clearly to the raw patch baseline on the key metric (`bpb`)
- the learned stream is easier in per-bit accuracy terms, but that does not outweigh the worse compression-style predictive quality
- this is a negative result for the current codec objective, not for the whole research direction

Updated conclusion:

- the current codec is good at reconstruction
- the current codec is not yet producing a downstream-superior patch stream
- the next step should be a downstream-aware codec redesign, not more rate sweeps or packing tweaks
