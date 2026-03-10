# Ambiguity Research

This memo covers the next two bottlenecks for Loopy:

1. ambiguity-aware encoding
2. context-aware decoding

The goal is to keep the training benefit of concept normalization without flattening too much meaning or style.

## Core problem split

### Input side

A surface word should not always be rewritten.

Example:

- `plan` in `upgrade your plan` may map to `<subscription>`
- `plan` in `what is the plan tomorrow` should stay normal text

### Output side

A concept code should not always decode to one fixed canonical word.

Example:

- `<customer>` may need to surface as `user`, `customer`, or `member`
- `<dashboard>` may need to surface as `admin panel` in some contexts

## What the literature says

### 1. Input ambiguity is a real standalone problem

Word sense disambiguation work is directly relevant here.

[GlossBERT](https://aclanthology.org/D19-1355/) treats disambiguation as a context-plus-definition matching problem and reports state-of-the-art WSD results. That matters because Loopy can treat each concept entry as a small sense definition and score whether a local mention really belongs to that concept.

More recent evaluation work suggests generic LLMs still struggle here. [On Functional Competence of LLMs for Linguistic Disambiguation](https://aclanthology.org/2024.conll-1.12/) reports that accuracy can drop below 70% on WSD datasets. [RoDEval](https://aclanthology.org/2025.emnlp-main.864/) argues that LLM WSD failures are often due to incomplete sense knowledge and overconfidence, and that scaling alone does not reliably fix this.

Inference for Loopy:

- do not assume a general LLM will solve rewrite ambiguity automatically
- use an explicit confidence gate
- if confidence is low, skip rewriting

### 2. Entity-like concept matching benefits from retrieve-then-rerank

[BLINK](https://aclanthology.org/2020.emnlp-main.519/) uses a two-stage entity linking setup:

- a fast bi-encoder retrieves candidates from context
- a cross-encoder reranks the shortlist

That is a strong template for Loopy input rewriting when the concept is closer to an entity or structured domain term.

Inference for Loopy:

- build a concept matcher that first proposes top candidate concepts cheaply
- then rerank them with more context
- do not brute-force every concept with the heaviest model

### 3. Output recovery is a lexical substitution problem

[BERT-based Lexical Substitution](https://aclanthology.org/P19-1328/) is highly relevant to Loopy decoding. The paper proposes and validates substitute candidates using context, and reports state-of-the-art results on lexical substitution benchmarks.

This maps almost exactly to Loopy output decoding:

- concept code gives the candidate set
- local sentence context ranks the surface form
- choose `user` vs `customer` vs `member` from context, not from a fixed canonical rule

Inference for Loopy:

- treat each concept alias list as a substitution candidate set
- rank candidates in context instead of always emitting the canonical form

### 4. Surface realization from symbolic forms is already a known useful split

[Designing a Symbolic Intermediate Representation for Neural Surface Realization](https://aclanthology.org/W19-2308/) is very close in spirit to Loopy. It argues that a symbolic intermediate representation can reduce failure modes in generation, and reports that the full system outperformed the winner of the E2E challenge.

[Generalising Multilingual Concept-to-Text NLG with Language Agnostic Delexicalisation](https://aclanthology.org/2021.acl-long.10/) is also relevant. It shows delexicalisation plus a post-editing relexicalisation step can outperform previous approaches, especially in low-resource conditions.

Inference for Loopy:

- splitting concept planning from surface realization is not a strange move
- a dedicated relexicalisation or post-editing step is a defensible design choice

### 5. Copying from context matters

[A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://aclanthology.org/E17-2075/) shows the usefulness of copying prior context into generation.

Inference for Loopy:

- if the original surface form is already in the prompt or recent context, decoding should be allowed to copy it
- this is one of the best ways to preserve style and avoid unnecessary canonicalization

### 6. Context size matters, but more context is not always better

[Context Matters in Semantically Controlled Language Generation for Task-oriented Dialogue Systems](https://aclanthology.org/2021.icon-main.18/) reports that contextual information improves generation, but longer context does not automatically help, while the immediate preceding utterance plays an essential role.

Inference for Loopy:

- your old idea of looking at nearby context is directionally right
- but only using the two words before may be too weak
- the best cheap baseline is likely a small local window plus immediate sentence context, not the whole document

### 7. Your context-tree intuition is mathematically legitimate

[The context-tree weighting method: basic properties](https://research.tue.nl/en/publications/the-context-tree-weighting-method-basic-properties/) shows context trees can model sequences efficiently with linear computational and storage complexity.

[An algorithm for universal lossless compression with side information](https://doi.org/10.1109/TIT.2006.880020) extends the idea to compression with side information and proves convergence to conditional entropy for stationary ergodic sources.

Inference for Loopy:

- a context tree is a valid cheap baseline for decoding concept codes back into surface words using nearby context as side information
- this is a mathematically clean and interpretable first decoder before heavier neural decoding

## Best design conclusions for Loopy

### Input side: ambiguity-aware encoding

Best first design:

1. exact alias hit proposes a concept
2. local context scores whether that concept is really intended
3. rewrite only if score is above threshold and margin from runner-up is large enough
4. otherwise keep the original text

Good baselines:

- rule-based skip lists and allowed-context rules
- concept descriptions plus a small context scorer, GlossBERT style
- retrieve-then-rerank for entity-like concepts, BLINK style

### Output side: context-aware decoding

Best first design:

1. concept code produces a candidate alias set
2. local context ranks candidates
3. if the original surface form exists in nearby context, allow copying
4. if confidence is low, fall back to the canonical form

Good baselines:

- local-window candidate ranking
- context tree over alias choices conditioned on neighboring words
- lexical-substitution scoring with a masked LM or small contextual scorer
- copy preference when the same surface word appears in the prompt or recent context

## Recommended build order

### Build first

- ambiguity-aware encoder gate
- context-aware decoder over alias lists
- copy-first fallback when the source wording is available

### Research later

- learned latent symbols again
- Mercury-style parallel refinement over concept sequences
- weight compression or quantization

## What to ignore for now

- do not jump into giant ontology systems
- do not use a full LLM in the loop for every rewrite decision
- do not treat one fixed canonical decode as acceptable long term
- do not combine this with Mercury-style generation before the ambiguity problem is handled

## Best next practical experiment

Use a real domain corpus and compare three systems:

1. plain text baseline
2. concept rewrite with canonical decode
3. concept rewrite with ambiguity-aware encoding plus context-aware decoding

That experiment will tell us whether the next layer of complexity is justified.
