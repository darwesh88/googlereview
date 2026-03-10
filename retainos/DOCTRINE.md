# RetainOS Doctrine

## Mission

RetainOS exists to help web-first subscription companies keep more good revenue at the exact moment a user tries to cancel.

We do this by owning one narrow surface: the cancel-intent flow.

We do not try to optimize the whole business at once. We win one revenue moment first.

## Product

RetainOS is a subscription monetization control plane.

When a user clicks `Cancel`, the customer opens a hosted RetainOS flow. RetainOS decides which approved path to show, measures the outcome, and keeps improving the flow through controlled experiments.

## Core Metric

The core metric is:

`30-day net retained revenue per cancel-intent user`

Simple meaning:

For every user who starts the cancel flow, how much real, healthy revenue is still alive 30 days later because of our system.

This metric matters more than:

- save rate
- click rate
- discount acceptance
- immediate conversion

Those can create fake wins. We optimize for durable revenue.

## Guardrail Metrics

Every experiment is judged by the core metric and blocked or penalized by guardrails.

Guardrails:

- refund rate
- chargeback rate
- repeat churn within 30 days
- support complaint rate
- offer abuse rate
- margin damage from discounts

If an experiment improves the main metric but breaks guardrails, it loses.

## Smallest Loop

This is the only loop that matters at the start:

1. User clicks `Cancel`.
2. Customer app opens the RetainOS hosted flow.
3. RetainOS shows one approved variant.
4. RetainOS records exposure and outcome events.
5. RetainOS scores the result after enough time passes.
6. Winners become the new baseline.
7. Losers are removed.

Nothing broader ships until this loop works.

## Allowed Actions

RetainOS may change only the approved cancel-intent surface.

Allowed actions:

- offer type
- offer order
- copy
- CTA framing
- discount amount within explicit limits
- pause duration within explicit limits
- downgrade routing
- annual switch offer
- follow-up timing tied to cancel intent
- simple segmentation using approved attributes

## Forbidden Actions

RetainOS must not do any of the following:

- make cancellation harder to complete
- hide or delay the cancel path deceptively
- use dark patterns
- exceed customer-defined discount caps
- use sensitive or prohibited attributes for targeting
- run individualized surveillance pricing
- edit the customer codebase by default
- optimize on short-term save rate alone
- act outside the cancel-intent surface in the first phase

If a strategy depends on tricking users, it is not part of this company.

## Product Shape

Version 1 is:

- hosted cancel-intent flow
- signed session token from customer backend
- experiment assignment engine
- event ingestion
- evaluator
- simple reporting

Version 1 is not:

- a billing provider
- a full experimentation suite
- a generic AI growth product
- an outbound marketing tool
- a full lifecycle CRM

## Ideal Customer Profile

RetainOS starts with:

- web-first subscription businesses
- clear recurring revenue
- enough cancel volume to learn fast
- existing billing stack such as Stripe or similar
- existing analytics or event instrumentation

Strong first customers have:

- 1,000+ cancellations per month
- one owner of monetization or retention
- willingness to test a hosted cancel flow
- enough event quality to measure outcomes cleanly

Poor first customers:

- low-volume SMBs
- mobile-first businesses dependent on app-store-native cancellation flows
- businesses without clean event data
- customers asking for a full custom services engagement

## Why This Wedge

We start with cancel intent because it has:

- a clear trigger
- a narrow action surface
- direct revenue impact
- measurable outcomes
- safe rollback

This is the closest business analogue to the Karpathy loop:

- doctrine defines the rules
- the system changes one narrow surface
- experiments run inside constraints
- a hard metric decides what survives

## Expansion Path

RetainOS expands only after cancel intent works.

Order:

1. Cancel intent
2. Win-back flows
3. Trial-to-paid conversion
4. Upgrade prompts

We do not skip ahead.

## Experiment Rules

Every experiment must:

- change a narrow set of variables
- have a clear control
- respect customer policy
- have a minimum sample threshold
- be reversible
- log both short-term and delayed outcomes

At the beginning:

- all experiments are human-approved before launch
- only low-risk variants may later auto-launch

## Trust Model

RetainOS wins only if customers trust it with revenue surfaces.

That means:

- strict approval gates at the start
- full audit logs
- clear discount and policy caps
- kill switch for every customer
- least-privilege access
- no direct production code changes by default

## Positioning

RetainOS is not "AI for churn."

RetainOS is the system that continuously tests and deploys better cancel-intent decisions, then keeps only the ones that increase good revenue.

## One-Sentence Pitch

RetainOS helps subscription companies keep more healthy revenue at cancel intent by running controlled experiments on hosted save flows and promoting only the variants that improve 30-day net retained revenue.
