---
name: saas-idea-validator
description: Research, compare, and validate SaaS business ideas. Use when evaluating 1 or more startup ideas, choosing between SaaS concepts, analyzing demand, competitors, ICP, pricing, GTM, or deciding whether an idea is simple enough to build first. Trigger on requests like "validate this SaaS idea", "research this business idea", "compare these startup ideas", "which SaaS should I build", "analyze this niche", or "help me pick the simplest promising product."
---

# SaaS Idea Validator

Use this skill to turn rough SaaS ideas into a ranked decision with explicit evidence, risks, and next validation steps.

Default to practical validation, not storytelling. Prefer small painful problems, narrow ICPs, visible buying intent, and fast distribution over large vague markets.

## Core Principles

- Favor a specific buyer over a broad audience.
- Favor a painful recurring workflow over a "nice to have".
- Favor existing budget and ugly workarounds over stated interest.
- Favor simple wedges over platforms, marketplaces, or network-effect bets.
- Favor ideas that can be validated in days or weeks, not months.
- State assumptions separately from evidence.

## What To Collect First

If the user provides multiple ideas, extract this for each one before scoring:

- One-sentence product definition
- Primary ICP
- Core painful job to be done
- Current workaround or substitute
- Who feels the pain and who pays
- Likely pricing model
- Why this can be reached or sold now

If information is missing, make the minimum reasonable assumption and label it clearly.

## Required Research Lenses

Research every idea through these lenses:

1. Problem severity
2. Buyer clarity
3. Market evidence
4. Competitive pressure
5. Distribution feasibility
6. Monetization willingness
7. Build complexity
8. Validation speed

Read [references/validation-scorecard.md](references/validation-scorecard.md) for the scoring rubric and output table format.
Read [references/research-checklist.md](references/research-checklist.md) when planning evidence collection or validation experiments.

## Workflow

### 1. Frame the ideas

Rewrite each idea into a crisp format:

`[product] for [ICP] that helps them [outcome] by [mechanism]`

Then identify:

- The triggering event that causes the buyer to look for this
- The highest-friction part of the current workflow
- Whether the buyer already spends money to solve it

### 2. Gather evidence

Use user-provided context first. If external research is available, prioritize:

- Competitor sites and pricing pages
- Review sites and community complaints
- Reddit, forums, GitHub issues, and job posts
- Search demand proxies
- Existing budgets, spend categories, or agency/service substitutes

Avoid declaring an idea validated from surface-level search volume alone.

### 3. Score each idea

Score each category from 1 to 5 using the rubric in [references/validation-scorecard.md](references/validation-scorecard.md).

Use this weighting:

- Problem severity: 20%
- Buyer clarity: 15%
- Market evidence: 15%
- Competitive pressure: 10%
- Distribution feasibility: 15%
- Monetization willingness: 10%
- Build complexity: 10%
- Validation speed: 5%

Adjust weights only if the user explicitly cares more about one constraint, such as bootstrapping speed or enterprise ACV.

### 4. Call out red flags

Explicitly flag ideas with any of these:

- The user cannot name a buyer with budget authority
- The pain is occasional or low urgency
- The idea depends on "if we get enough users"
- The wedge is too broad for version one
- Distribution depends on expensive paid acquisition with no margin room
- The market has strong incumbents and no clear wedge
- The workflow can be solved with existing general-purpose tools plus light services

### 5. Recommend the best first idea

Do not simply pick the highest theoretical upside. Prefer the best first business:

- Narrowest ICP with the clearest pain
- Fastest path to 5 to 10 user conversations
- Fastest path to a manual or MVP validation test
- Reasonable price relative to pain
- Lowest build complexity for meaningful value

If the best answer is "none of these yet," say so directly and explain what is missing.

### 6. Produce next-step experiments

End with a low-cost validation plan. Prefer 3 to 5 experiments such as:

- Ten ICP interviews with a sharp screening rule
- Concierge/manual prototype
- Landing page with a strong promise and call to action
- Direct outreach to buyers
- Paid test only if the channel is realistic for the ICP
- Simple MVP limited to the painful wedge

Each experiment must include:

- Objective
- Time cost
- Success threshold
- What decision the result unlocks

## Output Format

Use this structure unless the user asks for something else:

### Snapshot

- Ideas reviewed
- Best current pick
- Confidence level: low, medium, or high
- Biggest unresolved assumption

### Scorecard

Include the weighted table from [references/validation-scorecard.md](references/validation-scorecard.md).

### Evidence By Idea

For each idea, summarize:

- ICP
- Pain and current workaround
- Evidence for demand
- Competitor landscape
- Monetization angle
- Distribution path
- Main risks

### Recommendation

Pick one:

- Build now
- Validate first
- Park
- Reject

Then explain why in plain language.

### 7-Day Validation Plan

Provide the cheapest credible plan to reduce the biggest risks.

## Special Cases

- If the user has 2 to 3 ideas, compare them in one table and force-rank them.
- If the user has one idea, score it absolutely and explain what would have to be true for it to work.
- If the user asks for "simple", bias against ideas requiring integrations with many systems, complex workflows, marketplace liquidity, or heavy AI magic to be useful.
- If browsing or live research is unavailable, say that explicitly and base the analysis on user input, existing documents, and known heuristics.

## Decision Rules

Recommend moving forward only when most of these are true:

- Clear painful workflow
- Specific reachable ICP
- Existing spend or obvious willingness to pay
- Credible channel to reach first customers
- Small MVP can create value quickly
- Major risk can be tested cheaply

If those are not true, recommend a narrower wedge or a different idea instead of stretching the case.
