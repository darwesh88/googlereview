# OpenClaw Adaptation Notes

## Goal

Review the local `openclaw` repo and identify what should be adapted for the planned multi-country WhatsApp-first business workflow platform.

Target product:

- WhatsApp-first AI receptionist and booking / lead recovery platform
- Multi-country
- Starts in beauty / appointment businesses
- Must expand later into other service workflows without re-architecture

## Summary

OpenClaw is useful to us primarily as an **architecture reference**, not as a product template.

The most valuable reusable ideas are:

1. A single gateway / control-plane boundary
2. Channel adapters behind a plugin interface
3. Canonical inbound message normalization
4. Strong routing / identity separation
5. Skills and plugins with precedence + gating
6. Webhook hardening, dedupe, and keyed serialization
7. Config-schema driven plugins and UI hints

The least relevant parts are:

1. Personal assistant UX
2. General-purpose multi-agent orchestration
3. Companion device/node features
4. Broad consumer messaging support across dozens of channels

## What To Adapt Directly

### 1. Gateway as control plane

OpenClaw treats the gateway as the single owner of messaging surfaces, routing, sessions, and client connections.

Why it matters:

- Your product also needs one source of truth for webhooks, outbound sends, workflow state, approvals, and audit trails.
- This is the right shape for WhatsApp BSP webhooks, web app operators, and worker jobs.

References:

- `openclaw/docs/concepts/architecture.md`
- `openclaw/README.md`

Adaptation:

- Build one core backend that owns all inbound/outbound channel connections.
- Put your dashboard, workers, and automations behind that core boundary.
- Do not let individual workflow packs open their own provider sessions.

### 2. Channel adapters behind a plugin model

OpenClaw splits channel support into plugins and keeps normalized shared logic outside provider-specific implementations.

Why it matters:

- You want WhatsApp first, but not forever only WhatsApp.
- Later, Instagram DM, email, SMS, and web chat should be new adapters, not product rewrites.

References:

- `openclaw/docs/tools/plugin.md`
- `openclaw/src/channels/registry.ts`
- `openclaw/src/channels/plugins/load.ts`

Adaptation:

- Keep a `channels` boundary with interfaces for:
  - receive webhook events
  - normalize events
  - send outbound messages
  - report delivery / failure / health
- Start with a `whatsapp-bsp` adapter, not WhatsApp Web.

### 3. Canonical inbound envelope + normalization

OpenClaw normalizes channel-specific payloads into a shared inbound envelope and shared routing model.

Why it matters:

- Your workflows should not care whether a message came from WhatsApp, web chat, or later Instagram.
- This is critical for maintainability.

References:

- `openclaw/src/plugin-sdk/inbound-envelope.ts`
- `openclaw/src/channels/plugins/normalize/whatsapp.ts`
- `openclaw/docs/channels/whatsapp.md`

Adaptation:

- Define canonical objects like:
  - `InboundMessage`
  - `Contact`
  - `Conversation`
  - `Lead`
  - `WorkflowRun`
  - `ChannelEvent`
- Normalize all providers into those objects before any workflow logic runs.

### 4. Routing and identity resolution

OpenClaw has strong routing rules and stable session keys that separate channel/account/peer/agent identity.

Why it matters:

- Your product will need tenant, location, staff, customer, and conversation continuity.
- If you do not design identity and routing carefully now, later multi-location support becomes messy.

References:

- `openclaw/src/routing/resolve-route.ts`
- `openclaw/docs/index.md`

Adaptation:

- Replace agent/session routing with:
  - tenant routing
  - location routing
  - channel-account routing
  - workflow routing
- Keep stable keys for `tenant + location + contact + conversation`.

### 5. Skills and plugin precedence

OpenClaw supports bundled, managed, workspace, and plugin-provided skills with precedence and gating.

Why it matters:

- This is a strong model for your future vertical packs.
- You can use the same idea for industry-specific workflow packs without polluting the core.

References:

- `openclaw/docs/tools/skills.md`
- `openclaw/src/agents/skills/plugin-skills.ts`

Adaptation:

- Keep core platform generic.
- Package vertical-specific playbooks as data / packs:
  - beauty pack
  - clinic pack
  - home-services pack
- Each pack should define:
  - intake questions
  - service schemas
  - policy defaults
  - escalation rules
  - message templates

### 6. Webhook hardening, dedupe, and serialization

OpenClaw has good primitives for webhook safety, persistent dedupe, and serializing work per key.

Why it matters:

- WhatsApp BSPs, calendars, CRMs, and payment providers all send webhooks.
- Duplicate deliveries and concurrent updates are guaranteed problems.

References:

- `openclaw/src/plugin-sdk/webhook-request-guards.ts`
- `openclaw/src/plugin-sdk/persistent-dedupe.ts`
- `openclaw/src/plugin-sdk/keyed-async-queue.ts`

Adaptation:

- Use webhook guards for:
  - method restrictions
  - content-type checks
  - body size limits
  - timeout limits
  - rate limits
  - in-flight caps per tenant / conversation
- Use persistent dedupe for webhook event IDs.
- Serialize workflow execution per conversation or appointment key.

### 7. Config-schema + UI hints for modules

OpenClaw plugins define JSON-schema config and UI hints in manifests.

Why it matters:

- You will eventually need a partner / internal admin UI where connectors and modules can be configured cleanly.
- This pattern scales much better than hard-coded forms.

References:

- `openclaw/extensions/voice-call/openclaw.plugin.json`
- `openclaw/extensions/voice-call/index.ts`

Adaptation:

- Define each connector or module with:
  - machine-readable config schema
  - UI labels / placeholders
  - sensitive field markers
  - validation rules

### 8. Health/status boundaries

OpenClaw has explicit channel status and health monitoring surfaces.

Why it matters:

- Operators need to know when a channel is down, a webhook is failing, or a connector is unhealthy.
- This is especially important in SMB automation where reliability is the product.

References:

- `openclaw/src/gateway/channel-health-monitor.ts`
- `openclaw/src/channels/plugins/status.ts`
- `openclaw/src/gateway/server-methods/skills.ts`

Adaptation:

- Expose health for:
  - channel adapter
  - calendar connector
  - payment connector
  - worker queue
  - AI provider

## What To Adapt With Changes

### 1. Replace WhatsApp Web with official BSP messaging

OpenClaw uses WhatsApp Web / Baileys because it is a personal assistant.

That is not the right production choice for your B2B product.

Adaptation:

- Keep the adapter shape.
- Replace the transport with official WhatsApp Business Platform providers.

### 2. Replace multi-agent with multi-tenant workflow isolation

OpenClaw is designed around personal assistants and agent isolation.

Your system should instead isolate by:

- tenant
- location
- workflow
- contact

### 3. Replace generic skills with workflow packs

OpenClaw skills teach a general-purpose assistant how to use tools.

Your platform should use the same packaging idea for:

- business logic packs
- vertical playbooks
- compliance presets

### 4. Replace broad channel support with deep workflow support

OpenClaw wins by connecting everywhere.

Your product should win by being excellent at one workflow:

- booking
- reminders
- rescheduling
- lead recovery
- waitlist fill

Do not chase many channels too early.

## What To Ignore For Now

- device node system
- voice wake / canvas / mobile node complexity
- personal companion UX
- dozens of channels
- generic agent shell
- remote workstation orchestration

These are impressive, but not useful for V1.

## Best Immediate Reuse Pattern

If you borrow only one mental model from OpenClaw, borrow this:

`core gateway + adapters + normalized events + pluggable workflow packs`

That is the right backbone for your business.

## Recommended Build Mapping

Map OpenClaw ideas into your product like this:

| OpenClaw concept | Your product equivalent |
| --- | --- |
| Gateway | Core orchestration API |
| Channel plugin | WhatsApp / web / email adapter |
| Agent route | Tenant + location + workflow route |
| Skill | Vertical workflow pack |
| Plugin manifest | Connector / module manifest |
| Pairing / allowlist | Opt-in / consent / operator access |
| Session key | Conversation / lead / appointment key |
| Control UI | Operator dashboard |

## Final Recommendation

Do not fork OpenClaw for this product.

Instead:

1. Borrow the architecture patterns
2. Borrow the plugin and skill packaging philosophy
3. Borrow the webhook and routing discipline
4. Build a clean product-specific core around official business APIs

That gives you the upside of OpenClaw's architecture without inheriting the wrong product assumptions.
