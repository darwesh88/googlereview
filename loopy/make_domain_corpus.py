from __future__ import annotations

import argparse
import random
from pathlib import Path


CUSTOMER = ["customer", "user", "member", "account holder"]
SUBSCRIPTION = ["subscription", "plan", "membership", "tier"]
INVOICE = ["invoice", "bill", "billing statement"]
REFUND = ["refund", "credit", "reimbursement", "money back"]
PASSWORD_RESET = ["password reset", "reset link", "login reset", "sign-in reset"]
DASHBOARD = ["dashboard", "admin panel", "control panel", "console"]
ERROR = ["error message", "warning banner", "alert", "error", "warning"]
TICKET = ["support ticket", "ticket", "case", "support request"]
WORKSPACE = ["workspace", "project space", "team space", "account space"]
API_KEY = ["api key", "access token", "secret key"]
SYNC_JOB = ["sync job", "import run", "background sync", "sync task"]
INTEGRATION = ["integration", "connector", "app connection"]

TEAMS = ["finance", "sales", "support", "operations", "analytics", "success"]
TIME_HINTS = [
    "this morning",
    "after yesterday's deploy",
    "during the weekend",
    "after the latest release",
    "right before billing closed",
    "during the overnight run",
]
ACTIONS = ["opened", "updated", "closed", "reopened", "flagged", "escalated"]
SEVERITY = ["minor", "odd", "urgent", "recurring", "noisy", "intermittent"]
VALUE_WORDS = ["twice", "again", "for the third time", "without warning", "all at once"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a concept-dense domain corpus for Loopy.")
    parser.add_argument("--output", default="loopy/domain_support_corpus.txt")
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def build_templates() -> list[str]:
    return [
        "The {customer} {action} a {ticket} because the {invoice} for the {subscription} showed the wrong amount {time_hint}.",
        "A {customer} said the {password_reset} expired before the {customer} could enter the {dashboard}.",
        "The {team} team noticed a {severity} {error} in the {dashboard} whenever the {sync_job} touched a large {workspace}.",
        "One {customer} asked for {refund} after the {integration} duplicated an {invoice} in the same {workspace}.",
        "The {ticket} says the {api_key} works in staging but fails in the production {workspace}.",
        "Several {customers} reported that the {sync_job} finished successfully even though the {dashboard} still showed an {error}.",
        "A new {customer} upgraded the {subscription}, then opened a {ticket} because the {invoice} still reflected the old tier.",
        "The {integration} created a {severity} {error} when a {customer} rotated an {api_key} {value_word}.",
        "Support asked whether the {refund} should be automatic when a {customer} cancels a {subscription} in the first day.",
        "An internal {ticket} mentions that the {password_reset} email lands late whenever the {sync_job} queue is full.",
        "The {dashboard} hides the {error} on mobile, so the {customer} only sees the broken {integration} after refresh.",
        "A {customer} in the {team} team reopened the {ticket} because the {invoice} and {refund} appeared together.",
        "The {workspace} imported correctly, but the {integration} left one {severity} {error} beside the {api_key} settings.",
        "Another {customer} said the {password_reset} flow loops back to the {dashboard} without changing the password.",
        "The {ticket} from {time_hint} links the {sync_job}, the {integration}, and the repeated {invoice} issue.",
        "Billing asked why the {refund} was approved before the {ticket} was reviewed by the {support_team} lead.",
        "The {customer} keeps switching {subscriptions} because the {dashboard} labels in the {workspace} are inconsistent.",
        "A {severity} {error} appears when the {integration} writes into a locked {workspace} with an expired {api_key}.",
        "The {support_team} queue saw three {tickets} where the {password_reset} link worked only after a second attempt.",
        "A long-running {sync_job} made the {dashboard} look frozen, and one {customer} assumed the {refund} had failed.",
    ]


def make_line(rng: random.Random, templates: list[str]) -> str:
    line = pick(rng, templates).format(
        customer=pick(rng, CUSTOMER),
        customers=pick(rng, ["customers", "users", "members", "account holders"]),
        subscription=pick(rng, SUBSCRIPTION),
        subscriptions=pick(rng, ["subscriptions", "plans", "memberships", "tiers"]),
        invoice=pick(rng, INVOICE),
        refund=pick(rng, REFUND),
        password_reset=pick(rng, PASSWORD_RESET),
        dashboard=pick(rng, DASHBOARD),
        error=pick(rng, ERROR),
        ticket=pick(rng, TICKET),
        tickets=pick(rng, ["support tickets", "tickets", "cases", "support requests"]),
        workspace=pick(rng, WORKSPACE),
        api_key=pick(rng, API_KEY),
        sync_job=pick(rng, SYNC_JOB),
        integration=pick(rng, INTEGRATION),
        team=pick(rng, TEAMS),
        support_team=pick(rng, ["support", "customer success", "operations"]),
        action=pick(rng, ACTIONS),
        severity=pick(rng, SEVERITY),
        time_hint=pick(rng, TIME_HINTS),
        value_word=pick(rng, VALUE_WORDS),
    )
    return " ".join(line.split())


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    templates = build_templates()
    lines = [make_line(rng, templates) for _ in range(args.samples)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines)} samples to {output_path}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
