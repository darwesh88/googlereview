from __future__ import annotations

import argparse
import random
from pathlib import Path


CUSTOMER_COVERED = ["customer", "user", "member", "account holder"]
CUSTOMER_UNCOVERED = ["client", "subscriber", "buyer"]
SUBSCRIPTION_COVERED = ["subscription", "plan", "membership", "tier"]
SUBSCRIPTION_UNCOVERED = ["package", "bundle"]
INVOICE_COVERED = ["invoice", "bill", "billing statement"]
INVOICE_UNCOVERED = ["charge note", "receipt page"]
REFUND_COVERED = ["refund", "credit", "reimbursement", "money back"]
REFUND_UNCOVERED = ["repayment", "payback"]
PASSWORD_RESET_COVERED = ["password reset", "reset link", "login reset", "sign-in reset"]
PASSWORD_RESET_UNCOVERED = ["password help email", "recovery step"]
DASHBOARD_COVERED = ["dashboard", "admin panel", "control panel", "console"]
DASHBOARD_UNCOVERED = ["home screen", "ops view"]
ERROR_COVERED = ["error message", "warning banner", "alert", "error", "warning"]
ERROR_UNCOVERED = ["failure note", "red notice"]
TICKET_COVERED = ["support ticket", "ticket", "case", "support request"]
TICKET_UNCOVERED = ["help thread", "issue log"]
WORKSPACE_COVERED = ["workspace", "project space", "team space", "account space"]
WORKSPACE_UNCOVERED = ["instance", "shared room"]
API_KEY_COVERED = ["api key", "access token", "secret key"]
API_KEY_UNCOVERED = ["service credential", "private token"]
SYNC_JOB_COVERED = ["sync job", "import run", "background sync", "sync task"]
SYNC_JOB_UNCOVERED = ["data sweep", "transfer pass"]
INTEGRATION_COVERED = ["integration", "connector", "app connection"]
INTEGRATION_UNCOVERED = ["bridge app", "linked service"]

NAMES = ["Maya", "Jon", "Noor", "Elia", "Priya", "Sam", "Lina", "Omar"]
TEAMS = ["finance", "sales", "support", "operations", "analytics", "success"]
TIMES = [
    "this morning",
    "after yesterday's deploy",
    "during the weekend",
    "after the latest release",
    "right before billing closed",
    "during the overnight run",
]
TAILS = [
    "and nobody noticed for an hour.",
    "which made the report look worse than it was.",
    "before the on-call lead stepped in.",
    "while the customer was still on the page.",
    "so the team marked it for follow-up.",
    "and the second attempt worked.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a harder, partial-coverage support corpus.")
    parser.add_argument("--output", default="loopy/noisy_support_corpus.txt")
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--uncovered-ratio", type=float, default=0.3)
    return parser.parse_args()


def choose_variant(rng: random.Random, covered: list[str], uncovered: list[str], uncovered_ratio: float) -> str:
    if uncovered and rng.random() < uncovered_ratio:
        return rng.choice(uncovered)
    return rng.choice(covered)


def build_templates() -> list[str]:
    return [
        "{name} said the {password_reset} expired before the {customer} could reach the {dashboard}, {tail}",
        "The {customer} opened a {ticket} because the {invoice} for the {subscription} looked wrong {time_hint}.",
        "A {team} manager said the {sync_job} finished, but the {dashboard} still showed a {error}.",
        "Support approved a {refund} after the {integration} duplicated an {invoice} inside the same {workspace}.",
        "The {ticket} says the {api_key} works in staging but breaks in the production {workspace}.",
        "{name} from {team} reopened the {ticket} because the {refund} never appeared on the {dashboard}.",
        "The {integration} raised a {error} when the {customer} rotated an {api_key}, {tail}",
        "One {customer} changed {subscription} twice in one day and the {invoice} still reflected the old tier.",
        "The {workspace} imported correctly, but one {error} stayed beside the {api_key} settings.",
        "A long {sync_job} made the {dashboard} look frozen, so the {customer} thought the {refund} had failed.",
        "During review, {name} linked the {ticket}, the {integration}, and the repeated {invoice} issue.",
        "The {support_team} queue saw three {ticket_plural} where the {password_reset} only worked after a second try.",
        "A locked {workspace} plus an expired {api_key} triggered another {error} in the {integration}.",
        "The {customer} said the {dashboard} looked fine on desktop but the mobile page hid the {error}.",
        "{name} noticed the {invoice} and {refund} appeared together after the {sync_job} retried the same record.",
    ]


def make_line(rng: random.Random, templates: list[str], uncovered_ratio: float) -> str:
    line = rng.choice(templates).format(
        name=rng.choice(NAMES),
        team=rng.choice(TEAMS),
        support_team=rng.choice(["support", "customer success", "operations"]),
        time_hint=rng.choice(TIMES),
        tail=rng.choice(TAILS),
        customer=choose_variant(rng, CUSTOMER_COVERED, CUSTOMER_UNCOVERED, uncovered_ratio),
        subscription=choose_variant(rng, SUBSCRIPTION_COVERED, SUBSCRIPTION_UNCOVERED, uncovered_ratio),
        invoice=choose_variant(rng, INVOICE_COVERED, INVOICE_UNCOVERED, uncovered_ratio),
        refund=choose_variant(rng, REFUND_COVERED, REFUND_UNCOVERED, uncovered_ratio),
        password_reset=choose_variant(rng, PASSWORD_RESET_COVERED, PASSWORD_RESET_UNCOVERED, uncovered_ratio),
        dashboard=choose_variant(rng, DASHBOARD_COVERED, DASHBOARD_UNCOVERED, uncovered_ratio),
        error=choose_variant(rng, ERROR_COVERED, ERROR_UNCOVERED, uncovered_ratio),
        ticket=choose_variant(rng, TICKET_COVERED, TICKET_UNCOVERED, uncovered_ratio),
        ticket_plural=rng.choice(["tickets", "cases", "support tickets", "help threads"]),
        workspace=choose_variant(rng, WORKSPACE_COVERED, WORKSPACE_UNCOVERED, uncovered_ratio),
        api_key=choose_variant(rng, API_KEY_COVERED, API_KEY_UNCOVERED, uncovered_ratio),
        sync_job=choose_variant(rng, SYNC_JOB_COVERED, SYNC_JOB_UNCOVERED, uncovered_ratio),
        integration=choose_variant(rng, INTEGRATION_COVERED, INTEGRATION_UNCOVERED, uncovered_ratio),
    )
    return " ".join(line.split())


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    templates = build_templates()
    lines = [make_line(rng, templates, args.uncovered_ratio) for _ in range(args.samples)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines)} samples to {output_path}")
    print(f"Seed: {args.seed}")
    print(f"Uncovered ratio: {args.uncovered_ratio}")


if __name__ == "__main__":
    main()
