"""
CRRA Lab C4 - Audit Trail

Records every step of the review: what was recommended, on what evidence, and who
decided. The log is APPEND-ONLY by design: re-running the orchestrator adds to the
file and never replaces it, because a record that a re-run can overwrite cannot
serve as evidence. For a clean demo, delete logs/audit_trail.jsonl by hand.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # One id per orchestrator run, so entries from different runs can be told apart.
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.entries: list = []

    def log(self, node: str, action: str, contract_id: str, detail: str = "",
            approval_status: str = "N/A", actor: str = "system", **extra) -> dict:
        """Append one JSON object as one line, then echo a short form to the console."""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "node": node,
            "action": action,
            "contract_id": contract_id,
            "detail": detail,
            "approval_status": approval_status,
            "actor": actor,
            **extra,
        }
        # "a" = append mode. Never "w": that would wipe the history on every run.
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.entries.append(entry)

        short = detail if len(detail) <= 70 else detail[:67] + "..."
        print(f"  [AUDIT] {node}: {action} ({approval_status}, {actor}) - {short}")
        return entry

    def summary(self) -> None:
        print(f"\n{'=' * 74}\nAUDIT TRAIL - {len(self.entries)} entries this run  (run {self.run_id})\n{'=' * 74}")
        print(f"{'Time':<10}{'Contract':<11}{'Node':<15}{'Action':<20}{'Status':<10}Actor")
        print("-" * 74)
        for e in self.entries:
            print(f"{e['timestamp'][11:]:<10}{e['contract_id']:<11}{e['node']:<15}"
                  f"{e['action'][:19]:<20}{e['approval_status']:<10}{e['actor']}")
        print("-" * 74)
        print(f"Appended to: {self.log_path}")