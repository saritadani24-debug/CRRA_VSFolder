"""CRRA Lab C2 - Mock Contract API. Safe, repeatable stand-in for the contract system.
Run from the project root: python mcp_server/contract_shim.py  ->  http://localhost:5001/health
"""

import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
PORT = 5001

# One fixed "today" so every run and every participant sees identical results.
SIMULATED_TODAY = date(2025, 4, 1)

BAND_A_LIMIT = 1_000_000   # below this: band A
BAND_B_LIMIT = 5_000_000   # up to and including this: band B; above: band C
APPROACHING_DAYS = 30
PATCHABLE = {"status": str, "owner": str, "proposed_uplift_pct": (int, float)}

app = Flask(__name__)
app.json.sort_keys = False  # keep fields in a readable order

CONTRACTS: dict[str, dict] = {}


def notice_state(deadline: date, renewal: date) -> str:
    if renewal < SIMULATED_TODAY:
        return "EXPIRED"
    if deadline <= SIMULATED_TODAY:
        return "INSIDE_WINDOW"          # deadline passed, renewal not yet: leverage lost
    if (deadline - SIMULATED_TODAY).days <= APPROACHING_DAYS:
        return "APPROACHING"
    return "OPEN"


def approval_band(value: int) -> str:
    if value < BAND_A_LIMIT:
        return "A"
    return "B" if value <= BAND_B_LIMIT else "C"


def enrich(row: dict) -> dict:
    """Convert types and add the derived fields so the agent never does maths."""
    for key in ("annual_value_inr", "notice_days", "seats_purchased", "seats_active", "proposed_uplift_pct"):
        row[key] = int(row[key])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

    renewal = date.fromisoformat(row["renewal_date"])
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["notice_state"] = notice_state(deadline, renewal)

    # AMC / support contracts have no seats: utilisation is undefined, not zero.
    seats = row["seats_purchased"]
    row["utilisation_pct"] = round(100 * row["seats_active"] / seats) if seats else None
    row["approval_band"] = approval_band(row["annual_value_inr"])
    return row


def load_contracts() -> None:
    CONTRACTS.clear()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            CONTRACTS[row["contract_id"].upper()] = enrich(row)


def not_found(contract_id: str):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    """Optional filters: ?category= ?band= ?notice_state= (case-insensitive)."""
    filters = {"category": "category", "band": "approval_band", "notice_state": "notice_state"}
    results = list(CONTRACTS.values())
    for param, field in filters.items():
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.lower()]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    """Contracts renewing within ?days= (default 90), soonest first."""
    try:
        window = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be a whole number"}), 400
    results = sorted((c for c in CONTRACTS.values()
                      if 0 <= c["days_to_renewal"] <= window),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = CONTRACTS.get(contract_id.upper())
    return jsonify(contract) if contract else not_found(contract_id)


@app.get("/api/categories")
def categories():
    """Vendors grouped by category - the starting point for overlap analysis."""
    grouped: dict[str, list] = {}
    for c in CONTRACTS.values():
        grouped.setdefault(c["category"], []).append({
            "contract_id": c["contract_id"], "vendor": c["vendor"],
            "annual_value_inr": c["annual_value_inr"],
            "utilisation_pct": c["utilisation_pct"]})
    summary = [{"category": cat, "vendor_count": len(v),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in v),
                "vendors": v} for cat, v in sorted(grouped.items())]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only: restarting the server resets every change."""
    contract = CONTRACTS.get(contract_id.upper())
    if not contract:
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    updates = {k: v for k, v in payload.items() if k in PATCHABLE}
    if not updates:
        return jsonify({"error": f"Send at least one of: {', '.join(PATCHABLE)}"}), 400
    for key, value in updates.items():
        if not isinstance(value, PATCHABLE[key]) or isinstance(value, bool):
            return jsonify({"error": f"Invalid type for {key}"}), 400
    contract.update(updates)
    return jsonify({"updated": sorted(updates), "contract": contract})


load_contracts()

if __name__ == "__main__":
    print(f"Mock Contract API: {len(CONTRACTS)} contracts from {CSV_PATH.name}, "
          f"simulated today {SIMULATED_TODAY}\n  -> http://localhost:{PORT}/health")
    app.run(port=PORT, debug=False)