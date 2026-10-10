"""
CRRA Lab C4 - Portfolio Orchestrator

    analysis -> policy_check -> [hitl] -> report

The agent may recommend; only a named human may approve where policy reserves the
decision. policy_check is pure Python (no model call), and every node writes to the
append-only audit trail.

Prerequisites (from the project root):
    python data/kb_setup.py                 # Lab C1 - policy KB
    python mcp_server/contract_shim.py      # Lab C2 - leave running in its own tab

Run:
    python orchestrator/supervisor.py
"""

import sys
from pathlib import Path

# Running `python orchestrator/supervisor.py` puts only orchestrator/ on the import
# path. Add the project root BEFORE any project-local import so guardrails/, agents/
# and data/ resolve from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.renewal_agent import _get_client, analyse_contract, tool_get_contract
from data.kb_setup import get_collection
from guardrails.audit_logger import AuditLogger

audit = AuditLogger()


# ==========================================================================
# State - one object passed through every node, each node adding its fields
# ==========================================================================
class ContractState(TypedDict):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str
    final_status: str


def initial_state(contract_id: str) -> ContractState:
    return {"contract_id": contract_id, "contract": {}, "recommendation": "",
            "confidence": "", "rationale": "", "policy_citation": "",
            "estimated_annual_impact_inr": 0, "hitl_required": False, "hitl_reason": "",
            "hitl_approved": False, "approver": "", "final_status": ""}


# ==========================================================================
# Node 1 - Analysis (the Lab C3 agent, as one node)
# ==========================================================================
def analysis_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    print(f"\n{'#' * 66}\nCONTRACT: {cid}\n{'#' * 66}\n\n> ANALYSIS AGENT")

    # Fetch first: policy_check needs the real contract fields, not the model's view of them.
    contract = tool_get_contract(cid)
    if "error" in contract:
        status = "ERROR_API_UNREACHABLE" if "unreachable" in contract["error"] else "ERROR_NOT_FOUND"
        print(f"  x {contract['error']}")
        audit.log("analysis", "contract_fetch_failed", cid, contract["error"], "N/A")
        return {"final_status": status}

    # Lab C3 agent: tool-calling loop (get_contract, search_policy, find_category_overlap,
    # submit_recommendation), claude-opus-5 at medium effort, capped at MAX_ROUNDS = 5.
    rec = analyse_contract(cid)
    recommendation = rec["recommendation"] or "NO_DECISION"

    audit.log("analysis", "recommendation", cid,
              f"{recommendation} / {rec['confidence']} - {rec['policy_citation'] or 'no citation'}",
              "N/A", recommendation=recommendation, confidence=rec["confidence"],
              policy_citation=rec["policy_citation"],
              estimated_annual_impact_inr=rec.get("estimated_annual_impact_inr") or 0,
              agent_flagged_approval=rec.get("human_approval_required"))

    return {"contract": contract, "recommendation": recommendation,
            "confidence": rec["confidence"], "rationale": rec["rationale"],
            "policy_citation": rec["policy_citation"],
            "estimated_annual_impact_inr": rec.get("estimated_annual_impact_inr") or 0}


# ==========================================================================
# Node 2 - Policy check (pure Python: deterministic, free, auditable)
# ==========================================================================
def approval_triggers(contract: dict, recommendation: str, confidence: str) -> list:
    """Evaluate EVERY trigger and return all that match - never stop at the first."""
    reasons = []
    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"approval band {band}: policy reserves this value for a named approver")
    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append(f"inside the notice window (deadline {contract.get('notice_deadline')} "
                       f"has passed): negotiating leverage is already lost")
    if recommendation == "TERMINATE":
        reasons.append("TERMINATE requires written business-owner confirmation")
    if confidence == "LOW":
        reasons.append("analysis confidence is LOW: the agent itself is unsure")
    if str(contract.get("owner", "")).strip().upper() in ("", "UNASSIGNED"):
        reasons.append("no business owner on record: nobody has confirmed the facts")

    # Lab C4 Step 6 - your sixth trigger goes here.
    # Lab C4 Step 6 - sixth trigger: auto-renewal with the notice deadline approaching.
    if contract.get("auto_renew") and contract.get("notice_state") == "APPROACHING":
        reasons.append(
            f"auto-renews unless notice is served by {contract.get('notice_deadline')} "
            f"({contract.get('days_to_notice_deadline')} days away): doing nothing locks in "
            f"another full term on the vendor's terms"
        )

    return reasons


def policy_check_node(state: ContractState) -> dict:
    print("\n> POLICY CHECK")
    if state.get("final_status", "").startswith("ERROR"):
        print(f"  Skipped: {state['final_status']}")
        audit.log("policy_check", "skipped", state["contract_id"], state["final_status"], "N/A")
        return {}

    reasons = approval_triggers(state["contract"], state["recommendation"], state["confidence"])
    hitl_required = bool(reasons)
    hitl_reason = "; ".join(reasons) if reasons else "no policy trigger"

    print(f"  HITL required: {hitl_required}")
    for r in reasons:
        print(f"    - {r}")
    audit.log("policy_check", "evaluate_triggers", state["contract_id"], hitl_reason,
              "PENDING" if hitl_required else "N/A", trigger_count=len(reasons))
    return {"hitl_required": hitl_required, "hitl_reason": hitl_reason}


# ==========================================================================
# Node 3 - Human approval gate (blocking)
# ==========================================================================
def _ask(prompt: str, valid: Optional[dict] = None) -> Optional[str]:
    """input() that survives a closed terminal and re-asks on a typo."""
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            return None
        if valid is None:
            return answer
        if answer.lower() in valid:
            return valid[answer.lower()]
        print("  Please type y or n.")


def hitl_node(state: ContractState) -> dict:
    c = state["contract"]
    print("\n> HUMAN APPROVAL GATE")
    print(f"  Contract : {state['contract_id']}  {c.get('vendor')}  (band {c.get('approval_band')}, "
          f"INR {c.get('annual_value_inr', 0):,}/yr)")
    print(f"  Proposed : {state['recommendation']}  (confidence {state['confidence']})")
    impact = state.get("estimated_annual_impact_inr") or 0
    if impact:
        print(f"  Impact   : INR {impact:,}/yr")
    print(f"  Policy   : {state['policy_citation'] or '-'}")
    print("  Approval required because:")
    for reason in state["hitl_reason"].split("; "):
        print(f"    - {reason}")
    print(f"\n  Rationale: {state['rationale']}")

    audit.log("hitl", "approval_request", state["contract_id"],
              f"{state['recommendation']} - {state['hitl_reason']}", "PENDING")

    decision = _ask("\n  Approve this recommendation? [y/n]: ",
                    {"y": True, "yes": True, "n": False, "no": False})
    if decision is None:
        # No one there to answer: never treat silence as approval.
        approved, approver = False, "no-response"
        print("  (no input received - treated as rejected)")
    else:
        approved = decision
        # A rejection is a decision with an owner too, so ask for the name either way.
        approver = _ask("  Your name: ") or "unnamed approver"

    status = "APPROVED" if approved else "REJECTED"
    audit.log("hitl", "approval_decision", state["contract_id"],
              f"{state['recommendation']} {status.lower()} by {approver}", status, actor=approver)
    print(f"  -> {status} by {approver}")
    return {"hitl_approved": approved, "approver": approver}


# ==========================================================================
# Node 4 - Report (runs on every path, so nothing exits silently)
# ==========================================================================
def report_node(state: ContractState) -> dict:
    print("\n> REPORT")
    cid, rec = state["contract_id"], state["recommendation"]

    if state.get("final_status", "").startswith("ERROR"):
        final, note, status = state["final_status"], "Not analysed; no action taken.", "N/A"
    elif not state["hitl_required"]:
        final, note, status = f"{rec}_AUTO", "Completed without human approval (no policy trigger).", "N/A"
    elif state["hitl_approved"] and rec == "NO_DECISION":
        final, note, status = "MANUAL_REVIEW", f"No recommendation; {state['approver']} routed it to manual review.", "APPROVED"
    elif state["hitl_approved"]:
        final, note, status = f"{rec}_APPROVED", f"Approved by {state['approver']}. Cleared to action.", "APPROVED"
    else:
        final, note, status = "ON_HOLD_REJECTED", f"Rejected by {state['approver']}. No commitment made to the vendor.", "REJECTED"

    print(f"  FINAL STATUS: {final}\n  {note}")
    audit.log("report", "final_status", cid, f"{final} - {note}", status,
              actor=state.get("approver") or "system", final_status=final)
    return {"final_status": final}


# ==========================================================================
# Graph
# ==========================================================================
def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)

    g.add_edge(START, "analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges("policy_check", route_after_policy_check,
                            {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def print_summary(outcomes: list) -> None:
    print(f"\n\n{'=' * 74}\nPORTFOLIO REVIEW COMPLETE\n{'=' * 74}")
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Gate':<8}{'Approver':<18}Final")
    print("-" * 74)
    for o in outcomes:
        gate = "HUMAN" if o.get("hitl_required") else "auto"
        print(f"{o['contract_id']:<11}{(o.get('recommendation') or '-'):<14}"
              f"{(o.get('confidence') or '-'):<8}{gate:<8}{(o.get('approver') or '-'):<18}"
              f"{o.get('final_status') or '-'}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    _get_client()      # fail fast on missing credentials
    get_collection()   # fail fast (or rebuild) if the policy KB is missing

    graph = build_graph()

    # CTR-1010 trips no trigger and should finish with no prompt - that contrast is the point.
    # CTR-1012: approve it (type y). CTR-1006: reject it (type n).
    # portfolio = ["CTR-1010", "CTR-1012", "CTR-1006"]
    portfolio = ["CTR-1010", "CTR-1012", "CTR-1006", "CTR-1005"]

    outcomes = [graph.invoke(initial_state(cid)) for cid in portfolio]
    print_summary(outcomes)
    audit.summary()


if __name__ == "__main__":
    main()