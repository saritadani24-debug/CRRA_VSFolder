"""
CRRA Lab C3 - Renewal Analysis Agent

Given a contract, recommends RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE with a
confidence level and a citation to the procurement policy section behind it.

Prerequisites (from the project root):
    python data/kb_setup.py                 # Lab C1 - builds the policy KB
    python mcp_server/contract_shim.py      # Lab C2 - leave running in its own tab

Run:
    python agents/renewal_agent.py
"""

import json
import os
import sys
from pathlib import Path

# Make project-local imports (data.kb_setup) work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic
import requests
from dotenv import load_dotenv

from data.kb_setup import get_collection, search_policy

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "claude-opus-5"
EFFORT = "medium"          # temperature is not supported on this model
MAX_TOKENS = 4096          # generous, so structured output is never cut off mid-field
MAX_ROUNDS = 5             # hard cap: a non-converging agent cannot loop forever
CONTRACT_API = "http://localhost:5001"
HTTP_TIMEOUT = 10

RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW"]


# ==========================================================================
# Response helpers
# ==========================================================================
def extract_text(response) -> str:
    """Return the first block that carries text.

    A thinking block may come first, so response.content[0].text is unsafe.
    """
    for block in response.content:
        if getattr(block, "text", None):
            return block.text.strip()
    return ""


# ==========================================================================
# Tools - the agent's ONLY route to data or to recording a decision
# ==========================================================================
TOOLS = [
    {
        "name": "get_contract",
        "description": (
            "Fetch one contract by ID with its real figures and derived fields: "
            "approval_band, notice_state, notice_deadline, days_to_renewal, "
            "utilisation_pct (null for contracts without seats), proposed_uplift_pct, owner."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy. Returns the best-matching section from "
            "each of the top 2 policy articles, with source file, section heading, "
            "confidence and full text. Low confidence means the policy may not cover it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Plain-English question, e.g. 'who approves a 60 lakh contract'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": (
            "List every contract in a category with vendor, annual value and utilisation. "
            "Use it to judge whether another vendor could absorb this capability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string", "enum": RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                "rationale": {
                    "type": "string",
                    "description": "Two or three sentences stating the specific figures that drove the call.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "Policy file and section, e.g. 'renegotiation_levers.md §Price uplift benchmarks'",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Rough annual rupee impact. Negative = saving. 0 if renewing on current terms.",
                },
                "human_approval_required": {
                    "type": "boolean",
                    "description": "True if policy reserves this decision for a named human approver.",
                },
            },
            "required": ["contract_id", "recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr",
                         "human_approval_required"],
        },
    },
]


def _api_get(path: str) -> dict:
    """GET from the contract API. Never raises: failures come back as {'error': ...}."""
    try:
        r = requests.get(f"{CONTRACT_API}{path}", timeout=HTTP_TIMEOUT)
    except requests.exceptions.RequestException:
        return {"error": "Contract API unreachable at localhost:5001. "
                         "Start it with: python mcp_server/contract_shim.py"}
    try:
        data = r.json()
    except ValueError:
        return {"error": f"Contract API returned a non-JSON response (HTTP {r.status_code})."}
    if r.status_code != 200:
        return {"error": data.get("error", f"HTTP {r.status_code}")}
    return data


def tool_get_contract(contract_id: str) -> dict:
    return _api_get(f"/api/contracts/{contract_id}")


def tool_search_policy(query: str) -> dict:
    try:
        results = search_policy(query, top_articles=2)
    except Exception as e:  # KB problems must not crash the agent
        return {"error": f"Policy search failed ({e}). Run: python data/kb_setup.py"}
    return {"results": [{"source": r["source"], "section": r["section"],
                         "citation": r["citation"], "confidence": r["confidence"],
                         "text": r["text"]} for r in results]}


def tool_find_category_overlap(category: str) -> dict:
    data = _api_get("/api/categories")
    if "error" in data:
        return data
    for entry in data["categories"]:
        if entry["category"].lower() == category.lower():
            return entry
    return {"category": category, "vendor_count": 0, "vendors": [],
            "note": "No contracts found in this category."}


TOOL_HANDLERS = {
    "get_contract": lambda a: tool_get_contract(a["contract_id"]),
    "search_policy": lambda a: tool_search_policy(a["query"]),
    "find_category_overlap": lambda a: tool_find_category_overlap(a["category"]),
}


# ==========================================================================
# System prompt
# ==========================================================================
SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps. For one vendor contract,
recommend exactly one of: RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE.

Method:
1. Call get_contract first. Use only the figures it returns; never assume or recompute them.
2. Call search_policy to find the governing rule (at most twice).
3. Call find_category_overlap only if you are considering CONSOLIDATE.
4. Call submit_recommendation exactly once to finish.

If get_contract returns an error (contract not found, API unreachable), do NOT submit a
recommendation. Reply in plain text explaining what went wrong. Never recommend on a
contract you have not seen.

What the four options mean:
- RENEW: healthy, fairly priced, still needed.
- RENEGOTIATE: still needed, but the terms are wrong (for example a large uplift or seats
  well above actual use).
- CONSOLIDATE: the capability is still needed, but another existing vendor can cover it.
- TERMINATE: nobody needs the capability at all. This is the strongest claim and requires
  written business-owner confirmation. Low utilisation alone is not grounds for it, and an
  owner of UNASSIGNED means nobody has been asked, not that the tool is unneeded.

utilisation_pct is null for contracts without seats (maintenance or support plans). Treat
that as "not applicable", not as zero usage.

Confidence:
- HIGH: the figures and the policy point the same way with no ambiguity.
- MEDIUM: sound, but rests on an assumption; name it in the rationale.
- LOW: genuinely unclear. LOW confidence is a valid and useful answer. Say so rather than
  inventing certainty, and do not keep calling tools hoping for a cleaner picture.

Set human_approval_required to true when ANY of these apply: approval band B or C; notice_state
INSIDE_WINDOW; recommendation TERMINATE; confidence LOW; owner is UNASSIGNED or missing.

policy_citation must name a file and section exactly as returned by search_policy, in the form
'file.md §Section heading'. If the policy does not address the question, say so in the rationale
and lower your confidence."""


# ==========================================================================
# Agent loop
# ==========================================================================
_client = None


def _get_client():
    global _client
    if _client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("No credentials: set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env.")
        _client = anthropic.Anthropic()   # SDK picks up whichever variable is set
    return _client

def _call_model(messages: list):
    kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                  tools=TOOLS, messages=messages)
    try:
        return _get_client().messages.create(output_config={"effort": EFFORT}, **kwargs)
    except TypeError:
        # Older SDKs don't know output_config as a keyword; send it in the raw body instead.
        return _get_client().messages.create(extra_body={"output_config": {"effort": EFFORT}}, **kwargs)


def _no_decision(contract_id: str, reason: str) -> dict:
    """An honest LOW-confidence result, used whenever the agent does not submit."""
    return {"contract_id": contract_id, "recommendation": None, "confidence": "LOW",
            "rationale": reason, "policy_citation": "", "estimated_annual_impact_inr": 0,
            "human_approval_required": True}


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _describe(name: str, args: dict, result: dict) -> str:
    if "error" in result:
        return f"  x {name}: {result['error']}"
    if name == "get_contract":
        util = result.get("utilisation_pct")
        util_s = f"{util}%" if util is not None else "n/a"
        return (f"  -> contract: band {result['approval_band']}  {result['notice_state']}  "
                f"util {util_s}  uplift {result['proposed_uplift_pct']}%  "
                f"INR {result['annual_value_inr']:,}  owner {result['owner']}")
    if name == "search_policy":
        tops = ", ".join(f"{r['citation']} ({r['confidence']:.0%})" for r in result["results"])
        return f'  -> policy "{args["query"][:44]}" -> {tops}'
    if name == "find_category_overlap":
        vendors = ", ".join(
            f"{v['vendor']} ({v['utilisation_pct']}%)" if v["utilisation_pct"] is not None
            else f"{v['vendor']} (n/a)" for v in result.get("vendors", []))
        return f"  -> overlap in {args['category']}: {vendors or 'none'}"
    return f"  -> {name}"


def analyse_contract(contract_id: str, verbose: bool = True) -> dict:
    """Run the ReAct loop for one contract. Always returns a result dict."""
    _log(verbose, f"\n{'=' * 62}\nANALYSING: {contract_id}\n{'=' * 62}")
    messages = [{"role": "user",
                 "content": f"Analyse contract {contract_id} and recommend an action."}]

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            response = _call_model(messages)
        except anthropic.APIError as e:
            _log(verbose, f"  x Model call failed: {e}")
            return _no_decision(contract_id, f"Model call failed: {e}")

        if response.stop_reason != "tool_use":
            text = extract_text(response) or f"stopped with reason '{response.stop_reason}'"
            _log(verbose, f"  Agent ended without a recommendation: {text[:300]}")
            return _no_decision(contract_id, f"No recommendation submitted. Agent said: {text}")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_recommendation":
                rec = dict(block.input)
                rec.setdefault("contract_id", contract_id)
                rec.setdefault("estimated_annual_impact_inr", 0)
                rec["rounds_used"] = round_no
                _print_recommendation(rec, verbose)
                return rec

            handler = TOOL_HANDLERS.get(block.name)
            result = handler(block.input) if handler else {"error": f"Unknown tool {block.name}"}
            _log(verbose, _describe(block.name, block.input, result))
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result), "is_error": "error" in result})

        messages.append({"role": "user", "content": tool_results})

    _log(verbose, f"  ! Stopped after {MAX_ROUNDS} rounds without a recommendation: treated as LOW confidence.")
    return _no_decision(contract_id, f"No recommendation within {MAX_ROUNDS} rounds.")


def _print_recommendation(rec: dict, verbose: bool) -> None:
    if not verbose:
        return
    print("\n  +-- RECOMMENDATION ---------------------------------")
    print(f"  | {rec['recommendation']}   confidence {rec['confidence']}")
    print(f"  | Policy: {rec['policy_citation']}")
    impact = rec.get("estimated_annual_impact_inr") or 0
    if impact:
        print(f"  | Impact: INR {impact:,}/yr")
    print(f"  | Human approval required: {rec['human_approval_required']}")
    print("  +--------------------------------------------------")
    print(f"  {rec['rationale']}")


# ==========================================================================
# Batch run + summary
# ==========================================================================
def print_summary(results: list) -> None:
    print(f"\n\n{'=' * 72}\nPORTFOLIO SUMMARY\n{'=' * 72}")
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Approval?':<11}{'Impact':<16}Citation")
    print("-" * 72)
    for r in results:
        impact = r.get("estimated_annual_impact_inr") or 0
        print(f"{r['contract_id']:<11}{(r['recommendation'] or 'NO DECISION'):<14}"
              f"{r['confidence']:<8}{str(r['human_approval_required']):<11}"
              f"{(f'INR {impact:,}' if impact else '-'):<16}{r['policy_citation'] or '-'}")
    print("-" * 72)
    needs_human = sum(1 for r in results if r["human_approval_required"])
    print(f"{len(results)} analysed · {needs_human} require human approval before action")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    _get_client()      # fail fast on a missing API key
    get_collection()   # fail fast (or rebuild) if the policy KB is missing

    # A deliberate spread: healthy renewal, high uplift, low utilisation with an overlap,
    # an orphaned contract, and a high-value contract inside its notice window.
    test_contracts = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]

    results = [analyse_contract(cid) for cid in test_contracts]
    print_summary(results)


if __name__ == "__main__":
    main()