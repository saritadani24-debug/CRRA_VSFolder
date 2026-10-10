# CRRA - Contract Renewal Risk Advisor

Zensar BizOps AI Capstone. Built across four labs (C1-C4).

## Status
- [x] C1 - Policy knowledge base (`data/kb/`, `data/kb_setup.py`)
- [ ] C2 - Mock contract API
- [ ] C3 - Renewal analysis agent
- [ ] C4 - Orchestration, HITL and audit

## Setup (Windows)
```
python -m venv labenv
labenv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Lab C1
```
python data/kb_setup.py
```
Expect 19 chunks from 5 articles, then five test queries each showing
source file, section and confidence. The vector store is written to
`data/chroma_db/` and rebuilt on every run.

Later labs import from it:
```python
from data.kb_setup import get_collection, search_policy
search_policy("who approves a 60 lakh contract")
# -> best section from each of the top 2 articles, with citation + confidence
```
