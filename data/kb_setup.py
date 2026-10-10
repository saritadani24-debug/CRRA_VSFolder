"""
CRRA Lab C1 - Policy Knowledge Base
===================================

Loads the procurement policy articles in data/kb/ into a ChromaDB collection,
split at section level (## headings), so that later labs can retrieve the
exact rule that applies to a question and cite it by file and section.

Run from the project root:

    python data/kb_setup.py

Importable by later labs:

    from data.kb_setup import build_kb, get_collection, search_policy
"""

import sys
from pathlib import Path

import chromadb

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
KB_DIR = DATA_DIR / "kb"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "crra_policy"

# Cosine distance keeps scores in a predictable range, so (1 - distance)
# reads naturally as a confidence figure.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_article(path: Path) -> list[dict]:
    """
    Split one policy article into section-level chunks.

    - Walks the file line by line, watching for lines starting with '## '.
    - Each '## ' line closes the previous section and opens a new one.
    - The '# ' document title is skipped: it is a label, not a rule.
    - Each chunk keeps its heading so results can cite a section by name.
    """
    chunks = []
    heading = None
    body: list[str] = []

    def close_section():
        text = "\n".join(body).strip()
        if text:
            chunks.append({
                "source": path.name,
                "section": heading or "(no section)",
                # Heading is prepended so it contributes to the embedding.
                "text": f"{heading}\n\n{text}" if heading else text,
            })

    # utf-8 explicitly: the policy contains the rupee sign, and the Windows
    # default encoding would otherwise fail to read it.
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            close_section()
            heading = line[3:].strip()
            body = []
        elif line.startswith("# "):
            continue
        else:
            body.append(line)

    close_section()
    return chunks


# --------------------------------------------------------------------------
# ChromaDB access
# --------------------------------------------------------------------------
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_kb(verbose: bool = True):
    """Delete and rebuild the policy collection. Returns the collection."""
    client = _client()

    # Delete first so re-runs never duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

    files = sorted(KB_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy articles found in {KB_DIR}")

    if verbose:
        print("Loading policy articles")
        print("=" * 55)

    ids, documents, metadatas = [], [], []
    for path in files:
        chunks = chunk_article(path)
        for i, chunk in enumerate(chunks):
            ids.append(f"{path.stem}::{i:02d}")
            documents.append(chunk["text"])
            metadatas.append({"source": chunk["source"],
                              "section": chunk["section"]})
        if verbose:
            print(f"  {path.name:<32} {len(chunks)} chunks")

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    if verbose:
        print("=" * 55)
        print(f"  TOTAL: {len(ids)} chunks from {len(files)} articles\n")

    return collection


def get_collection(rebuild_if_missing: bool = True):
    """Return the policy collection, building it if it does not exist yet."""
    client = _client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    if not rebuild_if_missing:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run: python data/kb_setup.py"
        )
    return build_kb(verbose=False)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def search_policy(query: str, top_articles: int = 2, n_candidates: int = 8,
                  collection=None) -> list[dict]:
    """
    Search the policy KB and return whole, citable sections.

    Raw hits are grouped by source article and only the best section per
    article is kept, for the top `top_articles` articles. This gives the
    agent coherent context instead of fragments from five different files.

    Each result: {source, section, citation, confidence, text}
    """
    collection = collection or get_collection()
    n = min(n_candidates, collection.count())
    res = collection.query(query_texts=[query], n_results=n)

    best_per_article: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0],
                               res["metadatas"][0],
                               res["distances"][0]):
        confidence = round(1 - dist, 3)
        source = meta["source"]
        current = best_per_article.get(source)
        if current is None or confidence > current["confidence"]:
            best_per_article[source] = {
                "source": source,
                "section": meta["section"],
                "citation": f"{source} §{meta['section']}",
                "confidence": confidence,
                "text": doc,
            }

    ranked = sorted(best_per_article.values(),
                    key=lambda r: r["confidence"], reverse=True)
    return ranked[:top_articles]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def run_test_queries(collection, queries: list[str]):
    print("Testing retrieval")
    print("=" * 55)
    for q in queries:
        res = collection.query(query_texts=[q], n_results=1)
        meta = res["metadatas"][0][0]
        confidence = 1 - res["distances"][0][0]
        label = (q[:45] + "...") if len(q) > 45 else q
        print(f'  "{label}"')
        print(f"      -> {meta['source']}  §{meta['section']}"
              f"   confidence {confidence:.0%}")
    print("=" * 55)


if __name__ == "__main__":
    # Keep console output safe on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    collection = build_kb()

    test_queries = [
        "who approves a contract worth 60 lakh rupees",
        "contract auto renews next month and we missed the notice deadline",
        "vendor wants a 20 percent price increase at renewal",
        "two tools do the same job and one has 29% licence usage",
        "what do we need before terminating a vendor contract",
        # Step 5: add two of your own questions below, then re-run.
        # Try one the policy does NOT cover, and one spanning two articles.
    ]

    run_test_queries(collection, test_queries)
