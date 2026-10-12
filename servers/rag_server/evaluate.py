"""Before/after check for the RAG report: plain vector search vs. vector search + reranking.

    python -m servers.rag_server.evaluate

Each question has the doc page that should answer it. We report, per mode, how often that
page is ranked first (top-1) and appears in the top 3 (hit@3).
"""
from . import store

# (question, substring of the doc path that should answer it)
CASES = [
    ("How do I compute faithfulness of an answer?", "available_metrics/faithfulness"),
    ("What does context precision measure?", "available_metrics/context_precision"),
    ("What is context recall?", "available_metrics/context_recall"),
    ("Which metric checks whether the answer addresses the question?", "available_metrics/answer_relevance"),
    ("How is noise sensitivity calculated?", "available_metrics/noise_sensitivity"),
    ("How do I measure semantic similarity between answer and ground truth?", "available_metrics/semantic_similarity"),
    ("How do I generate a synthetic test set?", "testset"),
    ("How do I evaluate an agent's tool calls?", "agents"),
    ("How do I use Ragas with Langfuse?", "_langfuse"),
    ("How do I use Ragas with LangChain?", "langchain"),
]
K = 3


def main():
    top1 = {False: 0, True: 0}
    hit_k = {False: 0, True: 0}
    for q, want in CASES:
        print(f"\nQ: {q}   (expect ...{want}...)")
        for rerank in (False, True):
            srcs = [h["source"] for h in store.search(q, k=K, rerank=rerank)]
            t1 = want in srcs[0]
            hk = any(want in s for s in srcs)
            top1[rerank] += t1
            hit_k[rerank] += hk
            print(f"  {'rerank' if rerank else 'vector'}  top1={'Y' if t1 else 'n'} hit@{K}={'Y' if hk else 'n'}  {srcs[0]}")
    n = len(CASES)
    print(f"\n{'':14}top-1    hit@{K}")
    for rerank, name in ((False, "vector"), (True, "vector+rerank")):
        print(f"{name:14}{top1[rerank]}/{n}      {hit_k[rerank]}/{n}")


if __name__ == "__main__":
    main()
