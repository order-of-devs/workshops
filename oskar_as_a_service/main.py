from rank_bm25 import BM25Plus

docs = [
    "php is the best language for flow",
    "i love poland",
    "Flow use php",
    "egz yyyy"
]

tokenizer = BM25Plus(docs)
print(tokenizer.get_scores("poland"))