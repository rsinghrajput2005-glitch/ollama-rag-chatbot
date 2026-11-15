def retrieve_docs(question, vectorstore):
    retrieve_doc = vectorstore.similarity_search_with_relevance_scores(
    question,
    k=5
    )

    filtered_docs = [
        doc for doc, score in retrieve_doc
        if score >= 0.4
    ]

    return filtered_docs