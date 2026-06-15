
from langchain_experimental.text_splitter import SemanticChunker


def semantic_chunk_documents(tokens, embeddings):
    """
    Perform semantic chunking based on embedding similarity.
    Splits document at topic boundaries instead of fixed characters.
    """

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="gradient",   # Detects topic shifts
        buffer_size=1,                          # Adds neighboring sentences
        min_chunk_size=200                      # Avoids very small chunks
    )

    chunks = splitter.split_documents(tokens)

    return chunks










# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import spacy
# from langchain_core.documents import Document

# # Only for sentence splitting (light use)
# nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])


# def semantic_chunk_documents(documents, embeddings, buffer_size=1, min_chunk_size=200):
#     chunks = []

#     for doc in documents:
#         sentences = [s.text.strip() for s in nlp(doc.page_content).sents]

#         if len(sentences) <= 1:
#             chunks.append(doc)
#             continue

#         embs = embeddings.embed_documents(sentences)

#         sims = [
#             cosine_similarity([embs[i-1]], [embs[i]])[0][0]
#             for i in range(1, len(embs))
#         ]

#         grads = np.diff(sims)
#         threshold = np.mean(grads) - np.std(grads)

#         start = 0
#         for i, g in enumerate(grads):
#             if g < threshold:
#                 end = min(len(sentences), i+1 + buffer_size)
#                 text = " ".join(sentences[start:end])

#                 if len(text) >= min_chunk_size:
#                     chunks.append(Document(page_content=text, metadata=doc.metadata))

#                 start = i+1

#         # last chunk
#         if start < len(sentences):
#             text = " ".join(sentences[start:])
#             if len(text) >= min_chunk_size:
#                 chunks.append(Document(page_content=text, metadata=doc.metadata))

#     return chunks





