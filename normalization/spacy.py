import spacy
from langchain_core.documents import Document

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def normalize_text(text: str) -> str:
    doc = nlp(text)
    
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    
    return " ".join(tokens)


def normalize_documents(documents: list[Document]) -> list[Document]:
    normalized_docs = []

    for doc in documents:
        cleaned_text = normalize_text(doc.page_content)

        normalized_docs.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata
            )
        )
    return normalized_docs

