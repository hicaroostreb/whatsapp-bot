from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:
    def __init__(self, persist_directory="/app/chroma_data"):
        self.persist_directory = persist_directory

    def build_retriever(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=hf,
        )

        return vector_store.as_retriever(search_kwargs={"k": 5})
