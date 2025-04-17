import os
import time
import shutil
from decouple import config
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HUGGINGFACE_API_KEY"] = config("HUGGINGFACE_API_KEY")


def extrair_metadados_e_limpar_bloco(block: str):
    """
    Extrai os campos do bloco de texto no novo formato e retorna
    o texto limpo para embedding e os metadados.
    """
    linhas = block.strip().split("\n")
    categoria = ""
    pergunta = ""
    texto_formatado = []

    for linha in linhas:
        linha = linha.strip()
        if linha.lower().startswith("categoria:"):
            categoria = linha.replace("Categoria:", "").strip()
        elif linha.lower().startswith("pergunta:"):
            pergunta = linha.replace("Pergunta:", "").strip()
        texto_formatado.append(linha)

    return " ".join(texto_formatado), categoria, pergunta


def limpar_diretorio(diretorio):
    """
    Limpa o conteúdo de um diretório sem remover o diretório em si.
    Evita erros de "resource busy".
    """
    if os.path.exists(diretorio):
        print(f"🧹 Limpando conteúdo de: {diretorio}")
        for item in os.listdir(diretorio):
            item_path = os.path.join(diretorio, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"⚠️ Erro ao deletar {item_path}: {e}")


if __name__ == "__main__":
    file_path = "/app/Context/rag-context-consorcio-v2.md"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    faq_blocks = content.split("---")
    print(f"🔍 Total de blocos FAQ encontrados: {len(faq_blocks)}")

    documents = []
    for i, block in enumerate(faq_blocks):
        block = block.strip()
        if block:
            page_content, categoria, pergunta = extrair_metadados_e_limpar_bloco(block)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": file_path,
                        "index": i,
                        "categoria": categoria,
                        "pergunta": pergunta,
                    },
                )
            )

    print(f"✅ Total de documentos processados: {len(documents)}")

    # Embedding config
    embedding = HuggingFaceEmbeddings(
        model_name="neuralmind/bert-base-portuguese-cased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "truncate": True},
    )

    persist_directory = "/app/chroma_data"

    try:
        if os.path.exists(persist_directory):
            limpar_diretorio(persist_directory)
        else:
            os.makedirs(persist_directory, exist_ok=True)

        print(f"📂 Diretório de persistência pronto: {persist_directory}")

    except Exception as e:
        alt_directory = f"/tmp/chroma_data_{int(time.time())}"
        print(f"⚠️ Erro ao preparar o diretório. Usando alternativa: {alt_directory}")
        os.makedirs(alt_directory, exist_ok=True)
        persist_directory = alt_directory

    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(documents=batch)
            print(f"📦 Lote {i // batch_size + 1} adicionado ({len(batch)} documentos)")

        coll_count = vector_store._collection.count()
        print(f"📊 Total na base vetorial: {coll_count}")

        if coll_count != len(documents):
            print("⚠️ AVISO: Nem todos os documentos foram indexados corretamente!")
        else:
            print(f"✅ Base vetorial criada com sucesso em: {persist_directory}")

    except Exception as e:
        print(f"❌ Erro ao processar a base vetorial: {str(e)}")
