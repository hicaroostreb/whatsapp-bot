import os
import shutil
import time
import subprocess
from decouple import config
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HUGGINGFACE_API_KEY"] = config("HUGGINGFACE_API_KEY")

if __name__ == "__main__":
    file_path = "/app/Context/rag-context-consorcio.md"

    # Carregar o arquivo completo
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Dividir em blocos de FAQ usando o separador "---"
    faq_blocks = content.split("---")
    print(f"Número total de blocos FAQ encontrados: {len(faq_blocks)}")

    # Criar documentos para cada bloco de FAQ
    documents = []
    for i, block in enumerate(faq_blocks):
        block = block.strip()
        if block:  # Ignorar blocos vazios
            # Extrair categoria e pergunta para os metadados
            categoria = ""
            pergunta = ""

            categoria_match = block.find("## Categoria:")
            if categoria_match != -1:
                categoria_end = block.find("\n", categoria_match)
                if categoria_end != -1:
                    categoria = (
                        block[categoria_match:categoria_end]
                        .replace("## Categoria:", "")
                        .strip()
                    )

            pergunta_match = block.find("### Pergunta:")
            if pergunta_match != -1:
                pergunta_end = block.find("\n", pergunta_match)
                if pergunta_end != -1:
                    pergunta = (
                        block[pergunta_match:pergunta_end]
                        .replace("### Pergunta:", "")
                        .strip()
                    )

            # Criar documento com metadados úteis
            documents.append(
                Document(
                    page_content=block,
                    metadata={
                        "source": file_path,
                        "index": i,
                        "categoria": categoria,
                        "pergunta": pergunta,
                    },
                )
            )

    print(f"Número de documentos FAQ processados: {len(documents)}")

    # Configurar embedding
    embedding = HuggingFaceEmbeddings(
        model_name="neuralmind/bert-base-portuguese-cased",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "truncate": True},
    )

    # Preparar diretório para Chroma DB
    persist_directory = "/app/chroma_data"

    # Tentar corrigir o diretório com comandos de sistema
    try:
        # Primeiro, tente remover completamente usando comando do sistema
        print(f"Tentando remover e recriar o diretório: {persist_directory}")
        os.system(f"rm -rf {persist_directory}")

        # Verificar se ainda existe
        if os.path.exists(persist_directory):
            # Tentar mudar permissões e depois remover
            os.system(f"chmod -R 777 {persist_directory}")
            os.system(f"rm -rf {persist_directory}")

        # Criar o diretório com permissões adequadas
        os.makedirs(persist_directory, exist_ok=True)
        os.system(f"chmod -R 777 {persist_directory}")
        print(f"Diretório {persist_directory} preparado com sucesso")

    except Exception as e:
        print(f"Erro ao preparar diretório original: {str(e)}")
        # Usar diretório alternativo se o original falhar
        alt_directory = "/tmp/chroma_data_" + str(int(time.time()))
        print(f"Usando diretório alternativo: {alt_directory}")
        os.makedirs(alt_directory, exist_ok=True)
        persist_directory = alt_directory

    print(f"Usando diretório de persistência: {persist_directory}")

    # Criar e popular vector store
    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

        # Adicionar documentos em lotes para melhor performance
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            vector_store.add_documents(documents=batch)
            print(
                f"Adicionado lote {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} ({len(batch)} documentos)"
            )

        # Verificar total de documentos na base
        coll_count = vector_store._collection.count()
        print(f"TOTAL de documentos na base vetorial: {coll_count}")

        if coll_count != len(documents):
            print("AVISO: Nem todos os documentos foram adicionados corretamente!")
        else:
            print(
                f"Todos os {coll_count} documentos foram adicionados com sucesso em: {persist_directory}"
            )

    except Exception as e:
        print(f"Erro ao processar a base vetorial: {str(e)}")
