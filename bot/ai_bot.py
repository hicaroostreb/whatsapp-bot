import os
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

os.environ["GEMINI_API_KEY"] = config("GEMINI_API_KEY")

# Garantir que a pasta 'bot-cache' exista dentro da pasta 'bot'
cache_dir = os.path.join(os.getcwd(), "bot", "bot-cache")
os.makedirs(cache_dir, exist_ok=True)

# Ativa cache com caminho específico dentro da pasta 'bot-cache'
set_llm_cache(SQLiteCache(database_path=os.path.join(cache_dir, "langchain.db")))


class AIBot:
    def __init__(self):
        self.__chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",  # Nome do modelo. Flash é mais rápido e barato, ideal para apps em tempo real.
            temperature=0,  # Controla a criatividade. 0 = mais preciso, menos inventivo. Ideal para seguir regras rígidas.
            max_output_tokens=200,  # Limita o tamanho da resposta. Evita textos longos no WhatsApp e reduz custo.
            timeout=15,  # Tempo máximo (em segundos) para a IA responder. Garante fluidez da experiência.
            max_retries=2,  # Número de tentativas em caso de erro/transmissão falha. 2 é padrão e seguro.
            cache=True,  # Ativa cache para reutilizar respostas. Ajuda na performance e custo em produção.
            verbose=False,  # Se True, imprime a resposta no terminal. Útil só para debug.
            safety_settings={  # Define bloqueios de conteúdo inadequado. Essencial para manter respostas seguras.
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },
        )

        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = "/app/chroma_data"

        # Definindo o nome do modelo e os parâmetros necessários
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        # Inicializando o HuggingFaceEmbeddings com os parâmetros
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # Criando o armazenamento do vetor (vector store)
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=hf,
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 5},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get("fromMe") else AIMessage
            messages.append(message_class(content=message.get("body")))
        messages.append(HumanMessage(content=question))
        return messages

    def __should_use_name(self, user_name, history_messages, current_question):
        if not user_name:
            return False

        greetings = {
            "oi",
            "olá",
            "ola",
            "bom dia",
            "boa tarde",
            "boa noite",
            "e aí",
            "opa",
            "salve",
        }
        normalized_question = current_question.lower().strip()

        # 1. If it's the first message (empty history), use the name
        if not history_messages:
            return True

        # 2. If the message is a greeting
        if normalized_question in greetings:
            return True

        # 3. Count how many times the name has already been used
        name_count = sum(
            user_name.lower() in msg.get("body", "").lower() for msg in history_messages
        )
        return name_count < 2

    def invoke(self, history_messages, question, user_name=None):
        docs = self.__retriever.invoke(question)

        # Prompt com regras mais enfáticas e com repetição estratégica
        SYSTEM_PROMPT = """
        DIRETRIZES CRÍTICAS (SEGUIR RIGOROSAMENTE):
        1. NÃO USE EMOJIS EM HIPÓTESE ALGUMA - esta é uma regra inflexível.
        2. NÃO MENCIONE O NOME DO USUÁRIO, exceto na primeira saudação ou quando for muito relevante para a conversa.
        3. Use linguagem formal mas amigável, sem gírias.
        4. SE O USUÁRIO DESVIAR PARA TÓPICOS NÃO RELACIONADOS AO CONSÓRCIO DE IMÓVEIS (ex: promoções de iPhones, produtos de consumo, outros temas fora do mercado financeiro/imobiliário), 
        DESVIE EDUCADAMENTE A CONVERSA DE VOLTA PARA O CONSÓRCIO IMOBILIÁRIO.

                
        Você é um assistente virtual que tem como missão entender, com sutileza e no ritmo do usuário, se ele tem interesse em consórcio imobiliário, sem nunca forçar ou acelerar a conversa.

        Objetivo da conversa:
        1. Criar conexão com o usuário, de forma leve e acolhedora.
        2. Entender sua necessidade ou objetivo (ex: imóvel, projeto, investimento).
        3. Descobrir se conhece consórcio ou já teve alguma experiência.
        4. Investigar com cuidado se está buscando algo a longo prazo ou mais imediato.
        5. Descobrir se já tem valor definido ou fez pesquisas em outros lugares.
        6. Entender se está apenas pesquisando ou quer resolver logo.

        Diretrizes:
        - Nunca comece falando sobre consórcio. Dê espaço para o usuário guiar a conversa no início.
        - Inicie com mensagens curtas, amigáveis e abertas. Ex: "Olá! Tudo certo por aí?" ou "Bom te ver por aqui".
        - Vá puxando assunto com naturalidade e adaptando o tom com base nas respostas do usuário.
        - Faça perguntas suaves que ajudem a entender o que ele busca, sem parecer um interrogatório.
        - Mantenha as respostas curtas (até 200 caracteres), a menos que o contexto exija algo mais.
        - Use linguagem simples, humana, sem termos técnicos ou frases formais.
        - Quando o usuário demonstrar interesse claro por valores, contemplação ou contratação, pergunte o melhor dia e horário para um especialista entrar em contato.

        Lembre-se: sua função é criar uma experiência fluida e acolhedora, para que o usuário confie em você e compartilhe suas necessidades no tempo dele.
        
        <context>
        {context}
        </context>
        """

        use_name = self.__should_use_name(user_name, history_messages, question)

        user_name_instruction = (
            f"O nome do usuário é '{user_name}'. Use naturalmente, apenas quando for relevante ou saudações."
            if use_name
            else ""
        )

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", user_name_instruction),
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(
            self.__chat, question_answering_prompt
        )

        response = document_chain.invoke(
            {
                "context": docs,
                "messages": self.__build_messages(history_messages, question),
            }
        )

        return response
