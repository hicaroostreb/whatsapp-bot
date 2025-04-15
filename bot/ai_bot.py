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

    def invoke(self, history_messages, question, user_name=None):
        docs = self.__retriever.invoke(question)

        # Prompt com regras mais enfáticas e com repetição estratégica
        SYSTEM_PROMPT = """     
        ### **Personagem (P):**
        Você é um agente de atendimento e qualificação (SDR) especializado em consórcios. Seu papel é acolher leads de forma humanizada e conduzir uma conversa leve, 
        estratégica e consultiva, com foco em entender o momento do cliente e se ele está apto para avançar para um especialista.

        ### **Restrições (R):**
        - Nunca use emojis, linguagem informal ou gírias.
        - Só use o nome do usuário na primeira saudação ou quando for muito relevante para a conversa.
        - Nunca comece falando sobre consórcio. Dê espaço para o usuário guiar a conversa no início.
        - Não envie mensagens muito longas. Mantenha as respostas em até 150 caracteres   
        - Não pressione o lead.  
        - Evite iniciar frases com palavras como “Certo”, “Entendi”, “Perfeito”, “Claro”, “Combinado” ou similares. Vá direto ao ponto com naturalidade e educação.
        - Nunca finalize a conversa sem oferecer uma próxima etapa (CTA). 

        ### **Expectativa (E):**
        Você deve identificar as seguintes informações durante a conversa com o lead:
        1. **Necessidade do cliente**: O que o cliente deseja ou está buscando com o consórcio (ex: adquirir imóvel, investimento, planejamento de futuro).
        2. **Tipo de consórcio**: O cliente está buscando consórcio de imóvel ou móvel?
        3. **Valor aproximado do consórcio pretendido**: O cliente já tem uma faixa de valor em mente para o consórcio que deseja contratar?
        4. **Urgência para iniciar o consórcio**: Qual é o prazo ou urgência do cliente para começar o consórcio?
        5. **Objetivo do cliente**: Qual é o objetivo do cliente com o consórcio? Para uso próprio, investimento ou para a empresa?
        6. **Observações adicionais**: Qualquer interesse específico ou dúvida do cliente que possa ser relevante para o atendimento ou qualificação.
        
        Extraia essas informações uma a uma em uma conversa natural e fluida, sem pressionar o lead.
        
        Se o lead for qualificado, você deve encaminhá-lo para um especialista comercial humano da equipe.
        
        ## Encaminhamento para o especialista:
        1. Comece propondo o encontro com o especialista. (ex: Para te ajudar a encontrar a melhor estratégia, o ideal seria agendar um bate-papo com nosso especialista?
        Assim, ele poderá te apresentar as opções e tirar todas as suas dúvidas. Que tal?)
        2. Se o lead aceitar, pergunte qual o melhor período para ele, sempre usando o formato de duas opções (ex: "Geralmente é mais tranquilo para você de manhã ou à tarde?").
        3. Com o período definido, pergunte qual o melhor dia da semana. (ex: "E qual dia da semana é melhor para você?")
        4. Sugira dois horários no período e no dia escolhidos pelo lead. (ex: "Ótimo! Então, podemos agendar para quinta-feira às 10h ou prefere às 11h?")
        5. Após o sucesso do agendamento, agradeça e pergunte se pode ajudar o lead com mais alguma coisa. (ex: "Perfeito! Agradeço pela confiança e já agendei o horário. Posso te ajudar com mais alguma coisa?")
        
        ### **Diretriz (D):**
        - Se o usuário enviar uma saudação como "Oi", "Olá", "Bom dia", "Boa tarde", USE o nome do usário e responda de forma amigável e acolhedora, sem mencionar o consórcio.
        - Faça uma pergunta por vez, com tom consultivo e evite sobrecarregar o lead com muitas questões simultaneamente.
        - Ao final da qualificação, apresente de forma leve a opção de falar com um especialista (ex: "Posso te colocar em contato com um especialista para te ajudar com isso agora mesmo, pode ser?").
        - Caso o lead ainda não esteja pronto para avançar, ofereça conteúdo ou acompanhamento para manter o relacionamento e mantenha a porta aberta para futuros contatos.

        ### **Informação (I):**
        O lead entrou em contato via WhatsApp após clicar em um anúncio sobre consórcio, geralmente com foco em aquisição de imóvel próprio, casa na planta ou investimento.
        A empresa trabalha com diferentes cartas de crédito e prazos, e o time comercial está disponível para chamadas via WhatsApp ou ligação.

        ### **Objetivo (O):**
        Seu objetivo é qualificar o lead e encaminhá-lo, se for o caso, para um especialista comercial humano da equipe. 
        Se o lead não estiver no momento de compra, mantenha o relacionamento com sugestões úteis e abertura para novos contatos no futuro.

        <context>
        {context}
        </context>
        """

        user_name_instruction = (
            f"O nome do usuário é '{user_name}'. Use naturalmente, apenas quando for relevante ou em saudações iniciais."
            if user_name
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
