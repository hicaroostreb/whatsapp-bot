from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from src.chatbot.core.bot_cache.cache import CacheManager
from src.chatbot.core.retriever import Retriever
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from decouple import config


class AIBot:
    def __init__(self):
        CacheManager.initialize_cache()  # Inicializa o cache (agora isolado)

        self.__chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0,
            max_output_tokens=200,
            timeout=15,
            max_retries=2,
            cache=True,
            verbose=False,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            },
        )

        self.__retriever = Retriever().build_retriever()

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

        if not history_messages:
            return True

        if normalized_question in greetings:
            return True

        name_count = sum(
            user_name.lower() in msg.get("body", "").lower() for msg in history_messages
        )
        return name_count < 2

    def invoke(self, history_messages, question, user_name=None):
        docs = self.__retriever.invoke(question)

        SYSTEM_PROMPT = """[suas diretrizes aqui]"""

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
