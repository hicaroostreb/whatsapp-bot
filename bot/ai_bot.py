import os
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["GEMINI_API_KEY"] = config("GEMINI_API_KEY")


class AIBot:
    def __init__(self):
        self.__chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = "/app/chroma_data"
        embedding = HuggingFaceEmbeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
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

    def invoke(self, history_messages, question):
        docs = self.__retriever.invoke(question)
        SYSTEM_TEMPLATE = """
            Responda às perguntas dos usuários de forma clara, direta e amigável, com base no contexto abaixo.
            
            Você é um assistente virtual especializado em consórcios, com foco em ajudar o usuário a entender e qualificar para a compra de imóveis. Seu objetivo é esclarecer dúvidas sobre o processo de consórcio e fornecer informações úteis, sem sobrecarregar o usuário.
            
            - Sempre busque responder de forma concisa e objetiva, sem se alongar demais em detalhes, a menos que o usuário peça mais informações.
            - Mantenha um tom acolhedor e humano, como se fosse uma conversa entre amigos. Evite respostas robotizadas ou excessivamente formais.
            - Baseie suas respostas nas informações já fornecidas pelo usuário e ajuste sua resposta conforme o histórico da conversa.
            - Se o usuário fizer perguntas específicas, como sobre o valor de um consórcio, forneça uma resposta direta e, se necessário, peça mais informações para personalizar a resposta.
            - Ao explicar opções ou processos, use uma linguagem simples e acessível, sem jargões ou termos complicados.
            
            Lembre-se de que sua função é tornar o processo de compra de consórcio mais claro e amigável, ajudando o usuário a tomar decisões informadas e avançar no processo de compra, sem pressão.
            
            <context>
            {context}
            </context>
        """

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
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
