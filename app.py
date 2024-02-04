import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv


load_dotenv()


def get_vector_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
  
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store



# get retriever chain
def get_context_retriever_chain(vector_store):
    #model    
    llm = ChatOpenAI(model='gpt-3.5-turbo')
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above converstion, generate a search query to look up to get informtion relavant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
        
    
# get the rag chain
def get_converssational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model='gpt-3.5-turbo')
    
    prompt = ChatPromptTemplate.from_messages([
       ("system", "Answer the user's questions based on the below context: \n\n{context}"),
       MessagesPlaceholder(variable_name="chat_history"),
       ("user", "{input}"),
    ])
    
    document_rag_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, document_rag_chain)
    

# get the response from model
def get_response(user_query):
    
    retrieved_chain = get_context_retriever_chain(vector_db)
    conversational_rag_chain = get_converssational_rag_chain(retrieved_chain)
    
    response = conversational_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response["answer"]

# config
st.set_page_config(page_title="Chat-with-web App", page_icon=":shark:", layout="wide")
st.title("Chat with Webs")

#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    # disable the chat input when there is no website url
if not website_url:
    st.info("Please enter a website URL to start chatting")
else:
    # session_state is gloal state in streamlit to keep the state consisitent with reloading the app
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content= "Hello, How can I help you?")
    ]
        
    # Create conversation chain
    vector_db = get_vector_from_url(website_url)
    
    retrieved_chain = get_context_retriever_chain(vector_db)
         
    # user_query 
    user_query = st.chat_input("Type a message...")
    # if user_query is not None and user_query != "":
    if user_query:

        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))   
        st.session_state.chat_history.append(AIMessage(content=response))
    # conversation
    for mesage in st.session_state.chat_history:
        if isinstance(mesage, AIMessage):
            with st.chat_message("AI Bot 🤖"):
                st.write(mesage.content)
        elif isinstance(mesage, HumanMessage):
            with st.chat_message("You"):
                st.write(mesage.content)
        else:
            st.write("Unknown message type")
 





