import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from handle_user_input import handle_user_input
load_dotenv()


## Setting the pararmeters for API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf_api = os.getenv("HUGGINGFACEHUB_API_TOKEN")
pinecone_api = os.getenv("PINECONE_API_KEY")



# Title and Subheader
st.title("Personal File Assistant")

# File uploader for user to upload files
uploaded_files = st.file_uploader("Supported file types: TXT, CSV, PDF", accept_multiple_files=True)

if uploaded_files:
    docs = []
    st.markdown("---")
    st.markdown("### Chat with the Assistant")
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            if isinstance(uploaded_file, bytes):
                st.error("Uploaded file is not a valid file object.")
            else:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if uploaded_file.name.endswith(".pdf"):
                    pdf_loader = PyPDFLoader(file_path=file_path)
                    docs.extend(pdf_loader.load_and_split())
                elif uploaded_file.name.endswith(".csv"):
                    csv_loader = CSVLoader(file_path=file_path)
                    docs.extend(csv_loader.load())
                elif uploaded_file.name.endswith(".txt"):
                    text_loader = TextLoader(file_path=file_path)
                    docs.extend(text_loader.load())

        # Creating chunks of data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
        documents = text_splitter.split_documents(docs)  

        ## Vector Embedding And Vector Store
        embedding = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api, model_name="sentence-transformers/all-MiniLM-l6-v2")
        db = PineconeVectorStore.from_documents(documents, embedding, index_name="my-files")
        ## Load Model
        llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                task="text-generation",
                max_new_tokens=1000,
                do_sample=False,
                repetition_penalty=1.03,
        )

        ## Design ChatPrompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful Assistant. You will consider the provided context as well. <context> {context} </context>"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm, prompt)

        ## Retrieving chunks from the vector store
        retriever = db.as_retriever()
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Initialize chat history in Streamlit session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []



        user_input = st.text_input("Type your question here:", key="input", value="")

        if st.button("Send"):
            handle_user_input()
else:
    st.warning("⚠️ Please upload a file to proceed.")
