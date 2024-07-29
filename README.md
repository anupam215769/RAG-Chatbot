# Opensource RAG Chatbot for Personal Files
**LLM:** mistralai/Mixtral-8x7B-Instruct-v0.1

**Embeddings:** sentence-transformers/all-MiniLM-L6-v2

**Vectorstore:** Pinecone

**Framework:** LangChain, Streamlit

## Screenshots
![ss1](https://i.ibb.co/m8mzx4f/Screenshot-66.png)
![ss2](https://i.ibb.co/vPLbkT8/Screenshot-67.png)

## How To Setup

### Setup .env file
`HUGGINGFACEHUB_API_TOKEN`: can generated from https://huggingface.co/settings/tokens

`PINECONE_API_KEY` can generated from main console under API Keys

### Pincone Setup

1. Click on Indexes from sidebar
2. Click on create index
3. Enter index name (don't forget to change index_name field in app.py)
4. Enter dimensions (Depends upon the Embedding model, for [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) it is **384**. You can find for your model in their respective huggingface repo)
5. Keep metrics to cosine (you can change to euclidean or dotproduct if you want)
6. Keep Capacity mode, Cloud provider, Region to default (free accounts have limited usuage)
7. Finally click on create index and index will be ready in few seconds

### Change LLM, Embedding, Vectorstore (Optional)
Refer the LangChain documentation for using different services

[Supported Chat Model Classes](https://python.langchain.com/v0.2/docs/integrations/chat/)

[Supported Embedding Model Classes](https://python.langchain.com/v0.2/docs/integrations/text_embedding/)

[Supported Vector stores](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)


**Example on how to change all these things, I'm gonna use Azure OpenAI and Qdrant vectorstore here:**

> Import required libraries

```
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
```

> Creating vector embeddings

```
embedding = AzureOpenAIEmbeddings(azure_deployment="name given during deploying the Embedding model", chunk_size = 1000)
db = QdrantVectorStore.from_documents(documents, embedding, collection_name="RDK")
```

> LLM

```
llm = AzureChatOpenAI(azure_deployment="name given during deploying the LLM model")
```


## How To Run
> Note - Install requirements.txt before running the code

```
pip install -r requirements.txt
```

1. Clone the repo
  
2. Open terminal in the same directory as project folder

3. Then type `streamlit run app.py` in the terminal
