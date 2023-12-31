import os
import pinecone

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA


pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")


if __name__ == "__main__":

    loader = TextLoader("/Users/nmoureau/Desktop/intro-to-vector-db/mediumblogs/mediumblog1.txt")
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
    
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="blogs-embeddings-index")
    
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
   
    query="Enter your query here"
    
    result = qa.run(query)
    
    print(query,result)
    
