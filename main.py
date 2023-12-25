import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

file_path='faiss_index_openai.pkl'
main_placeholder = st.empty()

#from dotenv import load_dotenv
#load_dotenv()  # take environment variables from .env (especially openai api key)
key='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' 
os.environ['OPENAI_API_KEY']=key
llm = OpenAI(temperature=0.9, max_tokens=500) 

st.title("News Research Tool : ")
st.sidebar.title("News Articles URLS : ")

urls=[]
for i in range(2):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
process_url_clicked=st.sidebar.button("Process URLS")    

if process_url_clicked:
    #1.load the data 
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started ....")
    data=loader.load()
    
    #2.split the data 
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["/n/n","/n",",","."],
        chunk_size=1000
        )
    
    #-divide into the chunks 
    main_placeholder.text("Data Splitting Started ....")
    chunks = text_splitter.split_documents(data)
    # 3.create embeddings and save it to FAISS index

    embeddings=OpenAIEmbeddings()
    main_placeholder.text("Embedding vector strated building ....")
    #time.sleep(2)
    vectors_faiss_openai=FAISS.from_documents(chunks,embeddings)
    
    #dump into pickle format
    with open(file_path,'wb') as f:
        pickle.dump(vectors_faiss_openai,f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            #langchain.debug=True
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # write down sources if any :
            sources=result.get('sources', '')
            if sources:
                st.subheader("Sources : ")
                sources=sources.split("/n")
                for i in sources:
                    st.write(i)