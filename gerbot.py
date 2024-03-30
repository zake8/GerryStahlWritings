#!/usr/bin/env python

### GerBot project intendes to make http://gerrystahl.net/pub/index.html even more accessable; Generative AI "chat" about the gerrystahl.net writings
### Code by Zake Stahl
### March 2024
### Based on public/shared APIs and FOSS samples
### Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, more

from flask import Flask, redirect, url_for, render_template, request
import socket
import random
import logging
from langchain_community.llms import Ollama # from langchain.llms import Ollama # LangChainDeprecationWarning: Importing LLMs from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)

logging.basicConfig(
    filename='gerbot.log', 
    level=logging.INFO, 
    filemode='a', 
    format='%(asctime)s -%(levelname)s - %(message)s')

@app.route("/")
def home():
    webserver_hostname = socket.gethostname()
    return render_template('staging.html', webserver_hostname=webserver_hostname)

fullragchat_history = []

# TODO:
# implement context, conversation history forward !!!
# implement saving vector DB

@app.route("/reset_history")
def reset_fullragchat_history():
    global fullragchat_history
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    fullragchat_history.clear()
    fullragchat_history.append({'user':'-reset--', 'message':'reset'})
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model, 
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

def init_fullragchat_history():
    global fullragchat_history
    reset_fullragchat_history()
    fullragchat_history.clear()
    fullragchat_history.append({'user':'GerBot', 'message':'Hi!'}) 
    fullragchat_history.append({'user':'GerBot', 'message':"Lets chat about Gerry Stahl's writting."}) 
    fullragchat_history.append({'user':'GerBot', 'message':'Enter a question, and click query.'}) 

def pending_fullragchat_history():
    global fullragchat_history
    fullragchat_history.append({'user':'-reset', 'message':'pending - please wait for model inferences - small moving graphic on browser tab should indicate working'}) 

def unpending_fullragchat_history():
    global fullragchat_history
    fullragchat_history.pop()
    pass

@app.route("/fullragchat_init")
def fullragchat_init():
    global fullragchat_history
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    ### set initial values
    fullragchat_model = "fake_llm"
    fullragchat_temp = "0.7"
    fullragchat_stop_words = ""
    fullragchat_rag_source = ""
    fullragchat_embed_model = "nomic-embed-text"
    fullragchat_skin = "not yet implemented"
    fullragchat_music = "not yet implemented"
    init_fullragchat_history()
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history,
        fullragchat_model=fullragchat_model,
        fullragchat_temp=fullragchat_temp,
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

@app.route("/fullragchat_pending", methods=['POST'])
def fullragchat_pending():
    # fullragchat.html submits here
    global fullragchat_history
    global query
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    query = request.form['query']
    fullragchat_model = request.form['model']
    fullragchat_temp = request.form['temp']
    fullragchat_stop_words = request.form['stop_words']
    fullragchat_rag_source = request.form['rag_source']
    fullragchat_embed_model = request.form['embed_model']
    fullragchat_skin = request.form['skin']
    fullragchat_music = request.form['music']
    fullragchat_history.append({'user':'---user', 'message':query}) 
    logging.info(f'===> user: {query}')
    pending_fullragchat_history()
    return render_template('fullragchat_pending.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model,
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

def fake_llm(query):
    rand = random.randint(1, 9)
    if rand <= 3:
        answer = "The answers are in the questions - " + query + " - Think about it."
    elif rand <= 4:
        answer = "I don't understand, can you rephrase the question?" + query + "???"
    elif rand <= 5:
        answer = "That's a good question. Ask another."
    elif rand <= 6:
        answer = "Positive"
    elif rand <= 7:
        answer = "Negative"
    elif rand <= 8:
        answer = "I didn't quite catch that, again?"
    elif rand <= 9:
        answer = "Maybe, we'll see."
    else:
        answer = "Go away." + query + "Huh."
    return answer

def chat_query_return(
                    model, 
                    query, 
                    fullragchat_temp, 
                    fullragchat_stop_words, 
                    fullragchat_rag_source, 
                    fullragchat_embed_model,
                    ):
    stop_words_list = fullragchat_stop_words.split(', ')
    if stop_words_list == ['']: stop_words_list = None
    if model == "fake_llm":
        answer = fake_llm(query)
    elif (model == "orca-mini") or (model == "phi"):
        ### Instanciate LLM
        ollama = Ollama(
            model=model, 
            temperature=float(fullragchat_temp), 
            stop=stop_words_list, 
            verbose=True,
        )
        if fullragchat_rag_source: # have a rag doc to process
            extension = fullragchat_rag_source[-3:]
            if extension == "txt":
                loader = TextLoader(fullragchat_rag_source) # ex: /path/filename
            elif (extension == "tml") or (extension == "htm"): #html
                loader = WebBaseLoader(fullragchat_rag_source) # ex: https://url/file.html
            elif extension == "pdf":
                loader = OnlinePDFLoader(fullragchat_rag_source) # ex: https://url/file.pdf
            elif extension == "son": #json
                loader = JSONLoader(file_path=fullragchat_rag_source,
                    jq_schema='',
                    text_content=False
                )
                # https://python.langchain.com/docs/modules/data_connection/document_loaders/json
            else:
                answer = "Unable to make loader for " + fullragchat_rag_source
                return answer
            data = loader.load()
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            all_splits = text_splitter.split_documents(data)
            oembed = OllamaEmbeddings(model=fullragchat_embed_model)
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
            docs = vectorstore.similarity_search(query)
            vector_store_hits = len(docs)
            qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
            ### invoke model
            results = qachain.invoke({"query": query})
            answer = results['result'] 
        else: # simple chat
            ### invoke model
            context = ""
            answer = ollama(query)
            # answer = ollama.invoke(query) # is this prefered? how does the above know what to do?
    else:
        answer = "No LLM named " + model
    return answer

@app.route("/fullragchat_reply")
def fullragchat_reply():
    # fullragchat_pending.html refreshes here
    global fullragchat_history
    global query
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    unpending_fullragchat_history()
    logging.info(f'===> model info: model={fullragchat_model}, temp={fullragchat_temp}, stop={fullragchat_stop_words}, rag={fullragchat_rag_source}, embed={fullragchat_embed_model}')
    answer = chat_query_return(
        fullragchat_model, 
        query, 
        fullragchat_temp, 
        fullragchat_stop_words, 
        fullragchat_rag_source, 
        fullragchat_embed_model,
    )
    fullragchat_history.append({'user':'GerBot', 'message':answer})
    logging.info(f'===> GerBot: {answer}')
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model, 
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
        )

if __name__ == "__main__":
    app.run()
