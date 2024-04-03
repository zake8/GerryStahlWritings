#!/usr/bin/env python

# TODO:
# implement context, conversation history forward, into RAG chat !!!
# implement saving vector DB
# save down pdfs; pdf into txt chapters w/ book and chapter summeries
# what are max token in sizes per model? 
# agent-style answering:
        # query
        # based on query and context (chat history), what to pull for RAG within the token size? Reference these summeries.
        # based on query and context and RAG, answer

### GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessable
### Generative AI "chat" about the gerrystahl.net writings
### Code by Zake Stahl
### https://github.com/zake8/GerryStahlWritings
### March, April 2024
### Based on public/shared APIs and FOSS samples
### Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more

import logging
logging.basicConfig(
    filename='gerbot.log', 
    level=logging.INFO, 
    filemode='a', 
    format='%(asctime)s -%(levelname)s - %(message)s')

from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv('./.env')

# ConversationBufferWindowMemory setup
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=14)

import socket
import random
import os
# from langchain_community.document_loaders import OnlinePDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

@app.route("/")
def root():
    webserver_hostname = socket.gethostname()
    return render_template('staging.html', webserver_hostname=webserver_hostname)

def convo_mem_function(query):
    # ignoring query and generating text history from convo_mem dict
    history = f'<chat_history>\n'
    for line in fullragchat_history:
        history += f'{line}\n'
    history += f'</chat_history>\n'
    return history

def mistral_convochat(model, mkey, fullragchat_temp, query):
    large_lang_model = ChatMistralAI(
        model_name=model, 
        mistral_api_key=mkey, 
        temperature=float(fullragchat_temp), 
    )
    global memory
    chain = ConversationChain(llm=large_lang_model, memory=memory) # ConversationBufferWindowMemory leveraged by ConversationChain
    answer = chain.predict(input=query)
    return answer

def ollama_convochat(model, fullragchat_temp, stop_words_list, query):
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    global memory
    chain = ConversationChain(llm=ollama, memory=memory) # ConversationBufferWindowMemory leveraged by ConversationChain
    answer = chain.predict(input=query)
    return answer

def mistral_qachat(model, mkey, fullragchat_temp, query):
    # simple chat from https://docs.mistral.ai/platform/client/
    client = MistralClient(api_key=mkey)
    messages = [ ChatMessage(role="user", content=query) ]
    chat_response = client.chat(
            model=model,
            messages=messages,
            temperature=float(fullragchat_temp), 
    )
    answer = chat_response.choices[0].message.content
    return answer

def ollama_qachat(model, fullragchat_temp, stop_words_list, query):
    ### instanciate model
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    ### invoke model
    answer = ollama(query)
    # answer = ollama.invoke(query) # is this prefered? how does the above know what to do?
    return answer

def get_rag_text(fullragchat_rag_source, query):
    extension = fullragchat_rag_source[-3:]
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/json
    if extension == "txt":
        loader = TextLoader(fullragchat_rag_source) # ex: /path/filename
    elif (extension == "tml") or (extension == "htm"): #html
        loader = WebBaseLoader(fullragchat_rag_source) # ex: https://url/file.html
    elif extension == "pdf":
        # throws: ImportError: cannot import name 'open_filename' from 'pdfminer.utils'
        # loader = UnstructuredPDFLoader(fullragchat_rag_source)
        # loader = OnlinePDFLoader(fullragchat_rag_source) # ex: https://url/file.pdf
        answer = "Need to make a loader for pdf... " + fullragchat_rag_source
        return answer
    elif extension == "son": #json
        loader = JSONLoader(file_path=fullragchat_rag_source,
            jq_schema='.',
            text_content=False)
    else:
        answer = "Unable to make loader for " + fullragchat_rag_source
        return answer
    # from https://docs.mistral.ai/guides/basic-RAG/
    docs = loader.load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    rag_text = text_splitter.split_documents(docs)
    return rag_text

def mistral_convo_rag(fullragchat_rag_source, fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    documents = get_rag_text(fullragchat_rag_source, query)
    answer = f'mistral_convo_rag not yet implemented.'
    return answer

def mistral_rag(fullragchat_rag_source, fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    documents = get_rag_text(fullragchat_rag_source, query)
    # Define the embedding model
    embeddings = MistralAIEmbeddings(
                model=fullragchat_embed_model, 
                mistral_api_key=mkey
    )
    # Create the vector store 
    vector = FAISS.from_documents(documents, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    large_lang_model = ChatMistralAI(
                model_name=model, 
                mistral_api_key=mkey, 
                temperature=float(fullragchat_temp), 
    )
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
    """)
    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(large_lang_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    input_query = {}
    input_query['input'] = query
    # invoke chain
    response_dic = retrieval_chain.invoke(input_query)
    answer = response_dic['answer'] # parse return from LLM with input, context, and, answer into just answer
    return answer

def ollama_convo_rag(model, fullragchat_temp, stop_words_list, fullragchat_rag_source, fullragchat_embed_model, query):
    all_splits = get_rag_text(fullragchat_rag_source, query)
    oembed = OllamaEmbeddings(model=fullragchat_embed_model)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    docs = vectorstore.similarity_search(query)
    vector_store_hits = len(docs)
    history_runnable = RunnableLambda(convo_mem_function)
    setup_and_retrieval = RunnableParallel({
#        "context":  docs, 
        "question": RunnablePassthrough(),
        "history":  history_runnable})
    template = """
        Answer the question based primarily on this following authoritative context: 
        {context}
        
        Reference chat history for conversationality: 
        {history}
        
        Question: 
        {question}
        
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    output_parser = StrOutputParser()
    chain = ( setup_and_retrieval | prompt | ollama | output_parser )
    answer = chain.invoke(query)
    return answer

def ollama_rag(model, fullragchat_temp, stop_words_list, fullragchat_rag_source, fullragchat_embed_model, query):
    ### instanciate model
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    all_splits = get_rag_text(fullragchat_rag_source, query)
    oembed = OllamaEmbeddings(model=fullragchat_embed_model)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    docs = vectorstore.similarity_search(query)
    vector_store_hits = len(docs)
    ### create chain w/ model
    qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    ### invoke chain
    results = qachain.invoke({"query": query})
    answer = results['result'] 
    return answer

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
    elif (model == "orca-mini") or (model == "phi") or (model == "tinyllama"): # or Ollama served llama2-uncensored or mistral or mixtral 
        if fullragchat_rag_source:
            if fullragchat_loop_context == 'True':
                answer = ollama_convo_rag(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    fullragchat_rag_source=fullragchat_rag_source, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    query=query
                )
            else:
                answer = ollama_rag(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    fullragchat_rag_source=fullragchat_rag_source, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    query=query
                )
        else:
            if fullragchat_loop_context == 'True':
                answer = ollama_convochat(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    query=query, 
                )
            else:
                answer = ollama_qachat(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    query=query, 
                )
    elif (model == "open-mixtral-8x7b") or (model == "mistral-large-latest") or (model == "open-mistral-7b"):
        mkey = os.getenv('Mistral_API_key')
        if fullragchat_rag_source:
            if fullragchat_loop_context == 'True':
                answer = mistral_convo_rag(
                    fullragchat_rag_source=fullragchat_rag_source, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    mkey=mkey, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    query=query, 
                )
            else:
                answer = mistral_rag(
                    fullragchat_rag_source=fullragchat_rag_source, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    mkey=mkey, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    query=query, 
                )
        else:
            if fullragchat_loop_context == 'True':
                answer = mistral_convochat(
                    mkey=mkey, 
                    query=query, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                )
            else:
                answer = mistral_qachat(
                    mkey=mkey, 
                    query=query, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                )
    else:
        answer = "No LLM named " + model
    return answer

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
    global fullragchat_loop_context
    ### set initial values
    fullragchat_history = []
    fullragchat_model = "open-mixtral-8x7b"
    fullragchat_temp = "0.25"
    fullragchat_rag_source = ""
    fullragchat_embed_model = "mistral-embed"
    fullragchat_loop_context = "True"
    fullragchat_stop_words = ""
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
        fullragchat_loop_context=fullragchat_loop_context, 
    )

def pending_fullragchat_history():
    global fullragchat_history
    fullragchat_history.append({'user':'-reset', 'message':'pending - please wait for model inferences - small moving graphic on browser tab should indicate working'}) 

def unpending_fullragchat_history():
    global fullragchat_history
    if fullragchat_history: fullragchat_history.pop()

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
    global fullragchat_loop_context
    query = request.form['query']
    fullragchat_model = request.form['model']
    fullragchat_temp = request.form['temp']
    fullragchat_stop_words = request.form['stop_words']
    fullragchat_rag_source = request.form['rag_source']
    fullragchat_embed_model = request.form['embed_model']
    fullragchat_skin = request.form['skin']
    fullragchat_music = request.form['music']
    fullragchat_loop_context = request.form['loop_context']
    fullragchat_history.append({'user':'---User', 'message':query}) 
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
        fullragchat_loop_context=fullragchat_loop_context, 
    )

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
    global fullragchat_loop_context
    global memory
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
    memory.save_context({"input": query}, {"output": answer}) # ConversationBufferWindowMemory save of query and answer
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
        fullragchat_loop_context=fullragchat_loop_context, 
        )

if __name__ == "__main__":
    app.run()
