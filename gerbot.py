#!/usr/bin/env python

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import MistralAI
from llama_index.embeddings import MistralAIEmbedding
from llama_index import ServiceContext
from llama_index.query_engine import RetrieverQueryEngine

# Load data
reader = SimpleDirectoryReader(input_files=["essay.txt"])
documents = reader.load_data()

# Define LLM and embedding model
llm = MistralAI(api_key=api_key, model="mistral-medium")
embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Create vector store index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query(
    "What were the two main things the author worked on before college?"
)
print(str(response))
