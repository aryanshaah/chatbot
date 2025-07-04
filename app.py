from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a chatbot to help users with their queries."),
    HumanMessagePromptTemplate.from_template("Question: {question}")
])

st.title('Langchain Demo with gemma3 API')
input_text = st.text_input("Ask anything:")

llm = Ollama(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
