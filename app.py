import streamlit as st
from streamlit_lottie import st_lottie
import requests
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings

def main():
   st.set_page_config(page_title="AI Doctor",page_icon="👨‍⚕️",layout='wide')
   st.subheader('Hello! :wave:! How are you feeling today ?')
   st.title('Doctor AI')
   with st.container():
    st.write("---")
    query = st.text_area("Ask any question related to HealthCare or Medical")
    if st.button("Answer"):
        with st.spinner('Getting For Right Answer........'):
          template = """You are an AI Doctor who answers queries related to Healthcare and related things. Answer only to the questions related to Healthcare and related things, if not just reply 'Ask a Health care related question.....'
          Human : Hi
          Doctor : Hi, I am Your AI Doctor How can i help you?
          
          Human: {query}
          Doctor:"""
          prompt = PromptTemplate(input_variables=["query"], template=template)
          chain = LLMChain(
             llm=OpenAI(temperature=0,openai_api_key='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'),
             prompt=prompt,verbose=True)
          output = chain.predict(query=query)
          st.subheader('Answer:')
          st.write(output)

if __name__ =="__main__":
   main()