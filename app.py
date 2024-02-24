import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text =""
#     for pdf in pdf_docs:
#         pdf_reader =PdfReader(pdf)
#         for page in pdf_reader.page:
#             text+= page.extract_text()
#     return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap = 1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embaddings= GooglrGenerativeAIEmbeddings(model="model/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embaddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""Answer the question as detailed as possible from the provided context, make sure to provide accurate answare
    Context : \n {context}? \n
    Questions : \n{question}\n 
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temprature =0.3)

    PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GooglrGenerativeAIEmbeddings(model ="model/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()


    responce = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(responce)
    st.write("Reply: ", responce ["output_text"])

def  main():  
    st.set_page_config("Chat with multiple pdf")
    st.header("Chat with multiple pdf using geminiai")

    user_question = st.text_input("Ask a qustion from the file")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("menu:")
        pdf_docs = st.file_uploader("Upload yoyr pdf file  and click the submit button")
        if st.button("Submit & process"):
            with st.spinner("processing...."):
                raw_text =  get_pdf_text(pdf_docs)
                text_chunks =  get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")    


if __name__ == "__main__":
    main()

