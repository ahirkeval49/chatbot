import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextBaseLoader, PDFBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# Force pysqlite3 to replace sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import pandas as pd
import uuid
import os

# For text splitting
from langchain.text_splitter import CharacterTextSplitter

############################################
# SETUP
############################################

# Hide Streamlit's default footer
st.set_page_config(page_title="Groq Chatbot", page_icon="💬")

hide_footer = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_footer, unsafe_allow_html=True)

############################################
# MAIN APP
############################################

def main():
    st.title("💬 Chat with Groq (Docs-Aware)")

    # 1) Retrieve your hidden API key from Streamlit secrets
    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]  # Don't store keys in code
    
    # 2) Initialize session state (for conversation and vector store references)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [("user", "..."), ("assistant", "...")]
    if "collection" not in st.session_state:
        st.session_state.collection = None
    
    st.write("Upload a **PDF** or **Text** file, and ask questions about it!")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])
    
    # 3) If a document is uploaded, parse and store it in a local ChromaDB
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing the document..."):
            # Load the file (PDF or text)
            if uploaded_file.type == "application/pdf":
                loader = PDFBaseLoader(uploaded_file)
            else:
                # For text
                loader = TextBaseLoader(uploaded_file)
            
            data = loader.load()
            raw_text = "\n".join([doc.page_content for doc in data])

            # Basic chunking
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(raw_text)
            
            # Setup a local ChromaDB instance for storing these chunks
            client = chromadb.PersistentClient(path="docstore")  # will create docstore folder
            # Create a new collection or reuse existing
            collection = client.get_or_create_collection(name="my_doc_collection")

            # Clear the existing collection if needed
            # collection.delete(where={})  # optionally delete all existing docs

            # Add each chunk to the vector store
            for c in chunks:
                # We'll use the chunk text as "document" and a random ID
                collection.add(
                    documents=c,
                    metadatas={"chunk": c[:50]},  # store the first 50 chars for reference
                    ids=[str(uuid.uuid4())]
                )
            
            st.session_state.collection = collection
        
        st.success("Document is processed and stored in the vector database!")

    # 4) Chat interface
    user_input = st.text_input("You:", placeholder="Ask about your uploaded document...")
    if st.button("Send"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        else:
            if user_input.strip() == "":
                st.warning("Cannot send empty message.")
            else:
                # Add user message
                st.session_state.messages.append(("user", user_input))
                
                # 5) Retrieve relevant chunks from the doc using ChromaDB
                # We'll do a simple query for top 3 results
                results = st.session_state.collection.query(
                    query_texts=[user_input],
                    n_results=3
                )
                
                # Flatten the chunks
                retrieved_chunks = "\n\n".join([m["chunk"] for m in results["metadatas"][0]])
                
                # 6) Send the user message + context to ChatGroq
                llm = ChatGroq(
                    temperature=0.2,
                    groq_api_key=groq_api_key,
                    model_name="llama-3.1-70b-versatile"
                )
                
                # Create a basic prompt template that includes retrieved text
                prompt_template = f"""
                You are a helpful assistant with access to the following context from a user-uploaded document:
                {retrieved_chunks}

                The user asked: "{user_input}"

                Provide the best possible answer, referencing ONLY the context if possible. 
                If the context doesn't have an answer, say you don't know.
                Answer:
                """

                response = llm.invoke(prompt_template)
                assistant_msg = response.content.strip()
                
                # Save assistant message to session state
                st.session_state.messages.append(("assistant", assistant_msg))
    
    # 7) Display conversation
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Groq:** {msg}")


if __name__ == "__main__":
    main()
