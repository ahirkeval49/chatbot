import streamlit as st
import os
import sys

# 1) PATCH SQLITE3 WITH pysqlite3 (try/except to avoid KeyError)
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except (ImportError, KeyError):
    pass

import chromadb
import uuid
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

########################################
# STREAMLIT CONFIG
########################################

st.set_page_config(page_title="Doc Chatbot", page_icon=":books:", layout="centered")

HIDE_FOOTER = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(HIDE_FOOTER, unsafe_allow_html=True)

########################################
# HELPER FUNCTIONS
########################################

def process_document(uploaded_file):
    """Load PDF or text, chunk it, store in Chroma."""
    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)  # Requires pypdf
        else:
            loader = TextLoader(uploaded_file, encoding="utf-8")

        data = loader.load()
        raw_text = "\n".join([doc.page_content for doc in data])

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)

        client = chromadb.PersistentClient(path="docstore")
        collection = client.get_or_create_collection(name="doc_collection")
        # collection.delete(where={})  # If you want to clear old docs each time

        for chunk in chunks:
            collection.add(
                documents=chunk,
                metadatas={"preview": chunk[:60]},
                ids=[str(uuid.uuid4())]
            )
        return collection
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

def answer_question(collection, query, groq_api_key, temperature=0.2, top_n=3):
    """Retrieve top chunks from Chroma, use Groq to answer."""
    try:
        results = collection.query(query_texts=[query], n_results=top_n)
        if results["metadatas"]:
            retrieved_chunks = "\n\n".join([m["preview"] for m in results["metadatas"][0]])
        else:
            retrieved_chunks = ""

        llm = ChatGroq(
            temperature=temperature,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile"
        )

        prompt_template = f"""
        You are an AI assistant with context from a user-uploaded document:
        {retrieved_chunks}

        The user asked: "{query}"

        Provide the best answer using ONLY the context above. 
        If the context doesn't have an answer, say "I don't know."
        """
        response = llm.invoke(prompt_template)
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I encountered an error."

########################################
# MAIN APP
########################################

def main():
    st.title("Document Chatbot (Groq)")

    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None

    # Button to clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared.")

    # Document upload
    uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

    # Process doc
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            coll = process_document(uploaded_file)
            if coll:
                st.session_state.collection = coll
                st.success("Document processed & stored!")

    # Ask question
    user_query = st.text_input("Your question:", "")
    if st.button("Send Question"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        elif not user_query.strip():
            st.warning("Cannot send an empty query.")
        else:
            st.session_state.messages.append(("user", user_query))
            answer = answer_question(
                st.session_state.collection,
                user_query,
                groq_api_key
            )
            st.session_state.messages.append(("assistant", answer))

    # Display conversation
    st.write("---")
    for role, text in st.session_state.messages:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Groq:** {text}")

if __name__ == "__main__":
    main()
