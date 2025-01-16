import streamlit as st
# Remove pysqlite3 hacks
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# If you still want to attempt pysqlite3, do a try/except:
# try:
#     __import__('pysqlite3')
#     import sys
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# except KeyError:
#     pass

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

import chromadb
import uuid


def process_document(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
        else:
            loader = TextLoader(uploaded_file, encoding="utf-8")

        data = loader.load()
        raw_text = "\n".join([doc.page_content for doc in data])

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(raw_text)

        # Use local ChromaDB
        client = chromadb.PersistentClient(path="docstore")
        collection = client.get_or_create_collection(name="doc_collection")

        # Clear old docs if needed
        # collection.delete(where={})

        # Add new chunks
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

def answer_question(collection, query, groq_api_key):
    try:
        results = collection.query(query_texts=[query], n_results=3)
        # Combine chunk previews
        if results["metadatas"]:
            retrieved_chunks = "\n\n".join([m["preview"] for m in results["metadatas"][0]])
        else:
            retrieved_chunks = ""

        llm = ChatGroq(
            temperature=0.2,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile"
        )

        prompt_template = f"""
        You are an AI assistant with the following context from a user-uploaded document:
        {retrieved_chunks}

        The user asked: "{query}"

        Answer based only on the context. If the context is irrelevant or missing, say "I don't know."
        """
        response = llm.invoke(prompt_template)
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I encountered an error."

def main():
    st.set_page_config(page_title="Doc Chatbot", page_icon=":books:")
    st.title("Document Chatbot")

    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Cleared!")

    uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            collection = process_document(uploaded_file)
            if collection is not None:
                st.session_state.collection = collection
                st.success("Document processed!")

    user_question = st.text_input("Your question:", "")
    if st.button("Send Question"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        elif not user_question.strip():
            st.warning("Cannot send an empty query.")
        else:
            st.session_state.messages.append(("user", user_question))
            answer = answer_question(st.session_state.collection, user_question, groq_api_key)
            st.session_state.messages.append(("assistant", answer))

    st.write("---")
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Groq:** {msg}")

if __name__ == "__main__":
    main()
