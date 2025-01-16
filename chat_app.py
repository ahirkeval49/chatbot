import streamlit as st

# If you're seeing deprecation warnings for PyPDFLoader/TextLoader,
# feel free to ignore or switch to the recommended community loaders.
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# If you had code that replaced sqlite3 with pysqlite3, remove it to avoid KeyError.
# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import uuid

########################################
# STREAMLIT SETUP & STYLING
########################################

st.set_page_config(
    page_title="Doc Chatbot",
    page_icon=":books:",
    layout="centered"
)

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
    """
    Reads the uploaded PDF or text file, splits into chunks,
    and stores them in a local ChromaDB vector store.
    """
    try:
        # Determine loader based on file type
        if uploaded_file.type == "application/pdf":
            # Requires 'pypdf' or 'PyPDF2'
            loader = PyPDFLoader(uploaded_file)
        else:
            # Plain text
            loader = TextLoader(uploaded_file, encoding="utf-8")

        data = loader.load()
        raw_text = "\n".join([doc.page_content for doc in data])

        # Basic chunking
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(raw_text)

        # Setup a local ChromaDB instance
        client = chromadb.PersistentClient(path="docstore")  # folder named 'docstore'
        collection = client.get_or_create_collection(name="doc_collection")

        # Optionally clear old docs each time:
        # collection.delete(where={})

        # Add chunks to vector store
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
    """
    Retrieves the top N chunks from the ChromaDB collection
    and uses Groq to generate an answer from the context.
    """
    try:
        # Query the vector store for relevant chunks
        results = collection.query(query_texts=[query], n_results=top_n)
        if results["metadatas"]:
            retrieved_chunks = "\n\n".join([m["preview"] for m in results["metadatas"][0]])
        else:
            retrieved_chunks = ""

        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=temperature,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile"
        )

        prompt_template = f"""
        You are an AI assistant with the following context from a user-uploaded document:
        {retrieved_chunks}

        The user asked: "{query}"

        Provide the best possible answer using ONLY the context above.
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
    st.write("Upload a PDF or Text file, then ask questions about it!")
    
    # 1) Retrieve your hidden API key from Streamlit secrets
    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

    # 2) Manage state for conversation and collection
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared.")

    # Document upload
    uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

    # Process document
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            collection = process_document(uploaded_file)
            if collection is not None:
                st.session_state.collection = collection
                st.success("Document processed & stored!")

    # User input for questions
    user_query = st.text_input("Your question:", "")
    if st.button("Send Question"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        elif not user_query.strip():
            st.warning("Cannot send an empty query.")
        else:
            st.session_state.messages.append(("user", user_query))
            answer = answer_question(
                collection=st.session_state.collection,
                query=user_query,
                groq_api_key=groq_api_key
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
