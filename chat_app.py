import streamlit as st
import os
import uuid
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
import os
os.environ["CHROMA_SQLITE_ALLOW_UNSAFE"] = "1"

import chromadb

########################################
# STREAMLIT CONFIG & STYLING
########################################

st.set_page_config(page_title="Document Chatbot", page_icon="📄", layout="centered")

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
    Reads the uploaded document (PDF, text, Word, or Excel), writes it to a 
    temporary local file, then loads and chunks its content for ChromaDB.
    """
    try:
        # 1. Create a temporary file for the uploaded document
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())  # Write raw file bytes to temp file
            temp_filepath = tmp_file.name

        # 2. Select the appropriate loader based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_filepath)
        elif file_extension == ".txt":
            loader = TextLoader(temp_filepath, encoding="utf-8")
        elif file_extension in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(temp_filepath)
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(temp_filepath)
        else:
            st.error(f"Unsupported file type: {file_extension}. Supported types: PDF, TXT, DOCX, XLSX.")
            return None

        # 3. Load document content
        data = loader.load()
        raw_text = "\n".join([doc.page_content for doc in data])

        # 4. Chunk the document
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)

        # 5. Store chunks in ChromaDB
        client = chromadb.PersistentClient(path="docstore")
        collection = client.get_or_create_collection(name="doc_collection")
        
        for chunk in chunks:
            collection.add(
                documents=chunk,
                metadatas={"preview": chunk[:60]},  # Store preview metadata
                ids=[str(uuid.uuid4())]
            )

        return collection

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None


def answer_question(collection, query, groq_api_key, temperature=0.2, top_n=3):
    """
    Retrieves top N chunks from ChromaDB and uses Groq to generate an answer.
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

        # Prompt with context
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
    st.write("Upload a document (PDF, Word, Excel, or Text), then ask questions about it!")

    # 1. Retrieve your Groq API key from Streamlit secrets
    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

    # 2. Manage session state for conversation and collection
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None

    # 3. Button to clear the conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared.")

    # 4. Document upload
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "xlsx"])

    # Process the uploaded document
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            collection = process_document(uploaded_file)
            if collection:
                st.session_state.collection = collection
                st.success("Document processed and stored!")

    # 5. Handle user questions
    user_query = st.text_input("Your question:", "")
    if st.button("Send Question"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        elif not user_query.strip():
            st.warning("Cannot send an empty query.")
        else:
            # Add user question
            st.session_state.messages.append(("user", user_query))

            # Get the AI's answer
            answer = answer_question(
                st.session_state.collection,
                user_query,
                groq_api_key
            )
            st.session_state.messages.append(("assistant", answer))

    # 6. Display the conversation
    st.write("---")
    for role, text in st.session_state.messages:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Groq:** {text}")


if __name__ == "__main__":
    main()
