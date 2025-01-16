import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# Force pysqlite3 to replace sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import uuid

########################################
# STREAMLIT PAGE CONFIG & STYLING
########################################

st.set_page_config(
    page_title="AI Chatbot (Groq)",
    page_icon="🤖",
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
    
    Returns a ChromaDB collection if successful, otherwise None.
    """
    try:
        # Determine loader based on file type
        if uploaded_file.type == "application/pdf":
            loader = PDFBaseLoader(uploaded_file)
        else:
            loader = TextBaseLoader(uploaded_file)

        # Load doc pages
        data = loader.load()
        raw_text = "\n".join([doc.page_content for doc in data])

        # Basic text splitting
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(raw_text)

        # Create or use existing local ChromaDB
        client = chromadb.PersistentClient(path="docstore")  # folder named 'docstore'
        collection = client.get_or_create_collection(name="doc_collection")

        # (Optional) Clear old docs:
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

def answer_question(collection, user_query, groq_api_key, temperature=0.2, top_n=3):
    """
    Retrieves the top N chunks from the Chroma collection
    and uses Groq to generate an answer from the context.
    """
    try:
        # Query the vector store for relevant chunks
        results = collection.query(query_texts=[user_query], n_results=top_n)

        # Combine the top chunk previews into a single context string
        if results["metadatas"]:
            retrieved_chunks = "\n\n".join([m["preview"] for m in results["metadatas"][0]])
        else:
            retrieved_chunks = ""

        # Initialize the Groq LLM
        llm = ChatGroq(
            temperature=temperature,
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile"
        )

        # Create a prompt with context
        prompt_template = f"""
        You are an AI assistant with access to the following context from a user-uploaded document:
        {retrieved_chunks}

        The user asked: "{user_query}"

        Provide the best possible answer, referencing ONLY the context if possible.
        If the context doesn't have an answer, say "I don't know."
        Answer:
        """

        response = llm.invoke(prompt_template)
        return response.content.strip()

    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "I'm sorry; I ran into an error."

########################################
# MAIN APP
########################################

def main():
    st.title("🤖 AI Chatbot with Groq")
    st.write("Upload a **PDF** or **Text** file, then ask questions about it!")

    # Retrieve your Groq API key from .streamlit/secrets.toml or Streamlit Cloud
    groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

    # Keep track of conversation & collection
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [("user", "..."), ("assistant", "...")]
    if "collection" not in st.session_state:
        st.session_state.collection = None

    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared.")

    # File uploader
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])

    # Process document
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            collection = process_document(uploaded_file)
            if collection is not None:
                st.session_state.collection = collection
                st.success("Document successfully processed & stored!")

    # Chat input
    user_question = st.text_input("Your question:", value="", placeholder="Ask a question about the document...")
    if st.button("Send Question"):
        if not st.session_state.collection:
            st.warning("Please upload and process a document first.")
        elif user_question.strip() == "":
            st.warning("Cannot send empty question.")
        else:
            # Add user message
            st.session_state.messages.append(("user", user_question))

            # Get AI answer
            answer = answer_question(
                collection=st.session_state.collection,
                user_query=user_question,
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
