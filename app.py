
import streamlit as st

from modules.document_loader import extract_document
from modules.vector_store import (
    store_document,
    document_count,
    clear_database,
)
from modules.rag_engine import generate_answer


st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 SmartDoc AI")
st.caption(
    "Local RAG Document Assistant | "
    "Ollama + ChromaDB + Streamlit"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4,
    )

    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )

    st.divider()
    st.header("📂 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or Image",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please select documents first.")
        else:
            for uploaded_file in uploaded_files:
                try:
                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        text = extract_document(uploaded_file)

                        count = store_document(
                            text,
                            uploaded_file.name,
                        )

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{count} chunks indexed"
                    )

                except Exception as error:
                    st.error(str(error))

    st.metric(
        "Indexed Chunks",
        document_count(),
    )

    if st.button("Clear Vector Database"):
        clear_database()
        st.session_state.messages = []
        st.rerun()


st.subheader("💬 Chat With Your Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(source["source"])
                    st.caption(source["text"])


question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating AI answer..."):
                answer, sources = generate_answer(
                    question,
                    top_k=top_k,
                    temperature=temperature,
                )

            st.markdown(answer)

            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(source["source"])
                        st.caption(source["text"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"RAG Error: {error}")
