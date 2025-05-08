import spacy
import faiss
import os
import numpy as np
import networkx as nx
import streamlit as st
from sentence_transformers import SentenceTransformer
from langdetect import detect
import PyPDF2
from langfuse import Langfuse
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Initialize Langfuse (replace these with your actual keys)
langfuse = None
try:
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST")
    )
except Exception as e:
    st.warning(f"Langfuse setup failed: {e}")
# ---------- Config ----------
st.set_page_config(page_title="PDF Knowledge Graph Chatbot", layout="wide")
st.title("📚 Multilingual PDF Chatbot (English & Arabic)")

# ---------- Load Models ----------
@st.cache_resource
def load_spacy_model(lang):
    if lang == 'en':
        return spacy.load("en_core_web_sm")
    elif lang == 'ar':
        return spacy.load("xx_sent_ud_sm")
    else:
        return spacy.load("en_core_web_sm")  # fallback

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("distiluse-base-multilingual-cased-v1")

# ---------- PDF Text Extraction ----------
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return " ".join([page.extract_text() or "" for page in reader.pages])

def load_documents(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            texts.append(extract_text_from_pdf(full_path))
    return texts

# ---------- Text Chunking ----------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------- Entity Extraction ----------
def extract_entities(text_chunk, nlp):
    doc = nlp(text_chunk)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents]

# ---------- Knowledge Graph ----------
def build_knowledge_graph(chunks, nlp):
    G = nx.Graph()
    for i, chunk in enumerate(chunks):
        ents = extract_entities(chunk, nlp)
        sentence_node = f"chunk_{i}"
        G.add_node(sentence_node, type="chunk")
        for ent_text, label in ents:
            if ent_text:
                G.add_node(ent_text, type=label)
                G.add_edge(sentence_node, ent_text)
    return G

def find_chunks_by_entity(query, graph, chunks):
    matches = [node for node in graph.nodes if query.lower() in node.lower()]
    related_chunks = set()
    for node in matches:
        for neighbor in graph.neighbors(node):
            if neighbor.startswith("chunk_"):
                idx = int(neighbor.replace("chunk_", ""))
                related_chunks.add(chunks[idx])
    return list(related_chunks)

# ---------- Embeddings & Search ----------
def create_embeddings(chunks, model):
    return model.encode(chunks)

def retrieve_relevant_chunks(query, model, index, chunks, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    return [chunks[i] for i in I[0]]

# ---------- Language Detection ----------
def detect_language(text):
    try:
        lang = detect(text)
        return 'ar' if lang == 'ar' else 'en'
    except:
        return 'en'

# ---------- Main Chatbot Logic ----------
query = st.text_input("Ask a question:")

if query:
      # ----- Step 1: Prompt Injection Scan -----
    sanitized_prompt, is_valid, risk_score = scanner.scan(query)
    if not is_valid:
        st.error("🚫 Unsafe input detected. Please rephrase your question.")
        st.stop()
     # ----- Step 2: Langfuse Trace Start -----
    trace = None
    try:
        trace = langfuse.trace(
            name="user_query",
            input=sanitized_prompt,
            metadata={"risk_score": risk_score}
        )
    except Exception as e:
        print("Langfuse initialization error:", e)
        st.warning("⚠️ Langfuse monitoring failed to start.")

    # ----- Step 3: Language Detection -----
    
    # Detect language
    language = detect_language(query)
    st.markdown(f"🌐 Detected language: **{'Arabic' if language == 'ar' else 'English'}**")

    folder_path = f"data/{'arabic' if language == 'ar' else 'english'}"
    documents = load_documents(folder_path)

    if not documents:
        st.error("❌ No PDF documents found.")
    else:
        full_text = " ".join(documents)
        chunks = chunk_text(full_text, chunk_size=100)

        nlp = load_spacy_model(language)
        sentence_model = load_sentence_model()

        graph = build_knowledge_graph(chunks, nlp)
        chunk_embeddings = create_embeddings(chunks, sentence_model)

        index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        index.add(np.array(chunk_embeddings))

        # Find entity and semantic matches
        entity_results = find_chunks_by_entity(query, graph, chunks)
        semantic_results = retrieve_relevant_chunks(query, sentence_model, index, chunks)

        # Combine results
        combined_results = list(set(entity_results + semantic_results))[:3]

        st.markdown("### 🔍 Top Relevant Chunks:")
        for i, result in enumerate(combined_results):
            st.markdown(f"**{i+1}.** {result}")
          