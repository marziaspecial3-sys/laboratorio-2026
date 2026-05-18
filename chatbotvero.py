import streamlit as st
import pdfplumber

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title= "RagChatbot",
                   page_icon=":clown_face:")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #eb0c0c;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.header("Il tuo assistente di diritto costituzionale")

st.image("RAG classe/Chatbot.webp", width=300)

documento = "REAL CHATBOT/Costituzione_italiana.pdf"

openai_api_key=st.secrets["OPENAI_API_KEY"]

if documento is not None:
    with pdfplumber.open(documento) as pdf:
        # st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
        testo = ""
        for pagina in pdf.pages:
            testo = testo + pagina.extract_text() + "\n"
            # testo += pagina.extract_text() + "\n"
    # st.write(testo)

    taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
    
    frammenti = taglierina.split_text(testo)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    vettori = FAISS.from_texts(frammenti, embedding=embeddings)

    def invia():
        st.session_state.domanda_inviata = st.session_state.domanda_utente
        st.session_state.domanda_utente = ""

    st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
    domanda_utente = st.session_state.get("domanda_inviata", "")

    def formatta_documento(documenti):
        return "\n\n".join([documento.page_content for documento in documenti])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         '''Sei un esperto di diritto costituzionale. 
    Usa il contesto fornito per rispondere alla domanda in modo conciso. 
    Accedi a Internet, se non trovi informazioni pertinenti. 
    Se non conosci la risposta, dì semplicemente 'Putroppo, non riesco a trovare l'informazione richiesta'. 
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])
    
    comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})
    
    modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    catena = (
        # All'inizio mettiamo un dizionario che serve a costruire 
        # la struttura che il prompt vuol in input
        # Il comparatore produce i documenti (es. k=4) e li passa alla formattazione
        # RunnablePassthrough() vuol dire:
        # quando arriverà un input → passalo così com’è
        # Dobbiamo fare così perché ancora l'input concreto non c'è!  
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
        # StrOutputParser() prende l’output del modello 
        # e lo traforma in una stringa semplice (senza aggiunta di info ecc.)
    
    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)

