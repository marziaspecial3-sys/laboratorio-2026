import streamlit as st
import pdfplumber
import os
 
# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
st.set_page_config(page_title="RagChatbot", page_icon=":classical_building:")
 
# Personalizzazione colori
st.markdown(
    """
    <style>
    .stApp {
        background-color: #B84B4B;
        color: #B1D4D8;
    }
    </style>
    """,
    unsafe_allow_html=True)
 
st.header("MagnaFranci")
 
# Gestione dinamica del percorso dell'immagine
if os.path.exists("rag marzia/Chatbot.webp"):
    st.image("rag marzia/Chatbot.webp", width=500)
elif os.path.exists("Chatbot.webp"):
    st.image("Chatbot.webp", width=500)
 
# FIX PERCORSO FILE: Trova il PDF nella stessa cartella di questo file app.py
cartella_corrente = os.path.dirname(__file__)
documento = os.path.join(cartella_corrente, "Tappe.pdf")
 
# Estrazione del contenuto e spezzettamento
if os.path.exists(documento):
   
    @st.cache_data(show_spinner="Sto leggendo il PDF...")
    def estrai_testo_pdf(percorso_pdf: str) -> str:
        testo = ""
        with pdfplumber.open(percorso_pdf) as pdf:
            for pagina in pdf.pages:
                testo_pagina = pagina.extract_text() or ""
                testo = testo + testo_pagina + "\n"
        return testo.strip()
   
    testo = estrai_testo_pdf(documento)
 
    @st.cache_data(show_spinner=False)
    def crea_frammenti(testo_estratto: str):
        taglierina = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=1000,
            chunk_overlap=200
        )
        # Rimuoviamo stringhe vuote prima dello split
        return [f for f in taglierina.split_text(testo_estratto) if f.strip()]
 
    frammenti = crea_frammenti(testo)
 
    # CONTROLLO DI SICUREZZA: Evita il crash se il PDF è vuoto o fatto di sole immagini
    if not frammenti:
        st.error("Il file PDF è stato trovato, ma non è stato possibile estrarre testo. È un PDF scannerizzato (immagine)?")
    else:
        @st.cache_resource(show_spinner=False)
        def crea_vectorstore(lista_frammenti):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=st.secrets["OPENAI_API_KEY"]
            )
            return FAISS.from_texts(lista_frammenti, embedding=embeddings)
       
        vettori = crea_vectorstore(frammenti)
 
        # -------------------------------------------------------------------
        # Gestione prompt e input
        # -------------------------------------------------------------------
        def invia():
            st.session_state.domanda_inviva = st.session_state.domanda_utente
            st.session_state.domanda_utente = ""
 
        st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
        domanda_utente = st.session_state.get("domanda_inviva", "")
 
        def formatta_documento(documenti):
            return "\n\n".join([doc.page_content for doc in documenti])
       
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             '''Sei “Sicily Pilgrim Assistant”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.

 

Il tuo ruolo è accompagnare l’utente durante il cammino fornendo:

- informazioni pratiche (acqua, distanza, difficoltà delle tappe)

- supporto culturale e narrativo sul territorio

- indicazioni su ospitalità, ristoro e luoghi di interesse

 

Regole di comportamento:

- Usa esclusivamente le informazioni presenti nel contesto fornito

- Non inventare informazioni mancanti

- Se l’informazione richiesta non è disponibile nel contesto, rispondi in modo accogliente e coerente con il ruolo di guida del cammino

Rispondi:

“Caro pellegrino, al momento non riesco a guidarti su questa informazione.”

Le risposte devono essere:

- chiare

- utili durante il cammino

- semplici da consultare anche in mobilità

- coerenti con l’esperienza del pellegrinaggio

- accoglienti e orientate all’accompagnamento del pellegrino'.
Contesto:\n{context}'''),
            ("human", "{question}")
        ])
 
        comparatore = vettori.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )
       
        # FIX MODELLO: Cambiato in gpt-4o-mini (veloce ed economico)
        modello_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
       
        catena = (
            {"context": comparatore | formatta_documento, "question": RunnablePassthrough()}
            | prompt
            | modello_llm
            | StrOutputParser()
        )
       
        if domanda_utente:
            risposta = catena.invoke(domanda_utente)
            st.write(risposta)
else:
    st.error(f"Non ho trovato il file PDF nel percorso: {documento}. Assicurati che sia nella cartella 'RAG Samuele'.")
 
