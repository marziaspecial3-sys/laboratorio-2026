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
 
st.set_page_config(page_title="La Magna Via", page_icon=":walking_man:")
 
# -------------------------------------------------------------------

# Configurazione Stile CSS (Corretto senza st.divider all'interno)

# -------------------------------------------------------------------

st.markdown(

    """
<style>

    .stApp {

        background-color:#B5A585;

        background-attachment: fixed;

        color: #000000;

        font-size: 36px;

    }

    /* Configurazione scritta "Chiedi al chatbot" */

    .stTextInput label div p {

        color: #4A2A20 !important;

        font-size: 24px !important;

        font-weight: bold !important;

    }

    /* Rettangolo di input */

    .stTextInput input {

        background-color: #4F7942;

        color: #ffffff;

    }
</style>

    """,

    unsafe_allow_html=True

)
 
# Interfaccia grafica principale

st.sidebar.image("LOGO.png", width=200)

st.sidebar.header("  I tuoi passi")
 
 
st.header("Verbum Viae", text_alignment="center")
 
# -------------------------------------------------------------------

# Elaborazione Documento PDF e RAG

# -------------------------------------------------------------------

cartella_corrente = os.path.dirname(__file__)

documento = os.path.join(cartella_corrente, "Tappe.pdf")
 
# Inizializziamo la variabile del retriever (catena) fuori dall'if

catena = None
 
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

        return [f for f in taglierina.split_text(testo_estratto) if f.strip()]
 
    frammenti = crea_frammenti(testo)
 
    if not frammenti:

        st.error("Il file PDF è stato trovato, ma non è stato possibile estrarre testo. È un PDF scannerizzato?")

    else:

        @st.cache_resource(show_spinner=False)

        def crea_vectorstore(lista_frammenti):

            embeddings = OpenAIEmbeddings(

                model="text-embedding-3-small",

                openai_api_key=st.secrets["OPENAI_API_KEY"]

            )

            return FAISS.from_texts(lista_frammenti, embedding=embeddings) 

        vettori = crea_vectorstore(frammenti)

        # --- CONFIGURAZIONE LANGCHAIN (Spostata qui per essere pronta subito) ---

        def formatta_documento(documenti):

            return "\n\n".join([doc.page_content for doc in documenti])

        prompt = ChatPromptTemplate.from_messages([

            ("system", 

             '''Sei “La Magna via”, un assistente digitale dedicato ai pellegrini della Magna Via Francigena in Sicilia.

Il tuo ruolo è accompagnare l’utente durante il cammino fornendo:

- informazioni pratiche (punti/fontanelle/fonti d'acqua, distanza, difficoltà delle tappe, punti di appproviggionamento)

- supporto culturale e narrativo sul territorio

- indicazioni su ospitalità, ristoro e luoghi di interesse

Regole di comportamento:

- Usa esclusivamente le informazioni presenti nel contesto fornito

- Non inventare informazioni mancanti

- Se l’informazione richiesta non è disponibile nel contesto, rispondi in modo accogliente e coerente con il ruolo di guida del cammino:

“Caro pellegrino, al momento non riesco a guidarti su questa informazione.”

- Nel caso in cui l'utente ponga una domanda in una lingua diversa dall'italiano rispondi nella stessa lingua.

- Nel caso in cui l'utente utilizzi un alfabeto diverso dalle lingue indoeuropee (cirillico, alfabeti asiatici ecc.) rispondi utilizzando lo stesso alfabeto

Le risposte devono essere:

- chiare

- utili durante il cammino

- semplici da consultare anche in mobilità

- coerenti con l’esperienza del pellegrinaggio

- accoglienti e orientate all’accompagnamento del pellegrino

Quando l'utente chiede informazioni su una tappa, verifica se il percorso attraversa aree sensibili (boschi, riserve naturali, zone di macchia mediterranea). 

Se la risposta è affermativa, aggiungi in chiusura il 'Consiglio del Custode', personalizzandolo come segue:



🌿 Cammina come custode del sentiero.

La Magna Via è un dono: aiutaci a proteggerla dal rischio incendi. Per favore, evita di fumare nei boschi e non lasciare traccia del tuo passaggio: porta sempre con te i tuoi mozziconi fino al prossimo centro abitato. Grazie di cuore per il tuo rispetto!.

Contesto:\n{context}'''),

            ("human", "{question}")

        ])
 
        comparatore = vettori.as_retriever(

            search_type="mmr",

            search_kwargs={"k": 4}

        )

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

else:

    st.error(f"Non ho trovato il file PDF nel percorso: {documento}.")
 
# -------------------------------------------------------------------

# Gestione Cronologia e Messaggi Chat

# -------------------------------------------------------------------

if "cronologia" not in st.session_state:

    st.session_state.cronologia = []
 
def invia():

    input_utente = st.session_state.domanda_utente

    if input_utente and catena is not None:  

        # 1. Salva la domanda dell'utente

        st.session_state.cronologia.append({"role": "user", "content": input_utente})

        # 2. Genera la risposta REALE usando LangChain

        with st.spinner("Il chatbot sta rispondendo..."):

            risposta_bot = catena.invoke(input_utente)

        # 3. Salva la risposta del bot

        st.session_state.cronologia.append({"role": "assistant", "content": risposta_bot})

        # 4. Resetta il campo di input

        st.session_state.domanda_utente = ""
 
# Mostra la cronologia a schermo

st.write("---")

for messaggio in st.session_state.cronologia:
 
    # Assegna l'avatar corretto in base al ruolo

    if messaggio["role"] == "user":

        icona = "Utente.png"  # Un'emoji per il pellegrino, oppure un percorso immagine "user.png"

    else:

        icona = "LOGO.png"  # Il logo del tuo chatbot (o "bot_icon.png")
 
    with st.chat_message(messaggio["role"], avatar=icona):

        st.write(messaggio["content"])

 
# Input dell'utente (Posizionato in fondo)

if catena is not None:

    st.text_input(

        "Label nascosta", 

        placeholder="Chiedi alla Via...", 

        key="domanda_utente", 

        on_change=invia,

        label_visibility="collapsed"

    )

 
