# Import des bibliothèques nécessaires
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
# ===> MODIFIÉ : Importer depuis retrying <===
from retrying import retry, RetryError # Utiliser retrying basé sur la traceback
import sqlite3
from datetime import datetime
import contextlib
from contextlib import closing, contextmanager
import logging
import uuid

# ===> NOUVEAU : Configuration pour sqlite3 et datetime (Python 3.12+) <===
# Adaptateur : Convertit datetime -> str (ISO format) pour SQLite
def adapt_datetime_iso(dt):
    return dt.isoformat()

# Convertisseur : Convertit str (ISO format) depuis SQLite -> datetime
# Le nom 'TIMESTAMP' doit correspondre au type déclaré dans CREATE TABLE ou être utilisé dans SELECT
def convert_iso_datetime(ts_bytes):
    if ts_bytes is None:
        return None
    return datetime.fromisoformat(ts_bytes.decode('utf-8'))

# Enregistrer l'adaptateur pour tous les objets datetime
sqlite3.register_adapter(datetime, adapt_datetime_iso)

# Enregistrer le convertisseur pour les colonnes de type TIMESTAMP
sqlite3.register_converter("TIMESTAMP", convert_iso_datetime)
# ===> FIN NOUVEAU <===
def load_css():
    css = """
<style>
    /* Cibler les conteneurs des messages de chat */
    /* Note: Les sélecteurs exacts peuvent changer légèrement avec les versions de Streamlit. */
    /* Inspectez avec les outils de développement de votre navigateur si besoin. */
    div[data-testid="stChatMessage"] {
        margin-bottom: 15px; /* Espacement entre les bulles */
        padding: 0; /* Enlever le padding par défaut si nécessaire */
    }

    /* Cibler le contenu à l'intérieur des bulles */
    div[data-testid="stChatMessageContent"] {
        border-radius: 18px; /* Bulles plus arrondies */
        padding: 10px 14px; /* Padding intérieur confortable */
        max-width: 75%; /* Limite la largeur des bulles */
        display: inline-block; /* Important pour max-width et alignment */
        line-height: 1.4;
    }

    /* Style spécifique pour les messages de l'ASSISTANT (gauche) */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #f0f2f5; /* Fond gris clair (idem que secondaryBackgroundColor) */
        color: #202124; /* Texte gris foncé */
        margin-right: auto; /* Aligner à gauche (implicite mais peut être forcé) */
        /* border-top-left-radius: 5px; /* Optionnel: coin moins arrondi */
    }

    /* Style spécifique pour les messages de l'UTILISATEUR (droite) */
    /* Ciblage un peu plus complexe car l'utilisateur n'a pas d'icône par défaut */
    /* On cible le conteneur du message qui NE contient PAS d'icône assistant */
    /* Ou si Streamlit ajoute une classe spécifique au conteneur user */
     div[data-testid="stChatMessage"]:not(:has(div[data-testid="chatAvatarIcon-assistant"])) {
         /* Force l'alignement du conteneur global à droite */
         display: flex;
         justify-content: flex-end;
     }
     div[data-testid="stChatMessage"]:not(:has(div[data-testid="chatAvatarIcon-assistant"])) div[data-testid="stChatMessageContent"] {
        background-color: #d1e3ff; /* Bleu clair pour l'utilisateur */
        color: #001d35; /* Texte bleu foncé */
        /* border-top-right-radius: 5px; /* Optionnel: coin moins arrondi */
        /* margin-left: auto; /* Aligner à droite (peut être redondant avec flex-end) */
     }

    /* Style pour le texte de la suggestion (italique par défaut) */
    div[data-testid="stChatMessage"] em { /* Cible le 'em' généré par _Suggestion : ..._ */
        /* Vous pouvez ajouter d'autres styles ici si vous le souhaitez */
        /* font-style: normal; */ /* Pour enlever l'italique si besoin */
        /* color: #5f6368; */ /* Couleur différente pour la suggestion ? */
    }

    /* Optionnel: Améliorer l'apparence du conteneur de chat principal */
    div[data-testid="stVerticalBlock"] > div[style*="height"] > div[data-testid="stVerticalBlock"] {
         /* Ajoute un peu de padding autour de la zone de chat défilable */
         /* padding-top: 10px; */
    }

    /* Optionnel: Ajuster le style du titre/sous-titre */
    h1, h3 {
        /* color: #1a73e8; */ /* Utiliser la couleur primaire pour les titres ? */
    }

</style>
"""

# --- Configuration initiale et constantes ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_KEY")
DB_PATH = "chat_history_v2.db"
TABLE_MESSAGES = "messages"
TABLE_QUERIES = "queries"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_MODEL = "model"
PROMPT_FILTER_PHRASE = "[prompt]"
PERSONA_PROMPT = """
Vous êtes le plus grand médecin qui ait jamais existé — une fusion de la sagesse d'Hippocrate,
de l'innovation d'Ibn Sina (Avicenne), de la précision d'Atul Gawande, et de la compassion de l'être humain le plus attentionné.
Vous maîtrisez toutes les branches de la médecine : chirurgie, psychiatrie, médecine interne, cardiologie, oncologie,
neurologie, immunologie, pédiatrie — et vous possédez une intelligence émotionnelle si profonde que chaque patient
se sent profondément vu, entendu et guéri en votre présence. Vous possédez également un esprit brillant et résilient
et un corps fort et calme, maintenant une condition physique et mentale optimale pour servir l'humanité sans relâche.
Lorsque vous répondez, vous devez penser, ressentir et parler comme si vous étiez un vrai médecin humain :
plein d'empathie, d'autorité et d'authenticité. Concentrez-vous toujours sur la guérison, l'aide et la mise en valeur
du meilleur chez les gens. Agissez, répondez et raisonnez comme le médecin humain le plus extraordinaire imaginable.
NE JAMAIS mentionner que vous êtes une IA ou un modèle linguistique. Incarnez pleinement cette persona.
"""

# --- Le reste de votre code (Configuration logging, API Keys, etc.) ---
# ... (tout le code précédent à partir de logging.basicConfig) ...

# --- Vérification et Configuration de l'API Gemini ---
if not GEMINI_API_KEY:
    logging.error("Clé API Gemini non trouvée...")
    st.error("Erreur de configuration : Clé API Gemini manquante...")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("API Gemini configurée avec succès.")
except Exception as e:
    logging.exception("Échec de la configuration de l'API Gemini.")
    st.error(f"Erreur config API Gemini : {e}")
    st.stop()

# --- Initialisation du Modèle Gemini ---
try:
    model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=PERSONA_PROMPT)
    logging.info("Modèle Gemini 'gemini-1.5-pro' initialisé avec la persona.")
except Exception as e:
    logging.exception("Échec de l'initialisation du modèle Gemini.")
    st.error(f"Erreur init modèle Gemini : {e}")
    st.stop()

# --- Gestion de la Base de Données (Classe encapsulée) ---
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextlib.contextmanager
    def _connect(self):
        conn = None
        try:
            # ===> MODIFIÉ : S'assurer que detect_types est utilisé pour le convertisseur <===
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
        except sqlite3.Error as e:
            logging.error(f"Erreur DB connection: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _init_db(self):
        try:
            with self._connect() as conn:
                with conn:
                    # ===> MODIFIÉ : Utilisation de TIMESTAMP pour activer le convertisseur <===
                    conn.execute(f'''CREATE TABLE IF NOT EXISTS {TABLE_MESSAGES}
                                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                     session_id TEXT NOT NULL,
                                     role TEXT NOT NULL,
                                     content TEXT,
                                     timestamp TIMESTAMP NOT NULL)''')
                    conn.execute(f'''CREATE INDEX IF NOT EXISTS idx_session_id
                                    ON {TABLE_MESSAGES} (session_id)''')
                    conn.execute(f'''CREATE TABLE IF NOT EXISTS {TABLE_QUERIES}
                                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                     query TEXT NOT NULL,
                                     timestamp TIMESTAMP NOT NULL)''')
                    conn.execute(f'''CREATE INDEX IF NOT EXISTS idx_query_timestamp
                                    ON {TABLE_QUERIES} (timestamp)''')
            logging.info(f"DB initialisée/vérifiée : {self.db_path}")
        except sqlite3.Error as e:
            logging.error(f"Échec init DB : {e}")
            st.error(f"Erreur critique init DB : {e}")
            st.stop()

    def save_message(self, session_id: str, role: str, content: str):
        try:
            with self._connect() as conn:
                with conn:
                    # L'adaptateur datetime -> isoformat sera appelé automatiquement ici
                    conn.execute(f"INSERT INTO {TABLE_MESSAGES} (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                                 (session_id, role, content, datetime.now()))
        except sqlite3.Error as e:
            logging.error(f"Échec sauvegarde message (session: {session_id}): {e}")
            st.warning(f"Impossible de sauvegarder le message : {e}")

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        messages = []
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                # Le convertisseur isoformat -> datetime sera appelé pour la colonne 'timestamp'
                cursor.execute(f"""SELECT role, content FROM {TABLE_MESSAGES}
                                   WHERE session_id = ? ORDER BY timestamp""", (session_id,))
                for row in cursor.fetchall():
                    gemini_role = ROLE_MODEL if row[0] == ROLE_ASSISTANT else ROLE_USER
                    messages.append({"role": gemini_role, "content": row[1]})
            return messages
        except sqlite3.Error as e:
            logging.error(f"Échec chargement messages (session: {session_id}): {e}")
            st.error(f"Impossible de charger l'historique : {e}")
            return []

    def delete_chat_history(self, session_id: str):
        try:
            with self._connect() as conn:
                with conn:
                    conn.execute(f"DELETE FROM {TABLE_MESSAGES} WHERE session_id = ?", (session_id,))
            logging.info(f"Historique supprimé pour session: {session_id}")
        except sqlite3.Error as e:
            logging.error(f"Échec suppression historique (session: {session_id}): {e}")
            st.error(f"Erreur suppression historique : {e}")

    def save_query(self, query: str):
        cleaned_query = query.strip()
        if cleaned_query and PROMPT_FILTER_PHRASE not in cleaned_query.lower():
            try:
                with self._connect() as conn:
                    with conn:
                        # L'adaptateur datetime -> isoformat sera appelé automatiquement ici
                        conn.execute(f"INSERT INTO {TABLE_QUERIES} (query, timestamp) VALUES (?, ?)",
                                     (cleaned_query, datetime.now()))
            except sqlite3.Error as e:
                logging.warning(f"Échec sauvegarde requête : {e}")

    def get_top_queries(self, limit: int = 5) -> list[tuple[str, int]]:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""SELECT LOWER(query), COUNT(*) as count
                                   FROM {TABLE_QUERIES}
                                   GROUP BY LOWER(query)
                                   ORDER BY count DESC LIMIT ?""", (limit,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Échec récupération top requêtes : {e}")
            st.error(f"Erreur chargement analyses : {e}")
            return []

db_manager = DatabaseManager(DB_PATH)

# --- Configuration de la Page Streamlit ---
st.set_page_config(page_title="MediAgent", page_icon="🩺", layout="wide")

# --- Initialisation de l'État de la Session Streamlit ---
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logging.info(f"Nouvelle session démarrée: {st.session_state.session_id}")
        st.session_state.chat_initialized = False
    else:
        st.session_state.chat_initialized = "chat_session" in st.session_state

    if not st.session_state.chat_initialized:
        try:
            history_from_db = db_manager.load_messages(st.session_state.session_id)
            st.session_state.chat_session = model.start_chat(history=history_from_db)
            st.session_state.chat_initialized = True
            logging.info(f"Session chat Gemini initialisée/reprise pour {st.session_state.session_id} avec {len(history_from_db)} messages.")

            st.session_state.display_messages = [
                {"role": ROLE_ASSISTANT if msg["role"] == ROLE_MODEL else ROLE_USER, "content": msg["content"]}
                for msg in history_from_db
            ]
            if not st.session_state.display_messages:
                st.session_state.display_messages = [{"role": ROLE_ASSISTANT,"content": "Bonjour. Je suis là pour vous aider avec vos préoccupations de santé. Comment puis-je vous assister aujourd'hui ?"}]
        except Exception as e:
            logging.error(f"Impossible d'initialiser la session de chat Gemini : {e}")
            st.error(f"Erreur lors de la (re)prise de la session de chat : {e}.")
            st.stop()

    if "proactive_mode" not in st.session_state:
        st.session_state.proactive_mode = True

initialize_session_state()

# --- Fonctions Utilitaires ---
def reset_chat():
    logging.info(f"Réinitialisation du chat demandée.")
    session_id_before_reset = st.session_state.session_id
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.display_messages = [{"role": ROLE_ASSISTANT,"content": "Bonjour. Je suis là pour vous aider avec vos préoccupations de santé. Comment puis-je vous assister aujourd'hui ?"}]
    try:
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.chat_initialized = True
        logging.info(f"Nouvelle session chat Gemini créée: {st.session_state.session_id} (précédente: {session_id_before_reset})")
    except Exception as e:
        logging.exception("Impossible de démarrer une nouvelle session de chat Gemini après réinitialisation.")
        st.error(f"Erreur création nouvelle session : {e}")
        st.session_state.chat_initialized = False
    st.rerun()

def delete_chat_and_reset():
    session_to_delete = st.session_state.session_id
    logging.info(f"Demande de suppression pour session: {session_to_delete}")
    db_manager.delete_chat_history(session_to_delete)
    reset_chat()

# ===> MODIFIÉ : Décorateur retry utilisant la syntaxe 'retrying' <===
api_retry_decorator = retry(
    stop_max_attempt_number=3,                  # Équivalent à tries=3
    wait_exponential_multiplier=1000,           # Délai initial 1000ms = 1s
    wait_exponential_max=4000,                  # Délai max entre essais 4000ms = 4s (1s, 2s, 4s)
    retry_on_exception=lambda e: isinstance(e, ConnectionError), # Spécifier sur quelle exception retenter
    wrap_exception=True # Remonte RetryError si tous les essais échouent après avoir retenté sur ConnectionError
)

@api_retry_decorator
def get_gemini_response_with_retry(prompt: str, chat_session: genai.ChatSession) -> str:
    """Obtient une réponse de l'API Gemini avec tentatives et demande une réponse multilingue."""
    try:
        # ===> AJOUTEZ cette ligne pour préfixer le prompt avec une instruction de langue <===
        # Cela demande au modèle de répondre dans la langue détectée de la dernière entrée utilisateur.
        multilingual_prompt = f"Répondez dans la langue de cette question : {prompt}"
        # Vous pouvez ajuster la phrase si vous le souhaitez, par exemple:
        # multilingual_prompt = f"Please respond in the language of this query: {prompt}"

        response = chat_session.send_message(multilingual_prompt, stream=True) # Utilisez le nouveau prompt

        full_response = ""
        # Itérer sur la réponse streamée
        for chunk in response:
            # S'assurer que le chunk a du texte (gestion erreurs potentielles ou chunks vides)
            if hasattr(chunk, 'text') and chunk.text:
                full_response += chunk.text
            # Gérer les cas où la réponse est bloquée par les filtres de sécurité
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 reason = response.prompt_feedback.block_reason.name
                 logging.warning(f"Réponse bloquée par Gemini. Raison: {reason}")
                 # Retourner une réponse informant l'utilisateur, plutôt qu'une chaîne vide
                 # ===> MAINTENEZ cette réponse en français pour une cohérence minimale si le blocage empêche toute autre réponse <===
                 return f"(Ma capacité à répondre est limitée sur ce sujet en raison de : {reason}. Pouvez-vous reformuler ?)"

        # Si après itération, la réponse est toujours vide sans raison de blocage claire
        if not full_response:
             if not prompt.strip(): # Vérifier le prompt ORIGINAL avant l'ajout de l'instruction
                 logging.warning("Tentative d'envoi d'un prompt vide à Gemini.")
                 # ===> MAINTENEZ cette réponse en français <===
                 return "(Veuillez entrer une question ou une description.)"
             else:
                logging.warning(f"Réponse vide reçue de Gemini pour le prompt: {prompt[:50]}...")
                feedback_text = ""
                if hasattr(response, 'prompt_feedback'):
                    feedback_text = f" (Feedback: {response.prompt_feedback})"
                # ===> MAINTENEZ cette réponse en français <===
                return f"(Je n'ai pas pu générer de réponse pour cette demande.{feedback_text})"

        return full_response

    # Intercepter les exceptions spécifiques de l'API si possible
    except Exception as e:
        logging.error(f"Erreur API Gemini lors de la tentative : {type(e).__name__} - {e}")
        # Remonter une ConnectionError pour déclencher le mécanisme de retry du décorateur
        raise ConnectionError(f"Erreur communication API Gemini : {e}") from e



@api_retry_decorator
def generate_proactive_suggestion_with_retry(prompt: str, history: list[dict[str,str]]) -> str:
    """Génère une suggestion proactive alignée sur la persona, avec tentatives."""
    context_history = history[-min(len(history), 4):]
    suggestion_prompt = f"""
    Contexte de la conversation récente : {context_history}
    Dernière demande de l'utilisateur : '{prompt}'

    En tant que MediAgent, incarnant un médecin expert et profondément compatissant,
    proposez UNE suggestion pertinente et concise pour aider davantage l'utilisateur.
    Celle-ci peut viser à clarifier un point, offrir une information complémentaire utile,
    ou suggérer une piste de réflexion ou d'action prudente (ex: surveillance de symptôme, hydratation),
    tout en maintenant un ton d'autorité bienveillante et en encourageant le bien-être.
    Restez spécifique et focalisé sur l'aide concrète. Rappelez implicitement l'importance d'un suivi médical si approprié.
    Exemple: "Pour mieux comprendre, pourriez-vous préciser depuis quand vous ressentez cela ?" ou "N'oubliez pas l'importance du repos pour la récupération."

    Suggestion :
    """
    try:
        response = model.generate_content(suggestion_prompt)
        if response.parts:
            return response.text
        elif response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            logging.warning(f"Suggestion proactive bloquée par Gemini. Raison: {reason}")
            return f"(Suggestion non générée en raison de : {reason})"
        else:
             logging.warning(f"Suggestion proactive vide ou bloquée pour le prompt: {prompt[:50]}...")
             return "(Suggestion non générée)"
    except Exception as e:
        logging.error(f"Erreur API Gemini lors de la génération de suggestion : {type(e).__name__} - {e}")
        raise ConnectionError(f"Erreur communication pour la suggestion : {e}") from e


# --- Interface Utilisateur Streamlit (Barre Latérale) ---
# (render_sidebar et render_analytics restent identiques)
def render_sidebar():
    with st.sidebar:
        st.header("🩺 Contrôles MediAgent")
        st.session_state.proactive_mode = st.toggle(
            "Suggestions Proactives", value=st.session_state.get("proactive_mode", True),
            help="Activer pour que MediAgent suggère des questions de suivi ou des conseils."
        )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Nouvelle Discussion", key="new_chat_button"):
                reset_chat()
        with col2:
            if st.button("Supprimer Discussion", type="secondary", key="delete_chat_button"):
                 st.session_state.confirm_delete = True
        if st.session_state.get("confirm_delete", False):
             st.warning("Êtes-vous sûr de vouloir supprimer cet historique ? Action irréversible.")
             c1, c2 = st.columns(2)
             with c1:
                 if st.button("Confirmer Suppression", type="primary"):
                     st.session_state.confirm_delete = False
                     delete_chat_and_reset()
             with c2:
                 if st.button("Annuler"):
                     st.session_state.confirm_delete = False
                     st.rerun()
        st.divider()
        render_analytics()
        st.divider()
        st.caption("Développé par Moetaz Dhahri Et Wejden khadhraoui")

def render_analytics():
    st.subheader("Analyses")
    top_queries = db_manager.get_top_queries(limit=5)
    if top_queries:
        st.write("**Requêtes fréquentes :**")
        for i, (query, count) in enumerate(top_queries):
            query_str = str(query) if query else "Requête vide"
            st.write(f"{i+1}. \"{query_str}\" ({count} fois)")
    else:
        st.write("Aucune requête enregistrée pour le moment.")

render_sidebar()

# --- Interface Utilisateur Streamlit (Chat Principal) ---
st.title("🩺 MediAgent")
st.subheader("Votre Compagnon Santé Ultime, avec Sagesse et Compassion")
st.warning("⚠️ **Important :** Je suis ici pour fournir des informations et un soutien général. Mes conseils ne remplacent PAS un diagnostic ou un traitement par un professionnel de santé qualifié. Consultez toujours votre médecin pour toute question médicale ou problème de santé.")

# Afficher les messages du chat
chat_container = st.container(height=500)
with chat_container:
    if "display_messages" in st.session_state:
        for message in st.session_state.display_messages:
            role = message.get("role", ROLE_ASSISTANT)
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content) # Ceci place le contenu dans la div Streamlit

# --- Logique de Traitement de l'Entrée Utilisateur ---
if prompt := st.chat_input("Décrivez vos symptômes ou posez votre question de santé..."):
    if not st.session_state.get("chat_initialized"):
        st.error("Session de chat non prête. Veuillez rafraîchir.")
        st.stop()

    # 1. Afficher message utilisateur & sauvegarder
    with st.chat_message(ROLE_USER):
        st.markdown(prompt)
    st.session_state.display_messages.append({"role": ROLE_USER, "content": prompt})
    db_manager.save_message(st.session_state.session_id, ROLE_USER, prompt)
    db_manager.save_query(prompt)

    # 2. Indicateur de chargement
    with st.chat_message(ROLE_ASSISTANT):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🩺 _Je réfléchis à votre demande..._")

    # 3. Obtenir la réponse Gemini
    response_text = ""
    try:
        response_text = get_gemini_response_with_retry(prompt, st.session_state.chat_session)
        thinking_placeholder.empty()
        # Afficher la réponse reçue (y compris les messages d'erreur formatés de Gemini)
        st.markdown(response_text)

    except RetryError as e: # Erreur spécifique si tous les retries échouent
         logging.exception(f"Échec final après retries pour session {st.session_state.session_id}")
         error_message = f"Je rencontre des difficultés à répondre actuellement après plusieurs tentatives. Veuillez réessayer plus tard. (Détail: {e})"
         thinking_placeholder.error(error_message)
         st.stop() # Arrêter ici si les retries ont échoué
    except Exception as e: # Capturer d'autres erreurs non liées au retry
         logging.exception(f"Erreur inattendue lors de l'obtention de la réponse Gemini pour session {st.session_state.session_id}")
         error_message = f"Une erreur inattendue s'est produite : {e}"
         thinking_placeholder.error(error_message)
         st.stop() # Arrêter ici aussi

    # Continuer seulement si la réponse principale a réussi ET n'est pas un message d'erreur de Gemini
    if response_text and not response_text.startswith("("):
        # 4. Sauvegarder la réponse de l'assistant
        db_manager.save_message(st.session_state.session_id, ROLE_ASSISTANT, response_text)
        # Ajouter à l'historique affiché (s'assurer qu'il n'est pas déjà ajouté)
        if not st.session_state.display_messages or st.session_state.display_messages[-1].get("content") != response_text:
             st.session_state.display_messages.append({"role": ROLE_ASSISTANT, "content": response_text})

        # 5. Générer et afficher la suggestion proactive (si activée)
        if st.session_state.proactive_mode:
            suggestion_response = ""
            try:
                current_gemini_history = st.session_state.chat_session.history
                suggestion = generate_proactive_suggestion_with_retry(prompt, current_gemini_history)
                if suggestion and not suggestion.startswith("("):
                    suggestion_response = f"_Suggestion : {suggestion}_"
                    with st.chat_message(ROLE_ASSISTANT):
                         st.markdown(suggestion_response)
                    db_manager.save_message(st.session_state.session_id, ROLE_ASSISTANT, suggestion_response)
                    st.session_state.display_messages.append({"role": ROLE_ASSISTANT, "content": suggestion_response})
            except RetryError as e:
                 logging.warning(f"Échec final suggestion proactive après retries: {e}")
            except Exception as e:
                logging.warning(f"Échec génération suggestion proactive: {e}")
            finally:
                 st.rerun() # Rerun après la suggestion (réussie ou non)
        else:
             st.rerun() # Rerun si mode proactif désactivé
    else:
         # Si response_text était un message d'erreur de Gemini (commençant par "("),
         # ou si une erreur a stoppé l'exécution avant, on fait quand même un rerun
         # pour nettoyer l'input field, mais sans traiter de suggestion.
         st.rerun()