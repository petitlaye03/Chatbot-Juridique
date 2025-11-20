import streamlit as st
import sys
import queue
import sounddevice as sd
import vosk
import json
import os
import time
import threading
from datetime import datetime

# Import direct de votre fonction
from rag_generation import generate_answer_ollama, check_ollama

# Configuration
MODEL_PATH = "models\\fr\\vosk-model-small-fr-0.22"

# Configuration de la page
st.set_page_config(
    page_title="Chatbot Juridique Sénégalais", 
    page_icon="⚖️",
    layout="wide"
)

# CSS personnalisé avec thème sombre
st.markdown("""
<style>
    /* Fond principal sombre */
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
        color: #ffffff;
    }
    
    .user-message {
        background-color: #2d3748;
        border-left-color: #ff6b6b;
        color: #ffffff;
    }
    
    .bot-message {
        background-color: #1a365d;
        border-left-color: #4ecdc4;
        color: #ffffff;
    }
    
    .error-message {
        background-color: #742a2a;
        border-left-color: #fc8181;
        color: #fed7d7;
    }
    
    .recording-indicator {
        background-color: #744210;
        color: #faf089;
        border: 1px solid #d69e2e;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Sidebar sombre */
    .css-1d391kg {
        background-color: #2d3748;
    }
    
    /* Zone de texte sombre */
    .stTextArea textarea {
        background-color: #2d3748;
        color: #ffffff;
        border: 1px solid #4a5568;
    }
    
    /* Métriques */
    .metric-container {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Transcription box */
    .transcription-box {
        background-color: #2d5016;
        color: #c6f6d5;
        border: 1px solid #48bb78;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classe STT simplifiée - version thread-safe
class StreamlitSTT:
    def __init__(self):
        self.model = None
        self.is_available = False
        
        try:
            if os.path.exists(MODEL_PATH):
                self.model = vosk.Model(MODEL_PATH)
                self.is_available = True
                print("Modèle STT chargé avec succès")
        except Exception as e:
            print(f"Erreur chargement modèle STT: {e}")
    
    def transcribe_audio_continuous(self):
        """Enregistrer et transcrire l'audio - durée illimitée avec arrêt manuel"""
        if not self.is_available:
            return "Modèle STT non disponible"
        
        # Créer un nouveau recognizer pour ce thread
        try:
            rec = vosk.KaldiRecognizer(self.model, 16000)
        except Exception as e:
            return f"Erreur initialisation recognizer: {e}"
        
        q = queue.Queue()
        
        def audio_callback(indata, frames, timestamp, status):  # Renommé time -> timestamp
            if status:
                print(f"Status audio: {status}")
            q.put(bytes(indata))
        
        try:
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback
            ):
                print("Démarrage enregistrement continu...")
                transcription = ""
                final_result = ""
                
                # Enregistrer jusqu'à ce que le fichier stop soit créé
                while not os.path.exists("stop_recording.txt"):
                    try:
                        data = q.get(timeout=0.1)
                        
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            if result.get("text"):
                                final_result = result["text"]
                                print(f"Résultat final: {final_result}")
                        else:
                            partial = json.loads(rec.PartialResult())
                            if partial.get("partial"):
                                transcription = partial["partial"]
                                # Écrire la transcription partielle dans un fichier
                                with open("partial_transcription.txt", "w", encoding="utf-8") as f:
                                    f.write(transcription)
                    
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Erreur traitement audio: {e}")
                        break
                
                # Supprimer le fichier stop
                try:
                    os.remove("stop_recording.txt")
                except:
                    pass
                
                result_text = final_result if final_result else transcription
                print(f"Transcription terminée: '{result_text}'")
                return result_text.strip() if result_text else "Aucune parole détectée"
                
        except Exception as e:
            print(f"Erreur enregistrement: {e}")
            return f"Erreur enregistrement: {str(e)[:50]}"

# Variables globales pour éviter les problèmes de threading
STT_INSTANCE = StreamlitSTT()
TRANSCRIPTION_FILE = "temp_transcription.txt"

# Initialisation du session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_conversation_index" not in st.session_state:
    st.session_state.current_conversation_index = None

if "check_transcription" not in st.session_state:
    st.session_state.check_transcription = 0

if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Fonction de transcription avec arrêt manuel
def record_audio_background():
    """Enregistrer l'audio en arrière-plan avec arrêt manuel"""
    def record_thread():
        print("Thread d'enregistrement démarré")
        
        # Créer fichier de statut
        with open("recording_status.txt", "w") as f:
            f.write("recording")
        
        try:
            result = STT_INSTANCE.transcribe_audio_continuous()
            print(f"Transcription thread terminée: {result}")
            
            # Écrire le résultat dans un fichier
            with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
                f.write(result if result else "Aucune transcription")
                
        except Exception as e:
            print(f"Erreur dans thread: {e}")
            with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
                f.write(f"Erreur: {str(e)[:50]}")
        finally:
            # Supprimer le fichier de statut
            try:
                os.remove("recording_status.txt")
            except:
                pass

    if not os.path.exists("recording_status.txt"):
        thread = threading.Thread(target=record_thread)
        thread.daemon = True
        thread.start()

def stop_recording():
    """Arrêter l'enregistrement en créant un fichier signal"""
    with open("stop_recording.txt", "w") as f:
        f.write("stop")
    print("Signal d'arrêt envoyé")

def get_partial_transcription():
    """Récupérer la transcription partielle en cours"""
    if os.path.exists("partial_transcription.txt"):
        try:
            with open("partial_transcription.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except:
            return None
    return None

def get_transcription_result():
    """Vérifier s'il y a une nouvelle transcription"""
    if os.path.exists(TRANSCRIPTION_FILE):
        try:
            with open(TRANSCRIPTION_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            os.remove(TRANSCRIPTION_FILE)  # Supprimer après lecture
            return content
        except:
            return None
    return None

def is_recording():
    """Vérifier si l'enregistrement est en cours"""
    return os.path.exists("recording_status.txt")

# Interface principale
def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Chatbot Juridique Sénégalais</h1>
        <p>Assistant spécialisé en droit du travail</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations et historique
    with st.sidebar:
        # HISTORIQUE DES DISCUSSIONS
        st.subheader("Historique des discussions")
        
        if st.session_state.chat_history:
            for i, conversation in enumerate(st.session_state.chat_history):
                # Titre de la conversation basé sur la question
                conv_title = conversation['question'][:40] + "..." if len(conversation['question']) > 40 else conversation['question']
                
                # Bouton pour afficher cette conversation
                if st.button(
                    f"{conv_title}",
                    key=f"conv_{i}",
                    help=f"Cliquez pour afficher cette discussion"
                ):
                    st.session_state.current_conversation_index = i
                    st.rerun()
            
            st.divider()
            
            # Bouton nouvelle discussion
            if st.button("Nouvelle discussion"):
                st.session_state.current_conversation_index = None
                st.rerun()
            
            # Bouton effacer tout
            if st.button("Effacer tout l'historique"):
                st.session_state.chat_history = []
                st.session_state.current_conversation_index = None
                # Nettoyer les fichiers temporaires
                for file in ["temp_transcription.txt", "recording_status.txt"]:
                    try:
                        if os.path.exists(file):
                            os.remove(file)
                    except:
                        pass
                st.rerun()
        else:
            st.info("Aucune discussion pour le moment")
        
        st.divider()

        st.header("Informations système")
        
        # Statut Ollama
        if check_ollama():
            st.success("Ollama connecté")
        else:
            st.error("Ollama déconnecté")
            st.warning("Démarrez Ollama avec: ollama serve")
        
        # Statut STT
        if STT_INSTANCE.is_available:
            st.success("STT Vosk prêt")
        else:
            st.error("Modèle STT manquant")
            st.info("Téléchargez le modèle Vosk français")
        
        st.divider()
        
        # Options
        st.subheader("Options")
        
        # Test de transcription simplifié
        if st.button("Test Transcription (3s)"):
            if STT_INSTANCE.is_available:
                st.info("Test de 3 secondes - parlez maintenant...")
                # Créer un test rapide avec durée fixe
                rec = vosk.KaldiRecognizer(STT_INSTANCE.model, 16000)
                q = queue.Queue()
                
                def callback(indata, frames, timestamp, status):  # Renommé time -> timestamp
                    q.put(bytes(indata))
                
                with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback):
                    import time as time_module  # Import explicite pour éviter conflit
                    start = time_module.time()
                    result = ""
                    while time_module.time() - start < 3:
                        try:
                            data = q.get(timeout=0.1)
                            if rec.AcceptWaveform(data):
                                res = json.loads(rec.Result())
                                if res.get("text"):
                                    result = res["text"]
                        except:
                            continue
                
                st.write(f"Résultat test: {result if result else 'Aucune parole détectée'}")
            else:
                st.error("STT non disponible")
    
    # Interface de saisie - VERSION AVEC FICHIER TEMPORAIRE
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Vérifier s'il y a une nouvelle transcription
        transcription_result = get_transcription_result()
        
        if transcription_result:
            st.markdown(f"""
            <div style="background-color: #2d5016; color: #c6f6d5; border: 2px solid #48bb78; 
                       padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>🎤 TRANSCRIPTION:</strong><br><br>
                <span style="font-size: 16px; font-weight: bold;">{transcription_result}</span><br><br>
                <small>Copiez ce texte dans le champ ci-dessous</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Forcer le rafraîchissement pour maintenir l'affichage
            st.session_state.check_transcription = st.session_state.check_transcription + 1
        
        # Zone de saisie
        question_input = st.text_area(
            "Votre question juridique:",
            height=100,
            placeholder="Ex: Quels sont les droits du travailleur malade ?",
            help="Utilisez le bouton micro puis copiez la transcription qui apparaît ci-dessus",
            key=f"question_area_{st.session_state.check_transcription}"
        )
    
    with col2:
        st.write("")  
        
        # Bouton micro avec gestion d'état - Version manuelle start/stop
        currently_recording = is_recording()
        
        if not currently_recording:
            if st.button("🎤 Démarrer", disabled=not STT_INSTANCE.is_available, key="start_rec"):
                record_audio_background()
                st.rerun()
        else:
            # Afficher transcription partielle en temps réel si disponible
            partial = get_partial_transcription()
            if partial:
                st.markdown(f"""
                <div style="background-color: #744210; color: #faf089; border: 2px solid #d69e2e; 
                           padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-size: 12px;">
                    {partial}
                </div>
                """, unsafe_allow_html=True)
            
            # Indicateur d'enregistrement avec bouton stop
            st.markdown("""
            <div style="background-color: #744210; color: #faf089; border: 2px solid #d69e2e; 
                       padding: 1rem; border-radius: 8px; text-align: center; font-weight: bold;
                       animation: pulse 1.5s infinite;">
                ENREGISTREMENT...<br>
                PARLEZ MAINTENANT
            </div>
            """, unsafe_allow_html=True)
            
            # Bouton arrêter
            if st.button("Arrêter", key="stop_rec"):
                stop_recording()
                time.sleep(1)  # Laisser le temps au thread de finir
                st.rerun()
            
            # Auto-refresh pour afficher transcription partielle
            time.sleep(0.5)
            st.rerun()
    
    # Bouton d'envoi
    col_send1, col_send2, col_send3 = st.columns([1, 2, 1])
    with col_send2:
        if st.button("Envoyer la question", disabled=not bool(question_input.strip())):
            if question_input.strip():
                # Générer la réponse
                with st.spinner("Génération de la réponse en cours..."):
                    try:
                        # Utilisation directe de votre fonction
                        answer = generate_answer_ollama(question_input.strip())
                        
                        # Ajouter la conversation (question + réponse ensemble)
                        st.session_state.chat_history.append({
                            "question": question_input.strip(),
                            "answer": answer,
                            "timestamp": datetime.now()
                        })
                        
                        # Afficher cette nouvelle conversation
                        st.session_state.current_conversation_index = len(st.session_state.chat_history) - 1
                        st.success("Réponse générée!")
                        
                    except Exception as e:
                        error_msg = f"Erreur lors de la génération: {str(e)}"
                        
                        # Ajouter quand même avec erreur
                        st.session_state.chat_history.append({
                            "question": question_input.strip(),
                            "answer": error_msg,
                            "error": True,
                            "timestamp": datetime.now()
                        })
                        
                        st.session_state.current_conversation_index = len(st.session_state.chat_history) - 1
                        st.error(error_msg)
                
                st.rerun()
    
    # Affichage de la conversation sélectionnée
    st.subheader("Conversation")
    
    if st.session_state.current_conversation_index is not None:
        # Afficher la conversation sélectionnée
        conv = st.session_state.chat_history[st.session_state.current_conversation_index]
        
        # Bloc unique contenant question ET réponse
        is_error = conv.get("error", False)
        
        if is_error:
            st.markdown(f"""
            <div class="chat-message error-message">
                <strong>Question:</strong><br>
                {conv['question']}<br><br>
                <strong>Erreur:</strong><br>
                {conv['answer']}<br>
                <small>{conv['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Question:</strong><br>
                {conv['question']}<br><br>
                <strong>Réponse:</strong><br>
                {conv['answer']}<br>
                <small>{conv['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Posez une question pour démarrer une conversation!")

if __name__ == "__main__":
    main()