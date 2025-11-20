import requests
import json
import sys
import queue
import sounddevice as sd
import vosk
from hybrid_search import hybrid_search

# ==============================
# 1. Configuration Ollama
# ==============================
OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma2:2b"
MODEL_NAME = "qwen2.5:3b"

# ==============================
# 2. Vérification Ollama
# ==============================
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if MODEL_NAME in model_names:
                print(f"Modèle {MODEL_NAME} disponible")
                return True
            else:
                print(f"Modèle {MODEL_NAME} non trouvé")
                print(f"Modèles disponibles: {model_names}")
                return False
        else:
            print("Ollama ne répond pas")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Erreur connexion Ollama: {e}")
        print("Assurez-vous qu'Ollama est démarré avec: ollama serve")
        return False

# ==============================
# 3. Prompt juridique
# ==============================
def build_legal_prompt(question, passages):
    context = ""
    for i, p in enumerate(passages, 1):
        text_excerpt = p['text'][:500] + "..." if len(p['text']) > 500 else p['text']
        context += f"[{i}] {text_excerpt}\n"
        context += f"    Source: {p['source']} - Section {p['chunk_index']}\n\n"

    prompt = f"""Tu es un assistant juridique spécialisé dans le droit du travail sénégalais.

CONTEXTE JURIDIQUE :
{context}

QUESTION : {question}

INSTRUCTIONS :
- Réponds de manière précise et structurée
- Base-toi UNIQUEMENT sur les extraits fournis ci-dessus
- Cite tes sources
- Si l'information n'est pas dans le contexte, dis-le clairement
- Utilise un langage simple mais juridiquement correct

RÉPONSE :"""
    return prompt

# ==============================
# 4. Génération avec Ollama
# ==============================
def generate_answer_ollama(question, max_tokens=400, temperature=0.0):
    try:
        print("🔎 Recherche des passages pertinents...")
        passages = hybrid_search(question, top_k=3, alpha=0.5, return_passages=True)

        if not passages:
            return "Aucune information pertinente trouvée dans la base de données juridique."

        prompt = build_legal_prompt(question, passages)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return f"Erreur API Ollama: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Erreur inattendue: {e}"

# ==============================
# 5. STT avec Vosk
# ==============================
MODEL_PATH = "models\\fr\\vosk-model-small-fr-0.22"
try:
    vosk_model = vosk.Model(MODEL_PATH)
except Exception as e:
    print(f"Erreur chargement modèle Vosk: {e}")
    sys.exit(1)

q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen_and_transcribe():
    samplerate = 16000
    device = None
    rec = vosk.KaldiRecognizer(vosk_model, samplerate)

    print("Parlez maintenant (Ctrl+C pour arrêter)...")

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                           device=device, dtype="int16",
                           channels=1, callback=callback):
        text = ""
        try:
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"):
                        text = res["text"]
                        break
        except KeyboardInterrupt:
            print("\nArrêt manuel")
        return text

# ==============================
# 6. Interface interactive
# ==============================
def interactive_chat():
    print("=" * 70)
    print("CHATBOT JURIDIQUE SÉNÉGALAIS")
    print("=" * 70)
    print("Commandes: 'quit' pour quitter, 'help' pour l'aide")
    print("Modes: clavier (k) ou micro (m)")
    print()

    if not check_ollama():
        return

    while True:
        try:

            mode = input("\nUtiliser clavier ou micro ? (k/m): ").strip().lower()

            if mode == "m":
                question = listen_and_transcribe()
                print(f"\nVous avez dit: {question}")
            else:
                question = input("\nVotre question juridique: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'stop']:
                print("Merci d'avoir utilisé le chatbot !")
                break

            if question.lower() == 'help':
                print("""
    Exemples de questions:
    • Quels sont les droits du travailleur malade ?
    • Comment calculer l'indemnité de licenciement ?
    • Que dit le code du travail sur les congés payés ?
    """)
                continue

            print("Génération de la réponse...")
            answer = generate_answer_ollama(question)

            print("\nRéponse:\n" + "-"*50)
            print(answer)
            print("-"*50)

        except KeyboardInterrupt:
            print("\n\nArrêt du chatbot (Ctrl+C détecté)")
            break
        except Exception as e:
            print(f"\nErreur: {e}")


# ==============================
# 7. Point d'entrée
# ==============================
if __name__ == "__main__":
    interactive_chat()
