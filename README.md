Chatbot Juridique Sénégalais – RAG Hybride + LLM + Voix 

Assistant intelligent spécialisé dans le droit du travail sénégalais

1. Présentation générale
Ce projet consiste en la conception et l’implémentation d’un chatbot juridique, capable de répondre aux questions relatives au droit du travail sénégalais, avec citations des articles officiels.
Le système repose sur une architecture RAG hybride (Retrieval-Augmented Generation) combinant :
 - Recherche hybride dans Milvus (dense + sparse)
 - Fusions de résultats via RRF (Reciprocal Rank Fusion)
 - Citations automatiques des extraits
 - Génération avec un LLM local via Ollama (Qwen2.5:3B)
 - Entrée vocale avec Vosk STT
 - Interface utilisateur Streamlit moderne et multimodale

Le chatbot est totalement offline et open source, fonctionnant localement même sur une machine CPU-only.

2. Objectifs du projet
  - Faciliter l’accès à l’information juridique au Sénégal
  - Proposer un assistant local, fiable et sourcé
  - Combiner la recherche vectorielle, lexicale et la génération contrôlée
  - Offrir une interface simple, rapide, voix + texte
  - Fonctionner entièrement hors-ligne sans dépendance cloud

3. Architecture technique

Pipeline global :
  - Extraction → Nettoyage → Chunking des documents juridiques
  - Encodage denses + sparse via BGEM3
  - Indexation hybride dans Milvus
  - Recherche dense + lexicale
  - Fusion via RRF
  - LLM local (Ollama) pour générer une réponse structurée
  - Citations automatiques
  - Interface Streamlit : texte + voix (Vosk)

4. Fonctionnalités principales
  Recherche Juridique Hybride
    - Embeddings denses (sémantique)
    - Sparse embeddings (lexical/TILDE)
    - Fusion RRF
    - Résultats plus pertinents qu’une recherche simple

  Citations automatiques
    - Chaque réponse générée contient les références du Code du travail :
    - Article
    - Section
    - Source

 Génération LLM locale
    - Modèle sélectionné : Qwen2.5:3B
    - Exécution via Ollama  
    - Prompt juridique spécialisé
    - Style clair, structuré, sans hallucinations (guidage strict)

 Entrée vocale (push-to-talk)
    - Reconnaissance vocale avec Vosk
    - Langue : français
    - Mode batch → transcription après fin de parole
    - Injection automatique dans la zone de texte

 Interface Streamlit
  - Mode clavier + micro 
  - Thème sombre professionnel
  - Historique des échanges
  - Indicateur d’enregistrement
  - Affichage propre (question + réponse regroupées)

5. Installation et utilisation
  1. Cloner le dépôt
  git clone https://github.com/username/chatbot-juridique
  cd chatbot-juridique
  
  2. Installer les dépendances
  pip install -r requirements.txt
  
  3. Installer les modèles
  
  Dans le dossier /models/README_MODELS.md, tu préciseras :
  
  modèle Vosk FR
  
  modèle Ollama (ollama pull qwen2.5:3b)
  
  4. Lancer le chatbot
  streamlit run src/app.py
  
6. Exemples de questions juridiques
  
  « Quels sont les droits du travailleur malade ? »
  
  « Comment calculer l’indemnité de licenciement ? »
  
  « Que dit le Code du travail sur la durée légale du travail ? »
  
  « Quelles sont les obligations de l’employeur en cas d’accident ? »
  
7. Approche méthodologique (Sprints)
  
  Sprint 1 : préparation de la base documentaire
  
  Sprint 2 : indexation hybride Milvus (dense + sparse)
  
  Sprint 3 : génération LLM + citations
  
  Sprint 4 : interface Streamlit multimodale

8. Limites actuelles
  - Temps de génération lent sur CPU
  - Modèles >3B impossibles à charger avec 8 Go RAM
  - TTS non fonctionnel pour l’instant (compatibilité Windows)
  - Pas encore de gestion d'historique conversationnel pour le LLM
  
9. Pistes d’amélioration
  - Ajouter un GPU (ou utiliser GPU cloud)
  - Passer à un LLM quantizé (Qwen 1.8B, Phi-3 Mini)
  - Intégrer Piper TTS une fois stabilisé
  - Ajouter une API REST pour usages externes
  - Développer une version mobile (Flutter ou React Native)
  - Ajouter une base SQL pour statistiques et logs
  
10. Auteur
  
  Projet réalisé par : Abdoulaye Ndour
  Étudiant en Big Data – Dakar Institute of Technology
