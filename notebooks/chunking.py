import fitz
import json
import re
from pathlib import Path

# =============================
# 1. Fonctions utilitaires
# =============================

def extract_text_from_pdf(pdf_path):
    """Extrait tout le texte d'un PDF page par page"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_text_from_txt(txt_path):
    """Charge le texte depuis un fichier .txt (OCR déjà fait)"""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def clean_text(text):
    """Nettoyage de base du texte"""
    text = re.sub(r"\s+", " ", text)  # enlever espaces multiples
    text = re.sub(r"\n+", "\n", text)  # normaliser sauts de lignes
    return text.strip()

def split_into_chunks(text, chunk_size=512, overlap=50):
    """Découpe un texte en chunks avec overlap"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # reculer pour garder un peu de contexte
    
    return chunks

# =============================
# 2. Traitement principal
# =============================

def process_document(input_path, source_name, output_json, chunk_size=512, overlap=50, is_txt=False):
    # Extraire texte
    if is_txt:  # si on travaille avec un fichier OCR déjà transformé en txt
        raw_text = extract_text_from_txt(input_path)
    else:       # sinon, extraction directe du PDF
        raw_text = extract_text_from_pdf(input_path)

    cleaned_text = clean_text(raw_text)

    # Découper en chunks
    chunks = split_into_chunks(cleaned_text, chunk_size, overlap)

    # Structurer avec métadonnées
    data = []
    for i, chunk in enumerate(chunks):
        entry = {
            "id": f"{source_name}_{i+1}",
            "source": source_name,
            "chunk_index": i+1,
            "text": chunk
        }
        data.append(entry)

    # Sauvegarde JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{len(chunks)} chunks sauvegardés dans {output_json}")


# =============================
# 3. Exécution sur les documents
# =============================

if __name__ == "__main__":
    # Définir chemins
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    code_pdf = data_dir / "codedutravail.pdf"
    manuel_txt = data_dir / "manuel_ocr.txt" 

    # Traiter Code du travail (PDF texte)
    process_document(code_pdf, "Code_du_travail", data_dir / "code_travail_chunks.json", 
                     chunk_size=512, overlap=50, is_txt=False)

    # Traiter Manuel du travailleur (texte OCR)
    process_document(manuel_txt, "Manuel_du_travailleur", data_dir / "manuel_chunks.json", 
                     chunk_size=512, overlap=50, is_txt=True)
