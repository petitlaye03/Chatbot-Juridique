import pytesseract
from pdf2image import convert_from_path
import os
from multiprocessing import Pool
import time
import fitz  # PyMuPDF


# ===============================
# ÉTAPE 1: CONFIGURATION
# ===============================

# Spécifier le chemin de tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Configuration
PDF_PATH = "data/LE_MANUEL_DU_TRAVAILLEUR.pdf"
OUTPUT_PATH = "data/manuel_ocr.txt"
TEMP_DIR = "data/temp_pages"
LANG = "fra"
DPI = 300 

# Traitement par lots pour éviter les erreurs mémoire
BATCH_SIZE = 50  
NUM_PROCESSES = 6  

# ===============================
# ÉTAPE 2: FONCTIONS
# ===============================

def create_temp_directory():
    """Créer le dossier temporaire pour les images"""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    print(f"Dossier temporaire créé: {TEMP_DIR}")

def process_page(args):
    """Fonction pour traiter une page (sera exécutée en parallèle)"""
    page_image, page_num = args
    try:
        # OCR sur la page
        text = pytesseract.image_to_string(page_image, lang=LANG)
        print(f"Page {page_num + 1} traitée")
        return page_num, text
    except Exception as e:
        print(f"Erreur page {page_num + 1}: {e}")
        return page_num, f"[ERREUR PAGE {page_num + 1}]"

def get_pdf_page_count(pdf_path):
    """Obtenir le nombre total de pages du PDF"""
    # Méthode rapide pour compter les pages
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count

def process_batch(pdf_path, start_page, end_page, batch_num, total_batches):
    """Traiter un lot de pages"""
    print(f"Lot {batch_num}/{total_batches} (pages {start_page}-{end_page})")
    
    try:
        # Convertir seulement les pages de ce lot
        pages = convert_from_path(
            pdf_path, 
            dpi=DPI,
            first_page=start_page,
            last_page=end_page
        )
        
        print(f"{len(pages)} pages converties pour le lot {batch_num}")
        
        # Préparer les données pour le parallélisme
        page_data = [(page, start_page - 1 + i) for i, page in enumerate(pages)]
        
        # Traitement parallèle du lot
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(process_page, page_data)
        
        # Nettoyer immédiatement les images de la mémoire
        del pages
        
        return results
        
    except Exception as e:
        print(f"Erreur lot {batch_num}: {e}")
        return []

def cleanup_temp_files():
    """Nettoyer les fichiers temporaires"""
    try:
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                os.remove(os.path.join(TEMP_DIR, file))
            os.rmdir(TEMP_DIR)
        print("Fichiers temporaires nettoyés")
    except Exception as e:
        print(f"Erreur nettoyage: {e}")

# ===============================
# ÉTAPE 3: SCRIPT PRINCIPAL
# ===============================

def main():
    start_time = time.time()
    print("DÉMARRAGE DE L'OCR PARALLÈLE (TRAITEMENT PAR LOTS)")
    print("=" * 60)
    
    # Créer dossier temporaire
    create_temp_directory()
    
    print(f"Analyse du PDF: {PDF_PATH}")
    
    # ÉTAPE 3.1: Obtenir le nombre total de pages
    try:
        total_pages = get_pdf_page_count(PDF_PATH)
        print(f"PDF analysé: {total_pages} pages détectées")
        
        # Pour test: décommenter la ligne suivante
        # total_pages = 20
        
    except Exception as e:
        print(f"Erreur analyse PDF: {e}")
        return
    
    # ÉTAPE 3.2: Calculer les lots
    total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Division en {total_batches} lots de {BATCH_SIZE} pages maximum")
    print(f"Utilisation de {NUM_PROCESSES} processus parallèles")
    
    # ÉTAPE 3.3: Traitement lot par lot
    all_results = []
    
    for batch_num in range(1, total_batches + 1):
        start_page = (batch_num - 1) * BATCH_SIZE + 1
        end_page = min(batch_num * BATCH_SIZE, total_pages)
        
        print(f"\n{'='*40}")
        print(f"TRAITEMENT LOT {batch_num}/{total_batches}")
        print(f"{'='*40}")
        
        batch_results = process_batch(PDF_PATH, start_page, end_page, batch_num, total_batches)
        all_results.extend(batch_results)
        
        processed_pages = len(all_results)
        progress = (processed_pages / total_pages) * 100
        print(f"Progression globale: {processed_pages}/{total_pages} pages ({progress:.1f}%)")
        
        # Pause courte entre les lots pour libérer la mémoire
        time.sleep(1)
    
    # ÉTAPE 3.4: Assemblage des résultats
    print(f"\nAssemblage de {len(all_results)} pages...")
    
    # Trier les résultats par numéro de page
    all_results.sort(key=lambda x: x[0])
    
    # Extraire seulement le texte
    all_text = [result[1] for result in all_results]
    
    # ÉTAPE 3.5: Correction de l'encodage et sauvegarde
    try:
        # Corriger l'encodage des caractères français
        corrected_text = []
        for text in all_text:
            # Corrections communes des caractères mal encodés
            corrected = text.replace("Ã©", "é").replace("Ã¨", "è").replace("Ã ", "à")
            corrected = corrected.replace("Ã´", "ô").replace("Ã®", "î").replace("Ã¢", "â")
            corrected = corrected.replace("Ã§", "ç").replace("Ã¹", "ù").replace("Ãª", "ê")
            corrected = corrected.replace("Ã«", "ë").replace("Ã¯", "ï").replace("Ã»", "û")
            corrected = corrected.replace("Ã", "À").replace("Ã‰", "É").replace("Ã", "Ô")
            corrected_text.append(corrected)
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n\n=== NOUVELLE PAGE ===\n\n".join(corrected_text))
        
        print(f"Fichier sauvegardé avec encodage corrigé: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")
        return
    
    # ÉTAPE 3.6: Nettoyage et statistiques
    cleanup_temp_files()
    
    end_time = time.time()
    duration = end_time - start_time
    pages_per_minute = total_pages / (duration / 60)
    
    print("\n" + "=" * 60)
    print("TRAITEMENT TERMINÉ!")
    print(f"Pages traitées: {total_pages}")
    print(f"Lots traités: {total_batches}")
    print(f"Temps total: {duration:.1f} secondes ({duration/60:.1f} minutes)")
    print(f"Vitesse: {pages_per_minute:.1f} pages/minute")
    print(f"Fichier de sortie: {OUTPUT_PATH}")
    print(f"Taille estimée du fichier: ~{len(all_text) * 500 / 1024 / 1024:.1f} MB")

# ===============================
# ÉTAPE 4: EXÉCUTION
# ===============================

if __name__ == "__main__":
    # Protection pour Windows multiprocessing
    main()