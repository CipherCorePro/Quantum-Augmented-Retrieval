# translate_books_robust.py
import os
import time
from deep_translator import GoogleTranslator

# Konfiguration
SOURCE_DIR = "books_markdown"
TARGET_DIR = "books_markdown_de"
os.makedirs(TARGET_DIR, exist_ok=True)

# Optional: Caching für Übersetzungen (spart Anfragen bei Wiederholungen)
translation_cache = {}

def is_translatable_line(line: str) -> bool:
    """Gibt True zurück, wenn die Zeile übersetzt werden soll."""
    line = line.strip()
    return bool(line) and not line.startswith("#")

def safe_translate(text: str) -> str:
    """Versucht, eine Zeile zu übersetzen, mit Fehlerbehandlung und Cache."""
    text = text.strip()
    if text in translation_cache:
        return translation_cache[text]

    try:
        translated = GoogleTranslator(source='auto', target='de').translate(text)
        translation_cache[text] = translated
        return translated
    except Exception as e:
        print(f"⚠️ Übersetzungsfehler bei Zeile: '{text[:40]}...' → {e}")
        return "[Übersetzungsfehler]"

def translate_markdown_file(filepath: str, target_path: str):
    print(f"🔄 Übersetze Datei: {os.path.basename(filepath)}")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    for idx, line in enumerate(lines):
        if is_translatable_line(line):
            translated = safe_translate(line)
            translated_lines.append(translated + "\n")
        else:
            translated_lines.append(line)

        # Optional: kleine Pause einbauen, um API-Rate-Limits zu vermeiden
        time.sleep(0.05)

        if idx % 50 == 0:
            print(f"  ... {idx} Zeilen verarbeitet")

    with open(target_path, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)

    print(f"✅ Fertig: {os.path.basename(target_path)}")

def translate_all_books():
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".md")]
    if not files:
        print("⚠️ Keine Markdown-Dateien gefunden.")
        return

    for filename in files:
        src = os.path.join(SOURCE_DIR, filename)
        tgt = os.path.join(TARGET_DIR, filename)
        translate_markdown_file(src, tgt)

if __name__ == "__main__":
    print("📘 Starte Übersetzung der Bücher...")
    translate_all_books()
    print(f"🎉 Alle Übersetzungen abgeschlossen. Zielordner: {TARGET_DIR}")
