import requests
import re
from pathlib import Path

# ----------- Einstellungen -----------
BOOKS = {
    "Frankenstein": {
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
        "filename": "frankenstein.md",
        "tags": {
            "life": "#TECHNOLOGIE",
            "creation": "#TECHNOLOGIE",
            "ethics": "#ETHIK",
            "soul": "#BEWUSSTSEIN",
            "science": "#TECHNOLOGIE",
            "death": "#BEWUSSTSEIN",
        }
    },
    "Zarathustra": {
        "url": "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
        "filename": "zarathustra.md",
        "tags": {
            "power": "#WILLE_ZUR_MACHT",
            "man": "#ÜBERMENSCH",
            "god": "#ÜBERMENSCH",
            "truth": "#META_COGNITIO",
            "wisdom": "#META_COGNITIO",
            "spirit": "#SEELE",
        }
    },
    "Platon_Dialoge": {
        "url": "https://www.gutenberg.org/cache/epub/1656/pg1656.txt",
        "filename": "platon_dialoge.md",
        "tags": {
            "soul": "#SEELE",
            "truth": "#ERKENNTNIS",
            "virtue": "#PHILOSOPHIE",
            "death": "#SEELE",
            "knowledge": "#ERKENNTNIS",
        }
    },
}
SAVE_DIR = Path("books_markdown")
SAVE_DIR.mkdir(exist_ok=True)

# ----------- Funktionen -----------
def download_book(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_main_text(raw_text: str) -> str:
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*", raw_text)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*", raw_text)
    if not start or not end:
        raise ValueError("Textgrenzen nicht gefunden.")
    return raw_text[start.end():end.start()].strip()

def split_into_chapters(text: str) -> list[str]:
    chapters = re.split(r'\n(?=CHAPTER\s+\w+|Chapter\s+\w+)', text)
    return [c.strip() for c in chapters if c.strip()]

def assign_tags(text: str, tag_dict: dict) -> list[str]:
    tags = set()
    for keyword, tag in tag_dict.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
            tags.add(tag)
    return list(tags)

def write_markdown(book_title: str, chapters: list[str], tag_dict: dict, filename: str):
    filepath = SAVE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        for i, chapter in enumerate(chapters, 1):
            tags = assign_tags(chapter, tag_dict)
            f.write(f"# Kapitel {i}\n")
            if tags:
                f.write(" " + " ".join(tags) + "\n")
            f.write("\n" + chapter.strip() + "\n\n")
    print(f"✅ Markdown-Datei gespeichert: {filepath.absolute()}")

# ----------- Hauptablauf -----------
def main():
    for title, info in BOOKS.items():
        print(f"\n📘 Lade herunter: {title}")
        raw_text = download_book(info["url"])
        main_text = extract_main_text(raw_text)
        chapters = split_into_chapters(main_text)
        write_markdown(title, chapters, info["tags"], info["filename"])

if __name__ == "__main__":
    main()
