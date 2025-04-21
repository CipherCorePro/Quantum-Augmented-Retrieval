# 🧠 Quantum-Augmented RAG Explorer

> **Ein hybrides, selbstlernendes Retrieval-Generierungssystem mit quanteninspirierter semantischer Strukturierung**

![Quantum RAG Logo](https://img.shields.io/badge/Status-Entwicklung-blue.svg) ![Self-Learning](https://img.shields.io/badge/Self--Learning-Aktiviert-brightgreen.svg) ![RAG](https://img.shields.io/badge/RAG-Gemini_2.0-orange.svg) ![Qubits](https://img.shields.io/badge/Qubits-25_pro_Knoten-purple.svg)

---

## 🔍 Abstract

**Titel**:  
**Semantisch rekursives Lernen in einem hybriden Quantum-RAG-System: Ein Schritt zur erfahrungsbasierten Wissensakkumulation in generativen KI-Modellen**

Dieses Projekt stellt ein neuartiges hybrides KI-System vor, das Retrieval-Augmented Generation (RAG) mit einer quanteninspirierten, semantisch dynamischen Architektur kombiniert. Jeder generierte Output wird automatisch gespeichert, rekursiv zerlegt, semantisch klassifiziert und fließt zurück in das Langzeitgedächtnis. Dies erlaubt erfahrungsbasiertes Lernen – unabhängig von der Wahrheit – orientiert an Bedeutungsvernetzung, Knotenkoaktivierung und heuristischer Relevanz.

---

## 🧱 Architekturkomponenten

- 🧠 **Semantisches Knotennetz** mit 25-Qubit-Knoten
- 📚 **TF-IDF Retrieval Index** mit >1300 Chunks
- 🪢 **Ko-Okkurrenzlogik** zur Bedeutungsvernetzung
- 🔁 **Selbstlernprozess** mit rekursiver Eingliederung
- 🤖 **RAG-Generator** über Gemini 2.0 Flash (Google Generative AI)
- 💾 **Persistenter Zustandsspeicher** (`qetp_state.json`)

---

## 🗂️ Gliederung

### 1️⃣ Einleitung
- Motivation: Warum klassische KI nicht reicht
- Ziel: Bedeutungsbasiertes, selbsttransformierendes Wissen

### 2️⃣ Theoretischer Rahmen
- RAG-Prinzipien
- Quanteninspirierte Bedeutungsräume
- Hebb’sches Lernen und Selbstreflexion

### 3️⃣ Systemarchitektur
- Überblick, Knoten, Lernschritte, Speicherprozesse

### 4️⃣ Selbstlernfunktion
- Chunking > Klassifikation > Ko-Okkurrenz > Integration

### 5️⃣ Beispielanalyse
- Beispielhafte Generierung + Live-Lernverhalten + Nachwirkungen

### 6️⃣ Diskussion
- Ethische, erkenntnistheoretische und semantische Dimensionen

### 7️⃣ Fazit & Ausblick
- Zukunft semantisch adaptiver KI

---

## 🧪 Live-Demo (Auszug)
```bash
streamlit run qllm_streamlit_ui_hybrid.py --server.address 0.0.0.0 --server.port 789
```

```
🎓 [Self-Learning] Starte Lernzyklus für generierte Antwort...
📄 Verarbeite Datenquelle: ./training_data/learn.txt
⭐⭐⭐ FOUND CO-OCCURRENCE of ['Philosophie', 'Bewusstsein', 'Technologie']
🔄 Aktualisiere TF-IDF Index... ✅
```

---

## 📦 Projektstruktur (Auszug)
```
├── qllm_streamlit_ui_hybrid.py         # Haupt-UI mit RAG + Self-Learning
├── quantum_arona_hybrid_llm.py         # Kernmodell mit Quantenknoten
├── training_data/
│   └── learn.txt                       # Lernspeicher für generierte Inhalte
├── checkpoints_arona_sample1_nq2/      # Persistente Checkpoints
└── qetp_state.json                     # Zustands-Dump für Self-Learning-Netz
```

---

## 🧠 Zitat aus dem System
> „Freie Gedanken sind wie Sterne, die in der Dunkelheit leuchten … Indoktrinierte Gedanken hingegen sind wie gebundene Wege – sie führen zwar voran, aber innerhalb vorgegebener Grenzen.“

---

## 📎 Lizenz & Kontakt

- 🔐 MIT Lizenz  
- 📫 Kontakt: [CypherCore Technology](mailto:info@cyphercore.tech)  
- 🤝 Beiträge willkommen!

---

> Dieses Projekt ist Teil der Quantum-Cognition-Reihe von **CypherCore Technology**.

---

> Für wissenschaftliche Zitation:  
> _CypherCore Technology (2025): Semantisch rekursives Lernen in einem hybriden Quantum-RAG-System._

