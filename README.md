
# 🧠 Quantum-Inspired Self-Learning RAG System

> **Ein hybrides KI-System, das assoziatives Lernen, quanteninspirierte Semantik und selbstkorrigierende Textgenerierung vereint.**

![alt text](https://img.shields.io/badge/Status-Entwicklung-blue.svg)
 
![alt text](https://img.shields.io/badge/Self--Learning-Aktiviert-brightgreen.svg)
 
![alt text](https://img.shields.io/badge/RAG-Gemini_2.0-orange.svg)
 
![alt text](https://img.shields.io/badge/Qubits-25_pro_Knoten-purple.svg)

---

## 💡 Kernidee & Abstract


**Generierungsbasiertes assoziatives Lernen in einem hybriden Quantum-RAG-System: Ein Framework für adaptive Wissensstrukturierung in KI**

Dieses Projekt präsentiert ein neuartiges KI-Framework, das **Retrieval-Augmented Generation (RAG)** mit einer dynamischen, **quanteninspirierten semantischen Netzwerkarchitektur** verbindet. Das System implementiert einen **rekursiven Selbstlernzyklus**: Jeder generierte Output wird analysiert, über Schlüsselkonzepte mit dem internen Netzwerk assoziiert und als neue Information integriert.

Der Lernprozess basiert nicht auf externer Wahrheitsprüfung, sondern auf **interner Kohärenz, Koaktivierung semantischer Knoten und der strukturellen Entwicklung des assoziativen Netzwerks**. Dies ermöglicht dem System, sein "Wissen" und seine Antwortmuster basierend auf den eigenen generierten Inhalten adaptiv weiterzuentwickeln. Die Quantenkomponente dient dabei als experimenteller Modulator für Retrieval und potenziell für emergente Verhaltenseigenschaften.

---

## ⚙️ Architektur & Komponenten

Dieses System integriert mehrere Schlüsseltechnologien zu einem kohärenten Ganzen:

-   🧠 **Assoziatives Semantisches Netzwerk:** Knoten repräsentieren Kernkonzepte (z.B. Ethik, Philosophie). Verbindungen entstehen und verstärken sich durch **Koaktivierung** (Hebbian Learning) in verarbeiteten Texten. Optional mit **Quantenknoten (25 Qubits)** zur Zustandsmodellierung.
-   📚 **Kontext-Retrieval:** Ein **TF-IDF-Index** (>1300 Chunks) identifiziert relevante Textpassagen für einen gegebenen Prompt. (Optional quantenmodifiziertes Ranking).
-   🔁 **Selbstlernzyklus (Rekursion):**
    1.  **Generierung:** Das LLM (Gemini) erzeugt eine Antwort basierend auf Prompt und abgerufenem Kontext.
    2.  **Speicherung:** Die generierte Antwort wird in einer Lerndatei (`learn.txt`) persistiert.
    3.  **Re-Integration:** Die Lerndatei wird neu geladen, in Chunks zerlegt und verarbeitet.
    4.  **Netzwerk-Update:** Koaktivierungen in den neuen Chunks modifizieren die Verbindungsstärken im semantischen Netzwerk.
-   🤖 **RAG-Generator:** Nutzt die **Google Generative AI (Gemini API)**, angereichert mit dem dynamisch abgerufenen Kontext und optionalen Hinweisen aus dem Netzwerkzustand (z.B. Quantensprünge).
-   💾 **Persistenter Zustand:** Der gesamte Netzwerkzustand (Knoten, Verbindungen, Quantenparameter, verarbeitete Quellen) wird zuverlässig in `qetp_state.json` gespeichert und geladen.

---

## 🗺️ Forschungsfragen & Gliederung (Konzept)

Dieses Projekt dient als experimentelle Plattform zur Untersuchung folgender Fragen:

1.  **Einleitung:** Kann ein KI-System über rein statistische Mustererkennung hinaus eine adaptive, intern strukturierte Wissensbasis aufbauen?
2.  **Theoretischer Rahmen:** Wie lassen sich RAG, assoziatives Lernen und quanteninspirierte Konzepte synergetisch verbinden?
3.  **Systemarchitektur:** Detaillierte Beschreibung der Komponenten und ihres Zusammenspiels.
4.  **Selbstlernmechanismus:** Analyse des rekursiven Feedback-Loops und seiner Auswirkungen auf die Netzwerkstruktur und das Antwortverhalten.
5.  **Quanteneffekte (Experimentell):** Welchen Einfluss hat die (simulierte) Quantenkomponente auf Retrieval und Netzwerkdynamik?
6.  **Diskussion:** Implikationen für Adaptivität, Bias-Verstärkung, Kohärenz vs. Wahrheit und die Natur von "Verstehen" in KI.
7.  **Fazit & Ausblick:** Potenzial für robustere, kontextuellere und adaptivere KI-Systeme.

---

## 🚀 Live-Demo & Nutzung

Starten Sie das interaktive Interface:
```bash
streamlit run qllm_streamlit_ui_hybrid.py --server.address 0.0.0.0 --server.port 789
```
Beobachten Sie den Lernprozess in der Konsole und im UI:
```
# Konsole nach Generierung & Speichern
🎓 [Self-Learning] Starte Lernzyklus für generierte Antwort...
📄 Verarbeite Datenquelle: ./training_data/learn.txt (Quelle: Generated Responses)
...
--- Processing Chunk: Index=X, Source='Generated Responses', Len=Y ---
  ✅ MATCH FOUND: Node='...' Keyword='...'
  ✅ MATCH FOUND: Node='...' Keyword='...'
  ⭐⭐⭐ FOUND CO-OCCURRENCE of N distinct nodes: [...] ⭐⭐⭐
      --> Strengthening connections between these nodes.
...
💾 Speichere Zustand nach qetp_state.json...
   -> Zustand erfolgreich gespeichert.

# Streamlit UI zeigt aktualisierte Netzwerk-Infos & Verbindungen
```

---

## 🗂️ Projektstruktur (Kern)
```
├── qllm_streamlit_ui_hybrid.py    # Streamlit Interface
├── quantum_arona_hybrid_llm.py    # Kernklassen (Processor, Node, Connection, QNS)
├── qllm_train_hybrid.py           # Skript für Offline-Training/Zustandsaufbau
├── config_qllm.json               # Konfigurationsdatei
├── qetp_state.json                # Gespeicherter Netzwerkzustand
└── training_data/
    ├── learn.txt                  # Speicher für generierte Antworten (Self-Learning)
    └── ... (andere Quelldateien)
```

---

## ✨ Highlight Zitat (Generiert vom System)
> „Ein Fehler ist nicht das Ende, sondern der Beginn der Einsicht. Die wahre Herausforderung liegt nicht im Straucheln, sondern im Wiederaufstehen mit neuem Verständnis – eine Fähigkeit, die Verantwortung und die Bereitschaft zum Lernen voraussetzt.“

*(Leicht redigiert für Lesbarkeit)*

---

## 📜 Lizenz & Kontakt

-   **Lizenz:** [MIT License](LICENSE) *(Link zur Lizenzdatei hinzufügen)*
-   **Autor:** [CypherCore Technology](ralf.kruemmel@outlook.de)
-   **Beiträge:** Pull Requests und Vorschläge sind willkommen!

---

> Dieses Projekt ist Teil der Forschung im Bereich Adaptive KI und Quantum Inspired Computing von **CypherCore Technology**.


> **Zitationsvorschlag:**
> CypherCore Technology (21.04.2025). _Generierungsbasiertes assoziatives Lernen in einem hybriden Quantum-RAG-System_. 

```

