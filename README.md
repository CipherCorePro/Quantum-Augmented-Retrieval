# 🧠⚛️ Quantum-Arona RAG Explorer 🎭🎓

Willkommen beim Quantum-Arona RAG Explorer! Dieses Projekt ist ein Experimentierfeld für ein **hybrides System**, das klassische KI-Techniken mit quanten-inspirierten Konzepten und einem simulierten "emotionalen Zustand" verbindet, um Fragen auf Basis einer Wissensdatenbank zu beantworten.

Stell dir einen sehr belesenen Assistenten vor, der nicht nur Faktenwissen hat, sondern auch Stimmungen und Assoziationen nutzt, um Informationen zu finden und Antworten zu formulieren. Dieses Projekt simuliert einen solchen Assistenten.

---

## ✨ Hauptmerkmale

*   **🧠 Semantisches Netzwerk:** Baut ein Netzwerk aus Konzepten (Knoten) und deren Verbindungen auf, basierend auf Textdaten. Lernt, welche Themen oft gemeinsam auftreten.
*   **⚛️ Quanten-inspirierte Knoten:** Einige Konzept-Knoten nutzen simulierte Quantenschaltkreise (Parametrized Quantum Circuits - PQC), um ihr Verhalten zu modellieren. Dies bringt eine Ebene von Wahrscheinlichkeit und potenziellem "Quantenspringen" (plötzliche Zustandsänderungen) ins Spiel, die das Finden von Informationen beeinflussen können.
*   **📜 Retrieval-Augmented Generation (RAG):**
    1.  **Retrieval:** Findet die relevantesten Textabschnitte (Chunks) aus der Wissensdatenbank als Antwort auf eine Nutzerfrage (Prompt). Dieser Prozess wird durch die Aktivität im Netzwerk und den "emotionalen Zustand" beeinflusst.
    2.  **Generation:** Nutzt die gefundenen Textabschnitte und den internen Systemzustand (Konzepte, Emotionen, Quanten-Hinweise), um mithilfe eines großen Sprachmodells (LLM), hier Google Gemini, eine kohärente und kontextbezogene Antwort zu formulieren.
*   **🎭 Limbus Affektus (Emotionaler Modulator):** Ein spezieller Knoten simuliert einen globalen "emotionalen Zustand" des Systems basierend auf dem PAD-Modell (Pleasure, Arousal, Dominance). Dieser Zustand **moduliert aktiv** verschiedene Systemteile:
    *   🔍 **Retrieval:** Beeinflusst, wie streng nach relevanten Texten gesucht wird (Threshold) und wie Suchergebnisse gewichtet werden (Ranking Bias).
    *   🌡️ **Textgenerierung:** Passt die Kreativität/Zufälligkeit (Temperatur) des LLMs an.
    *   🎓 **Lernen:** Beeinflusst, wie schnell neue Verbindungen im Netzwerk gestärkt werden (Lernrate).
    *   ⚛️ **Quanteneffekte:** Moduliert den Einfluss von Quanten-Phänomenen (Varianz, Aktivierung) auf das Retrieval-Ranking.
*   **🎓 Selbstlernend:** Das System kann seine eigenen generierten (guten) Antworten nutzen, um sein Wissen zu erweitern und die Verbindungen im Netzwerk weiter zu lernen.
*   **⚙️ Konfigurierbar:** Viele Aspekte des Systems (Netzwerkstruktur, Quantenparameter, Lernraten, RAG-Einstellungen, Limbus-Einfluss) können über eine JSON-Datei angepasst werden.
*   **💾 Zustandspersistenz:** Der gesamte gelernte Zustand des Netzwerks (Knoten, Verbindungen, Parameter, verarbeitete Texte) kann gespeichert und geladen werden.
*   **🌐 Interaktive UI:** Eine Web-Oberfläche (basierend auf Streamlit) ermöglicht das einfache Interagieren mit dem System, das Anpassen von Parametern und das Beobachten des Systemzustands.

---

## 💡 Wie funktioniert es? (Konzeptuell)

1.  **Wissen aufbauen (Training):**
    *   Textdokumente (Bücher, Artikel etc.) werden in kleine Abschnitte (Chunks) zerlegt.
    *   Das System identifiziert Schlüsselkonzepte (z.B. "Ethik", "KI", "Bewusstsein") in diesen Chunks, die als Knoten im Netzwerk definiert sind.
    *   Wenn mehrere Konzepte gemeinsam in einem Chunk auftreten, wird eine Verbindung zwischen den entsprechenden Knoten im Netzwerk gestärkt. Dies geschieht über mehrere Durchläufe (Epochen).
    *   Einige Knoten sind als "Quantenknoten" definiert. Ihre interne "Berechnung" basiert auf einer Simulation von Quantenschaltungen, was zu probabilistischem Verhalten führt.
    *   Der "Limbus Affektus"-Knoten beobachtet die allgemeine Aktivität im Netzwerk und passt seinen eigenen PAD-Zustand (Pleasure, Arousal, Dominance) an.
2.  **Fragen beantworten (Inferenz/RAG):**
    *   **Frage (Prompt):** Der Nutzer stellt eine Frage.
    *   **Netzwerkaktivierung:** Die Frage aktiviert relevante Konzept-Knoten im Netzwerk. Aktivierung breitet sich über die gelernten Verbindungen aus. Quantenknoten werden simuliert. Der Limbus-Zustand wird aktualisiert.
    *   **Retrieval (Suche):** Basierend auf den aktivierten Knoten und *moduliert durch den Limbus-Zustand* sucht das System nach den Text-Chunks, die am besten zur Frage passen. Quanteneffekte und der Limbus-Zustand können das Ranking beeinflussen.
    *   **Kontextaufbau:** Die gefundenen Chunks, die aktivierten Konzepte, eventuelle Quantensprung-Hinweise und der *skalierte Limbus-Zustand* werden als Kontext für das LLM vorbereitet.
    *   **Generation (Antwort):** Das LLM (Google Gemini) erhält die Nutzerfrage und den aufbereiteten Kontext. Es wird angewiesen, eine natürliche Antwort zu formulieren und dabei den Kontext (inkl. der "Stimmung" aus dem Limbus) zu berücksichtigen. Die *durch den Limbus-Zustand modulierte* Temperatur beeinflusst den Stil der Antwort.
    *   **Selbstlernen (Optional):** Wenn die generierte Antwort als nützlich erachtet wird, wird sie zur Wissensbasis hinzugefügt und das Netzwerk lernt daraus neue Verbindungen oder stärkt bestehende (mit *Limbus-modulierter Lernrate*).
3.  **Zustand:** Der gesamte Zustand des Netzwerks (Knotenparameter, Verbindungsstärken, Limbus-Zustand, verarbeitete Chunks) wird gespeichert, damit das System beim nächsten Start dort weitermachen kann, wo es aufgehört hat.

---

## 📁 Projektstruktur

```
.
├── quantum_arona_hybrid_llm.py  # Hauptlogik: Klassen für Knoten, Quantensystem, Prozessor, RAG, Limbus etc.
├── qllm_train_hybrid.py         # Skript zum Trainieren/Initialisieren des Netzwerks aus Daten.
├── qllm_streamlit_ui_hybrid.py  # Interaktive Web-Oberfläche mit Streamlit.
├── config_qllm.json             # Konfigurationsdatei (Parameter, Knoten, Dateien).
├── qetp_state.json              # Gespeicherter Zustand des Netzwerks (wird vom Training erstellt/aktualisiert).
├── requirements.txt             # Liste der Python-Abhängigkeiten.
├── training_data/               # Verzeichnis für Trainingsdokumente.
│   ├── ethics_ai.md
│   ├── philosophy_basics.txt
│   └── ... (weitere Textdateien)
│   └── learn.txt                # Datei für selbstgelernte Antworten.
└── README.md                    # Diese Datei.
```

---

## 🚀 Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/CipherCorePro/Quantum-Augmented-Retrieval.git
    cd Quantum-Augmented-Retrieval
    ```
2.  **Virtuelle Umgebung (Empfohlen):**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Stelle sicher, dass `requirements.txt` existiert und die nötigen Pakete enthält: `numpy`, `scikit-learn`, `google-generativeai`, `streamlit`, `pandas`, `tqdm` (optional))*
4.  **Google Gemini API Key:**
    *   Besorge dir einen API Key vom [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Setze den Key als Umgebungsvariable:
        ```bash
        # Windows (Eingabeaufforderung)
        set GEMINI_API_KEY=DEIN_API_KEY
        # Windows (PowerShell)
        $env:GEMINI_API_KEY="DEIN_API_KEY"
        # Linux/Mac
        export GEMINI_API_KEY=DEIN_API_KEY
        ```
    *   Alternativ kannst du den Key als Streamlit Secret speichern, wenn du die App über Streamlit Cloud bereitstellst. Die App versucht, dies als Fallback zu nutzen.
5.  **Trainingsdaten:**
    *   Lege deine Trainings-Textdateien (z.B. `.txt`, `.md`) im Verzeichnis `training_data/` ab.
    *   Passe die Liste `training_files` in `config_qllm.json` an, um auf deine Dateien zu verweisen. Die Datei `learn.txt` wird automatisch für das Self-Learning erstellt.

---

## ▶️ Benutzung

1.  **(Optional) Konfiguration anpassen:** Bearbeite `config_qllm.json`, um Parameter wie die zu verwendenden Trainingsdateien, die Definition semantischer Knoten, Lernraten, Quantenparameter oder Limbus-Einstellungen anzupassen (siehe Abschnitt "Konfiguration").
2.  **Netzwerk trainieren/initialisieren:** Führe das Trainingsskript aus. Dies liest die Trainingsdaten, baut das Netzwerk auf und speichert den initialen Zustand.
    ```bash
    python qllm_train_hybrid.py [Optionen]
    ```
    *   `-c DATEI`: Pfad zur Konfigurationsdatei (default: `config_qllm.json`)
    *   `-s DATEI`: Pfad zur Zustandsdatei (default: `qetp_state.json`)
    *   `-f` oder `--force-rebuild`: Ignoriert eine vorhandene Zustandsdatei und startet komplett neu.
3.  **Interaktive UI starten:** Starte die Streamlit Web-App.
    ```bash
    streamlit run qllm_streamlit_ui_hybrid.py --server.address 0.0.0.0 --server.port 789
    ```
    *   Öffne die angezeigte URL (z.B. `http://<Deine-IP>:789`) in deinem Browser.
    *   In der App kannst du:
        *   Den Zustand laden/speichern.
        *   Prompts eingeben und Antworten generieren lassen.
        *   Den Netzwerkstatus (inkl. Limbus PAD-Zustand) einsehen.
        *   Temporär Konfigurationsparameter (z.B. Limbus-Einflussfaktoren) anpassen.
        *   Netzwerkschritte manuell simulieren.
        *   Gelernte Verbindungen inspizieren.

---

## ⚙️ Konfiguration (`config_qllm.json`)

Diese JSON-Datei steuert das Verhalten des Systems. Wichtige Abschnitte:

*   `training_files`: Liste der Pfade zu deinen Textdateien für das Training.
*   `semantic_nodes`: Definition der Konzept-Knoten und zugehöriger Schlüsselwörter.
*   `chunk_size`, `chunk_overlap`: Wie die Texte zerlegt werden.
*   `connection_learning_rate`, `connection_decay_rate`: Wie schnell Verbindungen gelernt und vergessen werden.
*   `use_quantum_nodes`, `default_num_qubits`, `simulation_n_shots`: Steuerung der Quanten-inspirierten Knoten.
*   `enable_rag`, `generator_model_name`, `generator_temperature`: Einstellungen für die Textgenerierung mit Gemini.
*   `enable_self_learning`, `self_learning_file_path`: Steuerung des Selbstlern-Mechanismus.
*   **`limbus_...` Parameter:**
    *   `limbus_num_qubits`, `limbus_emotion_decay`, `limbus_*_sensitivity`: Grundparameter für den Limbus-Knoten selbst.
    *   **`limbus_influence_...`**: **Steuern, wie stark der Limbus-Zustand andere Teile beeinflusst!**
        *   `..._prompt_level`: Wie stark der PAD-Zustand im LLM-Prompt erwähnt wird.
        *   `..._temperature_*`: Wie Arousal/Dominance die LLM-Temperatur verändern.
        *   `..._threshold_*`: Wie Arousal/Pleasure den Retrieval-Schwellenwert ändern.
        *   `..._ranking_bias_*`: Wie Pleasure das Ranking beeinflusst.
        *   `..._learning_rate_*`: Wie Emotionen die Lernrate beeinflussen.
        *   `..._variance_penalty`, `..._activation_boost`: Wie Emotionen die Quanten-Effekt-Parameter im Retrieval ändern.
    *   `limbus_min_*`, `limbus_max_*`: Grenzen für modulierte Werte (z.B. Temperatur, Threshold).

---

## 💾 Zustandsdatei (`qetp_state.json`)

Diese Datei ist entscheidend! Sie speichert den gesamten gelernten Zustand des Systems:

*   Alle Knoten (inkl. ihrer internen Parameter, auch Quantenparameter).
*   Alle gelernten Verbindungen und ihre Gewichte.
*   Alle verarbeiteten Text-Chunks und ihre zugeordneten Knoten.
*   Den aktuellen Limbus-Zustand.
*   Die Konfiguration, mit der der Zustand erstellt wurde.
*   Metadaten wie verarbeitete Quellen.

Wenn du das Training (`qllm_train_hybrid.py`) ohne `--force-rebuild` startest oder die UI (`qllm_streamlit_ui_hybrid.py`) lädst, wird versucht, aus dieser Datei zu laden, um den Lernfortschritt beizubehalten.

---

## 🔭 Zukünftige Ideen

*   Anbindung an echte Quantenhardware (statt Simulation).
*   Ausgefeiltere Quantenschaltkreise für Knoten.
*   Verbesserte Visualisierung des Netzwerks und der Aktivierungsdynamik.
*   Integration weiterer LLMs neben Gemini.
*   Feingranularere Steuerung der Limbus-Modulation (z.B. nicht-lineare Effekte).
*   Evaluierungsmetriken für die Qualität von Retrieval und Generierung.

---

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte erstelle einen Issue, um Bugs zu melden oder neue Features zu diskutieren. Pull Requests sind ebenfalls gerne gesehen.

---

## 📜 Lizenz

Dieses Projekt steht unter der [MIT Lizenz](LICENSE).
