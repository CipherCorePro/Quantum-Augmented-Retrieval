# -- coding: utf-8 --

# Filename: qllm_streamlit_ui_hybrid.py
# Description: Interaktives Interface für Quantum-Arona Hybrid LLM (RAG) mit Self-Learning.
# Version: 1.2 - Integrated LimbusAffektus Display
# Author: [CipherCore Technology] & Gemini & Your Input & History Maker

import streamlit as st
import json
import os
import sys
import pandas as pd
import traceback
import time
import numpy as np
from collections import defaultdict, deque
from typing import List, Optional, Dict, Any # Ergänzt Dict, Any

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
try:
    # Importiere alle benötigten Klassen
    from quantum_arona_hybrid_llm import (
        QuantumEnhancedTextProcessor,
        TextChunk,
        Node,
        LimbusAffektus, # Importiere LimbusAffektus
        Connection,     # Importiere Connection (wird intern verwendet)
        QuantumNodeSystem # Importiere QNS (wird intern verwendet)
    )
except ImportError:
    st.error(
        """
    **FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren.**

    Stellen Sie sicher, dass:
    1. Die Datei `quantum_arona_hybrid_llm.py` existiert und alle Klassen enthält.
    2. Sie sich im selben Verzeichnis wie dieses Skript befindet ODER im Python-Pfad liegt.
        """
    )
    st.stop()

# === Hilfsfunktion für Verbindungsanzeige ===
def show_connections_table(connections_data: List[Dict[str, Any]]) -> None:
    """Zeigt Verbindungsdaten als Tabelle an."""
    if connections_data:
        df_connections = pd.DataFrame(connections_data)
        # Runde Gewicht für bessere Anzeige
        if "Gewicht" in df_connections.columns:
             df_connections["Gewicht"] = df_connections["Gewicht"].round(4)
        st.dataframe(df_connections, use_container_width=True)
    else:
        st.info("Keine Verbindungen mit den aktuellen Filterkriterien gefunden.")

# === Zustandslade-Funktion ===
#@st.cache_resource(ttl=3600) # Caching kann bei Objektänderungen problematisch sein
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`.")
        return None
    try:
        # print(f"DEBUG: Attempting to load state from {state_path}")
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            # print(f"DEBUG: State loaded successfully, Processor ID: {id(processor)}")
            return processor
        else:
            st.error(f"❌ Fehler beim Laden des Zustands aus `{state_path}`.")
            return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler beim Laden des Zustands: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

# === Streamlit GUI ===
st.set_page_config(page_title="QAE RAG Explorer", layout="wide")

# --- Header ---
col1_title, col2_title = st.columns([1, 6])
with col1_title:
     # Optional: Logo oder Icon hier
     # st.image("path/to/your/logo.png", width=80)
     st.markdown("🧠", unsafe_allow_html=True) # Emoji als einfacher Platzhalter
with col2_title:
     st.title("Quantum-Augmented RAG Explorer")
     st.caption("Hybrides Quanten-Retrieval & Textgenerierungs-Interface mit Lernzyklus")

# --- Initialisierung des Session State ---
if 'processor' not in st.session_state: st.session_state['processor'] = None
if 'state_file_path' not in st.session_state: st.session_state['state_file_path'] = "qetp_state.json"
if 'last_retrieved_chunks' not in st.session_state: st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state: st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state: st.session_state['last_prompt'] = ""
# Debug-Checkbox wird direkt im UI gesetzt/abgefragt

# --- Lade initialen Zustand, wenn noch nicht vorhanden ---
if st.session_state.processor is None:
    # print("DEBUG: Initial processor load attempt.")
    st.session_state.processor = load_processor_state(st.session_state.state_file_path)
    # Kein st.rerun() hier, um Endlosschleife bei Ladefehler zu vermeiden

# === Seitenleiste: Steuerung & Infos ===
with st.sidebar:
    st.header("⚙️ Steuerung & Status")

    # --- Zustand Laden/Speichern ---
    current_state_path = st.text_input(
        "Pfad zur Zustandsdatei",
        value=st.session_state['state_file_path'],
        key="state_path_input",
        help="Der Pfad zur JSON-Datei, die den Netzwerkzustand enthält."
    )
    col1_load, col2_save = st.columns(2)
    with col1_load:
        if st.button("🔄 Laden", key="load_state_button", help="Lädt den Zustand aus der Datei."):
            with st.spinner(f"Lade Zustand aus `{current_state_path}`..."):
                st.session_state['processor'] = None # Reset state before loading
                st.session_state['last_retrieved_chunks'] = []
                st.session_state['last_generated_response'] = None
                processor_instance = load_processor_state(current_state_path)
                if processor_instance:
                    st.session_state['processor'] = processor_instance
                    st.session_state['state_file_path'] = current_state_path
                    st.success("Zustand geladen.")
                    st.rerun() # Force rerun to update UI after successful load
                else:
                    st.error("Laden fehlgeschlagen.")

    # Zugriff auf den (potenziell neu geladenen) Prozessor
    processor = st.session_state.get('processor')

    with col2_save:
        save_disabled = processor is None
        if st.button("💾 Speichern", key="save_state_button", disabled=save_disabled, help="Speichert den aktuellen Netzwerkzustand in die Datei."):
            if processor:
                with st.spinner("Speichere Zustand..."):
                    processor.save_state(st.session_state['state_file_path'])
                st.success("Zustand gespeichert.")
            else:
                st.warning("Kein Prozessor zum Speichern geladen.")

    st.markdown("---")

    # --- Netzwerk Status & Simulation ---
    if processor is not None:
        st.subheader("📊 Netzwerk Übersicht")
        try:
            # Hole aktuelle Summary Daten
            summary = processor.get_network_state_summary()
            col1_info, col2_info, col3_info = st.columns(3)
            with col1_info:
                st.metric("Knoten", summary.get('num_nodes', 0))
            with col2_info:
                st.metric("Verbindungen", summary.get('total_connections', 0))
            with col3_info:
                st.metric("Chunks", summary.get('num_chunks', 0))

            with st.expander("Details anzeigen (JSON)", expanded=False):
                st.json(summary) # Volle Summary im Expander

            # Statusanzeigen
            if not summary.get('rag_enabled'):
                st.warning("RAG (Gemini) ist deaktiviert.", icon="⚠️")
            if summary.get('self_learning_enabled'):
                st.success("Self-Learning ist aktiviert.", icon="🎓")
            else:
                st.info("Self-Learning ist deaktiviert.", icon="❌")

            # Simulations-Button
            if st.button("➡️ Schritt simulieren", key="simulate_button", help="Führt einen Simulationsschritt durch (Aktivierung, Decay, Emotion)."):
                with st.spinner("Simuliere Netzwerk..."):
                    processor.simulate_network_step(decay_connections=True)
                st.success("Simulationsschritt abgeschlossen.")
                st.rerun() # Update UI nach Simulation

        except Exception as e:
            st.error(f"Fehler beim Abrufen der Netzwerkinfo: {e}")

        # --- Parameter & Metriken ---
        st.markdown("---")
        st.subheader("🔧 Parameter & Metriken")

        with st.expander("Konfiguration anpassen", expanded=False):
            # 1. n_shots-Slider
            n_shots_val = processor.config.get("simulation_n_shots", 50)
            n_shots_new = st.slider(
                "Quanten-Messungen (n_shots)", min_value=1, max_value=200, value=n_shots_val, step=1,
                help="Beeinflusst die Genauigkeit/Stabilität der Quantenknoten-Aktivierung."
            )
            if n_shots_new != n_shots_val:
                processor.config["simulation_n_shots"] = n_shots_new
                st.info(f"n_shots auf {n_shots_new} geändert (temporär). Zum Speichern '💾 Speichern' klicken.")

            # Optional: Weitere Parameter hier hinzufügen (z.B. Decay Rate, Lernrate)
            # decay_rate_val = processor.config.get("connection_decay_rate", 0.001)
            # decay_rate_new = st.slider("Connection Decay Rate", 0.0, 0.1, decay_rate_val, 0.0001, format="%.4f")
            # if decay_rate_new != decay_rate_val:
            #     processor.config["connection_decay_rate"] = decay_rate_new
            #     st.info(f"Decay Rate auf {decay_rate_new:.4f} geändert (temporär).")

        # --- Aktuelle Metriken anzeigen ---
        st.caption("Aktueller Netzwerkzustand")
        # Durchschnittsaktivierung
        activations = [getattr(n, 'activation', 0.0) for n in processor.nodes.values()]
        valid_activations = [a for a in activations if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if valid_activations:
            avg_act = sum(valid_activations) / len(valid_activations)
            st.metric("Ø Knoten-Aktivierung", f"{avg_act:.4f}")
            # Balkendiagramm für Einzelaktivierungen
            valid_labels = [n.label for n in processor.nodes.values() if isinstance(getattr(n, 'activation', None), (float, np.number)) and np.isfinite(getattr(n, 'activation', None))]
            if len(valid_labels) == len(valid_activations):
                 df_activations = pd.DataFrame({"Aktivierung": valid_activations}, index=valid_labels)
                 st.bar_chart(df_activations, height=150)
        else: st.info("Keine gültigen Aktivierungsdaten.")

        # Limbus Affektus Zustand anzeigen (wenn vorhanden und korrekt implementiert)
        limbus_node = processor.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             st.caption("Globaler emotionaler Zustand (PAD)")
             emotion_state = getattr(limbus_node, 'emotion_state', {})
             if emotion_state:
                  col_p, col_a, col_d = st.columns(3)
                  with col_p: st.metric("Pleasure", f"{emotion_state.get('pleasure', 0.0):.3f}", delta=None)
                  with col_a: st.metric("Arousal", f"{emotion_state.get('arousal', 0.0):.3f}", delta=None)
                  with col_d: st.metric("Dominance", f"{emotion_state.get('dominance', 0.0):.3f}", delta=None)
             else: st.info("Emotionszustand nicht verfügbar.")

    else:
        # Meldung, wenn kein Prozessor geladen ist
        st.info("ℹ️ Kein Prozessor-Zustand geladen.")
        st.warning("Bitte laden Sie eine Zustandsdatei oder führen Sie das Trainingsskript aus.")

# === Hauptbereich: Prompt & Ergebnisse ===
processor = st.session_state.get('processor') # Stelle sicher, dass wir die aktuellste Version haben
if processor is not None:
    st.header("💬 Prompt & Antwort")
    prompt = st.text_area(
        "Geben Sie einen Prompt oder eine Frage ein:",
        height=100,
        key="prompt_input_main",
        value=st.session_state.get("last_prompt", ""),
        help="Ihre Frage oder Ihr Thema für das RAG-System."
    )

    # Button zum Generieren
    generate_disabled = not processor.rag_enabled or not prompt.strip()
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=generate_disabled, type="primary"):
        st.session_state['last_prompt'] = prompt # Speichere aktuellen Prompt
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None
        st.rerun() # Force rerun to show spinner immediately

    # Zeige Ergebnisse erst nach Button-Klick (oder wenn sie schon im State sind)
    if st.session_state.last_prompt and st.session_state.last_generated_response is None: # Nur ausführen, wenn generiert werden soll
        if processor.rag_enabled and st.session_state.last_prompt.strip():
            start_process_time = time.time()
            success_flag = False
            with st.spinner("🧠 Generiere Antwort (Retrieval + LLM)..."):
                try:
                    # Hole den letzten Prompt aus dem State
                    current_prompt = st.session_state.last_prompt
                    # Führe Generierung durch
                    generated_response = processor.generate_response(current_prompt)
                    st.session_state['last_generated_response'] = generated_response # Speichere Ergebnis
                    # Prüfe Gültigkeit
                    is_valid = (generated_response and
                                not generated_response.startswith("[Fehler") and
                                not generated_response.startswith("[Antwort blockiert"))
                    if is_valid: success_flag = True

                    # Hole Kontext für Anzeige (redundant, aber einfacher für UI)
                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(current_prompt)
                    st.success(f"Antwort generiert in {time.time() - start_process_time:.2f}s")

                    # Speichere Zustand nach Self-Learning
                    if processor.self_learning_enabled and success_flag:
                        # Dieser Teil wird nun nach der Anzeige der Antwort ausgeführt,
                        # da st.rerun() den Ablauf hier unterbricht.
                        # Besser: Speichern als separaten Button anbieten oder im Hintergrund?
                        # Fürs Erste lassen wir es hier, aber es wird erst nach dem nächsten rerun wirksam.
                         with st.spinner("💾 Speichere Zustand nach Lernzyklus (im nächsten Schritt)..."):
                              processor.save_state(st.session_state['state_file_path'])
                         st.success("Zustand nach Lernzyklus gespeichert (wird nach Aktualisierung wirksam).")

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state['last_generated_response'] = "[Fehler bei der Generierung]"
            # Erneutes Rerun, um die Ergebnisse anzuzeigen
            st.rerun()

        elif not processor.rag_enabled:
             st.error("Textgenerierung (RAG) ist nicht aktiviert.")
             # Optional: Nur Retrieval durchführen und anzeigen
             # st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(st.session_state.last_prompt)
             # st.rerun()

    # --- Anzeige der Ergebnisse ---
    if st.session_state.get('last_generated_response'):
         st.markdown("---")
         st.subheader("💡 Generierte Antwort")
         st.markdown(st.session_state['last_generated_response'])

    retrieved_chunks = st.session_state.get('last_retrieved_chunks', [])
    if retrieved_chunks:
         st.markdown("---")
         with st.expander(f"📚 Kontext ({len(retrieved_chunks)} Textabschnitt{'e' if len(retrieved_chunks) != 1 else ''})", expanded=False):
              for i, chunk in enumerate(retrieved_chunks):
                  st.markdown(f"**[{i+1}] Quelle:** `{chunk.source}` (`Index: {chunk.index}`)")
                  if hasattr(chunk, 'activated_node_labels') and chunk.activated_node_labels:
                      nodes_str = ", ".join(f"`{lbl}`" for lbl in chunk.activated_node_labels)
                      st.markdown(f"**Knoten:** {nodes_str}")
                  st.markdown(f"> _{chunk.text[:300]}..._") # Gekürzter Text für Übersicht
                  if st.button(f"Volltext {i+1}", key=f"chunk_text_{i}"):
                       st.markdown(chunk.text)
                  if i < len(retrieved_chunks) - 1:
                       st.markdown("---")

    # --- Netzwerkverbindungen anzeigen ---
    st.markdown("---")
    if st.checkbox("🕸️ Zeige Netzwerkverbindungen", key="show_connections_main", value=False):
        st.subheader("Gelernte Verbindungen (Top 50)")
        processor_ui_main = st.session_state.get('processor') # Sicherstellen, dass der aktuelle Prozessor verwendet wird
        if processor_ui_main and hasattr(processor_ui_main, 'nodes'):
            min_weight_thr = st.slider("Mindestgewicht", 0.0, 1.0, 0.1, 0.01, key="weight_slider_main")
            connections_data = []
            node_uuid_map = {n.uuid: n for n in processor_ui_main.nodes.values()}

            for node in processor_ui_main.nodes.values():
                if isinstance(getattr(node, 'connections', None), dict):
                    for conn_uuid, conn in node.connections.items():
                        if conn is None: continue
                        target_node = node_uuid_map.get(getattr(conn, 'target_node_uuid', None))
                        weight = getattr(conn, 'weight', None)
                        if (target_node and isinstance(weight, (float, np.number)) and
                            np.isfinite(weight) and weight >= min_weight_thr):
                            connections_data.append({
                                "Quelle": node.label,
                                "Ziel": target_node.label,
                                "Gewicht": weight,
                                "Typ": getattr(conn, 'conn_type', 'N/A')
                            })

            connections_data.sort(key=lambda x: x["Gewicht"], reverse=True)
            show_connections_table(connections_data[:50]) # Zeige Top 50
            if len(connections_data) > 50:
                 st.caption(f"Zeige Top 50 von {len(connections_data)} Verbindungen ≥ {min_weight_thr:.2f}.")

        else:
            st.warning("Prozessor oder Knoten nicht verfügbar.")

else:
    # Nachricht, wenn kein Prozessor im Session State ist
    st.info("ℹ️ Bitte laden Sie einen Prozessor-Zustand über die Seitenleiste, um zu beginnen.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("QAE-SL Interface v1.2")