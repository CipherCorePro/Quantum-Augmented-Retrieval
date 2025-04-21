# -- coding: utf-8 --

# Filename: qllm_streamlit_ui_hybrid.py
# Description: Interaktives Interface für Quantum-Arona Hybrid LLM (RAG).
# Version: 0.6 - Stabile Version mit RAG
# Author: [CipherCore Technology] & Gemini & Your Input

import streamlit as st
import json
import os
import sys
import pandas as pd
from collections import defaultdict
import traceback
import time
from typing import List, Optional # Importiere List und Optional
import numpy as np

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
# Passe den Pfad ggf. an.
# Beispiel: sys.path.append(os.path.abspath('../src'))
try:
    from quantum_arona_hybrid_llm import QuantumEnhancedTextProcessor, TextChunk, Node
except ImportError:
    st.error(
        """
    **FEHLER: Konnte 'QuantumEnhancedTextProcessor' nicht importieren.**

    Stellen Sie sicher, dass:
    1. Die Datei `quantum_arona_hybrid_llm.py` existiert.
    2. Sie sich im selben Verzeichnis wie dieses Streamlit-Skript befindet ODER im Python-Pfad liegt.
    """
    )
    st.stop() # Beende die Ausführung, da die Kernkomponente fehlt

# === Hilfsfunktionen ===
# @st.cache_resource(ttl=3600) # Caching kann Probleme verursachen, wenn das Objekt intern verändert wird
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`. Bitte zuerst das Trainingsskript (`qllm_train_hybrid.py`) ausführen.")
        return None
    try:
        print(f"INFO: Versuche, Zustand aus '{state_path}' zu laden...") # Logging für Konsole
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            # Erfolgsmeldung im UI statt nur in Konsole
            # st.success(f"✅ Prozessor-Zustand erfolgreich aus `{state_path}` geladen.")
            # Statusinfo wird jetzt in der Seitenleiste angezeigt
            return processor
        else:
            st.error(f"❌ Fehler beim Laden des Zustands aus `{state_path}`. Datei könnte korrupt oder leer sein.")
            return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler beim Laden des Zustands: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

# === Streamlit GUI ===
st.set_page_config(page_title="QAE RAG Explorer", layout="wide")
st.title("🧠 Quantum-Augmented RAG Explorer")
st.caption("Hybrides Quanten-Retrieval & Textgenerierungs-Interface")

# --- Initialisierung des Session State ---
if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'state_file_path' not in st.session_state:
    st.session_state['state_file_path'] = "qetp_state.json"
if 'last_retrieved_chunks' not in st.session_state:
     st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state:
     st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state:
     st.session_state['last_prompt'] = ""


# === Seitenleiste: Laden und Steuerung ===
with st.sidebar:
    st.header("⚙️ Steuerung")
    current_state_path = st.text_input(
        "Pfad zur Zustandsdatei",
        value=st.session_state['state_file_path'],
        key="state_path_input"
    )

    # Button zum Laden des Zustands
    if st.button("🔄 Zustand Laden/Aktualisieren", key="load_state_button", help="Lädt den Prozessorzustand aus der angegebenen Datei."):
        with st.spinner(f"Lade Zustand aus `{current_state_path}`..."):
            # Setze Prozessor zurück bevor neu geladen wird
            st.session_state['processor'] = None
            st.session_state['last_retrieved_chunks'] = []
            st.session_state['last_generated_response'] = None
            # Lade den Zustand
            processor_instance = load_processor_state(current_state_path)
            if processor_instance:
                st.session_state['processor'] = processor_instance
                st.session_state['state_file_path'] = current_state_path # Pfad speichern bei Erfolg
                st.success("Zustand erfolgreich geladen.")
                st.rerun() # Neu laden, um Status zu aktualisieren
            else:
                 # Fehlermeldung wird bereits in load_processor_state angezeigt
                 st.error("Laden fehlgeschlagen.")

    # Zeige Status nur an, wenn ein Prozessor geladen ist
    st.markdown("---")
    processor = st.session_state.get('processor')
    if processor is not None:
        st.subheader("Netzwerk Info")
        try:
            summary = processor.get_network_state_summary()
            st.json(summary, expanded=False)
            if not summary.get('rag_enabled'):
                 st.warning("RAG (Textgenerierung) ist deaktiviert oder konnte nicht geladen werden.", icon="⚠️")

            # Optional: Button für Simulationsschritt
            if st.button("➡️ Netzwerk-Schritt simulieren", help="Führt einen Simulationsschritt durch (Knotenaktivierung & Zerfall)."):
                with st.spinner("Simuliere Netzwerkaktivität..."):
                    processor.simulate_network_step(decay_connections=True)
                st.success("Netzwerk-Schritt abgeschlossen.")
                # Optional: Update der Summary nach Simulation erzwingen (kann aber langsam sein)
                st.rerun()

        except Exception as e:
            st.warning(f"Konnte Netzwerkinfo nicht abrufen: {e}")
    else:
        st.info("ℹ️ Kein Prozessor-Zustand geladen.")
        st.warning("Führen Sie zuerst das Trainingsskript (`qllm_train_hybrid.py`) aus, um eine Zustandsdatei zu erstellen, und laden Sie diese dann hier.")


# === Hauptbereich: Prompt-Eingabe und Ergebnisse ===
if processor is not None:

    st.header("💬 Prompt-Eingabe & Generierung")
    # Verwende Session State, um den Prompt zwischen Läufen zu behalten
    prompt = st.text_area(
        "Geben Sie einen Prompt oder eine Frage ein:",
        height=100,
        key="prompt_input_main",
        value=st.session_state.get("last_prompt","") # Lade letzten Prompt
        )

    # Button zum Generieren der Antwort
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=(not processor.rag_enabled or not prompt)):

        # Speichere den aktuellen Prompt
        st.session_state["last_prompt"] = prompt

        # Setze vorherige Ergebnisse zurück
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None

        if processor.rag_enabled and prompt:
            start_process_time = time.time()
            with st.spinner("🧠 Generiere Antwort (Retrieval + Generation)..."):
                try:
                    # Rufe generate_response auf - diese ruft intern respond_to_prompt
                    generated_response = processor.generate_response(prompt)
                    st.session_state['last_generated_response'] = generated_response

                    # Workaround: Rufe respond_to_prompt nochmal auf für die Anzeige der Chunks
                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(prompt)

                    end_process_time = time.time()
                    st.success(f"Antwort generiert in {end_process_time - start_process_time:.2f}s")

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state['last_retrieved_chunks'] = []
                    st.session_state['last_generated_response'] = "Fehler bei der Generierung."

        elif not processor.rag_enabled:
             st.error("Textgenerierung (RAG) ist nicht aktiviert. Nur Retrieval möglich.")
             with st.spinner("Suche relevante Textstellen (Retrieval)..."):
                 retrieved_chunks = processor.respond_to_prompt(prompt)
                 st.session_state['last_retrieved_chunks'] = retrieved_chunks

        # Force rerun, um die Ergebnisse anzuzeigen (alternativ Callback nutzen)
        st.rerun()

    # --- Anzeige der Ergebnisse (aus Session State) ---
    if st.session_state.get('last_generated_response') is not None:
         st.markdown("---")
         st.subheader("💡 Generierte Antwort")
         st.markdown(st.session_state['last_generated_response'])

    retrieved_chunks_to_show = st.session_state.get('last_retrieved_chunks', [])
    if retrieved_chunks_to_show:
         st.markdown("---")
         with st.expander(f"Kontext: {len(retrieved_chunks_to_show)} abgerufene(r) Textabschnitt(e)", expanded=False):
              for i, chunk in enumerate(retrieved_chunks_to_show):
                  st.markdown(f"**[{i+1}] Quelle:** `{chunk.source}` (Abschnitt: `{chunk.index}`)")
                  if chunk.activated_node_labels:
                      nodes_str = ", ".join([f"`{label}`" for label in chunk.activated_node_labels])
                      st.markdown(f"**Zugeordnete Knoten:** {nodes_str}")
                  # Sicherer Textzugriff
                  chunk_text = getattr(chunk, 'text', '*Text nicht verfügbar*')
                  st.markdown(f"> {chunk_text}")
                  if i < len(retrieved_chunks_to_show) - 1:
                       st.markdown("---") # Trennlinie zwischen Chunks

    # --- (Optional: Netzwerkverbindungen anzeigen - Mit Debugging) ---
    st.markdown("---")
    if st.checkbox("📊 Zeige Netzwerkverbindungen (Top 50)", key="show_connections", value=False):
        st.subheader("🕸️ Gelernte Verbindungen zwischen Knoten")
        connections_data = []
        # Sicherstellen, dass processor und nodes existieren
        if processor and hasattr(processor, 'nodes'):
            min_weight_threshold = st.slider("Mindestgewicht anzeigen", 0.0, 1.0, 0.1, 0.01, key="weight_slider")
            # st.info("ℹ️ Debug-Informationen werden in der Konsole ausgegeben, in der Streamlit gestartet wurde.") # Hinweis für den Benutzer
            try:
                print("\n--- DEBUGGING START: Netzwerkverbindungsanzeige ---") # Konsolen-Marker
                processed_conns = 0
                displayed_conns = 0

                for node in processor.nodes.values():
                    if not hasattr(node, 'connections') or not isinstance(node.connections, dict):
                        print(f"DEBUG Streamlit: Node '{node.label}' hat keine Connections oder ist kein Dict.")
                        continue

                    if not node.connections:
                        # print(f"DEBUG Streamlit: Node '{node.label}' hat leere Connections.") # Optional: Weniger verbose
                        continue

                    print(f"DEBUG Streamlit: Prüfe Node '{node.label}' mit {len(node.connections)} potenziellen Verbindungen.")
                    for target_uuid, conn in node.connections.items():
                        processed_conns += 1
                        if conn is None:
                            print(f"DEBUG Streamlit:  Skipping None connection entry for target UUID '{target_uuid}' from node '{node.label}'.")
                            continue # Überspringe falls None

                        target_node_obj = getattr(conn, 'target_node', None)
                        weight = getattr(conn, 'weight', None)
                        conn_type = getattr(conn, 'conn_type', 'N/A')
                        target_label = getattr(target_node_obj, 'label', 'N/A') # Sicherer Label-Zugriff

                        # --- DEBUGGING AUSGABEN HINZUGEFÜGT ---
                        print(f"\nDEBUG Streamlit: Checking conn from '{node.label}' to UUID '{target_uuid}'.")
                        print(f"  - Connection Object: {conn}") # Verwendet __repr__ von Connection
                        print(f"  - Target Node Object (via getattr): {target_node_obj}")
                        print(f"  - Target Node Label: {target_label}")
                        print(f"  - Weight (via getattr): {weight} (Type: {type(weight)})")
                        print(f"  - Connection Type: {conn_type}")

                        # Werte für die Bedingungsprüfung
                        cond_target_node_exists = target_node_obj is not None
                        cond_target_has_label = hasattr(target_node_obj, 'label')
                        cond_weight_exists = weight is not None
                        cond_weight_is_number = isinstance(weight, (float, np.number))
                        cond_weight_is_finite = False # Default
                        cond_weight_meets_threshold = False # Default

                        if cond_weight_is_number:
                            cond_weight_is_finite = np.isfinite(weight)
                            if cond_weight_is_finite:
                                cond_weight_meets_threshold = weight >= min_weight_threshold

                        # Gesamte Filterbedingung
                        filter_condition_met = (cond_target_node_exists and cond_target_has_label and
                                                cond_weight_exists and cond_weight_is_number and
                                                cond_weight_is_finite and cond_weight_meets_threshold)

                        print(f"  - Checks:")
                        print(f"    - Target Node Exists? {cond_target_node_exists}")
                        print(f"    - Target Has Label?   {cond_target_has_label}")
                        print(f"    - Weight Exists?      {cond_weight_exists}")
                        print(f"    - Weight Is Number?   {cond_weight_is_number}")
                        if cond_weight_is_number: # Nur drucken, wenn relevant
                             print(f"    - Weight Is Finite?   {cond_weight_is_finite}")
                             if cond_weight_is_finite:
                                 print(f"    - Weight >= Threshold ({min_weight_threshold:.2f})? {cond_weight_meets_threshold} ({weight:.3f})")
                        print(f"  - Filter condition met? {filter_condition_met}")
                        # --- ENDE DEBUGGING AUSGABEN ---

                        if filter_condition_met:
                             displayed_conns += 1
                             connections_data.append({
                                "Quelle": node.label,
                                "Ziel": target_label, # Sicher verwenden
                                "Gewicht": weight,
                                "Typ": conn_type
                             })
                        # else: # Optional: Grund für Ablehnung ausgeben
                        #    print(f"  - Reason for rejection: Filter condition not met.")


                print(f"\n--- DEBUGGING ENDE: Netzwerkverbindungsanzeige ---")
                print(f"Total connections checked in processor: {processed_conns}")
                print(f"Connections meeting filter criteria: {displayed_conns}")
                # --- Ende Debug-Zusammenfassung ---

                if connections_data:
                    connections_data.sort(key=lambda x: x["Gewicht"], reverse=True)
                    df_connections = pd.DataFrame(connections_data[:50])
                    # Sicherstellen, dass die Gewicht-Spalte existiert, bevor sie gerundet wird
                    if "Gewicht" in df_connections.columns:
                        df_connections["Gewicht"] = df_connections["Gewicht"].round(3)
                    st.dataframe(df_connections, use_container_width=True)
                    if len(connections_data) > 50:
                        st.caption(f"Zeige die Top 50 von {len(connections_data)} Verbindungen über dem Schwellwert.")
                else:
                    # Diese Meldung wird nun nur angezeigt, wenn nach dem Debugging keine Verbindungen übrig bleiben
                    st.info(f"Keine Verbindungen mit Gewicht ≥ {min_weight_threshold:.2f} gefunden (oder Filterkriterien nicht erfüllt).")

            except Exception as e:
                st.error(f"Fehler beim Anzeigen der Verbindungen: {e}")
                st.error("Traceback:")
                st.code(traceback.format_exc()) # Mehr Details im Fehlerfall
        else:
             st.warning("Prozessor oder Knoten nicht verfügbar für Verbindungsanzeige.")


else:
    # Diese Nachricht wird angezeigt, wenn kein Prozessor geladen ist
    st.info("ℹ️ Bitte laden Sie zuerst einen Prozessor-Zustand über die Seitenleiste.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Quantum-Arona RAG Interface v0.6")