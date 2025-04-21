# -- coding: utf-8 --

# Filename: qllm_streamlit_ui_hybrid.py
# Description: Interaktives Interface für Quantum-Arona Hybrid LLM (RAG) mit Self-Learning.
# Version: 1.1 - Bugfix: Anzeige von Netzwerkverbindungen und UI-Erweiterungen
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
from typing import List, Optional

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
# Annahme: Das Skript liegt im selben Verzeichnis
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
    st.stop()

# === Hilfsfunktion für Verbindungsanzeige ===
def show_connections_table(connections_data: List[dict[str, any]]) -> None:
    """
    Wandelt die Verbindungsdaten in einen Pandas DataFrame um
    und zeigt ihn in Streamlit an. Wenn keine Daten vorhanden sind,
    wird eine Info‑Meldung ausgegeben.
    """
    if connections_data:
        df_connections = pd.DataFrame(connections_data)
        st.dataframe(df_connections, use_container_width=True)
    else:
        st.info("Keine Verbindungen mit den aktuellen Filterkriterien gefunden.")

# === Zustandslade-Funktion ===
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`. Bitte zuerst das Trainingsskript (`qllm_train_hybrid.py`) ausführen.")
        return None
    try:
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
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
st.caption("Hybrides Quanten-Retrieval & Textgenerierungs-Interface mit Lernzyklus")

# --- Session State initialisieren ---
if 'processor' not in st.session_state: st.session_state['processor'] = None
if 'state_file_path' not in st.session_state: st.session_state['state_file_path'] = "qetp_state.json"
if 'last_retrieved_chunks' not in st.session_state: st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state: st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state: st.session_state['last_prompt'] = ""
if 'show_debug_output' not in st.session_state: st.session_state['show_debug_output'] = False

# === Seitenleiste: Steuerung ===
with st.sidebar:
    st.header("⚙️ Steuerung")
    current_state_path = st.text_input(
        "Pfad zur Zustandsdatei",
        value=st.session_state['state_file_path'],
        key="state_path_input"
    )
    if st.button("🔄 Zustand Laden/Aktualisieren", key="load_state_button"):
        with st.spinner(f"Lade Zustand aus `{current_state_path}`..."):
            st.session_state['processor'] = None
            st.session_state['last_retrieved_chunks'] = []
            st.session_state['last_generated_response'] = None
            processor_instance = load_processor_state(current_state_path)
            if processor_instance:
                st.session_state['processor'] = processor_instance
                st.session_state['state_file_path'] = current_state_path
                st.success("Zustand erfolgreich geladen.")
            else:
                st.error("Laden fehlgeschlagen.")

    st.markdown("---")
    processor = st.session_state.get('processor')
    if processor is not None:
        st.subheader("Netzwerk Info")
        try:
            summary = processor.get_network_state_summary()
            st.json(summary, expanded=False)
            if not summary.get('rag_enabled'):
                st.warning("RAG ist deaktiviert oder konnte nicht geladen werden.")
            if summary.get('self_learning_enabled'):
                st.info("Self-Learning ist aktiviert.")
            else:
                st.info("Self-Learning ist deaktiviert.")

            if st.button("➡️ Netzwerk-Schritt simulieren"):
                with st.spinner("Simuliere Netzwerkaktivität..."):
                    processor.simulate_network_step(decay_connections=True)
                st.success("Netzwerk-Schritt abgeschlossen.")

        except Exception as e:
            st.warning(f"Konnte Netzwerkinfo nicht abrufen: {e}")

        st.markdown("---")
        if st.button("💾 Zustand manuell speichern"):
            with st.spinner("Speichere aktuellen Zustand..."):
                processor.save_state(st.session_state['state_file_path'])
            st.success("Zustand gespeichert.")

        # === Neue UI‑Erweiterungen ===
        st.markdown("---")
        st.subheader("⚙️ Parameter & Metriken")

        # 1. n_shots-Slider
        n_shots = st.slider(
            "Anzahl der Messdurchläufe (n_shots)",
            min_value=1,
            max_value=200,
            value=processor.config.get("simulation_n_shots", 50),
            step=1
        )
        processor.config["simulation_n_shots"] = n_shots

        # 2. Durchschnittsaktivierung
        activations = [
            n.activation for n in processor.nodes.values()
            if isinstance(n.activation, (float, np.number))
        ]
        if activations:
            avg_act = sum(activations) / len(activations)
            st.metric("Ø Knoten-Aktivierung", f"{avg_act:.3f}")
            st.bar_chart(
                pd.DataFrame({"Aktivierung": activations}, index=[n.label for n in processor.nodes.values()])
            )

        # 3. Verbindungsgewicht-Verteilung
        weights = [
            conn.weight
            for n in processor.nodes.values()
            for conn in (n.connections or {}).values()
            if hasattr(conn, 'weight') and isinstance(conn.weight, (float, np.number))
        ]
        if weights:
            st.caption("Verteilungsdiagramm der Verbindungsgewichte")
            st.bar_chart(pd.DataFrame({"Gewicht": weights}))
        # === Ende UI‑Erweiterungen ===

    else:
        st.info("ℹ️ Kein Prozessor-Zustand geladen.")
        st.warning("Führen Sie zuerst das Trainingsskript (`qllm_train_hybrid.py`) aus, um eine Zustandsdatei zu erstellen.")

# === Hauptbereich: Prompt & Ergebnisse ===
processor = st.session_state.get('processor')
if processor is not None:
    st.header("💬 Prompt-Eingabe & Generierung")
    prompt = st.text_area(
        "Geben Sie einen Prompt oder eine Frage ein:",
        height=100,
        key="prompt_input_main",
        value=st.session_state.get("last_prompt", "")
    )

    generate_disabled = not processor.rag_enabled or not prompt
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=generate_disabled):
        st.session_state['last_prompt'] = prompt
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None

        if processor.rag_enabled and prompt:
            start_process_time = time.time()
            success_flag = False
            with st.spinner("🧠 Generiere Antwort..."):
                try:
                    generated_response = processor.generate_response(prompt)
                    st.session_state['last_generated_response'] = generated_response
                    is_valid = (generated_response and
                                not generated_response.startswith("[Fehler") and
                                not generated_response.startswith("[Antwort blockiert"))
                    if is_valid:
                        success_flag = True

                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(prompt)
                    st.success(f"Antwort generiert in {time.time() - start_process_time:.2f}s")

                    if processor.self_learning_enabled and success_flag:
                        with st.spinner("💾 Speichere Zustand nach Lernzyklus..."):
                            processor.save_state(st.session_state['state_file_path'])
                        st.success("Zustand nach Lernzyklus gespeichert.")
                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state['last_generated_response'] = "Fehler bei der Generierung."
        elif not processor.rag_enabled:
            st.error("Textgenerierung ist nicht aktiviert. Nur Retrieval möglich.")
            st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(prompt)

    if st.session_state.get('last_generated_response') is not None:
        st.markdown("---")
        st.subheader("💡 Generierte Antwort")
        st.markdown(st.session_state['last_generated_response'])

    retrieved = st.session_state.get('last_retrieved_chunks', [])
    if retrieved:
        st.markdown("---")
        with st.expander(f"Kontext: {len(retrieved)} abgerufene Textabschnitte"):
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**[{i+1}] Quelle:** `{chunk.source}` (Abschnitt: `{chunk.index}`)")
                if hasattr(chunk, 'activated_node_labels') and chunk.activated_node_labels:
                    nodes_str = ", ".join(f"`{lbl}`" for lbl in chunk.activated_node_labels)
                    st.markdown(f"**Zugeordnete Knoten:** {nodes_str}")
                st.markdown(f"> {chunk.text}")
                if i < len(retrieved)-1:
                    st.markdown("---")

    st.markdown("---")
    st.session_state['show_debug_output'] = st.checkbox("🐞 Debug-Ausgaben für Verbindungen in Konsole", key="debug_checkbox")

    if st.checkbox("📊 Zeige Netzwerkverbindungen (Top 50)", key="show_connections"):
        st.subheader("🕸️ Gelernte Verbindungen zwischen Knoten")
        connections_data = []
        processor_ui = processor
        if processor_ui and hasattr(processor_ui, 'nodes'):
            min_thr = st.slider("Mindestgewicht anzeigen", 0.0, 1.0, 0.1, 0.01)
            if st.session_state['show_debug_output']:
                print("\n--- DEBUG: Netzwerkverbindungsanzeige (Streamlit) ---")

            # UUID→Node Map für Lookup
            node_uuid_map = {n.uuid: n for n in processor_ui.nodes.values()}
            processed_conns = displayed_conns = 0

            for node in processor_ui.nodes.values():
                if not isinstance(getattr(node, 'connections', None), dict): continue
                for tgt_uuid, conn in node.connections.items():
                    processed_conns += 1
                    if conn is None: continue
                    target_node = node_uuid_map.get(conn.target_node_uuid)
                    weight = getattr(conn, 'weight', None)
                    conn_type = getattr(conn, 'conn_type', 'N/A')
                    if (target_node and isinstance(weight, (float, np.number)) and
                        np.isfinite(weight) and weight >= min_thr):
                        displayed_conns += 1
                        connections_data.append({
                            "Quelle": node.label,
                            "Ziel": target_node.label,
                            "Gewicht": weight,
                            "Typ": conn_type
                        })
            # Tabelle anzeigen
            show_connections_table(connections_data)
        else:
            st.warning("Prozessor oder Knoten nicht verfügbar für Verbindungsanzeige.")
else:
    st.info("ℹ️ Bitte laden Sie zuerst einen Prozessor-Zustand über die Seitenleiste.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("Quantum-Arona RAG Interface v1.1")