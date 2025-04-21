# -- coding: utf-8 --

# Filename: qllm_streamlit_ui_hybrid.py
# Version: 1.3 - Integrated LimbusAffektus Display & Parameters
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
from typing import List, Optional, Dict, Any

# Add directory containing quantum_arona_hybrid_llm.py if necessary
try:
    # Import required classes
    from quantum_arona_hybrid_llm import (
        QuantumEnhancedTextProcessor,
        TextChunk,
        Node,
        LimbusAffektus, # LimbusAffektus node class
        Connection,     # Used internally by processor
        QuantumNodeSystem # Used internally by processor
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

# === Helper function for connection display ===
def show_connections_table(connections_data: List[Dict[str, Any]]) -> None:
    """Displays connection data in a table."""
    if connections_data:
        df_connections = pd.DataFrame(connections_data)
        # Round weight for better display
        if "Gewicht" in df_connections.columns:
             df_connections["Gewicht"] = df_connections["Gewicht"].round(4)
        st.dataframe(df_connections, use_container_width=True)
    else:
        st.info("Keine Verbindungen mit den aktuellen Filterkriterien gefunden.")

# In qllm_streamlit_ui_hybrid.py

#@st.cache_resource(ttl=3600) # Caching kann bei Objektänderungen problematisch sein
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    print(f"DEBUG: Attempting to load state from {state_path}") # Debug Print 1
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`.")
        print(f"DEBUG: State file not found: {state_path}") # Debug Print 2
        return None
    try:
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            print(f"DEBUG: State loaded successfully. Processor ID: {id(processor)}") # Debug Print 3
            # Prüfe den rag_enabled Status direkt nach dem Laden
            print(f"DEBUG: Processor rag_enabled immediately after load_state: {getattr(processor, 'rag_enabled', 'Attribute Missing')}") # Debug Print 4
            print(f"DEBUG: Processor config['enable_rag'] after load_state: {processor.config.get('enable_rag', 'Key Missing')}") # Debug Print 5
            # -- ENTFERNT: Redundante Parameter-Updates nach dem Laden --
            # # Update RAG status based on loaded config and SDK availability
            # processor.rag_enabled = processor.config.get("enable_rag", False) and 'GEMINI_AVAILABLE' in globals() and GEMINI_AVAILABLE
            # # Update Limbus parameters from the potentially newer config file used during load
            # limbus_node = processor.nodes.get("Limbus Affektus")
            # if isinstance(limbus_node, LimbusAffektus):
            #     limbus_node.config = processor.config # Ensure reference is correct
            #     limbus_node.decay = processor.config.get("limbus_emotion_decay", 0.95)
            #     limbus_node.arousal_sens = processor.config.get("limbus_arousal_sensitivity", 1.5)
            #     limbus_node.pleasure_sens = processor.config.get("limbus_pleasure_sensitivity", 1.0)
            #     limbus_node.dominance_sens = processor.config.get("limbus_dominance_sensitivity", 1.0)
            # --- Ende ENTFERNT ---
            return processor
        else:
            st.error(f"❌ Fehler beim Laden des Zustands aus `{state_path}`.")
            print(f"DEBUG: QuantumEnhancedTextProcessor.load_state returned None for {state_path}") # Debug Print 6
            return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler beim Laden des Zustands: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        print(f"DEBUG: Exception during load_processor_state: {e}") # Debug Print 7
        return None

# === Streamlit GUI Setup ===
st.set_page_config(page_title="QAE RAG Explorer", layout="wide")

# --- Header ---
col1_title, col2_title = st.columns([1, 6])
with col1_title:
     # st.image("path/to/your/logo.png", width=80) # Placeholder for logo
     st.markdown("🧠", unsafe_allow_html=True) # Simple emoji placeholder
with col2_title:
     st.title("Quantum-Augmented RAG Explorer")
     st.caption("Hybrides Quanten-Retrieval & Textgenerierungs-Interface mit Lernzyklus")

# --- Session State Initialization ---
if 'processor' not in st.session_state: st.session_state['processor'] = None
if 'state_file_path' not in st.session_state: st.session_state['state_file_path'] = "qetp_state.json"
if 'last_retrieved_chunks' not in st.session_state: st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state: st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state: st.session_state['last_prompt'] = ""
# Debug checkbox is handled directly in the UI

# --- Initial State Load if not already loaded ---
if st.session_state.processor is None:
    st.session_state.processor = load_processor_state(st.session_state.state_file_path)
    # No st.rerun() here to avoid infinite loop on load error

# === Sidebar: Control & Info ===
with st.sidebar:
    st.header("⚙️ Steuerung & Status")

    # --- State Load/Save ---
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

    # Access the (potentially newly loaded) processor
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

    # --- Network Status & Simulation ---
    if processor is not None:
        st.subheader("📊 Netzwerk Übersicht")
        try:
            # Get current summary data
            summary = processor.get_network_state_summary()
            col1_info, col2_info, col3_info = st.columns(3)
            with col1_info:
                st.metric("Knoten", summary.get('num_nodes', 0))
            with col2_info:
                st.metric("Verbindungen", summary.get('total_connections', 0))
            with col3_info:
                st.metric("Chunks", summary.get('num_chunks', 0))

            with st.expander("Details anzeigen (JSON)", expanded=False):
                st.json(summary) # Full summary in expander

            # Status indicators
            if not summary.get('rag_enabled'):
                st.warning("RAG (Gemini) ist deaktiviert.", icon="⚠️")
            if summary.get('self_learning_enabled'):
                st.success("Self-Learning ist aktiviert.", icon="🎓")
            else:
                st.info("Self-Learning ist deaktiviert.", icon="❌")

            # Simulation Button
            if st.button("➡️ Schritt simulieren", key="simulate_button", help="Führt einen Simulationsschritt durch (Aktivierung, Decay, Emotion)."):
                with st.spinner("Simuliere Netzwerk..."):
                    processor.simulate_network_step(decay_connections=True) # Simulate with decay
                st.success("Simulationsschritt abgeschlossen.")
                st.rerun() # Update UI after simulation

        except Exception as e:
            st.error(f"Fehler beim Abrufen der Netzwerkinfo: {e}")

        # --- Parameters & Metrics ---
        st.markdown("---")
        st.subheader("🔧 Parameter & Metriken")

        with st.expander("Konfiguration anpassen (Temporär)", expanded=False):
            # 1. n_shots slider
            n_shots_val = processor.config.get("simulation_n_shots", 50)
            n_shots_new = st.slider(
                "Quanten-Messungen (n_shots)", min_value=1, max_value=200, value=n_shots_val, step=1,
                help="Beeinflusst die Genauigkeit/Stabilität der Quantenknoten-Aktivierung."
            )
            if n_shots_new != n_shots_val:
                processor.config["simulation_n_shots"] = n_shots_new
                st.info(f"n_shots auf {n_shots_new} geändert. Zum Speichern '💾 Speichern' klicken.")

            # --- Limbus Influence Parameters ---
            st.caption("Limbus Modulation (Retrieval)")
            thr_arousal_val = processor.config.get("limbus_influence_threshold_arousal", -0.03)
            thr_arousal_new = st.slider("Einfluss Arousal auf Threshold", -0.1, 0.1, thr_arousal_val, 0.005, format="%.3f")
            if thr_arousal_new != thr_arousal_val:
                processor.config["limbus_influence_threshold_arousal"] = thr_arousal_new
                st.info(f"Threshold(Arousal) Faktor auf {thr_arousal_new:.3f} geändert.")

            thr_pleasure_val = processor.config.get("limbus_influence_threshold_pleasure", 0.03)
            thr_pleasure_new = st.slider("Einfluss Pleasure auf Threshold", -0.1, 0.1, thr_pleasure_val, 0.005, format="%.3f")
            if thr_pleasure_new != thr_pleasure_val:
                processor.config["limbus_influence_threshold_pleasure"] = thr_pleasure_new
                st.info(f"Threshold(Pleasure) Faktor auf {thr_pleasure_new:.3f} geändert.")

            rank_bias_val = processor.config.get("limbus_influence_ranking_bias_pleasure", 0.02)
            rank_bias_new = st.slider("Einfluss Pleasure auf Ranking Bias", -0.1, 0.1, rank_bias_val, 0.005, format="%.3f")
            if rank_bias_new != rank_bias_val:
                processor.config["limbus_influence_ranking_bias_pleasure"] = rank_bias_new
                st.info(f"Ranking Bias(Pleasure) Faktor auf {rank_bias_new:.3f} geändert.")

            st.caption("Limbus Modulation (LLM Temp)")
            temp_arousal_val = processor.config.get("limbus_influence_temperature_arousal", 0.1)
            temp_arousal_new = st.slider("Einfluss Arousal auf Temperatur", -0.5, 0.5, temp_arousal_val, 0.01, format="%.2f")
            if temp_arousal_new != temp_arousal_val:
                processor.config["limbus_influence_temperature_arousal"] = temp_arousal_new
                st.info(f"Temperatur(Arousal) Faktor auf {temp_arousal_new:.2f} geändert.")

            temp_dominance_val = processor.config.get("limbus_influence_temperature_dominance", -0.1)
            temp_dominance_new = st.slider("Einfluss Dominance auf Temperatur", -0.5, 0.5, temp_dominance_val, 0.01, format="%.2f")
            if temp_dominance_new != temp_dominance_val:
                processor.config["limbus_influence_temperature_dominance"] = temp_dominance_new
                st.info(f"Temperatur(Dominance) Faktor auf {temp_dominance_new:.2f} geändert.")

            st.caption("Limbus Modulation (Lernrate)")
            lr_mult_val = processor.config.get("limbus_influence_learning_rate_multiplier", 0.1)
            lr_mult_new = st.slider("Einfluss (P+A)/2 auf Lernrate-Mult.", -0.5, 0.5, lr_mult_val, 0.01, format="%.2f")
            if lr_mult_new != lr_mult_val:
                processor.config["limbus_influence_learning_rate_multiplier"] = lr_mult_new
                st.info(f"Lernraten-Mult.(P+A) Faktor auf {lr_mult_new:.2f} geändert.")


        # --- Current Metrics Display ---
        st.caption("Aktueller Netzwerkzustand")
        # Average activation
        activations = [getattr(n, 'activation', 0.0) for n in processor.nodes.values()]
        valid_activations = [a for a in activations if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if valid_activations:
            avg_act = sum(valid_activations) / len(valid_activations)
            st.metric("Ø Knoten-Aktivierung", f"{avg_act:.4f}")
            # Bar chart for individual activations
            valid_labels = [n.label for n in processor.nodes.values() if isinstance(getattr(n, 'activation', None), (float, np.number)) and np.isfinite(getattr(n, 'activation', None))]
            if len(valid_labels) == len(valid_activations):
                 df_activations = pd.DataFrame({"Aktivierung": valid_activations}, index=valid_labels)
                 st.bar_chart(df_activations, height=150)
        else: st.info("Keine gültigen Aktivierungsdaten.")

        # Limbus Affektus State Display (if node exists)
        limbus_node = processor.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             st.caption("Globaler emotionaler Zustand (PAD)")
             emotion_state = getattr(limbus_node, 'emotion_state', {})
             if emotion_state:
                  col_p, col_a, col_d = st.columns(3)
                  # Display PAD values
                  with col_p: st.metric("Pleasure", f"{emotion_state.get('pleasure', 0.0):.3f}", delta=None)
                  with col_a: st.metric("Arousal", f"{emotion_state.get('arousal', 0.0):.3f}", delta=None)
                  with col_d: st.metric("Dominance", f"{emotion_state.get('dominance', 0.0):.3f}", delta=None)
             else: st.info("Emotionszustand nicht verfügbar.")

    else:
        # Message when no processor is loaded
        st.info("ℹ️ Kein Prozessor-Zustand geladen.")
        st.warning("Bitte laden Sie eine Zustandsdatei oder führen Sie das Trainingsskript aus.")

# === Main Area: Prompt & Results ===
processor = st.session_state.get('processor') # Ensure we use the current processor state
if processor is not None:
    st.header("💬 Prompt & Antwort")
    prompt = st.text_area(
        "Geben Sie einen Prompt oder eine Frage ein:",
        height=100,
        key="prompt_input_main",
        value=st.session_state.get("last_prompt", ""),
        help="Ihre Frage oder Ihr Thema für das RAG-System."
    )

    # Generate Button
    generate_disabled = not processor.rag_enabled or not prompt.strip()
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=generate_disabled, type="primary"):
        st.session_state['last_prompt'] = prompt # Store current prompt
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None
        st.rerun() # Force rerun to show spinner immediately

    # --- Generation Logic (runs after button click or if response is pending) ---
    if st.session_state.last_prompt and st.session_state.last_generated_response is None: # Only execute when generation is requested
        if processor.rag_enabled and st.session_state.last_prompt.strip():
            start_process_time = time.time()
            success_flag = False
            with st.spinner("🧠 Generiere Antwort (Retrieval + LLM)..."):
                try:
                    # Get the last prompt from state
                    current_prompt = st.session_state.last_prompt
                    # Perform generation (this now includes Limbus modulation)
                    generated_response = processor.generate_response(current_prompt)
                    st.session_state['last_generated_response'] = generated_response # Store result
                    # Check validity
                    is_valid = (generated_response and
                                not generated_response.startswith("[Fehler") and
                                not generated_response.startswith("[Antwort blockiert"))
                    if is_valid: success_flag = True

                    # Get context for display (redundant call, but simpler for UI state)
                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(current_prompt)
                    st.success(f"Antwort generiert in {time.time() - start_process_time:.2f}s")

                    # Save state AFTER self-learning (if enabled and successful)
                    # Note: Saving happens within generate_response during self-learning now.
                    # We might still want a manual save button for the overall state.

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state['last_generated_response'] = "[Fehler bei der Generierung]"
            # Rerun again to display the results
            st.rerun()

        elif not processor.rag_enabled:
             st.error("Textgenerierung (RAG) ist nicht aktiviert.")
             # Optionally perform and display only retrieval
             # st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(st.session_state.last_prompt)
             # st.rerun()

    # --- Results Display ---
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
                  st.markdown(f"> _{chunk.text[:300]}..._") # Shortened text for overview
                  if st.button(f"Volltext {i+1}", key=f"chunk_text_{i}"):
                       st.markdown(chunk.text)
                  if i < len(retrieved_chunks) - 1:
                       st.markdown("---")

    # --- Network Connection Display ---
    st.markdown("---")
    if st.checkbox("🕸️ Zeige Netzwerkverbindungen", key="show_connections_main", value=False):
        st.subheader("Gelernte Verbindungen (Top 50)")
        processor_ui_main = st.session_state.get('processor') # Ensure current processor is used
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
            show_connections_table(connections_data[:50]) # Show top 50
            if len(connections_data) > 50:
                 st.caption(f"Zeige Top 50 von {len(connections_data)} Verbindungen ≥ {min_weight_thr:.2f}.")

        else:
            st.warning("Prozessor oder Knoten nicht verfügbar.")

else:
    # Message if no processor is in session state
    st.info("ℹ️ Bitte laden Sie einen Prozessor-Zustand über die Seitenleiste, um zu beginnen.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("QAE-SL Interface v1.3")