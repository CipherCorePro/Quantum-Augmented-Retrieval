# -- coding: utf-8 --

# Filename: qllm_streamlit_ui_hybrid.py
# Description: Interaktives Interface für Quantum-Arona Hybrid LLM (RAG) mit Self-Learning und Limbus-Modulation.
# Version: 1.3 - Integrated LimbusAffektus Display & Parameter Tuning
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

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
try:
    # Importiere alle benötigten Klassen
    from quantum_arona_hybrid_llm import (
        QuantumEnhancedTextProcessor,
        TextChunk,
        Node,
        LimbusAffektus, # LimbusAffektus node class
        Connection,     # Used internally by processor
        QuantumNodeSystem # Used internally by processor
    )
    # Prüfe, ob die Gemini-Bibliothek verfügbar ist (wird im Processor-Code verwendet)
    from quantum_arona_hybrid_llm import GEMINI_AVAILABLE
except ImportError:
    st.error(
        """
    **FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren.**

    Stellen Sie sicher, dass:
    1. Die Datei `quantum_arona_hybrid_llm.py` (Version 1.2+) existiert und alle Klassen enthält.
    2. Sie sich im selben Verzeichnis wie dieses Skript befindet ODER im Python-Pfad liegt.
        """
    )
    st.stop()
except Exception as import_err:
     st.error(f"Anderer Importfehler: {import_err}")
     st.stop()


# === Hilfsfunktion für Verbindungsanzeige ===
def show_connections_table(connections_data: List[Dict[str, Any]]) -> None:
    """Zeigt Verbindungsdaten als Tabelle an."""
    if connections_data:
        df_connections = pd.DataFrame(connections_data)
        # Runde Gewicht für bessere Anzeige
        if "Gewicht" in df_connections.columns:
             df_connections["Gewicht"] = df_connections["Gewicht"].round(4)
        st.dataframe(df_connections, use_container_width=True, hide_index=True)
    else:
        st.info("Keine Verbindungen mit den aktuellen Filterkriterien gefunden.")

# === Zustandslade-Funktion ===
#@st.cache_resource(ttl=3600) # Caching kann bei Objektänderungen problematisch sein
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    # print(f"DEBUG: Attempting to load state from {state_path}") # Debug entfernt
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`.")
        return None
    try:
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            # print(f"DEBUG: State loaded successfully. Processor ID: {id(processor)}") # Debug entfernt
            # Prüfe RAG Status nach Laden (wird jetzt in __init__ / load_state gesetzt)
            # print(f"DEBUG: Processor rag_enabled after load_state: {getattr(processor, 'rag_enabled', 'N/A')}")
            return processor
        else:
            st.error(f"❌ Fehler beim Laden des Zustands aus `{state_path}`.")
            return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler beim Laden des Zustands: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

# === Streamlit GUI Setup ===
st.set_page_config(page_title="QAE RAG Explorer", layout="wide", initial_sidebar_state="expanded")

# --- Header ---
col1_title, col2_title = st.columns([1, 9]) # Gib Titel mehr Platz
with col1_title:
     st.markdown("🧠", unsafe_allow_html=True) # Emoji als einfacher Platzhalter
with col2_title:
     st.title("Quantum-Augmented RAG Explorer")
     st.caption("Hybrides Quanten-Retrieval & Textgenerierungs-Interface mit Lernzyklus & Limbus-Modulation")

# --- Session State Initialization ---
if 'processor' not in st.session_state: st.session_state['processor'] = None
if 'state_file_path' not in st.session_state: st.session_state['state_file_path'] = "qetp_state.json"
if 'last_retrieved_chunks' not in st.session_state: st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state: st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state: st.session_state['last_prompt'] = ""
# show_debug_output wird nicht mehr benötigt/verwendet

# --- Initial State Load if not already loaded ---
if st.session_state.processor is None:
    with st.spinner("Lade initialen Zustand..."):
        st.session_state.processor = load_processor_state(st.session_state.state_file_path)

# === Sidebar: Control & Info ===
with st.sidebar:
    st.header("⚙️ Steuerung & Status")

    # --- State Load/Save ---
    current_state_path = st.text_input(
        "Pfad zur Zustandsdatei",
        value=st.session_state['state_file_path'],
        key="state_path_input",
        help="JSON-Datei mit dem Netzwerkzustand."
    )
    col1_load, col2_save = st.columns(2)
    with col1_load:
        if st.button("🔄 Laden", key="load_state_button", help="Lädt den Zustand aus der Datei."):
            with st.spinner(f"Lade Zustand aus `{current_state_path}`..."):
                # Reset state before loading
                st.session_state['processor'] = None
                st.session_state['last_retrieved_chunks'] = []
                st.session_state['last_generated_response'] = None
                # Attempt to load
                processor_instance = load_processor_state(current_state_path)
                if processor_instance:
                    st.session_state['processor'] = processor_instance
                    st.session_state['state_file_path'] = current_state_path # Update path only on success
                    st.success("Zustand geladen.")
                    st.rerun() # Force rerun to update UI
                else:
                    st.error("Laden fehlgeschlagen.") # Error message is shown in load_processor_state

    # Access the (potentially newly loaded) processor from session state
    processor = st.session_state.get('processor')

    with col2_save:
        save_disabled = processor is None
        if st.button("💾 Speichern", key="save_state_button", disabled=save_disabled, help="Speichert den aktuellen Netzwerkzustand (inkl. temporärer Parameteränderungen)."):
            if processor:
                with st.spinner("Speichere Zustand..."):
                    try:
                         processor.save_state(st.session_state['state_file_path'])
                         st.success("Zustand gespeichert.")
                    except Exception as save_err:
                         st.error(f"Fehler beim Speichern: {save_err}")
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
            with col1_info: st.metric("Knoten", summary.get('num_nodes', 0))
            with col2_info: st.metric("Verbindungen", summary.get('total_connections', 0))
            with col3_info: st.metric("Chunks", summary.get('num_chunks', 0))

            with st.expander("Netzwerk Details (JSON)", expanded=False):
                st.json(summary) # Full summary in expander

            # Status indicators
            if not getattr(processor, 'rag_enabled', False): # Use processor attribute directly
                st.warning("RAG (Gemini) ist deaktiviert.", icon="⚠️")
            if getattr(processor, 'self_learning_enabled', False):
                st.success("Self-Learning ist aktiviert.", icon="🎓")
            else:
                st.info("Self-Learning ist deaktiviert.", icon="❌")

            # Simulation Button
            if st.button("➡️ Schritt simulieren", key="simulate_button", help="Führt einen Simulationsschritt durch (Aktivierung, Decay, Emotion)."):
                with st.spinner("Simuliere Netzwerk..."):
                    try:
                         processor.simulate_network_step(decay_connections=True) # Simulate with decay
                         st.success("Simulationsschritt abgeschlossen.")
                    except Exception as sim_err:
                         st.error(f"Simulationsfehler: {sim_err}")
                st.rerun() # Update UI after simulation

        except Exception as e:
            st.error(f"Fehler beim Abrufen der Netzwerkinfo: {e}")

        # --- Parameters & Metrics ---
        st.markdown("---")
        st.subheader("🔧 Parameter & Metriken")

        with st.expander("Konfiguration anpassen (Temporär)", expanded=False):
            st.caption("Änderungen hier sind nur für die aktuelle Sitzung wirksam. Klicken Sie auf '💾 Speichern', um sie persistent zu machen.")
            # 1. n_shots slider
            n_shots_val = processor.config.get("simulation_n_shots", 50)
            n_shots_new = st.slider(
                "Quanten-Messungen (n_shots)", min_value=1, max_value=200, value=n_shots_val, step=1,
                key="n_shots_slider_sidebar",
                help="Beeinflusst die Genauigkeit/Stabilität der Quantenknoten-Aktivierung."
            )
            if n_shots_new != n_shots_val: processor.config["simulation_n_shots"] = n_shots_new

            # --- Limbus Influence Parameters Sliders ---
            st.markdown("---")
            st.caption("Limbus Modulation (Retrieval)")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                thr_arousal_val = processor.config.get("limbus_influence_threshold_arousal", -0.03)
                thr_arousal_new = st.slider("Thr(Arousal)", -0.1, 0.1, thr_arousal_val, 0.005, format="%.3f", key="thr_a_slider", help="Einfluss Arousal auf Retrieval-Threshold")
                if thr_arousal_new != thr_arousal_val: processor.config["limbus_influence_threshold_arousal"] = thr_arousal_new
            with col_t2:
                thr_pleasure_val = processor.config.get("limbus_influence_threshold_pleasure", 0.03)
                thr_pleasure_new = st.slider("Thr(Pleasure)", -0.1, 0.1, thr_pleasure_val, 0.005, format="%.3f", key="thr_p_slider", help="Einfluss Pleasure auf Retrieval-Threshold")
                if thr_pleasure_new != thr_pleasure_val: processor.config["limbus_influence_threshold_pleasure"] = thr_pleasure_new

            rank_bias_val = processor.config.get("limbus_influence_ranking_bias_pleasure", 0.02)
            rank_bias_new = st.slider("Ranking Bias(Pleasure)", -0.1, 0.1, rank_bias_val, 0.005, format="%.3f", key="rank_bias_slider", help="Einfluss Pleasure auf Chunk-Ranking")
            if rank_bias_new != rank_bias_val: processor.config["limbus_influence_ranking_bias_pleasure"] = rank_bias_new

            st.markdown("---")
            st.caption("Limbus Modulation (LLM Temp)")
            col_temp1, col_temp2 = st.columns(2)
            with col_temp1:
                temp_arousal_val = processor.config.get("limbus_influence_temperature_arousal", 0.1)
                temp_arousal_new = st.slider("Temp(Arousal)", -0.5, 0.5, temp_arousal_val, 0.01, format="%.2f", key="temp_a_slider", help="Einfluss Arousal auf LLM-Temperatur")
                if temp_arousal_new != temp_arousal_val: processor.config["limbus_influence_temperature_arousal"] = temp_arousal_new
            with col_temp2:
                temp_dominance_val = processor.config.get("limbus_influence_temperature_dominance", -0.1)
                temp_dominance_new = st.slider("Temp(Dominance)", -0.5, 0.5, temp_dominance_val, 0.01, format="%.2f", key="temp_d_slider", help="Einfluss Dominance auf LLM-Temperatur")
                if temp_dominance_new != temp_dominance_val: processor.config["limbus_influence_temperature_dominance"] = temp_dominance_new

            st.markdown("---")
            st.caption("Limbus Modulation (Lernrate)")
            lr_mult_val = processor.config.get("limbus_influence_learning_rate_multiplier", 0.1)
            lr_mult_new = st.slider("LR-Mult(P+A)/2", -0.5, 0.5, lr_mult_val, 0.01, format="%.2f", key="lr_mult_slider", help="Einfluss Emotion auf Lernraten-Multiplikator")
            if lr_mult_new != lr_mult_val: processor.config["limbus_influence_learning_rate_multiplier"] = lr_mult_new

            # --- NEU: Limbus Modulation (Quanten Effekte) ---
            st.markdown("---")
            st.caption("Limbus Modulation (Quanten Effekte)")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                var_penalty_val = processor.config.get("limbus_influence_variance_penalty", 0.1)
                var_penalty_new = st.slider("VarPenalty(A-P)/2", -0.5, 0.5, var_penalty_val, 0.01, format="%.2f", key="var_penalty_slider", help="Einfluss Emotion auf Varianz-Malus")
                if var_penalty_new != var_penalty_val:
                    processor.config["limbus_influence_variance_penalty"] = var_penalty_new
                    st.info(f"Varianz-Malus Faktor auf {var_penalty_new:.2f} geändert.")
            with col_q2:
                act_boost_val = processor.config.get("limbus_influence_activation_boost", 0.05)
                act_boost_new = st.slider("ActBoost(P-A)/2", -0.5, 0.5, act_boost_val, 0.01, format="%.2f", key="act_boost_slider", help="Einfluss Emotion auf Aktivierungs-Bonus")
                if act_boost_new != act_boost_val:
                    processor.config["limbus_influence_activation_boost"] = act_boost_new
                    st.info(f"Aktivierungs-Bonus Faktor auf {act_boost_new:.2f} geändert.")
            # --- ENDE NEU ---

        # --- Current Metrics Display ---
        st.markdown("---")
        st.subheader("📈 Aktuelle Metriken")
        # Average activation
        activations = [getattr(n, 'activation', 0.0) for n in processor.nodes.values()]
        valid_activations = [a for a in activations if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if valid_activations:
            avg_act = sum(valid_activations) / len(valid_activations)
            st.metric("Ø Knoten-Aktivierung", f"{avg_act:.4f}")
            # Optional: Bar chart für Einzelaktivierungen im Expander
            with st.expander("Einzelne Knotenaktivierungen", expanded=False):
                 valid_labels = [n.label for n in processor.nodes.values() if isinstance(getattr(n, 'activation', None), (float, np.number)) and np.isfinite(getattr(n, 'activation', None))]
                 if len(valid_labels) == len(valid_activations):
                      df_activations = pd.DataFrame({"Aktivierung": valid_activations}, index=valid_labels)
                      st.bar_chart(df_activations, height=150)
                 else: st.warning("Inkonsistenz Aktivierungen/Labels")
        else: st.info("Keine gültigen Aktivierungsdaten.")

        # Limbus Affektus State Display (wenn vorhanden und korrekt implementiert)
        limbus_node = processor.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             st.caption("Globaler emotionaler Zustand (PAD)")
             emotion_state = getattr(limbus_node, 'emotion_state', {})
             if emotion_state:
                  col_p, col_a, col_d = st.columns(3)
                  with col_p: st.metric("Pleasure", f"{emotion_state.get('pleasure', 0.0):.3f}")
                  with col_a: st.metric("Arousal", f"{emotion_state.get('arousal', 0.0):.3f}")
                  with col_d: st.metric("Dominance", f"{emotion_state.get('dominance', 0.0):.3f}")
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
        placeholder="Was möchtest du wissen oder diskutieren?",
        help="Ihre Frage oder Ihr Thema für das RAG-System."
    )

    # Generate Button
    generate_disabled = not processor.rag_enabled or not prompt.strip()
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=generate_disabled, type="primary"):
        st.session_state['last_prompt'] = prompt # Store current prompt
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None # Clear previous response
        st.rerun() # Force rerun to show spinner immediately

    # --- Generation Logic (runs after button click or if response is pending) ---
    if st.session_state.last_prompt and st.session_state.last_generated_response is None: # Only execute when generation is requested
        if processor.rag_enabled and st.session_state.last_prompt.strip():
            start_process_time = time.time()
            success_flag = False
            with st.spinner("🧠 Generiere Antwort (Retrieval + LLM)..."):
                try:
                    current_prompt = st.session_state.last_prompt
                    # --- Perform generation (includes Limbus modulation) ---
                    generated_response = processor.generate_response(current_prompt)
                    st.session_state['last_generated_response'] = generated_response # Store result
                    # --- Get context for display AFTER generation ---
                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(current_prompt)

                    # Check validity for self-learning flag
                    is_valid = (generated_response and
                                not generated_response.startswith("[Fehler") and
                                not generated_response.startswith("[Antwort blockiert"))
                    if is_valid: success_flag = True

                    st.success(f"Antwort generiert in {time.time() - start_process_time:.2f}s")

                    # Save state AFTER self-learning (which happens inside generate_response)
                    # This saving might be slightly delayed visually due to rerun, but functional.
                    if processor.self_learning_enabled and success_flag:
                         # No spinner here as it happens after response display typically
                         # print("DEBUG: Attempting to save state after successful self-learning response.")
                         processor.save_state(st.session_state['state_file_path'])
                         # st.success("Zustand nach Lernzyklus gespeichert.") # Might feel redundant after response

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:")
                    st.code(traceback.format_exc())
                    st.session_state['last_generated_response'] = "[Fehler bei der Generierung]"
            # Rerun again to display the results now stored in session state
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
                  # Use columns for better layout of the button
                  cols_btn = st.columns([1, 5])
                  with cols_btn[0]:
                        if st.button(f"Volltext {i+1}", key=f"chunk_text_{i}"):
                             # Display full text in a modal or separate area if needed, for now just below
                             st.markdown(f"**Volltext {i+1}:**\n{chunk.text}")
                  if i < len(retrieved_chunks) - 1:
                       st.markdown("---") # Separator line

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
            show_connections_table(connections_data[:50]) # Show top 50 using helper
            if len(connections_data) > 50:
                 st.caption(f"Zeige Top 50 von {len(connections_data)} Verbindungen ≥ {min_weight_thr:.2f}.")

        else:
            st.warning("Prozessor oder Knoten nicht verfügbar.")

else:
    # Message if no processor is in session state
    st.info("ℹ️ Bitte laden Sie einen Prozessor-Zustand über die Seitenleiste, um zu beginnen.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"QAE-SL Interface v1.3")