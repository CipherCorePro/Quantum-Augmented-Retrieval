# -- coding: utf-8 --

# Filename: qllm_train_hybrid.py
# Description: Trainingsskript für das Quantum-Arona Hybrid LLM.
# Version: 1.2 - Corrected serialization logic and training flow.
# Author: [CipherCore Technology] & Gemini & Your Input

import os
import sys
import time
import json
import argparse
import copy
import random
import traceback
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime

# Stelle sicher, dass numpy importiert ist
import numpy as np

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
try:
    from quantum_arona_hybrid_llm import QuantumEnhancedTextProcessor, TextChunk, Node, Connection, QuantumNodeSystem
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs): return iterable
except ImportError:
    print("FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren.")
    print("Stelle sicher, dass 'quantum_arona_hybrid_llm.py' im selben Verzeichnis oder im Python-Pfad liegt.")
    sys.exit(1)

def train_hybrid_model(config_path: str, state_path: str, force_rebuild: bool = False):
    """
    Hauptfunktion zum Trainieren/Verarbeiten der Daten mit dem Hybridmodell
    über mehrere Epochen.
    """
    print("="*50)
    print(" Starte Training/Datenverarbeitung für Quantum-Arona Hybrid LLM")
    print(f" - Konfiguration: {config_path}")
    print(f" - Zustandsdatei: {state_path}")
    print(f" - Neuaubau erzwingen: {force_rebuild}")
    print("="*50)

    start_time = time.time()

    # Lade zuerst die aktuelle Config aus der Datei
    config_from_file = None
    print(f"INFO: Lese aktuelle Konfiguration aus {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_from_file = json.load(f)
    except Exception as e:
        print(f"FATALER FEHLER: Konnte Konfigurationsdatei '{config_path}' nicht laden: {e}")
        sys.exit(1)

    # 1. Lade Zustand oder initialisiere neu
    processor: Optional[QuantumEnhancedTextProcessor] = None
    if not force_rebuild and os.path.exists(state_path):
        print(f"\nVersuche, Zustand aus '{state_path}' zu laden...")
        processor = QuantumEnhancedTextProcessor.load_state(state_path) # Verwendet Config aus State
        if processor:
            print("INFO: Versuche Konfiguration im geladenen Prozessor zu aktualisieren...")
            try:
                processor.config.update(config_from_file) # Wichtig: Update mit aktueller Config
                # Aktualisiere abhängige Attribute im Prozessor-Objekt
                processor.self_learning_enabled = processor.config.get("enable_self_learning", False)
                processor.learn_file_path = processor.config.get("self_learning_file_path", "./training_data/learn.txt")
                processor.learn_source_name = processor.config.get("self_learning_source_name", "Generated Responses")
                # Aktualisiere RAG Status basierend auf aktueller Config und Verfügbarkeit
                processor.rag_enabled = processor.config.get("enable_rag", False) and GEMINI_AVAILABLE
                print(" -> Konfiguration im geladenen Prozessor mit aktueller Datei aktualisiert.")
            except Exception as e:
                print(f"WARNUNG: Konnte Konfiguration im geladenen Prozessor nicht vollständig aktualisieren: {e}")
        else:
            print(f" -> Laden des Zustands fehlgeschlagen oder Datei leer.")

    # Wenn kein Prozessor geladen wurde, initialisiere neu
    if processor is None:
        if force_rebuild: print(f"\nNeuerstellung erzwungen.")
        print(f"\nInitialisiere Modell mit Konfiguration aus '{config_path}'.")
        try:
            processor = QuantumEnhancedTextProcessor(config_dict=config_from_file) # Init mit aktueller Config
        except Exception as e:
             print(f"\nFATALER FEHLER: Initialisierung fehlgeschlagen: {e}"); sys.exit(1)
        if not processor or not hasattr(processor, 'config'):
            print("\nFATALER FEHLER: Prozessor-Objekt konnte nicht korrekt initialisiert werden."); sys.exit(1)

    # Stelle sicher, dass wir einen gültigen Prozessor haben
    if processor is None: print("\nFATALER FEHLER: Konnte keinen Prozessor laden oder initialisieren."); sys.exit(1)

    # --- Finale Prüfung vor Training ---
    print(f"\nINFO: Finale Qubit-Anzahl VOR dem Training: {processor.config.get('default_num_qubits')}")
    print(f"INFO: Self-Learning VOR dem Training: {'Aktiviert' if processor.self_learning_enabled else 'Deaktiviert'}")
    print(f"INFO: RAG Status VOR Training: {'Aktiviert' if processor.rag_enabled else 'Deaktiviert'}")
    print(f"INFO: Simulation Steps After Training VOR Training: {processor.config.get('simulation_steps_after_training')}")

    # Lese Anzahl der Epochen aus der Konfiguration
    num_epochs = processor.config.get("training_epochs", 1)
    print(f"\nAnzahl der Trainingsepochen: {num_epochs}")

    # 2. Verarbeite Trainingsdateien
    training_files = processor.config.get("training_files", [])
    if not training_files:
        print("\nWARNUNG: Keine Trainingsdateien in der Konfiguration gefunden ('training_files').")
    else:
        print(f"\n--- Schritt 1: Verarbeite/Aktualisiere Chunks aus {len(training_files)} Trainingsdatei(en) ---")
        files_processed_count = 0
        file_iterator = tqdm(training_files, desc="Verarbeite Dateien", leave=False) if TQDM_AVAILABLE else training_files
        for file_path in file_iterator:
            if os.path.exists(file_path):
                 try:
                     processor.load_and_process_file(file_path) # Chunking & Processing
                     files_processed_count += 1
                 except Exception as load_err:
                      print(f"FEHLER beim Verarbeiten von Datei '{file_path}': {load_err}")
                      traceback.print_exc(limit=1)
            else:
                 print(f"WARNUNG: Trainingsdatei '{file_path}' nicht gefunden. Übersprungen.")

        if files_processed_count > 0:
            print(f" -> {files_processed_count} vorhandene Trainingsdateien verarbeitet/aktualisiert.")
        else:
            print(" -> Keine vorhandenen Trainingsdateien gefunden oder verarbeitet.")

    # 3. Führe das eigentliche Training über Epochen durch
    if not processor.chunks:
        print("\nWARNUNG: Keine Chunks zum Trainieren vorhanden. Überspringe Epochen-Training.")
    else:
        all_chunk_ids = list(processor.chunks.keys())
        print(f"\n--- Schritt 2: Beginne Epochen-Training über {num_epochs} Epoche(n) für {len(all_chunk_ids)} Chunks ---")
        for epoch in range(1, num_epochs + 1):
            print(f"\n🚀 Epoche {epoch}/{num_epochs}")
            random.shuffle(all_chunk_ids)
            epoch_chunk_iterator = tqdm(all_chunk_ids, desc=f"Epoch {epoch}", leave=False) if TQDM_AVAILABLE else all_chunk_ids
            chunks_processed_in_epoch = 0
            for chunk_uuid in epoch_chunk_iterator:
                 if chunk_uuid in processor.chunks:
                      try:
                          processor.process_chunk(processor.chunks[chunk_uuid]) # Stärkt Verbindungen
                          chunks_processed_in_epoch += 1
                      except Exception as process_err:
                           print(f"FEHLER beim Verarbeiten von Chunk UUID {chunk_uuid}: {process_err}")
                           traceback.print_exc(limit=1)
            print(f"   -> Epoche {epoch} abgeschlossen ({chunks_processed_in_epoch} Chunks verarbeitet).")

    # 4. (Optional) Führe Netzwerk-Simulationsschritte nach dem Training durch
    simulation_steps = processor.config.get("simulation_steps_after_training", 0)
    if simulation_steps > 0 and processor.nodes:
        print(f"\n--- Schritt 3: Führe {simulation_steps} Netzwerk-Simulationsschritte durch ---")
        sim_iterator = tqdm(range(simulation_steps), desc="Simulationsschritte", leave=False) if TQDM_AVAILABLE else range(simulation_steps)
        for i in sim_iterator:
             try:
                 processor.simulate_network_step(decay_connections=True) # Mit Decay
             except Exception as sim_err:
                 print(f"FEHLER während Simulationsschritt {i+1}: {sim_err}")
                 traceback.print_exc(limit=1); break # Breche Simulation ab bei Fehler
        print("--- Simulation abgeschlossen ---")
    elif simulation_steps > 0:
         print("\nWARNUNG: Simulation übersprungen, da keine Knoten im Netzwerk vorhanden sind.")
    else:
         print("\nINFO: Keine Simulationsschritte nach dem Training konfiguriert (simulation_steps_after_training = 0).")

    # Berechne finale Aktivierungen einmal ohne Decay vor dem Speichern/Summary
    print("\n--- Berechne finale Knotenaktivierungen für Zustandsspeicherung/Summary ---")
    if processor.nodes:
        try:
            processor.simulate_network_step(decay_connections=False) # Ohne Decay
        except Exception as final_sim_err:
             print(f"FEHLER während finaler Aktivierungsberechnung: {final_sim_err}")
             traceback.print_exc(limit=1)
    else:
        print("   -> Übersprungen, keine Knoten vorhanden.")

    # --- Reihenfolge: Erst Summary, dann Speichern ---

    # Berechne die Summary zuerst (greift auf Live-Objekte zu)
    final_summary = {}
    print("\n--- Berechne finale Summary (aus In-Memory Zustand) ---")
    try:
        # Stelle sicher, dass die korrigierte Summary-Methode verwendet wird
        final_summary = processor.get_network_state_summary()
        print("   -> Summary erfolgreich berechnet.")
    except Exception as summary_err:
         print(f"FEHLER beim Erstellen der Netzwerk-Summary: {summary_err}")
         traceback.print_exc(limit=1)

    # Speichere den Zustand danach (ruft die korrigierte __getstate__ auf)
    print(f"\n--- Schritt 4: Speichere finalen Zustand nach '{state_path}' ---")
    processor.save_state(state_path) # Verwendet die korrigierte __getstate__

    # --- Ende Reihenfolge ---

    end_time = time.time()
    print("\n--- Zusammenfassung des Laufs ---")
    print(f" - Gesamtdauer: {end_time - start_time:.2f} Sekunden")
    print(" - Finaler Netzwerkstatus (aus In-Memory, VOR dem Speichern berechnet):")
    if isinstance(final_summary, dict) and final_summary: # Prüfe ob Dict und nicht leer
        print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    else:
        print("Fehler: Konnte finale Summary nicht als Dictionary erstellen oder sie ist leer.")
    print("="*50)
    print(" Training/Datenverarbeitung beendet.")
    print("="*50)

# --- Ende train_hybrid_model ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainiert das Quantum-Arona Hybrid LLM über mehrere Epochen.")
    parser.add_argument(
        "-c", "--config",
        default="config_qllm.json",
        help="Pfad zur JSON-Konfigurationsdatei (default: config_qllm.json)"
    )
    parser.add_argument(
        "-s", "--state",
        default="qetp_state.json",
        help="Pfad zur Zustandsdatei für Speichern/Laden (default: qetp_state.json)"
    )
    parser.add_argument(
        "-f", "--force-rebuild",
        action="store_true",
        help="Ignoriert einen vorhandenen Zustand und baut das Modell neu auf (inkl. Chunks)."
    )

    args = parser.parse_args()

    # Stelle sicher, dass die Konfigurationsdatei existiert
    if not os.path.exists(args.config):
        print(f"FEHLER: Konfigurationsdatei nicht gefunden: {args.config}")
        sys.exit(1)

    print(f"INFO: Verwende Konfigurationsdatei: {args.config}")
    print(f"INFO: Verwende Zustandsdatei: {args.state}")
    if args.force_rebuild:
        print("INFO: Neuerstellung des Zustands ist erzwungen (--force-rebuild).")

    # Rufe die Haupt-Trainingsfunktion auf
    try:
        train_hybrid_model(args.config, args.state, args.force_rebuild)
    except Exception as main_err:
        print(f"\nFATALER FEHLER im Haupt-Trainingsprozess: {main_err}")
        traceback.print_exc()
        sys.exit(1)