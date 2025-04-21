# -- coding: utf-8 --

# Filename: qllm_train_hybrid.py
# Description: Trainingsskript für das Quantum-Arona Hybrid LLM.
#              Lädt Daten, verarbeitet sie mit QuantumEnhancedTextProcessor
#              über mehrere Epochen und speichert den gelernten Zustand.
# Version: 0.5 - Epoch Training & Qubit Debugging
# Author: [CipherCore Technology] & Gemini & Your Input

import os
import sys
import time
import json
import argparse
import copy # Für deepcopy von Chunks
import random
from typing import Optional # Für Type Hinting

# Füge das Verzeichnis hinzu, in dem sich quantum_arona_hybrid_llm.py befindet
try:
    # Importiere die Hauptklasse und ggf. Typen für Type Hinting
    from quantum_arona_hybrid_llm import QuantumEnhancedTextProcessor, TextChunk
    # Optional: tqdm importieren
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs): # Fallback-Implementierung
            # print("Info: tqdm nicht gefunden, Fortschrittsbalken nicht verfügbar.") # Weniger verbose
            return iterable
except ImportError:
    print("FEHLER: Konnte 'QuantumEnhancedTextProcessor' nicht importieren.")
    print("Stelle sicher, dass 'quantum_arona_hybrid_llm.py' im selben Verzeichnis oder im Python-Pfad liegt.")
    sys.exit(1)

def train_hybrid_model(config_path: str, state_path: str, force_rebuild: bool = False):
    """
    Hauptfunktion zum Trainieren/Verarbeiten der Daten mit dem Hybridmodell
    über mehrere Epochen.

    Args:
        config_path (str): Pfad zur Konfigurationsdatei (JSON).
        state_path (str): Pfad zur Datei, in der der Zustand gespeichert/geladen wird.
        force_rebuild (bool): Wenn True, wird ein vorhandener Zustand ignoriert
                              und das Modell von Grund auf neu aufgebaut.
    """
    print("="*50)
    print(" Starte Training/Datenverarbeitung für Quantum-Arona Hybrid LLM")
    print(f" - Konfiguration: {config_path}")
    print(f" - Zustandsdatei: {state_path}")
    print(f" - Neuaubau erzwingen: {force_rebuild}")
    print("="*50)

    start_time = time.time()

    # Variable für die aus Datei geladene Config
    config_from_file = None

    # Lade zuerst die aktuelle Config aus der Datei
    print(f"INFO: Lese aktuelle Konfiguration aus {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_from_file = json.load(f)
        # --- DEBUGGING CONFIG AUS DATEI ---
        print(f"DEBUG: Wert für 'default_num_qubits' aus Datei {config_path}: {config_from_file.get('default_num_qubits')}")
        # --- ENDE DEBUGGING ---
    except Exception as e:
        print(f"FATALER FEHLER: Konnte Konfigurationsdatei '{config_path}' nicht laden: {e}")
        sys.exit(1) # Beenden, wenn die Haupt-Config nicht gelesen werden kann

    # 1. Lade Zustand oder initialisiere neu
    processor: Optional[QuantumEnhancedTextProcessor] = None
    if not force_rebuild and os.path.exists(state_path):
        print(f"\nVersuche, Zustand aus '{state_path}' zu laden...")
        # Übergebe die GERADE GELADENE Config an load_state, damit __init__ den korrekten Wert hat,
        # falls der Prozessor neu erstellt werden muss (obwohl das nicht der Fall sein sollte, wenn state existiert)
        # Wichtiger: Das Update NACH dem Laden.
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            # --- DEBUGGING NACH LADEN ---
            print(f"DEBUG: Qubits im Prozessor VOR Update (aus geladenem State): {processor.config.get('default_num_qubits')}")
            # --- ENDE DEBUGGING ---
            print("DEBUG: Versuche Konfiguration im geladenen Prozessor zu aktualisieren...")
            try:
                # Aktualisiere die Config des geladenen Prozessors mit der aus der Datei
                processor.config.update(config_from_file)
                # --- DEBUGGING NACH UPDATE ---
                print(f"DEBUG: Qubits im Prozessor NACH Update mit {config_path}: {processor.config.get('default_num_qubits')}")
                # --- ENDE DEBUGGING ---
                print(" -> Konfiguration im geladenen Prozessor mit aktueller Datei aktualisiert.")
            except Exception as e:
                print(f"WARNUNG: Konnte Konfiguration im geladenen Prozessor nicht aktualisieren: {e}")
        else:
            print(f" -> Laden des Zustands fehlgeschlagen oder Datei leer.")

    # Wenn kein Prozessor geladen wurde, initialisiere neu mit der Config aus der Datei
    if processor is None:
        if force_rebuild: print(f"\nNeuerstellung erzwungen.")
        print(f"\nInitialisiere Modell mit Konfiguration aus '{config_path}'.")
        try:
            # Initialisiere direkt mit der bereits geladenen config_from_file
            processor = QuantumEnhancedTextProcessor(config_dict=config_from_file)
            # --- DEBUGGING NACH INIT ---
            if processor:
                 print(f"DEBUG: Qubits nach Neu-Initialisierung: {processor.config.get('default_num_qubits')}")
            # --- ENDE DEBUGGING ---
        except Exception as e:
             print(f"\nFATALER FEHLER: Initialisierung fehlgeschlagen: {e}"); sys.exit(1)
        if not processor or not hasattr(processor, 'config'): print("\nFATALER FEHLER: Prozessor-Objekt konnte nicht korrekt initialisiert werden."); sys.exit(1)

    # Stelle sicher, dass wir einen gültigen Prozessor haben
    if processor is None: print("\nFATALER FEHLER: Konnte keinen Prozessor laden oder initialisieren."); sys.exit(1)

    # --- FINALE PRÜFUNG VOR TRAINING ---
    print(f"\nDEBUG: Finale Qubit-Anzahl VOR dem Training: {processor.config.get('default_num_qubits')}")
    # --- ENDE FINALE PRÜFUNG ---

    # Lese Anzahl der Epochen aus der (hoffentlich aktualisierten) Konfiguration
    num_epochs = processor.config.get("training_epochs", 1)
    print(f"\nAnzahl der Trainingsepochen: {num_epochs}")

    # 2. Verarbeite Trainingsdateien (Laden/Chunking) - nur einmal, wenn noch nicht geschehen
    training_files = processor.config.get("training_files", [])
    new_chunks_added_this_run = False
    if not training_files:
        print("\nWARNUNG: Keine Trainingsdateien in der Konfiguration gefunden ('training_files').")
    else:
        print(f"\n--- Schritt 1: Prüfe/Lade Chunks aus {len(training_files)} Trainingsdatei(en) ---")
        initial_chunk_count = len(processor.chunks)
        for file_path in training_files:
            processor.load_and_process_file(file_path)
        if len(processor.chunks) > initial_chunk_count:
            new_chunks_added_this_run = True
            print(" -> Neue Chunks wurden hinzugefügt oder erstmalig geladen.")
        else:
             print(" -> Keine neuen Chunks hinzugefügt (Dateien waren vermutlich schon bekannt).")

        # Stelle sicher, dass der TF-IDF Index aktuell ist
        # Überprüfung hinzugefügt, ob Vektorizer bereits existiert
        if not hasattr(processor, 'vectorizer') or processor.vectorizer is None or new_chunks_added_this_run:
             processor.update_tfidf_index()


    # 3. Führe das eigentliche Training über Epochen durch
    if not processor.chunks:
        print("\nWARNUNG: Keine Chunks zum Trainieren vorhanden. Überspringe Epochen-Training.")
    else:
        all_chunk_ids = list(processor.chunks.keys())
        print(f"\n--- Schritt 2: Beginne Training über {num_epochs} Epoche(n) für {len(all_chunk_ids)} Chunks ---")

        for epoch in range(1, num_epochs + 1):
            print(f"\n🚀 Epoche {epoch}/{num_epochs}")
            random.shuffle(all_chunk_ids)
            epoch_chunk_iterator = tqdm(all_chunk_ids, desc=f"Epoch {epoch}", leave=False) if TQDM_AVAILABLE else all_chunk_ids
            for chunk_uuid in epoch_chunk_iterator:
                 if chunk_uuid in processor.chunks:
                      processor.process_chunk(processor.chunks[chunk_uuid])

            print(f"   -> Epoche {epoch} abgeschlossen.")
            # Optional: Zwischenspeichern
            # if epoch % 10 == 0 and epoch < num_epochs:
            #    print(f"   -> Speichere Zwischenzustand nach Epoche {epoch}...")
            #    processor.save_state(f"{state_path}.epoch{epoch}")


    # 4. (Optional) Führe Netzwerk-Simulationsschritte nach dem Training durch
    simulation_steps = processor.config.get("simulation_steps_after_training", 0)
    if simulation_steps > 0 and processor.nodes:
        print(f"\n--- Schritt 3: Führe {simulation_steps} Netzwerk-Simulationsschritte durch ---")
        sim_iterator = tqdm(range(simulation_steps), desc="Simulationsschritte", leave=False) if TQDM_AVAILABLE else range(simulation_steps)
        for i in sim_iterator:
             processor.simulate_network_step(decay_connections=True)
        print("--- Simulation abgeschlossen ---")
    elif simulation_steps > 0:
         print("\nWARNUNG: Simulation übersprungen, da keine Knoten im Netzwerk vorhanden sind.")

    # Berechne finale Aktivierungen vor Summary/Speichern
    print("\n--- Berechne finale Knotenaktivierungen für Summary ---")
    if processor.nodes:
        processor.simulate_network_step(decay_connections=False)
    else:
        print("   -> Übersprungen, keine Knoten vorhanden.")

    # 5. Speichere den finalen Zustand
    print(f"\n--- Schritt 4: Speichere finalen Zustand nach '{state_path}' ---")
    processor.save_state(state_path)

    end_time = time.time()
    print("\n--- Zusammenfassung des Laufs ---")
    print(f" - Gesamtdauer: {end_time - start_time:.2f} Sekunden")
    # Rufe Summary NACH dem Speichern auf (oder davor, sollte keinen Unterschied machen)
    final_summary = processor.get_network_state_summary()
    print(" - Finaler Netzwerkstatus:")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print("="*50)
    print(" Training/Datenverarbeitung beendet.")
    print("="*50)


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

    # Stelle sicher, dass die Konfigurationsdatei existiert, bevor sie gelesen wird
    if not os.path.exists(args.config):
        print(f"FEHLER: Konfigurationsdatei nicht gefunden: {args.config}")
        sys.exit(1)

    print(f"INFO: Verwende Konfigurationsdatei: {args.config}")
    print(f"INFO: Verwende Zustandsdatei: {args.state}")

    train_hybrid_model(args.config, args.state, args.force_rebuild)