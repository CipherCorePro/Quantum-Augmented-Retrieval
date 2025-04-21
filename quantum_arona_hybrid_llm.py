# -- coding: utf-8 --

# Filename: quantum_arona_hybrid_llm.py
# Version: 1.2 - Implemented active LimbusAffektus modulation
# Author: [CipherCore Technology] & Gemini & Your Input & History Maker

import numpy as np
import pandas as pd
import random
from collections import deque, Counter, defaultdict
import json
# import sqlite3 # Vorerst nicht verwendet
import os
import time
import traceback
from typing import Optional, Callable, List, Tuple, Dict, Any, Generator
from datetime import datetime
import math # Hinzugefügt für LimbusAffektus (tanh)
import uuid as uuid_module
import re

# Text Processing / Retrieval specific imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Imports für Gemini API ---
try:
    import google.generativeai as genai
    # Optional: Importiere Typen für Fehlerbehandlung
    from google.api_core.exceptions import GoogleAPIError
    GEMINI_AVAILABLE = True
    print("INFO: Google Generative AI SDK gefunden.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNUNG: 'google-generativeai' nicht gefunden. RAG-Funktionalität (Textgenerierung) ist deaktiviert.")
    print("Installieren Sie es mit: pip install google-generativeai")
    genai = None
    GoogleAPIError = None
# --- Ende Imports ---

# Optional: Netzwerk-Visualisierung
try: import networkx as nx; NETWORKX_AVAILABLE = True
except ImportError: NETWORKX_AVAILABLE = False

# Optional: Fortschrittsbalken
try: from tqdm import tqdm; TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs): return iterable

# --- HILFSFUNKTIONEN & BASIS-GATES ---
# Definiere Quantengates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex) # Projektor |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex) # Projektor |1><1|

def _ry(theta: float) -> np.ndarray:
    """Erzeugt eine RY-Rotationsmatrix."""
    if not np.isfinite(theta): theta = 0.0 # Fallback für ungültige Winkel
    cos_t = np.cos(theta / 2); sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

def _rz(phi: float) -> np.ndarray:
    """Erzeugt eine RZ-Rotationsmatrix."""
    if not np.isfinite(phi): phi = 0.0 # Fallback für ungültige Winkel
    exp_m = np.exp(-1j * phi / 2); exp_p = np.exp(1j * phi / 2)
    return np.array([[exp_m, 0], [0, exp_p]], dtype=complex)

def _apply_gate(state_vector: np.ndarray, gate: np.ndarray, target_qubit: int, num_qubits: int) -> np.ndarray:
    """Wendet ein Single-Qubit-Gate auf den Zustandsvektor an."""
    if gate.shape != (2, 2): raise ValueError("Gate muss 2x2 sein.")
    if not (0 <= target_qubit < num_qubits): raise ValueError(f"Target qubit {target_qubit} out of range [0, {num_qubits-1}].")

    # Sicherstellen, dass der state_vector die korrekte Größe hat
    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        # Fallback: Initialisiere zu |0...0> wenn die Größe nicht stimmt
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0

    # Erstelle die vollständige Operator-Matrix für das gesamte System
    op_list = [I] * num_qubits
    op_list[target_qubit] = gate

    # Berechne das Kronecker-Produkt aller Operatoren
    full_matrix = op_list[0]
    for i in range(1, num_qubits):
        full_matrix = np.kron(full_matrix, op_list[i])

    # Wende die Matrix auf den Zustandsvektor an
    new_state = np.dot(full_matrix, state_vector)

    # Prüfe auf NaN/inf und normalisiere oder resette
    if not np.all(np.isfinite(new_state)):
        # print(f"WARNUNG: Nicht-endlicher Zustand nach Gate {gate} auf Qubit {target_qubit}. Reset zu |0...0>.")
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    # elif np.linalg.norm(new_state) < 1e-9: # Optional: Prüfung auf Nullvektor
    #    print(f"WARNUNG: Zustand wurde nach Gate {gate} auf Qubit {target_qubit} fast zu Null. Reset zu |0...0>.")
    #    new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0

    return new_state

def _apply_cnot(state_vector: np.ndarray, control_qubit: int, target_qubit: int, num_qubits: int) -> np.ndarray:
    """Wendet ein CNOT-Gate auf den Zustandsvektor an."""
    if not (0 <= control_qubit < num_qubits and 0 <= target_qubit < num_qubits): raise ValueError("Qubit index out of range.")
    if control_qubit == target_qubit: raise ValueError("Control and target must be different.")

    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0

    # Konstruiere CNOT-Matrix mittels Projektoren: |0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t
    op_list_p0 = [I] * num_qubits
    op_list_p1 = [I] * num_qubits

    op_list_p0[control_qubit] = P0 # Projektor |0><0| auf Kontroll-Qubit
    op_list_p1[control_qubit] = P1 # Projektor |1><1| auf Kontroll-Qubit
    op_list_p1[target_qubit] = X  # X-Gate auf Target-Qubit, wenn Kontrolle |1> ist

    # Berechne die Kronecker-Produkte für beide Terme
    term0_matrix = op_list_p0[0]; term1_matrix = op_list_p1[0]
    for i in range(1, num_qubits):
        term0_matrix = np.kron(term0_matrix, op_list_p0[i])
        term1_matrix = np.kron(term1_matrix, op_list_p1[i])

    # Addiere die Matrizen der beiden Terme
    cnot_matrix = term0_matrix + term1_matrix

    # Wende die CNOT-Matrix an
    new_state = np.dot(cnot_matrix, state_vector)

    if not np.all(np.isfinite(new_state)):
        # print(f"WARNUNG: Nicht-endlicher Zustand nach CNOT({control_qubit}, {target_qubit}). Reset zu |0...0>.")
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    # elif np.linalg.norm(new_state) < 1e-9: # Optional: Prüfung auf Nullvektor
    #    print(f"WARNUNG: Zustand wurde nach CNOT({control_qubit}, {target_qubit}) fast zu Null. Reset zu |0...0>.")
    #    new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0

    return new_state

# --- QUANTEN-ENGINE ---
class QuantumNodeSystem:
    """Simuliert das quantenbasierte Verhalten eines Knotens via PQC."""
    def __init__(self, num_qubits: int, initial_params: Optional[np.ndarray] = None):
        if not isinstance(num_qubits, int) or num_qubits <= 0: raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.num_params = num_qubits * 2 # Je ein RY und ein RZ Parameter pro Qubit
        self.state_vector_size = 2**self.num_qubits

        # Initialisiere Parameter sicher
        if initial_params is None:
            self.params = np.random.rand(self.num_params) * 2 * np.pi
        elif isinstance(initial_params, np.ndarray) and initial_params.shape == (self.num_params,):
             # Bereinige NaN oder Inf Werte, falls vorhanden
             safe_params = np.nan_to_num(initial_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
             self.params = np.clip(safe_params, 0, 2 * np.pi)
        else:
            print(f"WARNUNG: Ungültige initial_params (Typ: {type(initial_params)}, Shape: {getattr(initial_params, 'shape', 'N/A')}). Initialisiere zufällig.")
            self.params = np.random.rand(self.num_params) * 2 * np.pi

        # Initialisiere Zustandsvektor sicher als |0...0>
        self.state_vector = np.zeros(self.state_vector_size, dtype=complex); self.state_vector[0] = 1.0 + 0j
        self.last_measurement_results: List[Dict] = []
        self.last_applied_ops: List[Tuple] = []

    def _build_pqc_ops(self, input_strength: float) -> List[Tuple]:
        """Erstellt die Liste der Gate-Operationen für den PQC."""
        ops = []
        # Skaliere Input für Rotationswinkel (tanh begrenzt auf [-1, 1], dann mal Pi)
        scaled_input_angle = np.tanh(input_strength) * np.pi
        if not np.isfinite(scaled_input_angle): scaled_input_angle = 0.0 # Fallback

        # Layer 1: Hadamard auf alle Qubits
        for i in range(self.num_qubits): ops.append(('H', i))

        # Layer 2: Parametrisierte RY-Rotationen (beeinflusst durch Input)
        for i in range(self.num_qubits):
            theta = scaled_input_angle * self.params[2 * i] # Winkel skaliert mit Input & erstem Param
            ops.append(('RY', i, theta if np.isfinite(theta) else 0.0))

        # Layer 3: Parametrisierte RZ-Rotationen (nicht direkt vom Input abhängig)
        for i in range(self.num_qubits):
            phi = self.params[2 * i + 1] # Winkel ist der zweite Param
            ops.append(('RZ', i, phi if np.isfinite(phi) else 0.0))

        # Layer 4: CNOT-Verschränkung (einfache lineare Kette)
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1): ops.append(('CNOT', i, i + 1))

        return ops

    def activate(self, input_strength: float, n_shots: int = 100) -> Tuple[float, np.ndarray, List[Dict]]:
        """Führt den PQC aus und gibt die Aktivierungswahrscheinlichkeit zurück."""
        if not np.isfinite(input_strength): input_strength = 0.0
        if n_shots <= 0: n_shots = 1 # Mindestens ein Schuss für Wahrscheinlichkeiten

        pqc_ops = self._build_pqc_ops(input_strength)
        self.last_applied_ops = pqc_ops # Für Debugging speichern

        current_state = self.state_vector.copy()
        # Stelle sicher, dass der Startzustand normalisiert ist
        if not np.isclose(np.linalg.norm(current_state), 1.0):
            current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0

        gate_application_successful = True
        # Wende Gates an
        for op_index, op in enumerate(pqc_ops):
            try:
                op_type = op[0]
                if op_type == 'H': current_state = _apply_gate(current_state, H, op[1], self.num_qubits)
                elif op_type == 'RY': current_state = _apply_gate(current_state, _ry(op[2]), op[1], self.num_qubits)
                elif op_type == 'RZ': current_state = _apply_gate(current_state, _rz(op[2]), op[1], self.num_qubits)
                elif op_type == 'CNOT': current_state = _apply_cnot(current_state, op[1], op[2], self.num_qubits)

                # Prüfe nach jedem Gate auf Gültigkeit und normalisiere
                if not np.all(np.isfinite(current_state)): raise ValueError(f"Non-finite state after {op}")
                norm = np.linalg.norm(current_state)
                if norm > 1e-9: current_state /= norm # Normalisieren
                else: raise ValueError(f"Zero state after {op}")

            except Exception as e:
                # print(f"FEHLER: Gate-Fehler QNS bei Op {op_index} ({op}): {e}. Reset state vector.");
                current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0
                gate_application_successful = False
                break # Breche Gate-Anwendung ab

        # Aktualisiere den internen Zustand
        self.state_vector = current_state

        # --- Messung und Aktivierungsberechnung ---
        total_hamming_weight = 0
        measurement_log = []
        activation_prob = 0.0

        if n_shots > 0 and gate_application_successful and self.num_qubits > 0:
            # Berechne Wahrscheinlichkeiten
            probabilities = np.abs(current_state)**2
            # Numerische Stabilität: Setze sehr kleine negative Zahlen auf 0
            probabilities = np.maximum(0, probabilities)
            prob_sum = np.sum(probabilities)

            # Normalisiere Wahrscheinlichkeiten, falls nötig (sollte ~1 sein)
            if not np.isclose(prob_sum, 1.0, atol=1e-7):
                if prob_sum < 1e-9: # Fast Null, setze auf Basiszustand
                    probabilities.fill(0.0); probabilities[0] = 1.0
                else:
                    probabilities /= prob_sum # Normalisiere
                # Erneute Prüfung nach Division
                probabilities = np.maximum(0, probabilities)
                probabilities /= np.sum(probabilities) # Erzwinge Summe 1

            # Führe Messungen durch
            try:
                measured_indices = np.random.choice(self.state_vector_size, size=n_shots, p=probabilities)
                # Zähle Hamming-Gewichte für die Aktivierung
                for shot_idx, measured_index in enumerate(measured_indices):
                    state_idx_int = int(measured_index)
                    binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                    hamming_weight = binary_repr.count('1')
                    total_hamming_weight += hamming_weight
                    measurement_log.append({
                        "shot": shot_idx, "index": state_idx_int,
                        "binary": binary_repr, "hamming": hamming_weight,
                        "probability": probabilities[state_idx_int] # Wahrscheinlichkeit des gemessenen Zustands
                    })
            except ValueError as e: # Fehler bei np.random.choice (z.B. wegen Wahrscheinlichkeitssumme != 1 trotz Korrektur)
                 print(f"WARNUNG: np.random.choice Fehler in QNS ({e}). Fallback zu argmax.");
                 if np.any(probabilities): # Nur wenn es überhaupt Wahrscheinlichkeiten gibt
                     measured_index = np.argmax(probabilities)
                     state_idx_int = int(measured_index)
                     binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                     hamming_weight = binary_repr.count('1')
                     total_hamming_weight = hamming_weight * n_shots # Schätze Total basierend auf wahrscheinlichstem Zustand
                     measurement_log.append({"shot": 0, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight, "error": "ValueError, used argmax", "probability": probabilities[state_idx_int]})
                 else: # Keine Wahrscheinlichkeiten -> kein Hamming-Gewicht
                     measurement_log.append({"shot": 0, "index": 0, "binary": '0'*self.num_qubits, "hamming": 0, "error": "All probabilities zero", "probability": 0.0})

            # Berechne finale Aktivierungswahrscheinlichkeit (Durchschnittliches Hamming-Gewicht pro Qubit)
            if n_shots > 0 and self.num_qubits > 0:
                activation_prob = float(np.clip(total_hamming_weight / (n_shots * self.num_qubits), 0.0, 1.0))
                if not np.isfinite(activation_prob): activation_prob = 0.0 # Fallback

        elif not gate_application_successful:
             activation_prob = 0.0 # Keine Aktivierung bei Gate-Fehler
             measurement_log = [{"error": "PQC execution failed"}]

        self.last_measurement_results = measurement_log

        # Stelle sicher, dass der Rückgabewert ein gültiger float ist
        if not isinstance(activation_prob, (float, np.number)) or not np.isfinite(activation_prob):
            activation_prob = 0.0

        return activation_prob, self.state_vector, measurement_log

    def get_params(self) -> np.ndarray:
        """Gibt eine sichere Kopie der aktuellen Parameter zurück."""
        safe_params = np.nan_to_num(self.params.copy(), nan=np.pi, posinf=2*np.pi, neginf=0.0)
        return np.clip(safe_params, 0, 2 * np.pi)

    def set_params(self, params: np.ndarray):
        """Setzt die Parameter des Systems sicher."""
        if isinstance(params, np.ndarray) and params.shape == self.params.shape:
            safe_params = np.nan_to_num(params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
            self.params = np.clip(safe_params, 0, 2 * np.pi)
        # else: print("WARNUNG: Versuch, QNS Parameter mit ungültigem Array zu setzen.")

    def update_internal_params(self, delta_params: np.ndarray):
        """Aktualisiert die internen Parameter um delta_params."""
        if not isinstance(delta_params, np.ndarray) or delta_params.shape != self.params.shape: return
        safe_delta = np.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0) # Ignoriere ungültige Deltas
        new_params = self.params + safe_delta
        new_params_safe = np.nan_to_num(new_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        self.params = np.clip(new_params_safe, 0, 2 * np.pi) # Clip im gültigen Bereich

# --- NETZWERK-STRUKTUR & TEXT-CHUNKS ---
class Connection:
    """Repräsentiert eine gerichtete, gewichtete Verbindung zwischen zwei Knoten."""
    DEFAULT_WEIGHT_RANGE = (0.01, 0.5) # Standard-Initialisierungsgewicht
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_DECAY_RATE = 0.001

    def __init__(self, target_node: 'Node', weight: Optional[float] = None, source_node_label: Optional[str] = None, conn_type: str = "associative"):
        if target_node is None or not hasattr(target_node, 'uuid'): raise ValueError("Target node invalid or missing UUID.")
        # Speichere Zielknoten-UUID statt Objekt-Referenz zur einfacheren Serialisierung
        self.target_node_uuid: str = target_node.uuid
        self.source_node_label: Optional[str] = source_node_label
        self.conn_type: str = conn_type
        raw_weight = weight if weight is not None else random.uniform(*self.DEFAULT_WEIGHT_RANGE)
        self.weight: float = float(np.clip(raw_weight, 0.0, 1.0)) # Gewicht zwischen 0 und 1
        self.last_transmitted_signal: float = 0.0
        self.transmission_count: int = 0
        self.created_at: datetime = datetime.now()
        self.last_update_at: datetime = datetime.now()

    def update_weight(self, delta_weight: float, learning_rate: Optional[float] = None):
        """Aktualisiert das Gewicht der Verbindung."""
        lr = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        new_weight = self.weight + (delta_weight * lr)
        self.weight = float(np.clip(new_weight, 0.0, 1.0)) # Clip im gültigen Bereich
        self.last_update_at = datetime.now()

    def decay(self, decay_rate: Optional[float] = None):
        """Reduziert das Gewicht der Verbindung um einen Zerfallsfaktor."""
        dr = decay_rate if decay_rate is not None else self.DEFAULT_DECAY_RATE
        self.weight = max(0.0, self.weight * (1.0 - dr)) # Gewicht kann nicht negativ werden
        self.last_update_at = datetime.now()

    def transmit(self, source_activation: float) -> float:
        """Berechnet das übertragene Signal basierend auf der Quellaktivierung und dem Gewicht."""
        transmitted_signal = source_activation * self.weight
        self.last_transmitted_signal = transmitted_signal
        self.transmission_count += 1
        return transmitted_signal

    def __repr__(self) -> str:
        """Gibt eine String-Repräsentation der Verbindung zurück."""
        target_info = f"to_UUID:{self.target_node_uuid[:8]}..."
        source_info = f" from:{self.source_node_label}" if self.source_node_label else ""
        weight_info = f"W:{self.weight:.3f}" if hasattr(self, 'weight') else "W:N/A"
        count_info = f"Cnt:{self.transmission_count}" if hasattr(self, 'transmission_count') else "Cnt:N/A"
        return f"<Conn {target_info} {weight_info} {count_info}{source_info}>"

class Node:
    """Basisklasse für alle Knoten im Netzwerk."""
    DEFAULT_NUM_QUBITS = 10
    DEFAULT_ACTIVATION_HISTORY_LEN = 20
    DEFAULT_N_SHOTS = 50 # Wird jetzt aus der Config gelesen

    def __init__(self, label: str, num_qubits: Optional[int] = None, is_quantum: bool = True, neuron_type: str = "excitatory",
                 initial_params: Optional[np.ndarray] = None, uuid: Optional[str] = None):
        if not label: raise ValueError("Node label cannot be empty.")
        self.label: str = label
        self.uuid: str = uuid if uuid else str(uuid_module.uuid4())
        self.neuron_type: str = neuron_type
        self.is_quantum = is_quantum
        self.num_qubits = num_qubits if num_qubits is not None else self.DEFAULT_NUM_QUBITS

        # Sicherstellen, dass num_qubits gültig ist, wenn is_quantum=True
        if self.is_quantum and (not isinstance(self.num_qubits, int) or self.num_qubits <= 0):
            print(f"WARNUNG: Ungültige num_qubits ({self.num_qubits}) für Quantenknoten '{self.label}'. Setze auf klassisch.")
            self.is_quantum = False
            self.num_qubits = 0
        elif not self.is_quantum:
            self.num_qubits = 0 # Klassische Knoten haben 0 Qubits

        self.connections: Dict[str, Optional[Connection]] = {} # Speichert ausgehende Verbindungen {target_uuid: Connection}
        self.incoming_connections_info: List[Tuple[str, str]] = [] # Liste von (source_uuid, source_label)

        self.activation: float = 0.0 # Aktueller Aktivierungswert [0, 1]
        self.activation_sum: float = 0.0 # Summe der gewichteten Inputs in diesem Zeitschritt
        self.activation_history: deque = deque(maxlen=self.DEFAULT_ACTIVATION_HISTORY_LEN)

        self.q_system: Optional[QuantumNodeSystem] = None
        if self.is_quantum: # Nur erstellen, wenn is_quantum True UND num_qubits > 0 ist
            try:
                self.q_system = QuantumNodeSystem(num_qubits=self.num_qubits, initial_params=initial_params)
            except Exception as e:
                print(f"FEHLER bei Initialisierung des Quantensystems für Knoten '{self.label}': {e}. Setze auf klassisch.")
                self.is_quantum = False
                self.num_qubits = 0

        self.last_measurement_log: List[Dict] = []
        self.last_state_vector: Optional[np.ndarray] = None

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None, conn_type: str = "associative") -> Optional[Connection]:
        """Fügt eine ausgehende Verbindung zu einem Zielknoten hinzu oder gibt die bestehende zurück."""
        if target_node is None or not hasattr(target_node, 'uuid') or target_node.uuid == self.uuid:
            return None # Ungültiges Ziel oder Selbstverbindung nicht erlaubt
        target_uuid = target_node.uuid

        # Prüfe, ob die Verbindung bereits existiert
        if target_uuid not in self.connections:
            # print(f"  +++ DEBUG add_connection ({self.label}): Target '{target_node.label}' ({target_uuid}) NOT in connections dict. Adding new connection.") # Debug entfernt
            try:
                conn = Connection(target_node=target_node, weight=weight, source_node_label=self.label, conn_type=conn_type)
                self.connections[target_uuid] = conn
                # Prüfe direkt nach dem Hinzufügen (optional)
                # if target_uuid in self.connections and self.connections[target_uuid] is not None:
                #     print(f"      --> SUCCESS: Connection to '{target_node.label}' seems added. self.connections length: {len(self.connections)}")
                # else:
                #     print(f"      --> ❌ FAILURE?: Connection to '{target_node.label}' NOT found in dict immediately after adding! self.connections length: {len(self.connections)}")

                # Informiere Zielknoten (für eingehende Liste)
                if hasattr(target_node, 'add_incoming_connection_info'):
                    target_node.add_incoming_connection_info(self.uuid, self.label)
                return conn
            except Exception as e:
                 print(f"      --> ❌ EXCEPTION during connection creation/adding for {self.label} -> {target_node.label}: {e}")
                 traceback.print_exc(limit=1)
                 return None
        else:
            # Verbindung existiert bereits, gib sie zurück
            return self.connections.get(target_uuid)

    def add_incoming_connection_info(self, source_uuid: str, source_label: str):
        """Fügt Informationen über eine eingehende Verbindung hinzu (nur zur Info)."""
        if not any(info[0] == source_uuid for info in self.incoming_connections_info):
             self.incoming_connections_info.append((source_uuid, source_label))

    def strengthen_connection(self, target_node: 'Node', learning_signal: float = 0.1, learning_rate: Optional[float] = None):
        """Stärkt das Gewicht einer bestehenden Verbindung mit einer gegebenen Lernrate."""
        if target_node is None or not hasattr(target_node, 'uuid'): return
        target_uuid = target_node.uuid
        connection = self.connections.get(target_uuid)
        if connection is not None:
             # Die Lernrate wird hier DIREKT übergeben, nicht mehr der Klassen-Default
             connection.update_weight(delta_weight=learning_signal, learning_rate=learning_rate)

    def calculate_activation(self, n_shots: Optional[int] = None):
        """Berechnet den Aktivierungswert des Knotens für den aktuellen Zeitschritt."""
        # Hole n_shots aus Config wenn nicht übergeben (wird vom Processor-Objekt geholt)
        # Hier wird der Default der Klasse verwendet, wenn n_shots=None übergeben wird.
        # Im normalen Ablauf sollte simulate_network_step n_shots aus der Config übergeben.
        current_n_shots = n_shots if n_shots is not None else self.DEFAULT_N_SHOTS

        new_activation: float = 0.0
        if self.is_quantum and self.q_system:
            try:
                q_activation, q_state_vector, q_measure_log = self.q_system.activate(self.activation_sum, current_n_shots)
                new_activation = q_activation
                self.last_state_vector = q_state_vector
                self.last_measurement_log = q_measure_log
            except Exception as e:
                # print(f"FEHLER Quantenaktivierung für {self.label}: {e}")
                new_activation = 0.0 # Fallback bei Fehler
                self.last_state_vector = None
                self.last_measurement_log = [{"error": f"Activation failed: {e}"}]
        else:
            # Klassische Sigmoid-Aktivierung
            activation_sum_float = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
            # Vermeide Overflow/Underflow in exp()
            safe_activation_sum = np.clip(activation_sum_float, -700, 700)
            try:
                new_activation = 1 / (1 + np.exp(-safe_activation_sum))
            except FloatingPointError:
                new_activation = 1.0 if safe_activation_sum > 0 else 0.0 # Grenzwert bei extremen Inputs
            self.last_state_vector = None
            self.last_measurement_log = []

        # Stelle sicher, dass Aktivierung gültig ist und aktualisiere
        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation):
            self.activation = 0.0
        else:
            self.activation = float(np.clip(new_activation, 0.0, 1.0)) # Clip [0, 1]

        # Füge zur Historie hinzu und resette die Summe für den nächsten Schritt
        self.activation_history.append(self.activation)
        self.activation_sum = 0.0 # WICHTIG: Reset nach Berechnung

    def get_smoothed_activation(self, window: int = 3) -> float:
        """Gibt eine geglättete Aktivierung über die letzten 'window' Zeitschritte zurück."""
        if not self.activation_history: return self.activation
        # Nimm die letzten 'window' Einträge (oder weniger, wenn nicht verfügbar)
        hist = list(self.activation_history)[-window:]
        # Filtere ungültige Werte (sollte nicht vorkommen, aber sicher ist sicher)
        valid_hist = [a for a in hist if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if not valid_hist: return self.activation # Fallback zur aktuellen Aktivierung
        else: return float(np.mean(valid_hist)) # Durchschnitt der gültigen Werte

    def get_state_representation(self) -> Dict[str, Any]:
        """Gibt eine Dictionary-Repräsentation des aktuellen Zustands des Knotens zurück (für Debugging/Anzeige)."""
        state = {
            "label": self.label,
            "uuid": self.uuid,
            "activation": round(self.activation, 4),
            "smoothed_activation": round(self.get_smoothed_activation(), 4),
            "type": type(self).__name__,
            "neuron_type": self.neuron_type,
            "is_quantum": self.is_quantum,
        }
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits
            state["last_measurement_analysis"] = self.analyze_jumps(self.last_measurement_log)
        # Füge spezifische Zustände von Subklassen hinzu
        if hasattr(self, 'emotion_state'):
            state["emotion_state"] = getattr(self, 'emotion_state', {}).copy()
        # Zähle Verbindungen sicher
        state["num_connections"] = len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        return state

    def analyze_jumps(self, measurement_log: List[Dict]) -> Dict[str, Any]:
        """Analysiert die Messprotokolle auf signifikante Zustandsänderungen (Jumps)."""
        default_jump_info = {
            "shots_recorded": len(measurement_log),
            "jump_detected": False, "max_jump_abs": 0, "avg_jump_abs": 0.0,
            "state_variance": 0.0, "significant_threshold": 0.0,
            "error_count": sum(1 for m in measurement_log if m.get("error"))
        }
        if len(measurement_log) < 2: return default_jump_info # Mindestens 2 Messungen für Differenz nötig

        # Extrahiere gültige gemessene Zustandsindizes
        valid_indices = [m.get('index') for m in measurement_log if isinstance(m.get('index'), (int, np.integer))]
        if len(valid_indices) < 2:
             default_jump_info["shots_recorded"] = len(valid_indices); return default_jump_info

        indices_array = np.array(valid_indices, dtype=float)
        # Berechne absolute Differenzen zwischen aufeinanderfolgenden Messungen
        jumps = np.abs(np.diff(indices_array))
        # Berechne Varianz der gemessenen Zustände
        state_variance = np.var(indices_array) if len(indices_array) > 1 else 0.0

        max_jump = 0.0; avg_jump = 0.0; jump_detected = False; significant_threshold = 0.0
        if jumps.size > 0: # Nur wenn Sprünge berechnet werden konnten
            max_jump = np.max(jumps)
            avg_jump = np.mean(jumps)
            # Definiere Schwellwert für "signifikanten" Sprung (z.B. 1/4 des Zustandsraums)
            if self.is_quantum and self.q_system and self.num_qubits > 0:
                significant_threshold = (2**self.num_qubits) / 4.0
            else: significant_threshold = 1.0 # Standard für klassisch oder Fallback
            jump_detected = max_jump > significant_threshold

        return {
            "shots_recorded": len(valid_indices),
            "jump_detected": jump_detected,
            "max_jump_abs": int(max_jump), # Als Integer zurückgeben
            "avg_jump_abs": round(avg_jump, 3),
            "state_variance": round(state_variance, 3),
            "significant_threshold": round(significant_threshold, 1),
            "error_count": default_jump_info["error_count"]
        }

    def __repr__(self) -> str:
        """Gibt eine kompakte String-Repräsentation des Knotens zurück."""
        act_str = f"Act:{self.activation:.3f}"
        q_info = f" Q:{self.num_qubits}" if self.is_quantum and self.q_system else " (Cls)"
        conn_count = len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        conn_info = f" Conns:{conn_count}"
        return f"<{type(self).__name__} '{self.label}' {act_str}{q_info}{conn_info}>"


    # --- Korrigierte __getstate__ und __setstate__ für Persistenz ---

    def __getstate__(self):
        """Erstellt ein serialisierbares Dictionary für den Zustand des Knotens."""
        # Explizit ein neues Dictionary erstellen
        state_to_return = {}

        # 1. Basisattribute hinzufügen
        for key in ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum']:
             if hasattr(self, key):
                 state_to_return[key] = getattr(self, key)
        # incoming_connections_info
        if hasattr(self, 'incoming_connections_info'):
             info_list = getattr(self, 'incoming_connections_info')
             state_to_return['incoming_connections_info'] = info_list if isinstance(info_list, list) else []
        else:
             state_to_return['incoming_connections_info'] = []

        # 2. Quantenparameter serialisieren
        q_system = getattr(self, 'q_system', None)
        if q_system is not None and hasattr(q_system, 'get_params'):
            try:
                q_params = q_system.get_params()
                state_to_return['q_system_params'] = q_params.tolist() if isinstance(q_params, np.ndarray) else q_params
            except Exception as e_q:
                print(f"    ERROR getting/converting q_system_params for {self.label}: {e_q}")
                state_to_return['q_system_params'] = None
        else:
            state_to_return['q_system_params'] = None

        # 3. Verbindungen serialisieren
        connections_serializable = {}
        live_connections = getattr(self, 'connections', None)
        if isinstance(live_connections, dict):
            for target_uuid, conn in live_connections.items():
                if conn is None: continue
                try:
                    target_uuid_in_conn = getattr(conn, 'target_node_uuid', target_uuid)
                    if not target_uuid_in_conn: continue
                    conn_data = {
                        'weight': float(getattr(conn, 'weight', 0.0)),
                        'source_node_label': getattr(conn, 'source_node_label', self.label),
                        'conn_type': getattr(conn, 'conn_type', 'associative'),
                        'last_transmitted_signal': float(getattr(conn, 'last_transmitted_signal', 0.0)),
                        'transmission_count': int(getattr(conn, 'transmission_count', 0)),
                        'created_at': str(getattr(conn, 'created_at', datetime.now())),
                        'last_update_at': str(getattr(conn, 'last_update_at', datetime.now())),
                        'target_node_uuid': target_uuid_in_conn
                    }
                    connections_serializable[target_uuid_in_conn] = conn_data
                except Exception as e_ser:
                    print(f"    ERROR serializing connection object for UUID {target_uuid} from {self.label}: {e_ser}")
        state_to_return['connections_serializable'] = connections_serializable

        # 4. Aktivierungsverlauf serialisieren
        activation_hist = getattr(self, 'activation_history', None)
        if isinstance(activation_hist, deque):
            state_to_return['activation_history'] = list(activation_hist)
        else:
            state_to_return['activation_history'] = []

        # 5. Spezifische Attribute von Subklassen hinzufügen (z.B. LimbusAffektus)
        if isinstance(self, LimbusAffektus):
            state_to_return['emotion_state'] = getattr(self, 'emotion_state', INITIAL_EMOTION_STATE).copy()

        # Füge den Typ hinzu
        state_to_return['type'] = type(self).__name__

        return state_to_return


    def __setstate__(self, state: Dict[str, Any]):
        """Stellt den Zustand des Knotens aus einem Dictionary wieder her."""
        history_len = getattr(type(self), 'DEFAULT_ACTIVATION_HISTORY_LEN', 20)
        state['activation_history'] = deque(state.get('activation_history', []), maxlen=history_len)

        q_params_list = state.pop('q_system_params', None)
        num_qbits = state.get('num_qubits', getattr(type(self), 'DEFAULT_NUM_QUBITS', 10))
        is_q = state.get('is_quantum', True)
        self.q_system = None
        q_params_np = None

        if q_params_list is not None and isinstance(q_params_list, list):
             try:
                  q_params_np = np.array(q_params_list, dtype=float)
                  expected_shape = (num_qbits * 2,)
                  if num_qbits > 0:
                      if q_params_np.shape != expected_shape: q_params_np = None
                      elif not np.all(np.isfinite(q_params_np)): q_params_np = None
                  elif num_qbits == 0 and q_params_np.size != 0: q_params_np = None
             except Exception as e: q_params_np = None

        if is_q and num_qbits > 0:
             try:
                 global QuantumNodeSystem
                 self.q_system = QuantumNodeSystem(num_qubits=num_qbits, initial_params=q_params_np)
             except Exception as e:
                 print(f"FEHLER (__setstate__) Restore QNS für '{state.get('label', '?')}': {e}");
                 state['is_quantum'] = False; state['num_qubits'] = 0
        else:
             state['is_quantum'] = False; state['num_qubits'] = 0

        self.connections_serializable_temp = state.pop('connections_serializable', {})
        self.connections: Dict[str, Optional[Connection]] = {}

        # Spezifische Attribute für Subklassen wiederherstellen
        if state.get('type') == 'LimbusAffektus':
            self.emotion_state = state.pop('emotion_state', INITIAL_EMOTION_STATE.copy())
            # config und abhängige Parameter werden in load_state gesetzt

        # Aktualisiere restliche Attribute
        valid_attrs = ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum', 'incoming_connections_info', 'activation_history']
        for key, value in state.items():
             if key in valid_attrs: setattr(self, key, value)

        if not hasattr(self, 'uuid') or not self.uuid:
            import uuid as uuid_module_local
            self.uuid = str(uuid_module_local.uuid4())
        if not hasattr(self, 'incoming_connections_info') or not isinstance(self.incoming_connections_info, list):
            self.incoming_connections_info = []


# --- Emotionale Konstanten und LimbusAffektus Klasse ---
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}

class LimbusAffektus(Node):
    """
    Modelliert den globalen emotionalen Zustand des Netzwerks (PAD-Modell).
    Aktualisiert seinen Zustand basierend auf der Netzwerkaktivität.
    Beeinflusst aktiv andere Systemkomponenten (Modulation).
    """
    # Akzeptiere is_quantum als Argument, Default kann True bleiben
    def __init__(self, label: str = "Limbus Affektus", num_qubits: Optional[int] = None, is_quantum: bool = True, config: Optional[Dict] = None, **kwargs):
        """
        Initialisiert den LimbusAffektus-Knoten.

        Args:
            label (str): Label des Knotens.
            num_qubits (int): Anzahl der Qubits, WENN is_quantum True ist.
            is_quantum (bool): Flag, ob der Knoten Quanteneigenschaften haben soll.
            config (Optional[Dict]): Das Konfigurationsdictionary des Prozessors.
            **kwargs: Weitere Argumente für die Basisklasse Node.
        """
        # Bestimme die tatsächliche Anzahl der Qubits basierend auf dem is_quantum Flag
        actual_num_qubits = num_qubits if is_quantum else 0
        if actual_num_qubits is None and is_quantum: # Fallback falls num_qubits None ist aber Quanten gewünscht
            actual_num_qubits = 4 # Oder ein anderer Default für Limbus

        # Rufe super().__init__ mit dem übergebenen is_quantum Wert auf
        super().__init__(label=label,
                         num_qubits=actual_num_qubits,
                         is_quantum=is_quantum, # Verwende den übergebenen Wert!
                         neuron_type="affective_modulator",
                         **kwargs) # Übergib restliche kwargs an Node

        # Der Rest der Initialisierung bleibt gleich
        self.emotion_state = INITIAL_EMOTION_STATE.copy()
        self.config = config if config else {}
        self.decay = self.config.get("limbus_emotion_decay", 0.95)
        self.arousal_sens = self.config.get("limbus_arousal_sensitivity", 1.5)
        self.pleasure_sens = self.config.get("limbus_pleasure_sensitivity", 1.0)
        self.dominance_sens = self.config.get("limbus_dominance_sensitivity", 1.0)
        self.last_input_sum_for_pleasure = 0.0

    def calculate_activation(self, n_shots: Optional[int] = None):
        """Berechnet Aktivierung und speichert Input-Summe."""
        self.last_input_sum_for_pleasure = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
        super().calculate_activation(n_shots=n_shots)

    def update_emotion_state(self, all_nodes: List['Node']):
        """Aktualisiert den internen emotionalen Zustand (PAD)."""
        if not all_nodes: return

        other_node_activations = [
            n.activation for n in all_nodes
            if n.uuid != self.uuid and hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)
        ]
        avg_activation = np.mean(other_node_activations) if other_node_activations else 0.0
        # Sensitivitäts-Faktoren aus Config lesen
        arousal_sens = self.config.get("limbus_arousal_sensitivity", 1.5)
        pleasure_sens = self.config.get("limbus_pleasure_sensitivity", 1.0)
        dominance_sens = self.config.get("limbus_dominance_sensitivity", 1.0)
        decay_rate = self.config.get("limbus_emotion_decay", 0.95)

        # Update Berechnungen mit Sensitivität
        arousal_update = (avg_activation * 2 - 1) * arousal_sens
        pleasure_update = math.tanh(self.last_input_sum_for_pleasure * pleasure_sens)
        dominance_update = (self.activation * 2 - 1) * dominance_sens

        # Update PAD Werte mit Decay und Clipping
        self.emotion_state["pleasure"] = float(np.clip(self.emotion_state["pleasure"] * decay_rate + pleasure_update, -1.0, 1.0))
        self.emotion_state["arousal"] = float(np.clip(self.emotion_state["arousal"] * decay_rate + arousal_update, -1.0, 1.0))
        self.emotion_state["dominance"] = float(np.clip(self.emotion_state["dominance"] * decay_rate + dominance_update, -1.0, 1.0))

    # __getstate__ und __setstate__ wurden angepasst, um emotion_state zu behandeln
    # (siehe oben in der Node-Klasse, da sie von Node erbt und die Logik dort integriert ist)


# --- TextChunk Klasse ---
class TextChunk:
    """Repräsentiert einen Textabschnitt mit Metadaten."""
    def __init__(self, text: str, source: str, index: int, chunk_uuid: Optional[str]=None):
        self.uuid = chunk_uuid if chunk_uuid else str(uuid_module.uuid4())
        self.text: str = text
        self.source: str = source
        self.index: int = index
        self.activated_node_labels: List[str] = []
        self.embedding: Optional[np.ndarray] = None # Für zukünftige Embedding-Nutzung

    def __repr__(self) -> str:
        node_str = f" Nodes:[{','.join(self.activated_node_labels)}]" if self.activated_node_labels else ""
        return f"<Chunk {self.index} from '{self.source}' (UUID:{self.uuid[:4]}...) Len:{len(self.text)}{node_str}>"

# --- HAUPTPROZESSOR-KLASSE ---
class QuantumEnhancedTextProcessor:
    """Orchestriert Laden, Quantenknoten, Lernen und RAG, moduliert durch LimbusAffektus."""
    DEFAULT_CONFIG = {
        "embedding_dim": 128, "chunk_size": 500, "chunk_overlap": 100, "training_epochs": 1, "training_files": [],
        "semantic_nodes": { # Standard-Knotendefinitionen, können überschrieben werden
             "DefaultNode": []
        },
        "connection_learning_rate": 0.05, "connection_decay_rate": 0.001, "connection_strengthening_signal": 0.1,
        "max_prompt_results": 3, "relevance_threshold": 0.08, "tfidf_max_features": 5000,
        "use_quantum_nodes": True, "default_num_qubits": 10, "simulation_n_shots": 50,
        "simulation_steps_after_training": 0,
        "enable_rag": True, "generator_model_name": "models/gemini-1.5-flash-latest",
        "generator_max_length": 8192, "generator_temperature": 0.7,
        "quantum_effect_variance_penalty": 0.5, "quantum_effect_activation_boost": 0.3,
        "quantum_effect_jump_llm_trigger": True,
        "enable_self_learning": True, "self_learning_file_path": "./training_data/learn.txt",
        "self_learning_source_name": "Generated Responses",
        # Limbus Affektus Basisparameter
        "limbus_emotion_decay": 0.95,
        "limbus_arousal_sensitivity": 1.5,
        "limbus_pleasure_sensitivity": 1.0,
        "limbus_dominance_sensitivity": 1.0,
        "limbus_num_qubits": 4,

        # --- NEU: Limbus Einflussfaktoren ---
        # RAG Generation
        "limbus_influence_prompt_level": 0.5,      # Wie stark Emotionen den Prompt-Zusatz beeinflussen (0=kein Einfluss, 1=voller Wertebereich)
        "limbus_influence_temperature_arousal": 0.1, # Additiver Faktor pro Arousal-Punkt
        "limbus_influence_temperature_dominance": -0.1,# Additiver Faktor pro Dominance-Punkt
        "limbus_min_temperature": 0.3,             # Minimale erlaubte Temperatur
        "limbus_max_temperature": 1.0,             # Maximale erlaubte Temperatur
        # Retrieval
        "limbus_influence_threshold_arousal": -0.03, # Additiver Faktor pro Arousal-Punkt (negativ = niedriger Threshold bei hoher Arousal)
        "limbus_influence_threshold_pleasure": 0.03, # Additiver Faktor pro Pleasure-Punkt (positiv = höherer Threshold bei hoher Pleasure)
        "limbus_min_threshold": 0.02,              # Minimaler erlaubter Threshold
        "limbus_max_threshold": 0.2,               # Maximaler erlaubter Threshold
        "limbus_influence_ranking_bias_pleasure": 0.02, # Additiver Score-Bonus pro Pleasure-Punkt
        # Lernen
        "limbus_influence_learning_rate_multiplier": 0.1, # Multiplikativer Faktor pro (Arousal+Pleasure)/2 Punkt (0.1 = +/-10%)
        "limbus_min_lr_multiplier": 0.5,           # Minimaler Lernraten-Multiplikator
        "limbus_max_lr_multiplier": 1.5,           # Maximaler Lernraten-Multiplikator
        # Quanteneffekte (im Retrieval)
        "limbus_influence_variance_penalty": 0.1,  # Additiver Faktor pro (Arousal-Pleasure)/2 Punkt
        "limbus_influence_activation_boost": 0.05, # Additiver Faktor pro (Pleasure-Arousal)/2 Punkt
    }


    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict]=None):
        """Initialisiert den Prozessor mit einer Konfiguration."""
        # Lade oder setze Konfiguration
        if config_path: self.config = self._load_config(config_path)
        elif config_dict: self.config = {**self.DEFAULT_CONFIG, **config_dict}
        else: print("WARNUNG: Keine Konfig übergeben, nutze Defaults."); self.config = self.DEFAULT_CONFIG.copy()
        # Fülle fehlende Config-Werte mit Defaults
        for key, value in self.DEFAULT_CONFIG.items(): self.config.setdefault(key, value)

        self.nodes: Dict[str, Node] = {} # {label: Node object}
        self.chunks: Dict[str, TextChunk] = {} # {uuid: TextChunk object}
        self.sources_processed: set = set() # Verfolgt verarbeitete Dateipfade/Quellen

        self._initialize_semantic_nodes() # Initialisiert normale Knoten basierend auf Config

        # Erstelle LimbusAffektus Knoten, falls noch nicht durch Config erstellt oder geladen
        limbus_label = "Limbus Affektus"
        if limbus_label not in self.nodes:
             print(f"INFO: Erstelle '{limbus_label}' Knoten...")
             limbus_qubits = self.config.get("limbus_num_qubits", 4)
             use_quantum = self.config.get("use_quantum_nodes", True)
             try:
                  limbus_node = LimbusAffektus(label=limbus_label,
                                                num_qubits=limbus_qubits if use_quantum else 0,
                                                is_quantum=use_quantum,
                                                config=self.config) # Übergib config
                  self.nodes[limbus_label] = limbus_node
             except Exception as e:
                  print(f"FEHLER beim Erstellen des '{limbus_label}' Knotens: {e}")
        elif isinstance(self.nodes[limbus_label], LimbusAffektus):
             # Wenn er schon existiert (z.B. aus State geladen), setze die Config Referenz neu
             print(f"INFO: Setze Config-Referenz für geladenen '{limbus_label}' Knoten.")
             self.nodes[limbus_label].config = self.config
             # Lade Parameter neu aus Config, falls sie sich geändert haben
             self.nodes[limbus_label].decay = self.config.get("limbus_emotion_decay", 0.95)
             self.nodes[limbus_label].arousal_sens = self.config.get("limbus_arousal_sensitivity", 1.5)
             self.nodes[limbus_label].pleasure_sens = self.config.get("limbus_pleasure_sensitivity", 1.0)
             self.nodes[limbus_label].dominance_sens = self.config.get("limbus_dominance_sensitivity", 1.0)

        # Initialisiere TF-IDF Komponenten
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None # Type Any für spmatrix
        self.chunk_id_list_for_tfidf: List[str] = [] # Wichtig für Mapping

        # RAG & Self-Learning Setup
        self.gemini_model = None # Wird bei Bedarf initialisiert
        self.rag_enabled = self.config.get("enable_rag", False) and GEMINI_AVAILABLE
        self.self_learning_enabled = self.config.get("enable_self_learning", False)
        self.learn_file_path = self.config.get("self_learning_file_path", "./training_data/learn.txt")
        self.learn_source_name = self.config.get("self_learning_source_name", "Generated Responses")

        # Abschluss-Infos
        print(f"\nQuantumEnhancedTextProcessor initialisiert mit {len(self.nodes)} Knoten.")
        quantum_node_count = sum(1 for n in self.nodes.values() if n.is_quantum)
        if quantum_node_count > 0: print(f" -> Davon {quantum_node_count} Quantenknoten.")
        print(f" -> RAG (Textgenerierung via Gemini) {'AKTIVIERT' if self.rag_enabled else 'DEAKTIVIERT'}")
        print(f" -> Self-Learning {'AKTIVIERT' if self.self_learning_enabled else 'DEAKTIVIERT'} (Ziel: {self.learn_file_path})")
        if limbus_label in self.nodes: print(f" -> Limbus Affektus Modulation AKTIVIERT.")


    def _load_config(self, path: str) -> Dict:
        """Lädt Konfiguration aus JSON-Datei und merged mit Defaults."""
        try:
            with open(path, 'r', encoding='utf-8') as f: loaded_config = json.load(f)
            # Starte mit Defaults und überschreibe/ergänze mit geladenen Werten
            config = self.DEFAULT_CONFIG.copy()
            config.update(loaded_config)
            print(f"INFO: Konfiguration aus '{path}' geladen und mit Defaults gemischt.")
            return config
        except Exception as e:
            print(f"FEHLER Laden Config {path}: {e}. Nutze Defaults.")
            return self.DEFAULT_CONFIG.copy()

    def _initialize_semantic_nodes(self):
        """Initialisiert semantische Knoten basierend auf der Konfiguration."""
        semantic_node_definitions = self.config.get("semantic_nodes", {})
        use_quantum = self.config.get("use_quantum_nodes", True)
        num_qubits = self.config.get("default_num_qubits")

        for label in semantic_node_definitions.keys():
            if label not in self.nodes: # Nur erstellen, wenn nicht schon vorhanden (z.B. Limbus)
                # print(f"  Initialisiere semantischen Knoten '{label}'...") # Weniger verbose
                try:
                    # Normale Knoten erhalten keine spezielle Config-Referenz
                    node = Node(label=label,
                                is_quantum=use_quantum,
                                num_qubits=num_qubits if use_quantum else 0,
                                neuron_type="semantic")
                    self.nodes[label] = node
                except Exception as e:
                    print(f"FEHLER Erstellen Knoten '{label}': {e}. Übersprungen.")

    def _get_or_create_node(self, label: str, neuron_type: str = "semantic") -> Optional[Node]:
        """Holt einen Knoten oder erstellt ihn dynamisch (ohne spezielle Config)."""
        if not label: return None
        if label in self.nodes: return self.nodes[label]
        else:
            print(f"WARNUNG: Erstelle Knoten '{label}' dynamisch (nicht in Config gefunden).")
            try:
                use_quantum = self.config.get("use_quantum_nodes", True)
                num_qubits = self.config.get("default_num_qubits")
                node = Node(label=label,
                            is_quantum=use_quantum,
                            num_qubits=num_qubits if use_quantum else 0,
                            neuron_type=neuron_type)
                self.nodes[label] = node
                return node
            except Exception as e:
                print(f"FEHLER dyn. Erstellen Knoten '{label}': {e}"); return None

    def load_and_process_file(self, file_path: str, source_name: Optional[str] = None):
        """Lädt, chunkt und verarbeitet Text aus einer Datei."""
        if not os.path.exists(file_path):
             print(f"FEHLER: Datei nicht gefunden: {file_path}"); return
        effective_source_name = source_name if source_name else os.path.basename(file_path)

        # Verhindere erneute Verarbeitung, außer es ist die Self-Learning Datei
        if effective_source_name in self.sources_processed and effective_source_name != self.learn_source_name:
             # print(f"INFO: Quelle '{effective_source_name}' wurde bereits verarbeitet. Überspringe.") # Weniger verbose
             return

        print(f"\n📄 Verarbeite Datenquelle: {file_path} (Quelle: {effective_source_name})")
        try:
            chunks = self._load_chunks_from_file(file_path, effective_source_name)
            if not chunks: print(f"WARNUNG: Keine Chunks aus {file_path} geladen."); return
            print(f"   -> {len(chunks)} Chunks erstellt. Beginne Verarbeitung...")

            newly_added_chunk_ids = []
            chunk_iterator = tqdm(chunks, desc=f"Verarbeitung {effective_source_name}", leave=False) if TQDM_AVAILABLE else chunks
            for chunk in chunk_iterator:
                 # UUID wird in _load_chunks neu generiert, daher immer hinzufügen/überschreiben?
                 # Aktuell: Fügen immer hinzu, da UUID neu ist. Alte Chunks bleiben drin.
                 self.chunks[chunk.uuid] = chunk
                 self.process_chunk(chunk) # Assoziiert Knoten, stärkt Verbindungen (jetzt mit modulierter LR)
                 newly_added_chunk_ids.append(chunk.uuid)

            if effective_source_name != self.learn_source_name:
                self.sources_processed.add(effective_source_name)

            print(f"   -> Verarbeitung von {effective_source_name} abgeschlossen ({len(newly_added_chunk_ids)} Chunks verarbeitet). Gesamt Chunks: {len(self.chunks)}.")
            # TF-IDF Index muss aktualisiert werden, wenn neue Chunks dazukommen
            if newly_added_chunk_ids:
                self.update_tfidf_index()

        except Exception as e: print(f"FEHLER Verarbeitung Datei {file_path}: {e}"); traceback.print_exc(limit=2)

    def _load_chunks_from_file(self, path: str, source: str) -> List[TextChunk]:
        """Lädt Text und teilt ihn in überlappende Chunks."""
        chunk_size = self.config.get("chunk_size", 500)
        overlap = self.config.get("chunk_overlap", 100)
        chunks = []
        try:
            with open(path, 'r', encoding='utf-8') as f: text = f.read()
        except Exception as e: print(f"FEHLER Lesen Datei {path}: {e}"); return []
        if not text: return []

        start_index = 0; chunk_index = 0
        while start_index < len(text):
            end_index = start_index + chunk_size
            chunk_text = text[start_index:end_index]
            normalized_text = re.sub(r'\s+', ' ', chunk_text).strip() # Normalisiere Whitespace

            if normalized_text: # Nur nicht-leere Chunks hinzufügen
                # Generiere IMMER eine neue UUID für jeden geladenen Chunk
                chunk_uuid = str(uuid_module.uuid4())
                chunks.append(TextChunk(text=normalized_text, source=source, index=chunk_index, chunk_uuid=chunk_uuid))

            # Berechne nächsten Startpunkt
            next_start = start_index + chunk_size - overlap
            # Verhindere Endlosschleifen bei sehr kurzen Dateien/hohem Overlap
            if next_start <= start_index: start_index += 1
            else: start_index = next_start
            chunk_index += 1
        return chunks

    def process_chunk(self, chunk: TextChunk):
        """Verarbeitet Chunk, stärkt Verbindungen mit durch Limbus modulierter Lernrate."""
        activated_nodes_in_chunk: List[Node] = []
        semantic_node_definitions = self.config.get("semantic_nodes", {})
        chunk_text_lower = chunk.text.lower()
        chunk.activated_node_labels = [] # Reset für diesen Durchlauf

        # Hole Limbus Zustand (falls vorhanden)
        limbus_state = INITIAL_EMOTION_STATE.copy()
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        pleasure = limbus_state.get("pleasure", 0.0)
        arousal = limbus_state.get("arousal", 0.0)

        # --- MODULATION 4: Lernrate ---
        base_lr = self.config.get("connection_learning_rate", 0.05)
        lr_multiplier_factor = self.config.get("limbus_influence_learning_rate_multiplier", 0.1)
        min_lr_multiplier = self.config.get("limbus_min_lr_multiplier", 0.5)
        max_lr_multiplier = self.config.get("limbus_max_lr_multiplier", 1.5)
        # Berechne Multiplikator basierend auf Durchschnitt von Arousal und Pleasure
        lr_mod_input = (arousal + pleasure) / 2.0
        current_lr_multiplier = 1.0 + (lr_mod_input * lr_multiplier_factor)
        current_lr_multiplier = float(np.clip(current_lr_multiplier, min_lr_multiplier, max_lr_multiplier))
        current_learning_rate = base_lr * current_lr_multiplier
        # Optional: Debugging Output
        # if random.random() < 0.01: # Nur gelegentlich ausgeben
        #     print(f"DEBUG process_chunk: Base LR={base_lr:.4f}, Multiplier={current_lr_multiplier:.3f}, Current LR={current_learning_rate:.4f} (P:{pleasure:.2f}, A:{arousal:.2f})")

        # Finde aktivierte Knoten (Code bleibt gleich)
        for node_label, keywords in semantic_node_definitions.items():
            node = self.nodes.get(node_label)
            if not node: continue
            matched_keyword = None
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', chunk_text_lower):
                    matched_keyword = kw; break
            if matched_keyword:
                 activated_nodes_in_chunk.append(node)
                 if node.label not in chunk.activated_node_labels:
                     chunk.activated_node_labels.append(node.label)

        # Stärke Verbindungen mit MODULIERTER Lernrate
        unique_activated_nodes = list({node.uuid: node for node in activated_nodes_in_chunk}.values())
        if len(unique_activated_nodes) >= 2:
            learning_signal = self.config.get("connection_strengthening_signal", 0.1)
            # *** Verwende MODULIERTE Lernrate ***
            lr_to_use = current_learning_rate
            for i in range(len(unique_activated_nodes)):
                for j in range(i + 1, len(unique_activated_nodes)):
                    node_a = unique_activated_nodes[i]; node_b = unique_activated_nodes[j]
                    conn_ab = node_a.add_connection(node_b); conn_ba = node_b.add_connection(node_a)
                    # *** Übergib modifizierte Lernrate an strengthen_connection ***
                    if conn_ab: node_a.strengthen_connection(node_b, learning_signal=learning_signal, learning_rate=lr_to_use)
                    if conn_ba: node_b.strengthen_connection(node_a, learning_signal=learning_signal, learning_rate=lr_to_use)

    def update_tfidf_index(self):
        """Aktualisiert den TF-IDF Vektorizer und die Matrix basierend auf den aktuellen Chunks."""
        if not self.chunks: print("WARNUNG: Keine Chunks für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        print("🔄 Aktualisiere TF-IDF Index...")

        # Verwende IMMER die aktuellen Chunks aus self.chunks
        # Wichtig: Reihenfolge der IDs muss mit Reihenfolge der Texte übereinstimmen
        current_chunk_ids = list(self.chunks.keys())
        chunk_texts = [self.chunks[cid].text for cid in current_chunk_ids if cid in self.chunks and self.chunks[cid].text]
        # Filtere auch die ID-Liste, um sicherzustellen, dass sie zur Textliste passt
        self.chunk_id_list_for_tfidf = [cid for cid in current_chunk_ids if cid in self.chunks and self.chunks[cid].text]

        if not chunk_texts: print("WARNUNG: Keine gültigen Texte für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return

        try:
            max_features = self.config.get("tfidf_max_features", 5000)
            # Initialisiere Vektorizer JEDES MAL neu, um auf den aktuellen Korpus zu passen
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts) # Berechne Matrix neu

            # Sicherheitscheck: Länge der ID-Liste muss mit Zeilenzahl der Matrix übereinstimmen
            if self.tfidf_matrix.shape[0] != len(self.chunk_id_list_for_tfidf):
                 print(f"FATALER FEHLER: Inkonsistenz bei TF-IDF Erstellung. Matrix Zeilen ({self.tfidf_matrix.shape[0]}) != Chunk IDs ({len(self.chunk_id_list_for_tfidf)}).")
                 self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []
                 return # Abbruch, Index ist unbrauchbar
            print(f"   -> TF-IDF Index aktualisiert. Shape: {self.tfidf_matrix.shape}, Chunk IDs: {len(self.chunk_id_list_for_tfidf)}")
        except ValueError as ve:
             if "empty vocabulary" in str(ve): print("FEHLER TF-IDF Update: Leeres Vokabular.")
             else: print(f"FEHLER TF-IDF Update (ValueError): {ve}")
             self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []
        except Exception as e:
             print(f"FEHLER TF-IDF Update: {e}"); traceback.print_exc(limit=1)
             self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []


    def simulate_network_step(self, decay_connections: bool = True):
        """Führt einen Simulationsschritt für das gesamte Netzwerk durch."""
        if not self.nodes: return

        # 1. Decay Connections (Optional)
        if decay_connections:
            decay_rate = self.config.get("connection_decay_rate", 0.001)
            if decay_rate > 0:
                for node in self.nodes.values():
                    if hasattr(node, 'connections') and isinstance(node.connections, dict):
                        # Iteriere über Kopie der Schlüssel, falls Decay zum Entfernen führt (sollte nicht)
                        for target_uuid in list(node.connections.keys()):
                            conn = node.connections.get(target_uuid)
                            if conn: conn.decay(decay_rate=decay_rate)

        # 2. Calculate Node Activations für ALLE Knoten
        n_shots = self.config.get("simulation_n_shots", 50) # Hole n_shots aus Config
        for node in self.nodes.values():
            node.calculate_activation(n_shots=n_shots) # Übergib n_shots

        # 3. Transmit Signals and update activation_sum for NEXT step
        next_activation_sums = defaultdict(float)
        # Baue eine schnelle UUID -> Node Map für den Lookup
        node_uuid_map = {n.uuid: n for n in self.nodes.values()}

        for source_node in self.nodes.values():
             # Sende nur Signal, wenn Aktivierung über einem Schwellenwert liegt
             if hasattr(source_node, 'activation') and source_node.activation > 0.01:
                 source_output = source_node.get_smoothed_activation() # Geglättet senden
                 if hasattr(source_node, 'connections') and isinstance(source_node.connections, dict):
                     for target_uuid, connection in source_node.connections.items():
                          if connection and hasattr(connection, 'weight'):
                               # Finde Zielknoten über die Map (effizienter als Iteration)
                               target_node = node_uuid_map.get(target_uuid)
                               if target_node:
                                    # Addiere übertragenes Signal zur Summe des Zielknotens für den NÄCHSTEN Schritt
                                    next_activation_sums[target_node.uuid] += connection.transmit(source_output)

        # Weise die gesammelten Summen den Knoten für den nächsten Schritt zu
        for node_uuid, new_sum in next_activation_sums.items():
             target_node = node_uuid_map.get(node_uuid)
             if target_node:
                 target_node.activation_sum = new_sum

        # 4. Update Emotion State (NACH Berechnung der Aktivierungen)
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             try:
                  # Übergib die aktuelle Liste aller Knotenobjekte
                  limbus_node.update_emotion_state(list(self.nodes.values()))
             except Exception as e_limbus:
                  print(f"FEHLER beim Update des LimbusAffektus Zustands: {e_limbus}")


    def respond_to_prompt(self, prompt: str) -> List[TextChunk]:
        """Findet relevante Text-Chunks, moduliert durch LimbusAffektus (Threshold, Ranking, Quanten-Effekte)."""
        # Hole Basis-Konfigurationswerte
        base_max_results = self.config.get("max_prompt_results", 3)
        base_relevance_threshold = self.config.get("relevance_threshold", 0.1)
        base_variance_penalty_factor = self.config.get("quantum_effect_variance_penalty", 0.5)
        base_activation_boost_factor = self.config.get("quantum_effect_activation_boost", 0.3)

        # Hole Limbus Zustand (falls vorhanden)
        limbus_state = INITIAL_EMOTION_STATE.copy() # Default neutral
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        pleasure = limbus_state.get("pleasure", 0.0)
        arousal = limbus_state.get("arousal", 0.0)
        # dominance = limbus_state.get("dominance", 0.0) # Aktuell nicht direkt für Retrieval genutzt

        # --- MODULATION 1: Retrieval Threshold ---
        threshold_mod_arousal = self.config.get("limbus_influence_threshold_arousal", -0.03)
        threshold_mod_pleasure = self.config.get("limbus_influence_threshold_pleasure", 0.03)
        min_threshold = self.config.get("limbus_min_threshold", 0.02)
        max_threshold = self.config.get("limbus_max_threshold", 0.2)
        # Berechne modifizierten Threshold
        current_relevance_threshold = base_relevance_threshold + (arousal * threshold_mod_arousal) + (pleasure * threshold_mod_pleasure)
        current_relevance_threshold = float(np.clip(current_relevance_threshold, min_threshold, max_threshold))
        # Optional: Debugging Output
        # if random.random() < 0.05:
        #     print(f"DEBUG respond_to_prompt: Base Threshold={base_relevance_threshold:.3f}, Modulated Threshold={current_relevance_threshold:.3f} (P:{pleasure:.2f}, A:{arousal:.2f})")

        # --- MODULATION 2: Quanteneffekt-Parameter ---
        variance_mod = self.config.get("limbus_influence_variance_penalty", 0.1)
        activation_mod = self.config.get("limbus_influence_activation_boost", 0.05)
        # Berechne modifizierte Faktoren (Beispiel: Arousal erhöht Penalty, Pleasure erhöht Boost)
        current_variance_penalty_factor = base_variance_penalty_factor + (arousal - pleasure)/2 * variance_mod
        current_activation_boost_factor = base_activation_boost_factor + (pleasure - arousal)/2 * activation_mod
        current_variance_penalty_factor = float(np.clip(current_variance_penalty_factor, 0.0, 1.0)) # Clamp [0, 1]
        current_activation_boost_factor = float(np.clip(current_activation_boost_factor, 0.0, 1.0)) # Clamp [0, 1]
        # Optional: Debugging Output
        # if random.random() < 0.05:
        #     print(f"DEBUG respond_to_prompt: Modulated VarPenalty={current_variance_penalty_factor:.3f}, Modulated ActBoost={current_activation_boost_factor:.3f}")

        # --- MODULATION 3: Ranking Bias (einfach) ---
        ranking_bias_factor = self.config.get("limbus_influence_ranking_bias_pleasure", 0.02)
        ranking_bias = pleasure * ranking_bias_factor # Positives Pleasure -> kleiner Bonus, Negatives -> kleiner Malus
        # Optional: Debugging Output
        # if random.random() < 0.05:
        #     print(f"DEBUG respond_to_prompt: Ranking Bias={ranking_bias:.3f}")

        # --- Restlicher Retrieval Prozess ---
        prompt_lower = prompt.lower()
        semantic_node_definitions = self.config.get("semantic_nodes", {})

        # 1. Finde relevante Knoten (direkt + verbunden) - Code bleibt gleich
        directly_activated_nodes: List[Node] = []
        for node_label, keywords in semantic_node_definitions.items():
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords):
                node = self.nodes.get(node_label);
                if node: directly_activated_nodes.append(node)
        related_nodes: set[Node] = set(directly_activated_nodes)
        node_uuid_map = {n.uuid: n for n in self.nodes.values()}
        if directly_activated_nodes:
             queue = deque(directly_activated_nodes)
             processed_for_spread = set(n.uuid for n in directly_activated_nodes)
             while queue:
                  start_node = queue.popleft()
                  connections_dict = getattr(start_node, 'connections', {})
                  if not isinstance(connections_dict, dict): continue
                  strong_connections = sorted(
                      [conn for conn in connections_dict.values() if conn and getattr(conn, 'weight', 0) > 0.2],
                      key=lambda c: c.weight, reverse=True
                  )[:5]
                  for conn in strong_connections:
                      target_uuid = getattr(conn, 'target_node_uuid', None)
                      if target_uuid and target_uuid not in processed_for_spread:
                           target_node = node_uuid_map.get(target_uuid)
                           if target_node:
                                related_nodes.add(target_node)
                                processed_for_spread.add(target_uuid)
        relevant_node_labels = {node.label for node in related_nodes}

        # 2. Finde Kandidaten-Chunks - Code bleibt gleich
        candidate_chunks: List[TextChunk] = []
        if relevant_node_labels:
             candidate_chunks = [
                 chunk for chunk in self.chunks.values()
                 if chunk and hasattr(chunk, 'activated_node_labels') and any(label in chunk.activated_node_labels for label in relevant_node_labels)
             ]
        else: candidate_chunks = list(self.chunks.values())
        if not candidate_chunks: return []

        # 3. TF-IDF Ranking - Code bleibt gleich bis Score-Berechnung
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunk_id_list_for_tfidf:
             return candidate_chunks[:base_max_results] # Nutze Basis-MaxResults
        try:
             prompt_vector = self.vectorizer.transform([prompt])
             uuid_to_tfidf_index = {uuid: i for i, uuid in enumerate(self.chunk_id_list_for_tfidf)}
             candidate_matrix_indices = []
             valid_candidate_chunks_for_ranking = []
             for c in candidate_chunks:
                 if hasattr(c, 'uuid') and c.uuid in uuid_to_tfidf_index:
                      idx = uuid_to_tfidf_index[c.uuid]
                      candidate_matrix_indices.append(idx)
                      valid_candidate_chunks_for_ranking.append(c)
             if not candidate_matrix_indices: return candidate_chunks[:base_max_results]
             candidate_matrix = self.tfidf_matrix[candidate_matrix_indices, :]
             similarities = cosine_similarity(prompt_vector, candidate_matrix).flatten()

             # 4. Wende Quanten-Effekte & Ranking Bias auf Scores an
             scored_candidates = []
             for i, chunk in enumerate(valid_candidate_chunks_for_ranking):
                 base_score = similarities[i]
                 quantum_adjustment = 0.0
                 num_q_nodes = 0; sum_variance = 0.0; sum_activation = 0.0
                 if hasattr(chunk, 'activated_node_labels'):
                     for node_label in chunk.activated_node_labels:
                         node = self.nodes.get(node_label)
                         if node and node.is_quantum and node.q_system and hasattr(node, 'last_measurement_log'):
                             num_q_nodes += 1
                             analysis = node.analyze_jumps(node.last_measurement_log)
                             sum_variance += analysis.get("state_variance", 0.0)
                             sum_activation += node.activation
                 if num_q_nodes > 0:
                      avg_activation = sum_activation / num_q_nodes
                      avg_variance = sum_variance / num_q_nodes
                      # *** Verwende MODULIERTE Faktoren ***
                      quantum_adjustment = (avg_activation * current_activation_boost_factor) - (avg_variance * current_variance_penalty_factor)

                 # *** Addiere RANKING BIAS ***
                 biased_score = base_score + quantum_adjustment + ranking_bias
                 final_score = float(np.clip(biased_score, 0.0, 1.0)) # Score bleibt [0, 1]
                 scored_candidates.append({"chunk": chunk, "score": final_score})

             # 5. Finales Ranking und Auswahl mit MODULIERTEM Threshold
             ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
             # *** Verwende MODULIERTEN Threshold und Basis-MaxResults ***
             final_results = [item["chunk"] for item in ranked_candidates if item["score"] >= current_relevance_threshold][:base_max_results]

             # Fallback, wenn nichts über (moduliertem) Threshold
             if not final_results and ranked_candidates:
                 final_results = [ranked_candidates[0]["chunk"]] # Nimm den besten

             return final_results
        except Exception as e:
             print(f"FEHLER TF-IDF/Quantum Ranking: {e}"); traceback.print_exc(limit=1)
             return candidate_chunks[:base_max_results] # Fallback


    def generate_response(self, prompt: str) -> str:
        """Generiert Antwort mit RAG, Limbus-moduliertem Prompt & Temperatur."""
        # Prüfungen: SDK, RAG, API Key - bleiben gleich
        if not GEMINI_AVAILABLE: return "[Fehler: Gemini SDK fehlt]"
        if not self.rag_enabled: return "[Fehler: RAG deaktiviert]"
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: # Fallback Streamlit Secrets
            try: import streamlit as st; api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception: pass
        if not api_key: return "[Fehler: Gemini API Key fehlt]"

        # Initialisiere Gemini Modell (bleibt gleich)
        try:
             genai.configure(api_key=api_key)
             model_name = self.config.get("generator_model_name", "models/gemini-1.5-flash-latest")
             if not self.gemini_model or self.gemini_model.model_name != model_name:
                 print(f"INFO: Initialisiere Gemini Modell '{model_name}'...")
                 self.gemini_model = genai.GenerativeModel(model_name)
        except Exception as e: return f"[Fehler bei Gemini API Konfig: {e}]"
        if not self.gemini_model: return "[Fehler: Gemini Modell Init fehlgeschlagen]"

        # print(f"\n💬 [Generator] RAG für: '{prompt}'") # Debug entfernt

        # 1. Pre-Retrieval Simulation & Sprunganalyse (bleibt gleich)
        self.simulate_network_step(decay_connections=False)
        jump_trigger_active = False; significant_jump_nodes = []
        # Code zur Sprunganalyse bleibt gleich
        if self.config.get("quantum_effect_jump_llm_trigger", True):
            prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
            directly_activated_q_nodes: List[Node] = [
                node for node_label, keywords in semantic_node_definitions.items()
                if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords)
                if (node := self.nodes.get(node_label)) and node.is_quantum and hasattr(node, 'last_measurement_log')
            ]
            if directly_activated_q_nodes:
                for node in directly_activated_q_nodes:
                    analysis = node.analyze_jumps(node.last_measurement_log)
                    if analysis.get("jump_detected", False):
                        jump_trigger_active = True
                        jump_info_str = f"{node.label}(J:{analysis.get('max_jump_abs', 0)})"
                        if jump_info_str not in significant_jump_nodes: significant_jump_nodes.append(jump_info_str)


        # 2. Retrieval (ruft das modifizierte respond_to_prompt auf)
        retrieved_chunks = self.respond_to_prompt(prompt)

        # 3. Baue Kontext für LLM (inkl. Limbus-Status für Prompt)
        arona_context_parts = []
        # Finde relevante Knoten erneut für Konsistenz im Prompt - Code bleibt gleich
        relevant_node_labels_for_context = set()
        prompt_lower_ctx = prompt.lower(); semantic_defs_ctx = self.config.get("semantic_nodes", {})
        for node_label, keywords in semantic_defs_ctx.items():
             if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower_ctx) for kw in keywords):
                  if self.nodes.get(node_label): relevant_node_labels_for_context.add(node_label)

        if relevant_node_labels_for_context: arona_context_parts.append(f"Konzepte: {', '.join(sorted(list(relevant_node_labels_for_context)))}.")
        if jump_trigger_active: arona_context_parts.append(f"Q-Sprung Hinweis: {', '.join(significant_jump_nodes)}.")

        # *** NEU: Füge Limbus PAD zum Kontext hinzu ***
        limbus_state = INITIAL_EMOTION_STATE.copy()
        limbus_node = self.nodes.get("Limbus Affektus")
        prompt_influence_level = self.config.get("limbus_influence_prompt_level", 0.5)
        if isinstance(limbus_node, LimbusAffektus):
             limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
             # Skaliere den Einfluss auf den Prompt
             p = limbus_state.get('pleasure', 0.0) * prompt_influence_level
             a = limbus_state.get('arousal', 0.0) * prompt_influence_level
             d = limbus_state.get('dominance', 0.0) * prompt_influence_level
             emotion_summary = f"Emotion (PAD): P={p:.2f}, A={a:.2f}, D={d:.2f}" # Modulierter Zustand für Prompt
             arona_context_parts.append(emotion_summary)
             # Optional: Debugging Output
             # if random.random() < 0.05:
             #     print(f"DEBUG generate_response: Adding Emotion to context: {emotion_summary} (Base P:{limbus_state.get('pleasure',0):.2f}, A:{limbus_state.get('arousal',0):.2f}, D:{limbus_state.get('dominance',0):.2f})")

        # Füge abgerufene Chunks hinzu - Code bleibt gleich
        if retrieved_chunks:
            # Gekürzt für Prompt, um Token-Limit nicht zu sprengen
            context_text = "\n---\n".join([f"[{chunk.source[:15]}]: {chunk.text[:200]}..." for chunk in retrieved_chunks])
            arona_context_parts.append(f"Kontext ({len(retrieved_chunks)} Ausschnitte):")
            arona_context_parts.append(context_text)
        else: arona_context_parts.append("Kein spezifischer Text-Kontext gefunden.")
        arona_context_string = "\n".join(arona_context_parts)


        # 4. Baue finalen Prompt (mit Hinweis auf internen Zustand inkl. Emotionen)
        input_prompt_for_gemini = f"""
Du bist eine KI mit internem Zustand (Konzepte, Emotionen, Quanten-Hinweise).
Benutzer: "{prompt}"

Interner Zustand/Kontext:
{arona_context_string}

Anweisung: Beantworte die Frage des Benutzers. Nutze den internen Zustand (Konzepte, Emotionen, Q-Hinweise, Kontext-Ausschnitte), um den **Tonfall, Fokus oder die Perspektive** deiner Antwort subtil anzupassen. Erkläre den internen Prozess NICHT. Antworte natürlich.

Antwort:
"""

        # 5. Generation mit Gemini API (mit MODULIERTER Temperatur)
        generated_text = "[Generierungsfehler]"
        try:
            start_gen_time = time.time()

            # *** NEU: Moduliere Temperatur ***
            base_temp = self.config.get("generator_temperature", 0.7)
            temp_mod_arousal = self.config.get("limbus_influence_temperature_arousal", 0.1)
            temp_mod_dominance = self.config.get("limbus_influence_temperature_dominance", -0.1)
            min_temp = self.config.get("limbus_min_temperature", 0.3)
            max_temp = self.config.get("limbus_max_temperature", 1.0)
            # Hole aktuelle (unskalierte) Emotionen für Temperatur-Modulation
            # limbus_state wurde bereits oben geholt
            # pleasure = limbus_state.get("pleasure", 0.0) # Wird hier nicht direkt genutzt, aber zur Info
            arousal = limbus_state.get("arousal", 0.0)
            dominance = limbus_state.get("dominance", 0.0)
            # Berechne modulierte Temperatur
            current_temperature = base_temp + (arousal * temp_mod_arousal) + (dominance * temp_mod_dominance)
            current_temperature = float(np.clip(current_temperature, min_temp, max_temp))
            # Optional: Debugging Output
            # if random.random() < 0.05:
            #     print(f"DEBUG generate_response: Base Temp={base_temp:.2f}, Modulated Temp={current_temperature:.2f} (A:{arousal:.2f}, D:{dominance:.2f})")

            generation_config = genai.types.GenerationConfig(
                # *** Verwende MODULIERTE Temperatur ***
                temperature=current_temperature,
                max_output_tokens=self.config.get("generator_max_length", 8192)
            )
            # Safety Settings bleiben gleich
            safety_settings=[ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            response = self.gemini_model.generate_content(
                input_prompt_for_gemini,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Restliche Antwortverarbeitung und Self-Learning - Code bleibt gleich
            if not response.candidates:
                 reason = "Unbekannt"; ratings_str = "N/A"
                 if hasattr(response, 'prompt_feedback'):
                     if hasattr(response.prompt_feedback, 'block_reason'): reason = response.prompt_feedback.block_reason.name
                     if hasattr(response.prompt_feedback, 'safety_ratings'): ratings_str = ", ".join([f"{r.category.name}:{r.probability.name}" for r in response.prompt_feedback.safety_ratings])
                 print(f"WARNUNG: Gemini-Antwort blockiert. Grund: {reason}. Ratings: [{ratings_str}]")
                 generated_text = f"[Antwort blockiert: {reason}]"
            else:
                 # Sicherer Zugriff auf Text
                 generated_text = getattr(response, 'text', None)
                 if generated_text is None and hasattr(response, 'parts'): # Fallback für Multi-Part
                     try: generated_text = "".join(part.text for part in response.parts)
                     except Exception: generated_text = "[Fehler Parts]"
                 generated_text = generated_text.strip() if generated_text else "[Leere Antwort]"
                 print(f"   -> Generierung mit Gemini in {time.time() - start_gen_time:.2f}s (Mod. Temp: {current_temperature:.2f})")

            # 6. Self-Learning Schritt (Logik bleibt gleich)
            is_valid_response = (generated_text and not generated_text.startswith("[") and not generated_text.endswith("]"))
            if self.self_learning_enabled and is_valid_response:
                 print(f"\n🎓 [Self-Learning] Starte Lernzyklus...")
                 self._save_and_reprocess_response(generated_text)

            return generated_text

        # Fehlerbehandlung bleibt gleich
        except GoogleAPIError as api_err:
            print(f"FEHLER bei der Gemini API Anfrage: {api_err}")
            # Versuche, den Grund zu extrahieren, falls vorhanden
            reason = getattr(api_err, 'reason', '?') if hasattr(api_err, 'reason') else '?'
            message = getattr(api_err, 'message', str(api_err))
            return f"[Fehler: Google API Problem (Reason: {reason}, Message: {message[:100]}...)]"
        except Exception as e:
            print(f"FEHLER während der Textgenerierung: {e}"); traceback.print_exc(limit=2)
            return "[Fehler: Interner Fehler bei Generierung]"


    def _save_and_reprocess_response(self, response_text: str):
        """Speichert Antwort und verarbeitet Lerndatei neu."""
        if not response_text: return
        learn_file = self.learn_file_path; learn_source = self.learn_source_name
        try:
            os.makedirs(os.path.dirname(learn_file), exist_ok=True)
            with open(learn_file, 'a', encoding='utf-8') as f:
                f.write("\n\n---\n\n" + response_text) # Mit Trenner
            print(f"   -> [Self-Learning] Antwort an '{learn_file}' angehängt.")
            # Verarbeite die gesamte Lerndatei neu
            print(f"   -> [Self-Learning] Verarbeite '{learn_file}' neu...")
            self.load_and_process_file(learn_file, source_name=learn_source)
            print(f"   -> [Self-Learning] Neuverarbeitung abgeschlossen.")
        except Exception as e:
            print(f"FEHLER im Self-Learning Zyklus: {e}"); traceback.print_exc(limit=1)


    def get_network_state_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung des aktuellen Netzwerkzustands zurück."""
        summary = {
            "num_nodes": len(self.nodes),
            "num_quantum_nodes": sum(1 for n in self.nodes.values() if n.is_quantum),
            "num_chunks": len(self.chunks),
            "sources_processed": sorted(list(self.sources_processed)), # Sortiert für Konsistenz
            "self_learning_enabled": getattr(self, 'self_learning_enabled', False),
            "tfidf_index_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            "rag_enabled": getattr(self, 'rag_enabled', False),
            "generator_model": self.config.get("generator_model_name") if getattr(self, 'rag_enabled', False) else None
        }
        # Durchschnittliche Aktivierung
        activations = [n.activation for n in self.nodes.values() if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)]
        summary["average_node_activation"] = round(np.mean(activations), 4) if activations else 0.0

        # Limbus Zustand hinzufügen
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             summary["limbus_state_PAD"] = {k: round(v, 3) for k, v in limbus_node.emotion_state.items()}

        # Verbindungen zählen und Top finden
        total_valid_connections = 0; all_connections_found = []; nodes_with_connections_count = 0
        node_uuid_map = {n.uuid: n for n in self.nodes.values()} # Effizienter Lookup

        for source_node in self.nodes.values():
             connections_dict = getattr(source_node, 'connections', None)
             if not isinstance(connections_dict, dict): continue
             if connections_dict: nodes_with_connections_count += 1

             for target_uuid, conn in connections_dict.items():
                 if conn is None: continue
                 target_node_obj = node_uuid_map.get(getattr(conn, 'target_node_uuid', None))
                 weight = getattr(conn, 'weight', None)
                 # Prüfe ob Ziel & Gewicht gültig sind
                 if (target_node_obj and hasattr(target_node_obj, 'label') and
                     weight is not None and isinstance(weight, (float, np.number)) and np.isfinite(weight)):
                      total_valid_connections += 1
                      all_connections_found.append({
                          "source": source_node.label, "target": target_node_obj.label,
                          "weight": round(weight, 4) # Runde für Anzeige
                      })

        summary["total_connections"] = total_valid_connections
        all_connections_found.sort(key=lambda x: x["weight"], reverse=True)
        summary["top_connections"] = all_connections_found[:10] # Top 10
        summary["_debug_nodes_with_connections"] = nodes_with_connections_count # Debug-Info
        return summary


    def save_state(self, filepath: str) -> None:
        """Speichert den aktuellen Zustand in eine JSON-Datei."""
        print(f"💾 Speichere Zustand nach {filepath}...")
        try:
            # 1. Bereinige ungültige Verbindungen (optional aber empfohlen)
            existing_uuids = {node.uuid for node in self.nodes.values()}
            for node in self.nodes.values():
                if isinstance(getattr(node, 'connections', None), dict):
                    # Erstelle neues Dict nur mit gültigen Verbindungen
                    valid_connections = {}
                    for tgt_uuid, conn in node.connections.items():
                         if conn and getattr(conn, 'target_node_uuid', tgt_uuid) in existing_uuids:
                              valid_connections[tgt_uuid] = conn
                    node.connections = valid_connections # Ersetze altes Dict

            # 2. Serialisiere Chunks
            chunks_to_save = {
                c_uuid: {
                    "uuid": c_obj.uuid, "text": c_obj.text, "source": c_obj.source,
                    "index": c_obj.index, "activated_node_labels": getattr(c_obj, 'activated_node_labels', [])
                } for c_uuid, c_obj in self.chunks.items()
                  if hasattr(c_obj, 'text') # Grundlegende Prüfung
            }

            # 3. Serialisiere Knoten (ruft __getstate__ auf)
            nodes_data_for_json = {label: node.__getstate__() for label, node in self.nodes.items()}

            # 4. Baue state_data zusammen
            state_data = {
                # WICHTIG: Speichere die zur Laufzeit verwendete Config, NICHT die Default Config
                "config": self.config,
                "nodes": nodes_data_for_json,
                "chunks": chunks_to_save,
                "sources_processed": list(self.sources_processed),
                "chunk_id_list_for_tfidf": self.chunk_id_list_for_tfidf
            }

            # 5. Schreibe JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                # Sicherer Serializer (unverändert)
                def default_serializer(obj: Any) -> Any:
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (datetime, deque)): return str(obj)
                    if isinstance(obj, set): return list(obj)
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                    if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
                    if isinstance(obj, (np.bool_)): return bool(obj)
                    if isinstance(obj, (np.void)): return None # Kann nicht sinnvoll serialisiert werden
                    try: return repr(obj)
                    except Exception: return f"<SerializationError: {type(obj)}>"
                json.dump(state_data, f, indent=2, ensure_ascii=False, default=default_serializer)

            print("   -> Zustand erfolgreich gespeichert.")
        except Exception as e:
            print(f"FEHLER Speichern Zustand: {e}"); traceback.print_exc(limit=2)


    @classmethod
    def load_state(cls, filepath: str) -> Optional['QuantumEnhancedTextProcessor']:
        """Lädt den Prozessorzustand aus einer JSON-Datei."""
        print(f"📂 Lade Zustand von {filepath}...")
        if not os.path.exists(filepath): print(f"FEHLER: Zustandsdatei {filepath} nicht gefunden."); return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f: state_data = json.load(f)

            # Erstelle Instanz mit gespeicherter Config
            # Stelle sicher, dass die geladene Config alle NEUEN Default-Keys enthält
            config_from_state = state_data.get("config", {})
            merged_config = cls.DEFAULT_CONFIG.copy() # Starte mit aktuellen Defaults
            merged_config.update(config_from_state)  # Überschreibe mit gespeicherten Werten
            instance = cls(config_dict=merged_config) # __init__ wird aufgerufen mit gemischter Config
            if not instance: print("FEHLER: Instanzerstellung fehlgeschlagen."); return None

            # Lade Chunks
            loaded_chunks = {}
            for uuid_key, chunk_data in state_data.get("chunks", {}).items():
                 if isinstance(chunk_data, dict):
                     try:
                         new_chunk = TextChunk(
                             chunk_uuid=chunk_data.get('uuid', uuid_key),
                             text=chunk_data.get('text', ''),
                             source=chunk_data.get('source', 'Unknown'),
                             index=chunk_data.get('index', -1)
                         )
                         if new_chunk.text: # Nur nicht-leere Chunks
                             new_chunk.activated_node_labels = chunk_data.get('activated_node_labels', [])
                             loaded_chunks[new_chunk.uuid] = new_chunk
                     except Exception as e: print(f"FEHLER Erstellen Chunk UUID {uuid_key}: {e}")
            instance.chunks = loaded_chunks
            print(f"INFO (load_state): {len(instance.chunks)} Chunks geladen.")

            # Lade Knoten (ruft __setstate__ auf)
            loaded_node_states = state_data.get("nodes", {})
            instance.nodes = {} # Leere das von __init__ erstellte Dict
            node_uuid_map = {}
            for node_label, node_state_dict in loaded_node_states.items():
                 if isinstance(node_state_dict, dict):
                     node_type_name = node_state_dict.get('type', 'Node')
                     node_class = globals().get(node_type_name, Node) # Finde Klasse global
                     try:
                         node = node_class.__new__(node_class) # Erstelle leeres Objekt
                         node.__setstate__(node_state_dict) # Setze Zustand
                         instance.nodes[node.label] = node # Füge zum Prozessor hinzu
                         node_uuid_map[node.uuid] = node # Baue UUID Map
                     except Exception as e: print(f"FEHLER Restore Knoten '{node_label}': {e}"); traceback.print_exc(limit=1)

            # Setze Config Referenz und Parameter für Limbus Affektus explizit NEU
            # basierend auf der (potentiell aktualisierten) Config der Instanz
            limbus_label = "Limbus Affektus"
            if limbus_label in instance.nodes and isinstance(instance.nodes[limbus_label], LimbusAffektus):
                limbus_node_loaded = instance.nodes[limbus_label]
                limbus_node_loaded.config = instance.config # Setze aktuelle Config-Referenz
                # Aktualisiere Parameter aus der aktuellen Config
                limbus_node_loaded.decay = instance.config.get("limbus_emotion_decay", 0.95)
                limbus_node_loaded.arousal_sens = instance.config.get("limbus_arousal_sensitivity", 1.5)
                limbus_node_loaded.pleasure_sens = instance.config.get("limbus_pleasure_sensitivity", 1.0)
                limbus_node_loaded.dominance_sens = instance.config.get("limbus_dominance_sensitivity", 1.0)
                limbus_node_loaded.last_input_sum_for_pleasure = 0.0 # Reset bei Load
                print(f"INFO: Limbus Node '{limbus_label}' Parameter aus aktueller Config übernommen.")

            # Stelle Verbindungen wieder her (NACHDEM alle Knoten geladen sind)
            total_connections_restored = 0
            for node in instance.nodes.values():
                if hasattr(node, 'connections_serializable_temp') and isinstance(node.connections_serializable_temp, dict):
                     node.connections = {} # Initialisiere leeres Live-Dict
                     for target_uuid, conn_dict in node.connections_serializable_temp.items():
                          target_node = node_uuid_map.get(target_uuid)
                          if target_node and isinstance(conn_dict, dict):
                               try:
                                   conn = Connection.__new__(Connection) # Leeres Objekt
                                   # Setze Attribute aus conn_dict sicher
                                   conn.weight = float(conn_dict.get('weight', 0.0))
                                   conn.source_node_label = conn_dict.get('source_node_label', node.label)
                                   conn.conn_type = conn_dict.get('conn_type', 'associative')
                                   conn.last_transmitted_signal = float(conn_dict.get('last_transmitted_signal', 0.0))
                                   conn.transmission_count = int(conn_dict.get('transmission_count', 0))
                                   try: conn.created_at = datetime.fromisoformat(conn_dict.get('created_at', ''))
                                   except: conn.created_at = datetime.now()
                                   try: conn.last_update_at = datetime.fromisoformat(conn_dict.get('last_update_at', ''))
                                   except: conn.last_update_at = datetime.now()
                                   conn.target_node_uuid = target_uuid # Wichtig
                                   # conn.target_node = target_node # Setze KEINE Live-Referenz mehr hier, nur UUID
                                   # Füge wiederhergestellte Verbindung hinzu
                                   node.connections[target_uuid] = conn
                                   total_connections_restored += 1
                                   # Informiere Zielknoten (für eingehende Liste)
                                   if hasattr(target_node, 'add_incoming_connection_info'):
                                       target_node.add_incoming_connection_info(node.uuid, node.label)

                               except Exception as conn_e: print(f"FEHLER Restore Conn '{node.label}'->'{target_uuid}': {conn_e}")
                     del node.connections_serializable_temp # Entferne temporäres Dict

            # Lade restliche Metadaten
            instance.sources_processed = set(state_data.get("sources_processed", []))
            instance.chunk_id_list_for_tfidf = state_data.get("chunk_id_list_for_tfidf", [])

            # Aktualisiere TF-IDF Index basierend auf geladenen Chunks und IDs
            instance.update_tfidf_index()

            print(f"\n✅ Zustand geladen ({len(instance.nodes)} Knoten, {len(instance.chunks)} Chunks, {total_connections_restored} Verbindungen wiederhergestellt).")
            return instance

        except json.JSONDecodeError as json_err: print(f"FEHLER: JSON Decode Error in {filepath}: {json_err}"); return None
        except Exception as e: print(f"FEHLER Laden Zustand {filepath}: {e}"); traceback.print_exc(limit=2); return None

# --- Beispielnutzung __main__ ---
if __name__ == "__main__":
    # --- Dieser Teil bleibt unverändert zum Testen der Klasse ---
    print("="*50 + "\n Starte Quantum-Enhanced Text Processor Demo (v1.2 - Limbus Modulation) \n" + "="*50)
    CONFIG_FILE = "config_qllm.json"; STATE_FILE = "qetp_state.json"

    # Lade Zustand oder initialisiere neu
    processor = QuantumEnhancedTextProcessor.load_state(STATE_FILE)
    if processor is None:
        print(f"\nInitialisiere neu mit '{CONFIG_FILE}'.")
        processor = QuantumEnhancedTextProcessor(config_path=CONFIG_FILE)
        if processor is None or not hasattr(processor, 'config'): print("\nFATALER FEHLER: Init fehlgeschlagen."); exit()
        # Führe initiale Datenverarbeitung durch
        training_files = processor.config.get("training_files", [])
        if not training_files: print("\nWARNUNG: Keine Trainingsdateien in Config.")
        else:
             print("\n--- Initiale Datenverarbeitung ---")
             for file in training_files: processor.load_and_process_file(file)
             print("--- Initiale Verarbeitung abgeschlossen ---")
             processor.save_state(STATE_FILE) # Speichere initialen Zustand
    else: print(f"\nZustand aus '{STATE_FILE}' geladen.")

    # ---- Interaktive Schleife ----
    print("\n--- Aktueller Netzwerkstatus ---")
    try:
        summary = processor.get_network_state_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e: print(f"Fehler beim Abrufen der Summary: {e}")

    print("\n--- Interaktive Abfrage (Typ 'exit' zum Beenden) ---")
    if processor.rag_enabled and not os.environ.get("GEMINI_API_KEY"):
         # Versuche Fallback zu Streamlit Secrets vor der Warnung
         try:
             import streamlit as st
             if not st.secrets.get("GEMINI_API_KEY"):
                 print("\nWARNUNG: RAG aktiviert, aber GEMINI_API_KEY fehlt (weder Umgebungsvariable noch Streamlit Secret).")
         except ImportError:
              print("\nWARNUNG: RAG aktiviert, aber GEMINI_API_KEY fehlt (Umgebungsvariable).")
         except Exception: # Falls st.secrets nicht existiert oder Fehler wirft
              print("\nWARNUNG: RAG aktiviert, aber GEMINI_API_KEY fehlt (Umgebungsvariable, Streamlit Secrets nicht prüfbar).")


    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.lower() == 'exit': break
            if not prompt: continue

            # Führe vor der Antwort einen Simulationsschritt durch, um Emotionen etc. zu aktualisieren
            # print("INFO: Simuliere Netzwerk-Schritt vor Antwort...") # Optional Info
            processor.simulate_network_step(decay_connections=True) # Decay aktivieren im laufenden Betrieb

            # Generiere die Antwort
            generated_response = processor.generate_response(prompt)
            print("\n--- Generierte Antwort ---"); print(generated_response); print("-" * 25)

            # Zeige aktuellen Limbus-Zustand nach der Interaktion
            limbus_node = processor.nodes.get("Limbus Affektus")
            if isinstance(limbus_node, LimbusAffektus):
                pad_state = {k: round(v, 3) for k, v in limbus_node.emotion_state.items()}
                print(f"--- Aktueller Limbus Zustand: {pad_state} ---")

            # Speichern nach erfolgreicher Antwort (inkl. Self-Learning falls aktiv)
            # Self-Learning passiert jetzt IN generate_response, wenn aktiv und Antwort valide
            # Speichern wir hier trotzdem, um den Netzwerk/Emotionszustand zu sichern
            print("\n--- Speichere Zustand nach Interaktion ---")
            processor.save_state(STATE_FILE)

        except KeyboardInterrupt: print("\nUnterbrochen."); break
        except Exception as e: print(f"\nFehler in der Hauptschleife: {e}"); traceback.print_exc(limit=1)

    print("\n--- Speichere finalen Zustand ---"); processor.save_state(STATE_FILE)
    print("\n" + "="*50 + "\n Demo beendet. \n" + "="*50)

