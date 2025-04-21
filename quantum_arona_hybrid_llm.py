# -- coding: utf-8 --

# Filename: quantum_arona_hybrid_llm.py
# Version: 1.0 - Self-Learning Cycle Integration
# Author: [CipherCore Technology] & Gemini & Your Input & History Maker

import numpy as np
import pandas as pd
import random
from collections import deque, Counter, defaultdict
import json
import sqlite3
import os
import time
import traceback
from typing import Optional, Callable, List, Tuple, Dict, Any, Generator
from datetime import datetime
import math
import uuid as uuid_module
import re

# Text Processing / Retrieval specific imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NEU: Imports für Gemini API ---
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
# --- Ende NEU: Imports ---

# Optional: Netzwerk-Visualisierung
try: import networkx as nx; NETWORKX_AVAILABLE = True
except ImportError: NETWORKX_AVAILABLE = False

# Optional: Fortschrittsbalken
try: from tqdm import tqdm; TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # print("Warnung: tqdm nicht gefunden.")
    def tqdm(iterable, *args, **kwargs): return iterable

# --- HILFSFUNKTIONEN & BASIS-GATES ---
# ... ( unverändert bis QuantumNodeSystem ) ...
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)
def _ry(theta: float) -> np.ndarray:
    if not np.isfinite(theta): theta = 0.0
    cos_t = np.cos(theta / 2); sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)
def _rz(phi: float) -> np.ndarray:
    if not np.isfinite(phi): phi = 0.0
    exp_m = np.exp(-1j * phi / 2); exp_p = np.exp(1j * phi / 2)
    return np.array([[exp_m, 0], [0, exp_p]], dtype=complex)
def _apply_gate(state_vector: np.ndarray, gate: np.ndarray, target_qubit: int, num_qubits: int) -> np.ndarray:
    if gate.shape != (2, 2): raise ValueError("Gate must be 2x2.")
    if not (0 <= target_qubit < num_qubits): raise ValueError(f"Target qubit {target_qubit} out of range [0, {num_qubits-1}].")
    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0
    op_list = [I] * num_qubits; op_list[target_qubit] = gate
    full_matrix = op_list[0];
    for i in range(1, num_qubits): full_matrix = np.kron(full_matrix, op_list[i])
    new_state = np.dot(full_matrix, state_vector)
    if not np.all(np.isfinite(new_state)):
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    return new_state
def _apply_cnot(state_vector: np.ndarray, control_qubit: int, target_qubit: int, num_qubits: int) -> np.ndarray:
    if not (0 <= control_qubit < num_qubits and 0 <= target_qubit < num_qubits): raise ValueError("Qubit index out of range.")
    if control_qubit == target_qubit: raise ValueError("Control and target must be different.")
    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0
    op_list_p0 = [I] * num_qubits; op_list_p1 = [I] * num_qubits
    op_list_p0[control_qubit] = P0; op_list_p1[control_qubit] = P1; op_list_p1[target_qubit] = X
    term0_matrix = op_list_p0[0]; term1_matrix = op_list_p1[0]
    for i in range(1, num_qubits):
        term0_matrix = np.kron(term0_matrix, op_list_p0[i])
        term1_matrix = np.kron(term1_matrix, op_list_p1[i])
    cnot_matrix = term0_matrix + term1_matrix
    new_state = np.dot(cnot_matrix, state_vector)
    if not np.all(np.isfinite(new_state)):
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    return new_state

# --- QUANTEN-ENGINE ---
class QuantumNodeSystem:
    """Simuliert das quantenbasierte Verhalten eines Knotens via PQC."""
    def __init__(self, num_qubits: int, initial_params: Optional[np.ndarray] = None):
        if num_qubits <= 0: raise ValueError("num_qubits must be positive.")
        self.num_qubits = num_qubits; self.num_params = num_qubits * 2
        self.state_vector_size = 2**self.num_qubits
        if initial_params is None:
            self.params = np.random.rand(self.num_params) * 2 * np.pi
        elif isinstance(initial_params, np.ndarray) and initial_params.shape == (self.num_params,):
            if not np.all(np.isfinite(initial_params)):
                self.params = np.random.rand(self.num_params) * 2 * np.pi
            else: self.params = np.clip(np.nan_to_num(initial_params, nan=np.pi), 0, 2 * np.pi)
        else:
            self.params = np.random.rand(self.num_params) * 2 * np.pi
        self.state_vector = np.zeros(self.state_vector_size, dtype=complex); self.state_vector[0] = 1.0 + 0j
        self.last_measurement_results: List[Dict] = []; self.last_applied_ops: List[Tuple] = []

    def _build_pqc_ops(self, input_strength: float) -> List[Tuple]:
        ops = []; scaled_input_angle = np.tanh(input_strength) * np.pi
        if not np.isfinite(scaled_input_angle): scaled_input_angle = 0.0
        for i in range(self.num_qubits): ops.append(('H', i))
        for i in range(self.num_qubits):
            theta = scaled_input_angle * self.params[2 * i]
            ops.append(('RY', i, theta if np.isfinite(theta) else 0.0))
        for i in range(self.num_qubits):
            phi = self.params[2 * i + 1]
            ops.append(('RZ', i, phi if np.isfinite(phi) else 0.0))
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1): ops.append(('CNOT', i, i + 1))
        return ops

    def activate(self, input_strength: float, n_shots: int = 100) -> Tuple[float, np.ndarray, List[Dict]]:
        if not np.isfinite(input_strength): input_strength = 0.0
        if n_shots <= 0: n_shots = 1
        pqc_ops = self._build_pqc_ops(input_strength); self.last_applied_ops = pqc_ops
        current_state = self.state_vector.copy()
        if not np.isclose(np.linalg.norm(current_state), 1.0):
            current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0
        gate_application_successful = True
        for op_index, op in enumerate(pqc_ops):
            try:
                op_type = op[0]
                if op_type == 'H': current_state = _apply_gate(current_state, H, op[1], self.num_qubits)
                elif op_type == 'RY': current_state = _apply_gate(current_state, _ry(op[2]), op[1], self.num_qubits)
                elif op_type == 'RZ': current_state = _apply_gate(current_state, _rz(op[2]), op[1], self.num_qubits)
                elif op_type == 'CNOT': current_state = _apply_cnot(current_state, op[1], op[2], self.num_qubits)
                if not np.all(np.isfinite(current_state)): raise ValueError(f"Non-finite state after {op}")
                norm = np.linalg.norm(current_state)
                if norm > 1e-9: current_state /= norm
                else: raise ValueError(f"Zero state after {op}")
            except Exception as e:
                # print(f"FEHLER: Gate-Fehler QNS bei {op}: {e}. Reset."); # Weniger verbose
                current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0
                gate_application_successful = False; break
        self.state_vector = current_state
        total_hamming_weight = 0; measurement_log = []; activation_prob = 0.0
        if n_shots > 0 and gate_application_successful and self.num_qubits > 0:
            probabilities = np.abs(current_state)**2; probabilities = np.maximum(0, probabilities)
            prob_sum = np.sum(probabilities)
            if not np.isclose(prob_sum, 1.0, atol=1e-7):
                if prob_sum < 1e-9: probabilities.fill(0.0); probabilities[0] = 1.0
                else: probabilities /= prob_sum
                probabilities = np.maximum(0, probabilities); probabilities /= np.sum(probabilities)
            try:
                measured_indices = np.random.choice(self.state_vector_size, size=n_shots, p=probabilities)
                for shot_idx, measured_index in enumerate(measured_indices):
                    state_idx_int = int(measured_index); binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                    hamming_weight = binary_repr.count('1'); total_hamming_weight += hamming_weight
                    measurement_log.append({"shot": shot_idx, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight, "probability": probabilities[state_idx_int]})
            except ValueError as e:
                 # print(f"FEHLER np.random.choice QNS: {e}. Fallback zu argmax."); # Weniger verbose
                 if np.any(probabilities):
                     measured_index = np.argmax(probabilities); state_idx_int = int(measured_index)
                     binary_repr = format(state_idx_int, f'0{self.num_qubits}b'); hamming_weight = binary_repr.count('1')
                     total_hamming_weight = hamming_weight * n_shots
                     measurement_log.append({"shot": 0, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight, "error": "ValueError, used argmax", "probability": probabilities[state_idx_int]})
                 else: measurement_log.append({"shot": 0, "index": 0, "binary": '0'*self.num_qubits, "hamming": 0, "error": "All probs zero", "probability": 0.0})
            if n_shots > 0 and self.num_qubits > 0:
                activation_prob = float(np.clip(total_hamming_weight / (n_shots * self.num_qubits), 0.0, 1.0))
                if not np.isfinite(activation_prob): activation_prob = 0.0
        elif not gate_application_successful:
             activation_prob = 0.0; measurement_log = [{"error": "PQC execution failed"}]
        self.last_measurement_results = measurement_log
        if not isinstance(activation_prob, (float, np.number)) or not np.isfinite(activation_prob): activation_prob = 0.0
        return activation_prob, self.state_vector, measurement_log

    def get_params(self) -> np.ndarray:
        safe_params = np.nan_to_num(self.params.copy(), nan=np.pi, posinf=2*np.pi, neginf=0.0)
        return np.clip(safe_params, 0, 2 * np.pi)

    def set_params(self, params: np.ndarray):
        if isinstance(params, np.ndarray) and params.shape == self.params.shape:
            safe_params = np.nan_to_num(params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
            self.params = np.clip(safe_params, 0, 2 * np.pi)

    def update_internal_params(self, delta_params: np.ndarray):
        if not isinstance(delta_params, np.ndarray) or delta_params.shape != self.params.shape: return
        if not np.all(np.isfinite(delta_params)): delta_params = np.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0)
        new_params = self.params + delta_params
        new_params_safe = np.nan_to_num(new_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        self.params = np.clip(new_params_safe, 0, 2 * np.pi)


# --- NETZWERK-STRUKTUR & TEXT-CHUNKS ---
class Connection:
    """Repräsentiert eine gerichtete, gewichtete Verbindung."""
    DEFAULT_WEIGHT_RANGE = (0.01, 0.5); DEFAULT_LEARNING_RATE = 0.05; DEFAULT_DECAY_RATE = 0.001
    def __init__(self, target_node: 'Node', weight: Optional[float] = None, source_node_label: Optional[str] = None, conn_type: str = "associative"):
        if target_node is None or not hasattr(target_node, 'uuid'): raise ValueError("Target node invalid."); self.target_node: 'Node' = target_node
        self.source_node_label: Optional[str] = source_node_label; self.conn_type: str = conn_type
        raw_weight = weight if weight is not None else random.uniform(*self.DEFAULT_WEIGHT_RANGE)
        self.weight: float = float(np.clip(raw_weight, 0.0, 1.0)); self.last_transmitted_signal: float = 0.0
        self.transmission_count: int = 0; self.created_at: datetime = datetime.now(); self.last_update_at: datetime = datetime.now()
        self.target_node_uuid: str = target_node.uuid

    def update_weight(self, delta_weight: float, learning_rate: Optional[float] = None):
        lr = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        new_weight = self.weight + (delta_weight * lr); self.weight = float(np.clip(new_weight, 0.0, 1.0)); self.last_update_at = datetime.now()
    def decay(self, decay_rate: Optional[float] = None):
        dr = decay_rate if decay_rate is not None else self.DEFAULT_DECAY_RATE
        self.weight = max(0.0, self.weight * (1.0 - dr)); self.last_update_at = datetime.now()
    def transmit(self, source_activation: float) -> float:
        transmitted_signal = source_activation * self.weight; self.last_transmitted_signal = transmitted_signal; self.transmission_count += 1; return transmitted_signal
    def __repr__(self) -> str:
        target_info = f"to_UUID:{self.target_node_uuid[:8]}..." # Zeige Anfang der UUID
        source_info = f" from:{self.source_node_label}" if self.source_node_label else ""
        weight_info = f"W:{self.weight:.3f}" if hasattr(self, 'weight') else "W:N/A"
        count_info = f"Cnt:{self.transmission_count}" if hasattr(self, 'transmission_count') else "Cnt:N/A"

        return f"<Conn {target_info} {weight_info} {count_info}{source_info}>"

class Node:
    """Basisklasse für alle Knoten."""
    DEFAULT_NUM_QUBITS = 10; DEFAULT_ACTIVATION_HISTORY_LEN = 20; DEFAULT_N_SHOTS = 50
    def __init__(self, label: str, num_qubits: Optional[int] = None, is_quantum: bool = True, neuron_type: str = "excitatory",
                 initial_params: Optional[np.ndarray] = None, uuid: Optional[str] = None):
        if not label: raise ValueError("Node label cannot be empty.");
        self.label: str = label
        self.uuid: str = uuid if uuid else str(uuid_module.uuid4()); self.neuron_type: str = neuron_type; self.is_quantum = is_quantum
        self.connections: Dict[str, Optional[Connection]] = {} # Standard Dictionary
        self.incoming_connections_info: List[Tuple[str, str]] = []
        self.activation: float = 0.0; self.activation_sum: float = 0.0
        self.activation_history: deque = deque(maxlen=self.DEFAULT_ACTIVATION_HISTORY_LEN)
        self.num_qubits = num_qubits if num_qubits is not None else self.DEFAULT_NUM_QUBITS
        self.q_system: Optional[QuantumNodeSystem] = None
        if self.is_quantum and self.num_qubits > 0:
            try: self.q_system = QuantumNodeSystem(num_qubits=self.num_qubits, initial_params=initial_params)
            except Exception as e: print(f"FEHLER init QNS für {self.label} mit {self.num_qubits} Qubits: {e}"); self.q_system = None; self.is_quantum = False
        elif self.is_quantum: self.is_quantum = False
        self.last_measurement_log: List[Dict] = []; self.last_state_vector: Optional[np.ndarray] = None

    # In der Klasse Node in quantum_arona_hybrid_llm.py

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None, conn_type: str = "associative") -> Optional[Connection]:
        # Grundlegende Prüfungen
        if target_node is None or not hasattr(target_node, 'uuid') or target_node.uuid == self.uuid:
            # print(f"DEBUG add_connection ({self.label}): Invalid target or self-loop. Target: {target_node}") # Optional
            return None
        target_uuid = target_node.uuid

        # Prüfe, ob die Verbindung bereits existiert
        if target_uuid not in self.connections:
            # --- DEBUGGING add_connection ---
            print(f"  +++ DEBUG add_connection ({self.label}): Target '{target_node.label}' ({target_uuid}) NOT in connections dict. Adding new connection.")
            # --- END DEBUGGING ---
            try:
                # Erstelle das neue Connection-Objekt
                conn = Connection(target_node=target_node, weight=weight, source_node_label=self.label, conn_type=conn_type)

                # === DER ENTSCHEIDENDE PUNKT: Das Hinzufügen zum Dictionary ===
                self.connections[target_uuid] = conn
                # ============================================================

                # --- DEBUGGING add_connection ---
                # Prüfe direkt nach dem Hinzufügen
                if target_uuid in self.connections and self.connections[target_uuid] is not None:
                    print(f"      --> SUCCESS: Connection to '{target_node.label}' seems added. self.connections length: {len(self.connections)}")
                else:
                    print(f"      --> ❌ FAILURE?: Connection to '{target_node.label}' NOT found in dict immediately after adding! self.connections length: {len(self.connections)}")
                # --- END DEBUGGING ---

                # Informiere den Zielknoten über die eingehende Verbindung (für Summary etc.)
                if hasattr(target_node, 'add_incoming_connection_info'):
                    target_node.add_incoming_connection_info(self.uuid, self.label)

                return conn # Gib die neu erstellte Verbindung zurück
            except Exception as e:
                 print(f"      --> ❌ EXCEPTION during connection creation/adding for {self.label} -> {target_node.label}: {e}")
                 traceback.print_exc(limit=1)
                 return None
        else:
            # Verbindung existiert bereits, gib die bestehende zurück (für strengthen_connection wichtig)
            # print(f"  --- DEBUG add_connection ({self.label}): Target '{target_node.label}' ({target_uuid}) ALREADY in connections dict. Returning existing.") # Optional: Weniger verbose
            return self.connections.get(target_uuid) # .get() ist sicherer

    def add_incoming_connection_info(self, source_uuid: str, source_label: str):
         if not any(info[0] == source_uuid for info in self.incoming_connections_info): self.incoming_connections_info.append((source_uuid, source_label))

    def strengthen_connection(self, target_node: 'Node', learning_signal: float = 0.1, learning_rate: Optional[float] = None):
        if target_node is None or not hasattr(target_node, 'uuid'): return
        target_uuid = target_node.uuid
        connection = self.connections.get(target_uuid)
        if connection is not None:
             connection.update_weight(delta_weight=learning_signal, learning_rate=learning_rate)

    def calculate_activation(self, n_shots: Optional[int] = None):
        current_n_shots = n_shots if n_shots is not None else self.DEFAULT_N_SHOTS; new_activation: float = 0.0
        if self.is_quantum and self.q_system:
            try:
                q_activation, q_state_vector, q_measure_log = self.q_system.activate(self.activation_sum, current_n_shots)
                new_activation = q_activation; self.last_state_vector = q_state_vector; self.last_measurement_log = q_measure_log
            except Exception as e:
                # print(f"FEHLER Quantenaktivierung für {self.label}: {e}") # Weniger verbose
                new_activation = 0.0; self.last_state_vector = None; self.last_measurement_log = [{"error": f"Activation failed: {e}"}]
        else:
            activation_sum_float = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
            safe_activation_sum = np.clip(activation_sum_float, -700, 700)
            try: new_activation = 1 / (1 + np.exp(-safe_activation_sum))
            except FloatingPointError: new_activation = 1.0 if safe_activation_sum > 0 else 0.0
            self.last_state_vector = None; self.last_measurement_log = []
        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation): self.activation = 0.0
        else: self.activation = float(np.clip(new_activation, 0.0, 1.0))
        self.activation_history.append(self.activation); self.activation_sum = 0.0

    def get_smoothed_activation(self, window: int = 3) -> float:
        if not self.activation_history: return self.activation
        hist = list(self.activation_history)[-window:]; valid_hist = [a for a in hist if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if not valid_hist: return self.activation
        else: return float(np.mean(valid_hist))

    def get_state_representation(self) -> Dict[str, Any]:
        state = {"label": self.label, "uuid": self.uuid, "activation": round(self.activation, 4), "smoothed_activation": round(self.get_smoothed_activation(), 4),
                 "type": type(self).__name__, "neuron_type": self.neuron_type, "is_quantum": self.is_quantum}
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits; state["last_measurement_analysis"] = self.analyze_jumps(self.last_measurement_log)
        if hasattr(self, 'emotion_state'): state["emotion_state"] = getattr(self, 'emotion_state', {}).copy()
        if hasattr(self, 'strategy_state'): state["strategy_state"] = getattr(self, 'strategy_state', {}).copy()
        state["num_connections"] = len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        return state

    def analyze_jumps(self, measurement_log: List[Dict]) -> Dict[str, Any]:
        default_jump_info = {"shots_recorded": len(measurement_log), "jump_detected": False, "max_jump_abs": 0, "avg_jump_abs": 0.0, "state_variance": 0.0, "significant_threshold": 0.0, "error_count": sum(1 for m in measurement_log if m.get("error"))}
        if len(measurement_log) < 2: return default_jump_info
        valid_indices = [m.get('index') for m in measurement_log if isinstance(m.get('index'), (int, np.integer))]
        if len(valid_indices) < 2: default_jump_info["shots_recorded"] = len(valid_indices); return default_jump_info
        indices_array = np.array(valid_indices, dtype=float); jumps = np.abs(np.diff(indices_array))
        state_variance = np.var(indices_array) if len(indices_array) > 1 else 0.0
        max_jump = 0; avg_jump = 0.0; jump_detected = False; significant_threshold = 0.0
        if jumps.size > 0:
            max_jump = np.max(jumps); avg_jump = np.mean(jumps)
            if self.is_quantum and self.q_system and self.num_qubits > 0: significant_threshold = (2**self.num_qubits) / 4.0
            else: significant_threshold = 1.0
            jump_detected = max_jump > significant_threshold
        return {"shots_recorded": len(valid_indices), "jump_detected": jump_detected, "max_jump_abs": int(max_jump), "avg_jump_abs": round(avg_jump, 3),
                "state_variance": round(state_variance, 3), "significant_threshold": round(significant_threshold, 1), "error_count": default_jump_info["error_count"]}

    def __repr__(self) -> str:
        act_str = f"Act:{self.activation:.3f}"; q_info = ""
        if self.is_quantum and self.q_system: q_info = f" Q:{self.num_qubits}"
        elif not self.is_quantum: q_info = " (Cls)"
        conn_count = len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        conn_info = f" Conns:{conn_count}"
        return f"<{type(self).__name__} '{self.label}' {act_str}{q_info}{conn_info}>"
    

    def __getstate__(self):
        """Erstellt ein serialisierbares Dictionary für den Zustand des Knotens."""
        # Explizit ein neues Dictionary erstellen
        state_to_return = {}

        # 1. Basisattribute hinzufügen (die sicher serialisierbar sind)
        # Schließt 'connections' und 'activation_history' hier aus, da sie speziell behandelt werden
        for key in ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum']:
             if hasattr(self, key):
                 state_to_return[key] = getattr(self, key)

        # Füge incoming_connections_info hinzu (Liste von Tupeln ist serialisierbar)
        if hasattr(self, 'incoming_connections_info'):
             info_list = getattr(self, 'incoming_connections_info')
             state_to_return['incoming_connections_info'] = info_list if isinstance(info_list, list) else []
        else:
             state_to_return['incoming_connections_info'] = []

        # 2. Quantenparameter serialisieren
        q_system = getattr(self, 'q_system', None)
        if q_system is not None and hasattr(q_system, 'get_params'):
            try:
                # Wandelt NumPy Array in Liste um für JSON Serialisierung
                q_params = q_system.get_params()
                state_to_return['q_system_params'] = q_params.tolist() if isinstance(q_params, np.ndarray) else q_params
            except Exception as e_q:
                print(f"    ERROR getting/converting q_system_params for {self.label}: {e_q}")
                state_to_return['q_system_params'] = None
        else:
            state_to_return['q_system_params'] = None

        # 3. Verbindungen serialisieren
        connections_serializable = {}
        live_connections = getattr(self, 'connections', None) # Holt das Live-Attribut
        if isinstance(live_connections, dict):
            for target_uuid, conn in live_connections.items():
                if conn is None: continue
                try:
                    # Erstelle ein sauberes Dict für jede Verbindung
                    target_uuid_in_conn = getattr(conn, 'target_node_uuid', target_uuid) # Priorisiere UUID aus Conn-Objekt
                    if not target_uuid_in_conn: # Überspringe, wenn keine UUID gefunden wird
                         print(f"    WARNUNG (__getstate__): Fehlende target_node_uuid für Verbindung von {self.label}. Übersprungen.")
                         continue

                    conn_data = {
                        'weight': float(getattr(conn, 'weight', 0.0)), # Stelle sicher, dass es float ist
                        'source_node_label': getattr(conn, 'source_node_label', self.label),
                        'conn_type': getattr(conn, 'conn_type', 'associative'),
                        'last_transmitted_signal': float(getattr(conn, 'last_transmitted_signal', 0.0)),
                        'transmission_count': int(getattr(conn, 'transmission_count', 0)),
                        'created_at': str(getattr(conn, 'created_at', datetime.now())), # Zu String
                        'last_update_at': str(getattr(conn, 'last_update_at', datetime.now())), # Zu String
                        'target_node_uuid': target_uuid_in_conn # Verwende die gefundene UUID
                    }
                    connections_serializable[target_uuid_in_conn] = conn_data # Verwende UUID als Schlüssel

                except Exception as e_ser:
                    print(f"    ERROR serializing connection object for UUID {target_uuid} from {self.label}: {e_ser}")

        state_to_return['connections_serializable'] = connections_serializable

        # 4. Aktivierungsverlauf serialisieren
        activation_hist = getattr(self, 'activation_history', None)
        if isinstance(activation_hist, deque):
            state_to_return['activation_history'] = list(activation_hist)
        else:
            state_to_return['activation_history'] = []

        # Füge den Typ hinzu, damit load_state weiß, welche Klasse es instanziieren soll
        state_to_return['type'] = type(self).__name__

        return state_to_return # Gib das neu erstellte, saubere Dictionary zurück


    def __setstate__(self, state: Dict[str, Any]):
        """Stellt den Zustand des Knotens aus einem Dictionary wieder her."""
        # Aktivierungsverlauf wiederherstellen
        # Nutze DEFAULT_ACTIVATION_HISTORY_LEN der Klasse, falls im state nicht vorhanden
        history_len = getattr(type(self), 'DEFAULT_ACTIVATION_HISTORY_LEN', 20) # Zugriff über type(self)
        state['activation_history'] = deque(state.get('activation_history', []), maxlen=history_len)

        # Quantensystem wiederherstellen
        q_params_list = state.pop('q_system_params', None)
        # Nutze DEFAULT_NUM_QUBITS der Klasse als Fallback
        num_qbits = state.get('num_qubits', getattr(type(self), 'DEFAULT_NUM_QUBITS', 10))
        is_q = state.get('is_quantum', True)
        self.q_system = None # Sicherstellen, dass es zurückgesetzt ist
        q_params_np = None

        # Konvertiere Parameterliste zurück zu NumPy Array
        if q_params_list is not None and isinstance(q_params_list, list):
             try:
                  q_params_np = np.array(q_params_list, dtype=float)
                  expected_shape = (num_qbits * 2,)
                  # Prüfe Shape nur wenn Qubits > 0
                  if num_qbits > 0:
                      if q_params_np.shape != expected_shape:
                           print(f"WARNUNG (__setstate__): QNS Param Shape mismatch for '{state.get('label', '?')}' (Expected {expected_shape}, Got {q_params_np.shape}). Resetting params.")
                           q_params_np = None # Setze zurück, QNS wird mit Random initialisiert
                      elif not np.all(np.isfinite(q_params_np)):
                           print(f"WARNUNG (__setstate__): QNS Params enthalten non-finite Werte für '{state.get('label', '?')}'. Resetting params.")
                           q_params_np = None
                  # Prüfe, ob Array leer ist, wenn Qubits == 0
                  elif num_qbits == 0 and q_params_np.size != 0:
                      print(f"WARNUNG (__setstate__): QNS Param nicht leer für 0 Qubits bei '{state.get('label', '?')}'. Ignoriere Params.")
                      q_params_np = None # Ignoriere die Parameter
                  # Falls num_qubits > 0 aber q_params_np jetzt None ist (wegen Fehler/Mismatch)
                  elif num_qbits > 0 and q_params_np is None:
                       print(f"INFO (__setstate__): QNS für '{state.get('label', '?')}' wird mit zufälligen Parametern initialisiert.")

             except Exception as e:
                 print(f"FEHLER (__setstate__) Konvertierung QNS Params für '{state.get('label', '?')}': {e}"); q_params_np = None

        # Erstelle QNS nur wenn is_quantum=True und num_qubits > 0
        if is_q and num_qbits > 0:
             try:
                 # Stelle sicher, dass QuantumNodeSystem verfügbar ist (global oder importiert)
                 global QuantumNodeSystem # Annahme: Klasse ist global verfügbar
                 self.q_system = QuantumNodeSystem(num_qubits=num_qbits, initial_params=q_params_np)
             except NameError:
                  print(f"FEHLER (__setstate__): Klasse QuantumNodeSystem nicht gefunden für Knoten '{state.get('label', '?')}'.")
                  state['is_quantum'] = False
                  state['num_qubits'] = 0
             except Exception as e:
                 print(f"FEHLER (__setstate__) Restore QNS für '{state.get('label', '?')}': {e}");
                 state['is_quantum'] = False
                 state['num_qubits'] = 0
        else:
             # Stelle sicher, dass is_quantum und num_qubits konsistent sind, wenn QNS nicht erstellt wird
             state['is_quantum'] = False
             state['num_qubits'] = 0


        # Speichere die serialisierten Verbindungen temporär
        # Das eigentliche Wiederherstellen der Connection-Objekte erfolgt später in load_state
        self.connections_serializable_temp = state.pop('connections_serializable', {})
        # Initialisiere das live connections Dictionary als leer; wird in load_state gefüllt
        self.connections: Dict[str, Optional[Connection]] = {}

        # Aktualisiere die restlichen Attribute aus dem state dict
        # Felder wie 'type' werden ignoriert, da sie nur für die Klassenwahl in load_state relevant waren
        valid_attrs = ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum', 'incoming_connections_info', 'activation_history']
        for key, value in state.items():
             if key in valid_attrs:
                 setattr(self, key, value)
             # Optional: Warnung für unbekannte Attribute
             # elif key not in ['connections_serializable_temp', 'type']: # Ignoriere bekannte, entfernte Schlüssel
             #     print(f"WARNUNG (__setstate__): Unbekanntes Attribut '{key}' im State für Knoten '{state.get('label', '?')}' gefunden.")


        # Stelle sicher, dass eine UUID existiert
        if not hasattr(self, 'uuid') or not self.uuid:
            try:
                 import uuid as uuid_module_local
                 self.uuid = str(uuid_module_local.uuid4())
                 print(f"WARNUNG (__setstate__): Fehlende UUID für Knoten '{getattr(self,'label','?')}' wiederhergestellt.")
            except ImportError:
                 print(f"FEHLER (__setstate__): Konnte UUID Modul nicht importieren, um UUID für '{getattr(self,'label','?')}' zu generieren.")
                 # Setze ggf. einen Platzhalter oder werfe einen Fehler
                 self.uuid = f"missing_uuid_{random.randint(1000, 9999)}"


        # Stelle sicher, dass incoming_connections_info eine Liste ist
        if not hasattr(self, 'incoming_connections_info') or not isinstance(self.incoming_connections_info, list):
            self.incoming_connections_info = []


# --- Ende Methoden in Node-Klasse ---

# Emotionale Konstanten
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}

class LimbusAffektus(Node): # Beispielhafte Implementierung
    """Modelliert den emotionalen Zustand."""
    def __init__(self, label: str = "Limbus Affektus", num_qubits: int = 4, **kwargs):
        super().__init__(label, num_qubits=num_qubits, is_quantum=True, neuron_type="affective_modulator", **kwargs)
        self.emotion_state = INITIAL_EMOTION_STATE.copy()
    # update_emotion_state muss hier implementiert sein, z.B.:
    def update_emotion_state(self, all_nodes: List['Node']):
        pass # Hier Logik einfügen

class TextChunk:
    """Repräsentiert einen Textabschnitt."""
    def __init__(self, text: str, source: str, index: int, chunk_uuid: Optional[str]=None):
        self.uuid = chunk_uuid if chunk_uuid else str(uuid_module.uuid4()); self.text: str = text; self.source: str = source
        self.index: int = index; self.activated_node_labels: List[str] = []; self.embedding: Optional[np.ndarray] = None
    def __repr__(self) -> str:
        node_str = f" Nodes:[{','.join(self.activated_node_labels)}]" if self.activated_node_labels else ""
        return f"<Chunk {self.index} from '{self.source}' (UUID:{self.uuid[:4]}...) Len:{len(self.text)}{node_str}>"

# --- HAUPTPROZESSOR-KLASSE ---
class QuantumEnhancedTextProcessor:
    """Orchestriert Laden, Quantenknoten, Lernen und RAG."""
    DEFAULT_CONFIG = {
        "embedding_dim": 128, "chunk_size": 500, "chunk_overlap": 100, "training_epochs": 1, "training_files": [], "semantic_nodes": {},
        "connection_learning_rate": 0.08, "connection_decay_rate": 0.002, "connection_strengthening_signal": 0.15, "max_prompt_results": 3,
        "relevance_threshold": 0.08, "tfidf_max_features": 5000, "use_quantum_nodes": True, "default_num_qubits": 10, "simulation_n_shots": 50,
        "simulation_steps_after_training": 0, "enable_rag": True, "generator_model_name": "google/flan-t5-small", "generator_max_length": 200,
        "generator_num_beams": 4, "generator_temperature": 0.7, "generator_repetition_penalty": 1.2,
        # --- Konfiguration für Quanteneffekte ---
        "quantum_effect_variance_penalty": 0.5,
        "quantum_effect_activation_boost": 0.3,
        "quantum_effect_jump_llm_trigger": True,
        # --- NEU: Konfiguration für den Lernzyklus ---
        "enable_self_learning": True,
        "self_learning_file_path": "./training_data/learn.txt",
        "self_learning_source_name": "Generated Responses",
    }

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict]=None):
        if config_path: self.config = self._load_config(config_path)
        elif config_dict: self.config = {**self.DEFAULT_CONFIG, **config_dict}
        else: print("WARNUNG: Keine Konfig übergeben, nutze Defaults."); self.config = self.DEFAULT_CONFIG.copy()
        for key, value in self.DEFAULT_CONFIG.items(): self.config.setdefault(key, value)
        self.nodes: Dict[str, Node] = {}; self.chunks: Dict[str, TextChunk] = {}; self.sources_processed: set = set()
        self._initialize_semantic_nodes()
        self.vectorizer: Optional[TfidfVectorizer] = None; self.tfidf_matrix: Optional[np.ndarray] = None; self.chunk_id_list_for_tfidf: List[str] = []
        self.generator_model = None; self.generator_tokenizer = None; self.gemini_model = None # Für Gemini
        self.rag_enabled = self.config.get("enable_rag", False) and GEMINI_AVAILABLE # Prüfe Gemini Lib
        # NEU: Self-Learning Schalter
        self.self_learning_enabled = self.config.get("enable_self_learning", False)
        self.learn_file_path = self.config.get("self_learning_file_path", "./training_data/learn.txt")
        self.learn_source_name = self.config.get("self_learning_source_name", "Generated Responses")

        if self.rag_enabled:
            # Gemini Initialisierung wird in generate_response gemacht, braucht API Key
            print(f"INFO: RAG aktiviert. Gemini Modell '{self.config.get('generator_model_name', 'gemini-1.5-flash-latest')}' wird bei Bedarf initialisiert.")
        else: print(f"INFO: RAG {'deaktiviert (Config)' if not self.config.get('enable_rag') else 'deaktiviert (google-generativeai fehlt)'}.")
        print(f"INFO: Self-Learning {'AKTIVIERT' if self.self_learning_enabled else 'DEAKTIVIERT'} (Ziel: {self.learn_file_path})") # Info über Lernzyklus
        print(f"\nQuantumEnhancedTextProcessor initialisiert mit {len(self.nodes)} semantischen Knoten.")
        if self.config.get("use_quantum_nodes"): print(f" -> Davon {sum(1 for n in self.nodes.values() if n.is_quantum)} Quantenknoten mit je {self.config.get('default_num_qubits')} Qubits.")
        print(f" -> RAG (Textgenerierung via Gemini) {'AKTIVIERT' if self.rag_enabled else 'DEAKTIVIERT'}")


    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f: loaded_config = json.load(f)
            config = self.DEFAULT_CONFIG.copy(); config.update(loaded_config)
            return config
        except Exception as e: print(f"FEHLER Laden Config {path}: {e}. Nutze Defaults."); return self.DEFAULT_CONFIG.copy()

    def _initialize_semantic_nodes(self):
        semantic_node_definitions = self.config.get("semantic_nodes", {}); use_quantum = self.config.get("use_quantum_nodes", True)
        num_qubits = self.config.get("default_num_qubits")
        # print(f"DEBUG _initialize_semantic_nodes: Verwende num_qubits = {num_qubits}") # Debug entfernt
        for label in semantic_node_definitions.keys():
            if label not in self.nodes:
                # print(f"  Initialisiere Knoten '{label}' mit num_qubits={num_qubits}") # Weniger verbose
                try:
                    node = Node(label=label, is_quantum=use_quantum, num_qubits=num_qubits if use_quantum else 0, neuron_type="semantic")
                    self.nodes[label] = node
                except Exception as e: print(f"FEHLER Erstellen Knoten '{label}': {e}. Übersprungen.")

    def _get_or_create_node(self, label: str, neuron_type: str = "semantic") -> Optional[Node]:
        if not label: return None
        if label in self.nodes: return self.nodes[label]
        else:
            try:
                use_quantum = self.config.get("use_quantum_nodes", True); num_qubits = self.config.get("default_num_qubits")
                node = Node(label=label, is_quantum=use_quantum, num_qubits=num_qubits if use_quantum else 0, neuron_type=neuron_type)
                self.nodes[label] = node; return node
            except Exception as e: print(f"FEHLER dyn. Erstellen Knoten '{label}': {e}"); return None

    def load_and_process_file(self, file_path: str, source_name: Optional[str] = None):
        """
        Lädt Text aus einer Datei, zerlegt ihn in Chunks und verarbeitet diese.
        Verhindert erneute Verarbeitung derselben Quelle in einer Sitzung (außer self-learning).
        """
        if not os.path.exists(file_path): print(f"FEHLER: Datei nicht gefunden: {file_path}"); return
        effective_source_name = source_name if source_name else os.path.basename(file_path)

        # Verhindere erneute Verarbeitung, AUSSER es ist die Self-Learning Datei
        if effective_source_name in self.sources_processed and effective_source_name != self.learn_source_name:
             # print(f"INFO: Quelle '{effective_source_name}' wurde bereits verarbeitet. Überspringe.") # Weniger verbose
             return

        print(f"\n📄 Verarbeite Datenquelle: {file_path} (Quelle: {effective_source_name})")
        try:
            chunks = self._load_chunks_from_file(file_path, effective_source_name)
            if not chunks: print(f"WARNUNG: Keine Chunks aus {file_path} geladen."); return
            print(f"   -> {len(chunks)} Chunks erstellt. Beginne Verarbeitung/Aktualisierung...")
            newly_added_chunk_ids = []
            # Wenn es die Lern-Datei ist, müssen wir ggf. alte Chunks dieser Quelle entfernen/ignorieren?
            # Einfacher Ansatz: Immer alle Chunks aus der Datei verarbeiten.
            # self.chunks kann alte Chunks derselben Quelle enthalten.
            # Wir überschreiben sie nicht direkt, sondern fügen neue hinzu, wenn sich UUIDs ändern.
            # Oder: Wir löschen alte Chunks dieser Quelle vor dem Hinzufügen? -> Nein, zu komplex/riskant.
            # Wir fügen neue UUIDs hinzu und aktualisieren den TF-IDF Index.

            chunk_iterator = tqdm(chunks, desc=f"Verarbeitung {effective_source_name}", leave=False) if TQDM_AVAILABLE else chunks
            for chunk in chunk_iterator:
                # Wir prüfen nicht mehr, ob die UUID bereits existiert,
                # da wir bei der Lern-Datei immer die neuesten Daten wollen.
                # Die UUID wird bei jedem Lauf von _load_chunks_from_file neu generiert.
                # Das bedeutet, alte Chunks aus learn.txt bleiben theoretisch im Speicher,
                # aber die neuen werden für TF-IDF etc. verwendet. Besser wäre es,
                # alte Chunks mit derselben Quelle zu identifizieren und zu entfernen.
                # --> VEREINFACHUNG: Wir gehen davon aus, dass process_chunk die Knoten-Assoziationen
                #     für neue Chunks korrekt anlegt und der TF-IDF Index sich aktualisiert.

                 # Fügen wir den Chunk immer hinzu (neue UUID)
                 self.chunks[chunk.uuid] = chunk
                 self.process_chunk(chunk) # Assoziiert Knoten
                 newly_added_chunk_ids.append(chunk.uuid)


            if effective_source_name != self.learn_source_name:
                self.sources_processed.add(effective_source_name)

            print(f"   -> Verarbeitung von {effective_source_name} abgeschlossen ({len(newly_added_chunk_ids)} Chunks verarbeitet/hinzugefügt). Gesamt Chunks: {len(self.chunks)}.")
            # TF-IDF Index muss immer aktualisiert werden, wenn neue Chunks (auch aus learn.txt) dazukommen
            if newly_added_chunk_ids:
                self.update_tfidf_index()

        except Exception as e: print(f"FEHLER Verarbeitung Datei {file_path}: {e}"); traceback.print_exc(limit=2)

    def _load_chunks_from_file(self, path: str, source: str) -> List[TextChunk]:
        chunk_size = self.config.get("chunk_size", 500); overlap = self.config.get("chunk_overlap", 100); chunks = []
        try:
            with open(path, 'r', encoding='utf-8') as f: text = f.read()
        except Exception as e: print(f"FEHLER Lesen Datei {path}: {e}"); return []
        if not text: return []
        start_index = 0; chunk_index = 0
        while start_index < len(text):
            end_index = start_index + chunk_size; chunk_text = text[start_index:end_index]
            # Wichtig: Text erst normalisieren, dann prüfen ob leer!
            normalized_text = re.sub(r'\s+', ' ', chunk_text).strip()
            if normalized_text:
                # Generiere IMMER eine neue UUID für Chunks aus dieser Funktion
                chunk_uuid = str(uuid_module.uuid4())
                chunks.append(TextChunk(text=normalized_text, source=source, index=chunk_index, chunk_uuid=chunk_uuid))
            # Berechne nächsten Startpunkt
            next_start = start_index + chunk_size - overlap
            # Stelle sicher, dass der Index vorwärts geht, um Endlosschleifen bei sehr kurzen Dateien zu vermeiden
            if next_start <= start_index:
                start_index += 1 # Mindestens ein Zeichen weitergehen
            else:
                start_index = next_start
            chunk_index += 1
        return chunks


    # In quantum_arona_hybrid_llm.py

    def process_chunk(self, chunk: TextChunk):
        activated_nodes_in_chunk: List[Node] = []
        semantic_node_definitions = self.config.get("semantic_nodes", {})
        chunk_text_lower = chunk.text.lower()
        chunk.activated_node_labels = [] # Reset

        # --- START DEBUGGING process_chunk ---
        print(f"\n--- Processing Chunk: Index={chunk.index}, Source='{chunk.source}', Len={len(chunk.text)} ---")
        # Optional: Ganzen Chunk-Text ausgeben (kann viel sein!)
        # print(f"Chunk Text:\n'''\n{chunk.text}\n'''")
        print(f"Chunk Text Sample: '{chunk_text_lower[:200]}...'") # Kurzer Sample
        nodes_matched_this_chunk_labels = []
        # --- END DEBUGGING ---

        for node_label, keywords in semantic_node_definitions.items():
            node = self.nodes.get(node_label)
            if not node: continue # Überspringe, wenn Knoten nicht existiert

            matched_keyword = None
            # Prüfe jedes Keyword für den aktuellen Knoten
            for kw in keywords:
                # Verwende Wortgrenzen \b für genauere Treffer
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', chunk_text_lower):
                    matched_keyword = kw
                    break # Erstes passendes Keyword für diesen Knoten reicht

            if matched_keyword:
                 # --- DEBUGGING process_chunk ---
                 print(f"  ✅ MATCH FOUND: Node='{node_label}', Keyword='{matched_keyword}'")
                 nodes_matched_this_chunk_labels.append(node_label)
                 # --- END DEBUGGING ---
                 activated_nodes_in_chunk.append(node)
                 # Füge Label zur Chunk-Info hinzu (nur einmal pro Knoten)
                 if node.label not in chunk.activated_node_labels:
                     chunk.activated_node_labels.append(node.label)

        # --- DEBUGGING process_chunk ---
        print(f"  --> Nodes activated in this specific chunk: {sorted(list(set(nodes_matched_this_chunk_labels)))}") # Zeige eindeutige aktivierte Knoten
        if len(activated_nodes_in_chunk) >= 2:
            # Wichtig: Prüfen, ob MEHRERE UNTERSCHIEDLICHE Knoten aktiviert wurden
            unique_activated_labels = set(n.label for n in activated_nodes_in_chunk)
            if len(unique_activated_labels) >= 2:
                print(f"  ⭐⭐⭐ FOUND CO-OCCURRENCE of {len(unique_activated_labels)} distinct nodes: {sorted(list(unique_activated_labels))} ⭐⭐⭐")
                print(f"      --> Strengthening connections between these nodes.")
                # --- Original Strengthening Logic ---
                learning_signal = self.config.get("connection_strengthening_signal", 0.1); lr = self.config.get("connection_learning_rate", 0.05)
                for i in range(len(activated_nodes_in_chunk)):
                    for j in range(i + 1, len(activated_nodes_in_chunk)):
                        node_a = activated_nodes_in_chunk[i]; node_b = activated_nodes_in_chunk[j]
                        # Stelle sicher, dass es unterschiedliche Knoten sind (sollte durch unique_activated_labels oben abgedeckt sein, aber doppelt prüfen schadet nicht)
                        if node_a.uuid == node_b.uuid: continue
                        # print(f"      Strengthening: {node_a.label} <-> {node_b.label}") # Optional: Noch detaillierter
                        conn_ab = node_a.add_connection(node_b); conn_ba = node_b.add_connection(node_a)
                        if conn_ab: node_a.strengthen_connection(node_b, learning_signal=learning_signal, learning_rate=lr)
                        if conn_ba: node_b.strengthen_connection(node_a, learning_signal=learning_signal, learning_rate=lr)
            else:
                 print(f"  --> Only one distinct node type activated ({unique_activated_labels}). No co-occurrence.")
        else:
            print(f"  --> Less than 2 node activations ({len(activated_nodes_in_chunk)}). No connections strengthened for this chunk.")
        # --- END DEBUGGING ---


    def update_tfidf_index(self):
        if not self.chunks: print("WARNUNG: Keine Chunks für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        print("🔄 Aktualisiere TF-IDF Index...")
        # Wichtig: Verwende IMMER die aktuellen Chunks aus self.chunks
        self.chunk_id_list_for_tfidf = list(self.chunks.keys())
        chunk_texts = [self.chunks[cid].text for cid in self.chunk_id_list_for_tfidf if cid in self.chunks and self.chunks[cid].text] # Nur nicht-leere Texte

        if not chunk_texts: print("WARNUNG: Keine gültigen Texte für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        try:
            max_features = self.config.get("tfidf_max_features", 5000)
            # Initialisiere Vektorizer neu, um veraltete Features zu entfernen
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            # Stelle sicher, dass die ID-Liste die gleiche Länge hat wie die Zeilen der Matrix
            if self.tfidf_matrix.shape[0] != len(self.chunk_id_list_for_tfidf):
                 print(f"WARNUNG: Diskrepanz TF-IDF Matrix Zeilen ({self.tfidf_matrix.shape[0]}) und Chunk-ID-Liste ({len(self.chunk_id_list_for_tfidf)}). Index könnte inkonsistent sein.")
                 # Fallback: Verwende nur IDs, für die Texte transformiert wurden? Schwierig.
                 # Sicherer ist, es neu zu versuchen oder einen Fehler zu werfen.
                 # Fürs Erste nur Warnung.
            print(f"   -> TF-IDF Index aktualisiert. Shape: {self.tfidf_matrix.shape}, Chunk IDs: {len(self.chunk_id_list_for_tfidf)}")
        except ValueError as ve:
             if "empty vocabulary" in str(ve):
                  print("FEHLER TF-IDF Update: Leeres Vokabular. Möglicherweise enthalten alle Chunks nur Stoppwörter oder sind zu kurz.")
             else: print(f"FEHLER TF-IDF Update (ValueError): {ve}")
             self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []
        except Exception as e: print(f"FEHLER TF-IDF Update: {e}"); traceback.print_exc(limit=1); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []


    def simulate_network_step(self, decay_connections: bool = True):
        if not self.nodes: return
        if decay_connections:
            decay_rate = self.config.get("connection_decay_rate", 0.001)
            if decay_rate > 0:
                for node in self.nodes.values():
                    # Prüfe, ob connections ein Dict ist
                    if hasattr(node, 'connections') and isinstance(node.connections, dict):
                        for target_uuid in list(node.connections.keys()):
                            conn = node.connections.get(target_uuid)
                            if conn: conn.decay(decay_rate=decay_rate)
        n_shots = self.config.get("simulation_n_shots", 50)
        for node in self.nodes.values(): node.calculate_activation(n_shots=n_shots)
        next_activation_sums = defaultdict(float)
        for source_node in self.nodes.values():
             if hasattr(source_node, 'activation') and source_node.activation > 0.01:
                 # Verwende geglättete Aktivierung für stabilere Übertragung
                 source_output = source_node.get_smoothed_activation()
                 if hasattr(source_node, 'connections') and isinstance(source_node.connections, dict):
                     for target_uuid, connection in source_node.connections.items():
                          if connection is None: continue
                          # Hole Zielknoten über UUID aus Haupt-Dictionary für Konsistenz
                          target_node = self.nodes.get(target_uuid)
                          if target_node:
                               # Stelle sicher, dass die Verbindung ein Gewicht hat
                               if hasattr(connection, 'weight'):
                                    next_activation_sums[target_node.uuid] += connection.transmit(source_output)
                               else:
                                    # print(f"WARNUNG: Verbindung von {source_node.label} zu {target_uuid} hat kein Gewicht-Attribut.") # Weniger verbose
                                    pass

        for node_uuid, new_sum in next_activation_sums.items():
             if node_uuid in self.nodes: self.nodes[node_uuid].activation_sum = new_sum

    def respond_to_prompt(self, prompt: str) -> List[TextChunk]:
        """Findet relevante Text-Chunks basierend auf dem Prompt und Quanteneffekten."""
        max_results = self.config.get("max_prompt_results", 3); relevance_threshold = self.config.get("relevance_threshold", 0.1)
        variance_penalty_factor = self.config.get("quantum_effect_variance_penalty", 0.5)
        activation_boost_factor = self.config.get("quantum_effect_activation_boost", 0.3)
        variance_penalty_factor = float(variance_penalty_factor) if isinstance(variance_penalty_factor, (int, float)) else 0.5
        activation_boost_factor = float(activation_boost_factor) if isinstance(activation_boost_factor, (int, float)) else 0.3
        prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
        # print(f"\n🔍 [Retriever] Prompt: '{prompt}' (Quantum Effects: VarPenalty={variance_penalty_factor:.2f}, ActBoost={activation_boost_factor:.2f})") # Weniger verbose

        # 1. Finde direkt aktivierte und verwandte Knoten
        directly_activated_nodes: List[Node] = []
        for node_label, keywords in semantic_node_definitions.items():
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords):
                node = self.nodes.get(node_label);
                if node: directly_activated_nodes.append(node)
        # print(f"   -> Direkt aktivierte Knoten: {[n.label for n in directly_activated_nodes]}") # Weniger verbose
        related_nodes: set[Node] = set(directly_activated_nodes)
        if directly_activated_nodes:
             for start_node in directly_activated_nodes:
                 connections_dict = getattr(start_node, 'connections', {})
                 if not isinstance(connections_dict, dict): continue
                 # Berücksichtige nur starke Verbindungen für die Ausbreitung
                 strong_connections = sorted(
                     [conn for conn in connections_dict.values() if conn and hasattr(conn, 'weight') and conn.weight > 0.2], # Schwellwert ggf. anpassen
                     key=lambda c: c.weight, reverse=True
                 )[:5] # Begrenze Ausbreitung
                 for conn in strong_connections:
                      target_node = self.nodes.get(getattr(conn, 'target_node_uuid', None))
                      if target_node and target_node not in related_nodes: related_nodes.add(target_node)
        relevant_node_labels = {node.label for node in related_nodes}
        # print(f"   -> Relevante Knoten (inkl. Ausbreitung): {list(relevant_node_labels)}") # Weniger verbose

        # 2. Finde Kandidaten-Chunks basierend auf relevanten Knoten
        candidate_chunks: List[TextChunk] = []
        if relevant_node_labels:
             candidate_chunks = [
                 chunk for chunk in self.chunks.values()
                 if chunk and hasattr(chunk, 'activated_node_labels') and any(label in chunk.activated_node_labels for label in relevant_node_labels)
             ]
             # print(f"   -> {len(candidate_chunks)} Kandidaten-Chunks (via Knoten).") # Weniger verbose
        else:
             # Wenn keine Knoten relevant sind, nutze TF-IDF auf allen Chunks
             candidate_chunks = list(self.chunks.values())
             # print("   -> Keine relevanten Knoten, nutze alle Chunks für TF-IDF.") # Weniger verbose

        if not candidate_chunks: print("   -> Keine Kandidaten-Chunks gefunden."); return []

        # 3. TF-IDF-basiertes Ranking (wenn Index verfügbar)
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunk_id_list_for_tfidf:
             print("WARNUNG: TF-IDF Index nicht verfügbar. Gebe ungerankte Kandidaten zurück.")
             # Gebe die ersten max_results zurück, optional nach Index oder Quelle sortiert?
             return candidate_chunks[:max_results]

        try:
             prompt_vector = self.vectorizer.transform([prompt])
             # Ordne Kandidaten-Chunks den Zeilen im TF-IDF Index zu
             candidate_indices_in_matrix = []; valid_candidate_chunks_for_tfidf = []
             # Erstelle eine Map von UUID zu Index in der TF-IDF-Liste für schnellen Zugriff
             uuid_to_tfidf_index = {uuid: i for i, uuid in enumerate(self.chunk_id_list_for_tfidf)}

             for c in candidate_chunks:
                  # Prüfe, ob der Chunk eine UUID hat und im Index ist
                  if hasattr(c, 'uuid') and c.uuid in uuid_to_tfidf_index:
                       idx = uuid_to_tfidf_index[c.uuid]
                       candidate_indices_in_matrix.append(idx)
                       valid_candidate_chunks_for_tfidf.append(c)
                  # else: print(f"DEBUG: Chunk {c.uuid} nicht im TF-IDF Index gefunden.") # Nur bei Bedarf

             if not candidate_indices_in_matrix:
                  print("WARNUNG: Keiner der Kandidaten-Chunks ist im aktuellen TF-IDF Index. Gebe ungerankte Kandidaten zurück.")
                  return candidate_chunks[:max_results]

             # Wähle nur die relevanten Zeilen aus der TF-IDF Matrix
             candidate_matrix = self.tfidf_matrix[candidate_indices_in_matrix, :]
             similarities = cosine_similarity(prompt_vector, candidate_matrix).flatten()

             # 4. Wende Quanten-Effekte auf Scores an
             scored_candidates = []
             # print("   -> Applying Quantum Effects to Ranking:") # Weniger verbose
             for i, chunk in enumerate(valid_candidate_chunks_for_tfidf):
                 base_score = similarities[i]
                 quantum_adjustment = 0.0
                 num_quantum_nodes_in_chunk = 0; sum_variance = 0.0; sum_activation = 0.0

                 if hasattr(chunk, 'activated_node_labels'):
                     for node_label in chunk.activated_node_labels:
                         node = self.nodes.get(node_label)
                         # Prüfe, ob der Knoten Quanten-Eigenschaften hat
                         if node and node.is_quantum and node.q_system and hasattr(node, 'last_measurement_log'):
                             num_quantum_nodes_in_chunk += 1
                             node_activation = node.activation # Aktuelle Aktivierung
                             analysis = node.analyze_jumps(node.last_measurement_log)
                             variance = analysis.get("state_variance", 0.0)
                             sum_activation += node_activation
                             sum_variance += variance

                 # Berechne durchschnittliche Effekte für den Chunk
                 avg_activation = (sum_activation / num_quantum_nodes_in_chunk) if num_quantum_nodes_in_chunk > 0 else 0.0
                 avg_variance = (sum_variance / num_quantum_nodes_in_chunk) if num_quantum_nodes_in_chunk > 0 else 0.0

                 # Wende Faktoren an
                 variance_penalty = avg_variance * variance_penalty_factor
                 activation_boost = avg_activation * activation_boost_factor
                 quantum_adjustment = activation_boost - variance_penalty

                 # Debug-Ausgabe nur bei aktivem Effekt
                 # if num_quantum_nodes_in_chunk > 0: print(f"      - Chunk {chunk.index} ({chunk.source}): Base={base_score:.3f}, QAdj={quantum_adjustment:+.3f} (Act:{avg_activation:.2f}, Var:{avg_variance:.2f})") # Weniger verbose

                 final_score = np.clip(base_score + quantum_adjustment, 0.0, 1.0)
                 scored_candidates.append({"chunk": chunk, "score": final_score, "base_score": base_score, "q_adjust": quantum_adjustment})

             # 5. Finales Ranking und Auswahl
             ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
             final_results = []
             # print("   -> Final Ranking (Quantum Adjusted):"); # Weniger verbose
             for item in ranked_candidates:
                 # Wähle Top-Kandidaten über dem Schwellenwert
                 if item["score"] >= relevance_threshold and len(final_results) < max_results:
                     final_results.append(item["chunk"])
                     # print(f"      - Score: {item['score']:.4f} (Base: {item['base_score']:.4f}, QAdj: {item['q_adjust']:+.4f}), Chunk: {item['chunk'].source} ({item['chunk'].index})") # Weniger verbose
                 elif len(final_results) >= max_results: break # Genug Ergebnisse

             # Fallback: Wenn nichts über der Schwelle ist, nimm den besten Treffer (falls vorhanden)
             if not final_results and ranked_candidates:
                 # print(f"   -> Fallback zum besten Treffer (unter Schwelle {relevance_threshold}).") # Weniger verbose
                 best_fallback = ranked_candidates[0]['chunk']; final_results = [best_fallback]
                 # print(f"         - Score: {ranked_candidates[0]['score']:.4f} (Base: {ranked_candidates[0]['base_score']:.4f}, QAdj: {ranked_candidates[0]['q_adjust']:+.4f}), Chunk: {best_fallback.source} ({best_fallback.index})") # Weniger verbose
             elif not final_results:
                 print(f"   -> Keine Chunks über Schwelle {relevance_threshold} gefunden.") # Kein Fallback nötig, wenn keine Kandidaten

             # print(f"   -> {len(final_results)} Chunks für Kontext ausgewählt (Quantum Ranked).") # Weniger verbose
             return final_results

        except Exception as e: print(f"FEHLER TF-IDF/Quantum Ranking: {e}"); traceback.print_exc(limit=1); return candidate_chunks[:max_results] # Fallback


    # --- *** generate_response mit Aufruf des Lernzyklus *** ---
    def generate_response(self, prompt: str) -> str:
        """Generiert Antwort mit Gemini API, Quanteneffekten, Persona-Prompt und optionalem Lernzyklus."""

        # Prüfe globale Abhängigkeiten und RAG-Schalter
        if not GEMINI_AVAILABLE: return "Fehler: Google Generative AI SDK (google-generativeai) nicht installiert."
        if not self.rag_enabled: return "Fehler: RAG (Gemini) ist in der Konfiguration deaktiviert."

        # Prüfe API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        # ... (Streamlit Secret Check bleibt gleich) ...
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except (ImportError, AttributeError): pass
            except Exception as e: print(f"WARNUNG: Fehler beim Lesen von Streamlit Secrets: {e}"); pass
        if not api_key: return "Fehler: Gemini API Key nicht gefunden (Setze GEMINI_API_KEY Umgebungsvariable oder Streamlit Secret)."

        # --- Konfiguriere API und initialisiere Modell ---
        try:
            genai.configure(api_key=api_key)
            config_model_name = self.config.get("generator_model_name", "models/gemini-1.5-flash-latest")
            model_name_to_use = config_model_name if config_model_name.startswith("models/") else "models/gemini-1.5-flash-latest"
            if not hasattr(self, 'gemini_model') or self.gemini_model is None or self.gemini_model.model_name != model_name_to_use:
                print(f"INFO: Initialisiere Gemini Modell '{model_name_to_use}'...")
                # Füge Systemanweisung hinzu, falls unterstützt und gewünscht
                # system_instruction = "Du bist ein hilfreicher Assistent, der von Quanten-NeuroPersona inspiriert wird."
                self.gemini_model = genai.GenerativeModel(
                    model_name_to_use,
                    # system_instruction=system_instruction # Aktuellere Modelle unterstützen das
                    )
                print(f"INFO: Gemini Modell '{model_name_to_use}' initialisiert.")
        # ... (Fehlerbehandlung für API bleibt gleich) ...
        except NameError: return "Fehler: Google Generative AI SDK (genai) nicht verfügbar."
        except GoogleAPIError as api_err: return f"Fehler: Problem bei der Google API Initialisierung ({api_err.reason if hasattr(api_err, 'reason') else 'Unbekannt'}). Prüfen Sie Key/Modellnamen."
        except Exception as e: return f"Fehler bei der Konfiguration der Gemini API: {e}"
        if not hasattr(self, 'gemini_model') or self.gemini_model is None: return "Fehler: Gemini-Modellobjekt konnte nicht initialisiert werden."


        print(f"\n💬 [Generator] RAG für: '{prompt}'")
        # 1. Pre-Retrieval Simulation
        print("   -> Führe Pre-Retrieval Netzwerk-Simulation durch...")
        self.simulate_network_step(decay_connections=False) # Keine Decay während der Abfrage

        # 2. Sprunganalyse
        jump_trigger_active = False; significant_jump_nodes = []
        if self.config.get("quantum_effect_jump_llm_trigger", True):
            prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
            directly_activated_q_nodes: List[Node] = [
                node for node_label, keywords in semantic_node_definitions.items()
                if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords)
                if (node := self.nodes.get(node_label)) and node.is_quantum and hasattr(node, 'last_measurement_log') # Sicherstellen, dass Log existiert
            ]
            if directly_activated_q_nodes:
                for node in directly_activated_q_nodes:
                    analysis = node.analyze_jumps(node.last_measurement_log)
                    if analysis.get("jump_detected", False):
                        jump_trigger_active = True
                        jump_info_str = f"{node.label} (MaxJump: {analysis.get('max_jump_abs', 0)})"
                        if jump_info_str not in significant_jump_nodes: significant_jump_nodes.append(jump_info_str)
                        print(f"      -> Signifikanter Sprung in Knoten '{node.label}' detektiert!")

        # 3. Retrieval (holt die Chunks mit Quanten-Ranking)
        retrieved_chunks = self.respond_to_prompt(prompt)

        # --- 4. Baue den "arona_context" für den Gemini-Prompt ---
        arona_context_parts = []
        # Finde relevante Knoten nochmal (konsistenter)
        relevant_node_labels_for_context = set()
        prompt_lower_ctx = prompt.lower(); semantic_defs_ctx = self.config.get("semantic_nodes", {})
        directly_activated_nodes_for_context: List[Node] = [
            node for node_label, keywords in semantic_defs_ctx.items()
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower_ctx) for kw in keywords)
            if (node := self.nodes.get(node_label))
        ]
        related_nodes_for_context: set[Node] = set(directly_activated_nodes_for_context)
        if directly_activated_nodes_for_context:
             for start_node in directly_activated_nodes_for_context:
                 connections_dict = getattr(start_node, 'connections', {})
                 if not isinstance(connections_dict, dict): continue
                 strong_connections = sorted([conn for conn in connections_dict.values() if conn and getattr(conn, 'weight', 0) > 0.2], key=lambda c: c.weight, reverse=True)[:3] # Top 3 starke Verbindungen
                 for conn in strong_connections:
                      target_node = self.nodes.get(getattr(conn, 'target_node_uuid', None))
                      if target_node and target_node not in related_nodes_for_context: related_nodes_for_context.add(target_node)
        relevant_node_labels_for_context = {node.label for node in related_nodes_for_context}

        if relevant_node_labels_for_context: arona_context_parts.append(f"Identifizierte relevante Kernkonzepte: {', '.join(sorted(list(relevant_node_labels_for_context)))}.")
        if jump_trigger_active: arona_context_parts.append(f"Quantensprung-Hinweis: Möglicher Perspektivwechsel in Bezug auf {', '.join(significant_jump_nodes)}.")

        if not retrieved_chunks:
            arona_context_parts.append("Keine spezifischen Text-Kontexte gefunden.")
            print("   -> Keine Chunks gefunden, Generierung ohne spezifischen Text-Kontext.")
        else:
            context_text = "\n---\n".join([f"[Abschnitt {idx+1} - Quelle: {chunk.source}]:\n{chunk.text}" for idx, chunk in enumerate(retrieved_chunks)])
            arona_context_parts.append(f"Relevanter Kontext ({len(retrieved_chunks)} Abschnitte):")
            arona_context_parts.append(context_text)
            print(f"   -> Kontext aus {len(retrieved_chunks)} Chunks für LLM vorbereitet.")

        arona_context_string = "\n".join(arona_context_parts)
        # --- Ende Kontext-Aufbau ---

        # 5. Baue den finalen Prompt für Gemini
        input_prompt_for_gemini = f"""
Du bist ein KI-Assistent, inspiriert von einem Quanten-NeuroPersona-Modell.
Der Benutzer fragt: "{prompt}"

NeuroPersonas Analyse dazu liefert folgenden Kontext:
{arona_context_string}

Deine Aufgabe:
1. Beantworte die *Frage des Benutzers* präzise.
2. Nutze NeuroPersonas Kontext (Konzepte, Sprünge, Textpassagen), um deine Antwort *inhaltlich zu formen* oder die *Perspektive anzupassen*.
3. Erkläre NeuroPersonas internen Prozess NICHT. Nutze ihn implizit.
4. Antworte natürlich und kohärent.

Antwort auf die Benutzerfrage unter Berücksichtigung des Kontexts:
"""
        # print(f"   -> Finaler Prompt für Gemini (gekürzt):\n{input_prompt_for_gemini[:600]}...\n") # Optional

        # 6. Generation mit Gemini API
        generated_text = "" # Initialisieren
        try:
            start_gen_time = time.time()
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.get("generator_temperature", 0.7),
                 max_output_tokens=self.config.get("generator_max_length", 8192) # Nutzen des Config-Werts
            )
            safety_settings=[ # Standard-Sicherheitseinstellungen
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = self.gemini_model.generate_content(
                input_prompt_for_gemini,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Verbesserte Fehlerbehandlung für Blockierung
            if not response.candidates:
                 block_reason = "Unbekannt"; safety_ratings_str = "N/A"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                         block_reason = response.prompt_feedback.block_reason.name
                     if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings:
                         safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in response.prompt_feedback.safety_ratings])
                 print(f"WARNUNG: Gemini-Antwort blockiert. Grund: {block_reason}. Ratings: [{safety_ratings_str}]")
                 generated_text = f"[Antwort blockiert durch Sicherheitsfilter: {block_reason}]" # Informative Fehlermeldung zurückgeben
            else:
                 # Sicherer Zugriff auf den Text
                 if hasattr(response, 'text'):
                      generated_text = response.text.strip()
                 else:
                      # Fallback, wenn .text nicht existiert, aber parts schon
                      try:
                           generated_text = "".join(part.text for part in response.parts).strip()
                      except Exception:
                           generated_text = "[Fehler beim Extrahieren der Antwort aus Gemini-Response-Objekt]"
                 end_gen_time = time.time()
                 print(f"   -> Generierung mit Gemini in {end_gen_time - start_gen_time:.2f}s.")
                 print(f"   -> Generierte Antwort (gekürzt): {generated_text[:200]}...") # Kürzen für Log

            # --- *** NEU: Self-Learning Schritt *** ---
            # Nur wenn Self-Learning aktiviert ist UND eine gültige Antwort generiert wurde
            # (keine Fehlermeldung, nicht leer, nicht blockiert)
            is_valid_response = (generated_text and
                                not generated_text.startswith("[Fehler") and
                                not generated_text.startswith("[Antwort blockiert"))

            if self.self_learning_enabled and is_valid_response:
                 print(f"\n🎓 [Self-Learning] Starte Lernzyklus für generierte Antwort...")
                 self._save_and_reprocess_response(generated_text)
            # --- *** Ende Self-Learning Schritt *** ---

            return generated_text # Gib die generierte (oder Fehler-)Antwort zurück

        # ... (Fehlerbehandlung für API-Kommunikation bleibt gleich) ...
        except GoogleAPIError as api_err:
            print(f"FEHLER bei der Gemini API Anfrage: {api_err}")
            return f"Fehler: Problem bei der Kommunikation mit der Gemini API ({api_err.reason if hasattr(api_err, 'reason') else 'Unbekannt'}). Prüfen Sie Key/Modellnamen."
        except Exception as e:
            print(f"FEHLER während der Textgenerierung mit Gemini: {e}")
            traceback.print_exc(limit=2)
            return "Fehler: Entschuldigung, ich konnte keine Antwort generieren."
    # --- *** ENDE generate_response *** ---

    # --- *** NEUE METHODE: _save_and_reprocess_response *** ---
    def _save_and_reprocess_response(self, response_text: str):
        """Speichert die generierte Antwort und verarbeitet die Lerndatei neu."""
        if not response_text:
            print("   -> [Self-Learning] Übersprungen: Leere Antwort.")
            return

        learn_file = self.learn_file_path
        learn_source = self.learn_source_name

        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(learn_file), exist_ok=True)

            # Hänge die Antwort an die Lerndatei an
            with open(learn_file, 'a', encoding='utf-8') as f:
                f.write("\n\n---\n\n") # Trennlinie für Lesbarkeit
                f.write(response_text)
            print(f"   -> [Self-Learning] Antwort erfolgreich an '{learn_file}' angehängt.")

            # Verarbeite die gesamte Lerndatei neu
            print(f"   -> [Self-Learning] Verarbeite '{learn_file}' neu...")
            # Wichtig: source_name übergeben, damit load_and_process_file sie korrekt identifiziert
            self.load_and_process_file(learn_file, source_name=learn_source)
            print(f"   -> [Self-Learning] Neuverarbeitung abgeschlossen.")

        except Exception as e:
            print(f"FEHLER im Self-Learning Zyklus: {e}")
            traceback.print_exc(limit=1)
    # --- *** ENDE _save_and_reprocess_response *** ---



    def get_network_state_summary(self) -> Dict[str, Any]:
         # Grundlegende Metriken (Initialisierung bleibt gleich)
         summary = {
             "num_nodes": len(self.nodes),
             "num_quantum_nodes": sum(1 for n in self.nodes.values() if n.is_quantum),
             "num_chunks": len(self.chunks),
             "sources_processed": list(self.sources_processed),
             "self_learning_enabled": getattr(self, 'self_learning_enabled', False), # Sicherer Zugriff
             "tfidf_index_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
             "rag_enabled": getattr(self, 'rag_enabled', False), # Sicherer Zugriff
             "generator_model": self.config.get("generator_model_name") if getattr(self, 'rag_enabled', False) else None
             }

         # Durchschnittliche Aktivierung berechnen
         activations = [n.activation for n in self.nodes.values() if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)]
         summary["average_node_activation"] = round(np.mean(activations), 4) if activations else 0.0

         # Verbindungen zählen und Top-Verbindungen finden
         total_valid_connections = 0
         all_connections_found = []
         nodes_with_connections_count = 0

         # print("\n--- DEBUG: Entering get_network_state_summary Loop ---") # Kann entfernt werden

         for source_node in self.nodes.values():
             connections_dict = getattr(source_node, 'connections', None)
             if not isinstance(connections_dict, dict):
                 continue

             if connections_dict:
                 nodes_with_connections_count += 1

             # Iteriere durch die Verbindungen dieses Knotens
             for target_uuid_from_conn_attr, conn in connections_dict.items():
                 # print(f"  DEBUG SUMMARY INNER LOOP: Processing Conn from {source_node.label} -> UUID {target_uuid_from_conn_attr}") # Kann entfernt werden
                 if conn is None:
                     continue

                 # --- KORRIGIERTER ZIELKNOTEN-LOOKUP ---
                 target_node_obj = None
                 # Hole die UUID aus dem Connection-Objekt selbst (sicherer)
                 target_uuid_from_conn = getattr(conn, 'target_node_uuid', None)
                 if target_uuid_from_conn:
                     # Iteriere durch ALLE Knoten im Haupt-Dictionary und vergleiche die UUIDs
                     for node_in_processor in self.nodes.values():
                         if node_in_processor.uuid == target_uuid_from_conn:
                             target_node_obj = node_in_processor
                             break # Gefunden, Schleife verlassen
                 # --- ENDE KORRIGIERTER ZIELKNOTEN-LOOKUP ---

                 weight = getattr(conn, 'weight', None)

                 # Jetzt sollte die Bedingung korrekt prüfen
                 cond_target_exists = target_node_obj is not None
                 cond_target_has_label = hasattr(target_node_obj, 'label') if cond_target_exists else False
                 cond_weight_exists = weight is not None
                 cond_weight_is_number = isinstance(weight, (float, np.number))
                 cond_weight_is_finite = np.isfinite(weight) if cond_weight_is_number else False

                 if (cond_target_exists and cond_target_has_label and
                     cond_weight_exists and cond_weight_is_number and cond_weight_is_finite):
                     # --- Bedingung ERFÜLLT ---
                     # print(f"    --> SUCCESS: Condition met for Conn {source_node.label} -> {target_node_obj.label}. Adding.") # Kann entfernt werden
                     total_valid_connections += 1
                     all_connections_found.append({
                         "source": source_node.label,
                         "target": target_node_obj.label,
                         "weight": weight
                         })

         # Zuweisungen zur Summary
         summary["total_connections"] = total_valid_connections
         all_connections_found.sort(key=lambda x: x["weight"], reverse=True)
         summary["top_connections"] = all_connections_found[:10]
         summary["_debug_nodes_with_connections"] = nodes_with_connections_count # Behalte diese Info vorerst
         return summary
    def save_state(self, filepath: str) -> None:
        """
        Speichert den aktuellen Zustand des QuantumEnhancedTextProcessor in einer JSON-Datei.
        Bereinigt dabei zuerst alle ungültigen Verbindungen, serialisiert dann Knoten, Chunks
        und Metadaten und schreibt alles in die angegebene Datei.
        """
        print(f"💾 Speichere Zustand nach {filepath}...")
        try:
            # 1. Bereinige ungültige Verbindungen vor dem Speichern:
            #    Wir sammeln alle zurzeit existierenden Node-UUIDs in einer Menge.
            existing_uuids = {node.uuid for node in self.nodes.values()}

            #    Iteriere über alle Knoten und filtere deren Verbindungen.
            for node_label, node in self.nodes.items():
                # Struktur-Pattern-Matching (PEP 634) auf das connections-Attribut:
                match getattr(node, 'connections', None):
                    case dict() as connections:
                        # Verwende Union-Typ-Operator (PEP 604) für die Typannotation:
                        valid_connections: dict[str, Connection] = {}
                        for target_uuid, conn in connections.items():
                            # Behalte nur Verbindungen, bei denen
                            # 1) ein Connection-Objekt existiert und
                            # 2) die Ziel-UUID in existing_uuids enthalten ist.
                            if conn is not None and target_uuid in existing_uuids:
                                valid_connections[target_uuid] = conn
                        # Ersetze das alte connections-Dict durch das gefilterte.
                        node.connections = valid_connections
                    case _:
                        # Kein gültiges connections-Dict vorhanden → überspringen
                        continue

            # 2. Serialisiere alle Chunks in ein JSON-kompatibles Format
            chunks_to_save: dict[str, dict[str, Any]] = {}
            for c_uuid, c_obj in self.chunks.items():
                if hasattr(c_obj, 'text') and hasattr(c_obj, 'source') and hasattr(c_obj, 'index'):
                    chunks_to_save[c_uuid] = {
                        "uuid": c_obj.uuid,
                        "text": c_obj.text,
                        "source": c_obj.source,
                        "index": c_obj.index,
                        "activated_node_labels": getattr(c_obj, 'activated_node_labels', [])
                    }

            # 3. Erzeuge die Debug-Ausgaben und hole __getstate__ von jedem Node
            print("\n--- DEBUG save_state: Checking node states before adding to state_data ---")
            nodes_data_for_json: dict[str, dict[str, Any]] = {}
            for label, node in self.nodes.items():
                print(f"  Getting state for node: '{label}' (calling __getstate__ now)")
                node_state = node.__getstate__()
                print(f"    State dict returned by __getstate__ for '{label}'. Checking 'connections_serializable'...")
                if isinstance(node_state, dict) and 'connections_serializable' in node_state:
                    conn_serial_len = len(node_state['connections_serializable'])
                    print(f"      'connections_serializable' FOUND in returned state. Length: {conn_serial_len}")
                    # Spezieller Alarm für den Knoten "Philosophie"
                    if label == "Philosophie" and conn_serial_len == 0 and len(getattr(node, 'connections', {})) > 0:
                        print(f"      🚨🚨🚨 ALARM: '{label}' hat live-Verbindungen ({len(node.connections)}) "
                              f"aber ZERO serialized connections!")
                else:
                    print(f"      WARNUNG: 'connections_serializable' fehlt oder state ist kein dict für '{label}'!")
                nodes_data_for_json[label] = node_state
            print("--- END DEBUG save_state check ---")

            # 4. Baue das komplette state_data-Dictionary zusammen
            state_data: dict[str, Any] = {
                "config": self.config,
                "nodes": nodes_data_for_json,
                "chunks": chunks_to_save,
                "sources_processed": list(self.sources_processed),
                "chunk_id_list_for_tfidf": self.chunk_id_list_for_tfidf
            }

            # 5. Letzter Debug-Check vor dem Schreiben
            print("\n--- DEBUG save_state: Checking final state_data before json.dump ---")
            if "Philosophie" in state_data["nodes"]:
                philosophie_state_final = state_data["nodes"]["Philosophie"]
                if isinstance(philosophie_state_final, dict) and 'connections_serializable' in philosophie_state_final:
                    final_len = len(philosophie_state_final['connections_serializable'])
                    print(f"  Final check of 'Philosophie' -> 'connections_serializable' length: {final_len}")
                    if final_len == 0 and len(self.nodes['Philosophie'].connections) > 0:
                        print(f"  🚨🚨🚨 ALARM: Philosophie state_data hat ZERO serialized connections, "
                              f"live object hat {len(self.nodes['Philosophie'].connections)}!")
                else:
                    print("  WARNUNG: 'connections_serializable' fehlt in finalem 'Philosophie' state_data.")
            else:
                print("  WARNUNG: 'Philosophie' node data fehlt in final state_data.")
            print("--- END DEBUG save_state final check ---\n")

            # 6. Schreibe den JSON-Dump mit einem Default-Serializer für spezielle Typen
            with open(filepath, 'w', encoding='utf-8') as f:
                def default_serializer(obj: Any) -> Any:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (datetime, deque)):
                        return str(obj)
                    if isinstance(obj, set):
                        return list(obj)
                    if isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    if hasattr(obj, '__getstate__'):
                        try:
                            return obj.__getstate__()
                        except Exception:
                            pass
                    try:
                        return repr(obj)
                    except Exception:
                        return f"<SerializationError: {type(obj)}>"
                json.dump(state_data, f, indent=2, ensure_ascii=False, default=default_serializer)

            print("   -> Zustand erfolgreich gespeichert.")
        except Exception as e:
            print(f"FEHLER Speichern Zustand: {e}")
            traceback.print_exc(limit=2)


    @classmethod
    def load_state(cls, filepath: str) -> Optional['QuantumEnhancedTextProcessor']:
        print(f"📂 Lade Zustand von {filepath}...")
        if not os.path.exists(filepath): print(f"FEHLER: Zustandsdatei {filepath} nicht gefunden."); return None
        instance = None
        try:
            with open(filepath, 'r', encoding='utf-8') as f: state_data = json.load(f)

            # Erstelle Instanz mit der gespeicherten Konfiguration
            instance = cls(config_dict=state_data.get("config"))
            if not instance: print("FEHLER: Instanzerstellung fehlgeschlagen."); return None
            print(f"INFO (load_state): RAG im geladenen Prozessor {'AKTIVIERT' if instance.rag_enabled else 'DEAKTIVIERT'}.")
            print(f"INFO (load_state): Self-Learning im geladenen Prozessor {'AKTIVIERT' if instance.self_learning_enabled else 'DEAKTIVIERT'}.")

            # Lade Chunks (bleibt gleich)
            loaded_chunks = {}; raw_chunk_data = state_data.get("chunks", {})
            for uuid_key, chunk_data_dict in raw_chunk_data.items():
                if not isinstance(chunk_data_dict, dict): continue
                loaded_chunk_uuid = chunk_data_dict.get('uuid', uuid_key)
                text = chunk_data_dict.get('text', '')
                source = chunk_data_dict.get('source', 'Unknown')
                index = chunk_data_dict.get('index', -1)
                if not text: continue
                try:
                    new_chunk = TextChunk(chunk_uuid=loaded_chunk_uuid, text=text, source=source, index=index)
                    new_chunk.activated_node_labels = chunk_data_dict.get('activated_node_labels', [])
                    loaded_chunks[new_chunk.uuid] = new_chunk
                except Exception as e: print(f"FEHLER Erstellen Chunk UUID {loaded_chunk_uuid}: {e}")
            instance.chunks = loaded_chunks
            print(f"INFO (load_state): {len(instance.chunks)} Chunks geladen.")


            # Lade Knoten
            print("\n--- Lade/Aktualisiere Knoten (load_state) ---")
            loaded_node_states = state_data.get("nodes", {})
            final_nodes_dict = {}; node_uuid_map = {} # Map von UUID zu Knotenobjekt
            print(f"INFO (load_state): Gefundene Knotenzustände im State: {len(loaded_node_states)}")
            for node_label, node_state_dict in loaded_node_states.items():
                 if not isinstance(node_state_dict, dict): continue
                 original_uuid = node_state_dict.get("uuid")
                 if not original_uuid: continue
                 node_type_name = node_state_dict.get('type', 'Node'); node_class = globals().get(node_type_name, Node)
                 try:
                     node = node_class.__new__(node_class)
                     node.__setstate__(node_state_dict) # Ruft __setstate__ auf
                     if node.uuid != original_uuid: node.uuid = original_uuid
                     final_nodes_dict[node.label] = node
                     node_uuid_map[node.uuid] = node # Fülle die UUID-Map
                     # print(f"  -> Knoten '{node.label}' (UUID: {node.uuid}) wiederhergestellt.") # Weniger verbose
                 except Exception as e: print(f"FEHLER Restore Knoten '{node_label}': {e}"); traceback.print_exc(limit=1)

            # Füge Knoten hinzu, die in Config, aber nicht im State sind
            for label_in_config in instance.config.get("semantic_nodes", {}).keys():
                 if label_in_config not in final_nodes_dict:
                      print(f"  -> Knoten '{label_in_config}' neu erstellen (nicht im State gefunden)...")
                      new_node = instance._get_or_create_node(label_in_config)
                      if new_node:
                          final_nodes_dict[new_node.label] = new_node
                          node_uuid_map[new_node.uuid] = new_node
                      else: print(f"     FEHLER Neuerstellen von '{label_in_config}'.")
            instance.nodes = final_nodes_dict
            print(f"INFO (load_state): {len(instance.nodes)} Knoten im Prozessor nach Laden/Erstellen.")
            print(f"INFO (load_state): Node UUID Map erstellt mit {len(node_uuid_map)} Einträgen.")


            # --- START DEBUGGING: Verbindungen wiederherstellen ---
            print(f"\n--- DEBUG: Verbindungen wiederherstellen (load_state) ---")
            total_connections_to_restore = 0
            total_connections_restored_successfully = 0
            nodes_processed_for_connections = 0

            for node_label, node in instance.nodes.items(): # Iteriere über die gerade geladenen Knoten
                nodes_processed_for_connections += 1
                # Prüfe, ob das temporäre Dict aus __setstate__ existiert
                if hasattr(node, 'connections_serializable_temp') and isinstance(node.connections_serializable_temp, dict):
                     print(f"  Processing connections FOR node '{node_label}' (UUID: {node.uuid}). Found {len(node.connections_serializable_temp)} serialized connections.")
                     # Stelle sicher, dass das connections-Attribut ein leeres Dict ist (sollte durch __setstate__ passiert sein)
                     if not hasattr(node, 'connections') or not isinstance(node.connections, dict):
                         node.connections = {}
                         print(f"    WARNUNG: node.connections war kein Dict, wurde neu initialisiert für '{node_label}'.")

                     num_restored_for_this_node = 0
                     for target_uuid, conn_dict in node.connections_serializable_temp.items():
                          total_connections_to_restore += 1
                          print(f"    Attempting to restore connection: {node_label} -> Target UUID {target_uuid}")
                          if not isinstance(conn_dict, dict):
                              print(f"      ERROR: Serialized connection data for target {target_uuid} is not a dict. Skipping.")
                              continue

                          # Finde das Ziel-Node-Objekt über die UUID-Map
                          target_node = node_uuid_map.get(target_uuid)

                          if target_node: # Nur wenn Zielknoten existiert
                               print(f"      Target Node '{target_node.label}' found in map.")
                               try:
                                   # Erstelle leeres Connection-Objekt und setze Zustand
                                   conn = Connection.__new__(Connection)
                                   conn_attrs = {k:v for k,v in conn_dict.items() if k != 'target_node'}
                                   created_at_str = conn_attrs.get('created_at'); last_update_at_str = conn_attrs.get('last_update_at')
                                   try: conn_attrs['created_at'] = datetime.fromisoformat(created_at_str) if isinstance(created_at_str, str) else datetime.now()
                                   except: conn_attrs['created_at'] = datetime.now()
                                   try: conn_attrs['last_update_at'] = datetime.fromisoformat(last_update_at_str) if isinstance(last_update_at_str, str) else datetime.now()
                                   except: conn_attrs['last_update_at'] = datetime.now()
                                   conn_attrs.setdefault('target_node_uuid', target_uuid)
                                   conn_attrs.setdefault('source_node_label', node.label)
                                   conn.__dict__.update(conn_attrs)
                                   conn.target_node = target_node # Wichtig: Live-Objekt setzen

                                   # === Der entscheidende Schritt ===
                                   node.connections[target_uuid] = conn
                                   # ================================

                                   # Prüfe direkt danach
                                   if target_uuid in node.connections and node.connections[target_uuid] is not None:
                                       print(f"      ✅ SUCCESS: Connection {node_label} -> {target_node.label} added to node.connections.")
                                       total_connections_restored_successfully += 1
                                       num_restored_for_this_node += 1
                                   else:
                                       print(f"      ❌ FAILURE?: Connection {node_label} -> {target_node.label} NOT found in node.connections after assignment!")

                               except Exception as conn_e:
                                   print(f"      ❌ EXCEPTION during connection restore for {node.label} -> {target_uuid}: {conn_e}")
                                   traceback.print_exc(limit=1)
                          else:
                               print(f"      ❌ FAILURE: Target Node with UUID '{target_uuid}' not found in node_uuid_map. Skipping connection.")
                     print(f"    Finished restoring for node '{node_label}'. Successfully restored: {num_restored_for_this_node}. Current total length of node.connections: {len(node.connections)}")
                     # Lösche das temporäre Dict, nachdem es verarbeitet wurde
                     del node.connections_serializable_temp
                else:
                     print(f"  Node '{node_label}' has no 'connections_serializable_temp' attribute or it's not a dict.")

            print(f"\n--- DEBUG: Connection Restore Summary ---")
            print(f"  Nodes processed for connections: {nodes_processed_for_connections}")
            print(f"  Total serialized connections found: {total_connections_to_restore}")
            print(f"  Total connections restored successfully: {total_connections_restored_successfully}")
            print(f"--- END DEBUG: Connection Restore ---")
            # --- ENDE DEBUGGING ---


            # Lade restliche Metadaten (bleibt gleich)
            instance.sources_processed = set(state_data.get("sources_processed", [])); instance.chunk_id_list_for_tfidf = state_data.get("chunk_id_list_for_tfidf", [])

            # Aktualisiere TF-IDF Index basierend auf geladenen Chunks und IDs
            instance.update_tfidf_index()

            print(f"\n✅ Zustand geladen ({len(instance.nodes)} Knoten, {len(instance.chunks)} Chunks, {total_connections_restored_successfully} Verbindungen explizit wiederhergestellt).")
            return instance

        except json.JSONDecodeError as json_err: print(f"FEHLER: JSON Decode Error in {filepath}: {json_err}"); return None
        except Exception as e: print(f"FEHLER Laden Zustand {filepath}: {e}"); traceback.print_exc(limit=2); return None


# --- Beispielnutzung __main__ ---
if __name__ == "__main__":
    print("="*50 + "\n Starte Quantum-Enhanced Text Processor Demo \n" + "="*50)
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
             # Speichere den initialen Zustand direkt
             processor.save_state(STATE_FILE)
    else: print(f"\nZustand aus '{STATE_FILE}' geladen.")

    # ---- Interaktive Schleife ----
    print("\n--- Aktueller Netzwerkstatus ---")
    summary = processor.get_network_state_summary(); print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n--- Interaktive Abfrage (Typ 'exit' zum Beenden) ---")

    # API Key Check
    if processor.rag_enabled and not os.environ.get("GEMINI_API_KEY"):
         print("\nWARNUNG: RAG ist aktiviert, aber die Umgebungsvariable GEMINI_API_KEY ist nicht gesetzt.")

    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.lower() == 'exit': break
            if not prompt: continue

            # Generiere Antwort (inklusive möglichem Self-Learning)
            generated_response = processor.generate_response(prompt)
            print("\n--- Generierte Antwort ---"); print(generated_response); print("-" * 25)

            # === WICHTIG: Zustand nach Self-Learning speichern ===
            # Das Speichern erfolgt nun nach JEDER Generierung, wenn Self-Learning aktiv ist,
            # um die neuen Daten aus learn.txt persistent zu machen.
            if processor.self_learning_enabled and generated_response and not generated_response.startswith("[Fehler") and not generated_response.startswith("[Antwort blockiert"):
                print("\n--- Speichere Zustand nach Lernzyklus ---")
                processor.save_state(STATE_FILE)
            # ====================================================

        except KeyboardInterrupt: print("\nUnterbrochen."); break
        except Exception as e: print(f"\nFehler in der Hauptschleife: {e}"); traceback.print_exc(limit=1)

    # ---- Ende interaktive Schleife ----

    # Speichere den finalen Zustand (wird nur erreicht, wenn Self-Learning aus ist oder Schleife normal beendet wird)
    print("\n--- Speichere finalen Zustand ---"); processor.save_state(STATE_FILE)
    print("\n" + "="*50 + "\n Demo beendet. \n" + "="*50)