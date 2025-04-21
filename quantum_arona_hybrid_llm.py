# -- coding: utf-8 --

# Filename: quantum_arona_hybrid_llm.py
# Version: 0.9 - Gemini API Integration für RAG
# Author: [CipherCore Technology] & Gemini & Your Input

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
# ... (Code für _ry, _rz, _apply_gate, _apply_cnot - unverändert) ...
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
        target_label = getattr(self.target_node, 'label', self.target_node_uuid[:4]+'...'); source_info = f" from:{self.source_node_label}" if self.source_node_label else ""
        return f"<Conn to:{target_label} W:{self.weight:.3f} Cnt:{self.transmission_count}{source_info}>"

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

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None, conn_type: str = "associative") -> Optional[Connection]:
        if target_node is None or not hasattr(target_node, 'uuid') or target_node.uuid == self.uuid: return None
        target_uuid = target_node.uuid
        if target_uuid not in self.connections:
            conn = Connection(target_node=target_node, weight=weight, source_node_label=self.label, conn_type=conn_type)
            self.connections[target_uuid] = conn
            if hasattr(target_node, 'add_incoming_connection_info'): target_node.add_incoming_connection_info(self.uuid, self.label)
            return conn
        else: return self.connections.get(target_uuid)

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
        state = self.__dict__.copy()
        if 'q_system' in state and state['q_system'] is not None: state['q_system_params'] = state['q_system'].get_params(); del state['q_system']
        else: state['q_system_params'] = None
        if 'last_state_vector' in state: del state['last_state_vector']
        connections_serializable = {}
        if 'connections' in state and isinstance(state['connections'], dict):
            for target_uuid, conn in state['connections'].items():
                if conn is None: continue
                conn_dict = conn.__dict__.copy()
                if 'target_node' in conn_dict: del conn_dict['target_node']
                conn_dict['target_node_uuid'] = getattr(conn, 'target_node_uuid', target_uuid)
                connections_serializable[target_uuid] = conn_dict
            del state['connections']
        state['connections_serializable'] = connections_serializable
        if 'activation_history' in state: state['activation_history'] = list(state['activation_history'])
        return state

    def __setstate__(self, state):
        state['activation_history'] = deque(state.get('activation_history', []), maxlen=self.DEFAULT_ACTIVATION_HISTORY_LEN)
        q_params_list = state.pop('q_system_params', None); num_qbits = state.get('num_qubits', self.DEFAULT_NUM_QUBITS)
        is_q = state.get('is_quantum', True); self.q_system = None; q_params_np = None
        if q_params_list is not None and isinstance(q_params_list, list):
             try:
                  q_params_np = np.array(q_params_list, dtype=float)
                  expected_shape = (num_qbits * 2,)
                  if num_qbits > 0 and q_params_np.shape != expected_shape: print(f"WARNUNG: QNS Param Shape mismatch for '{state.get('label', '?')}'"); q_params_np = None
                  elif num_qbits == 0 and q_params_np.size != 0: print(f"WARNUNG: QNS Param nicht leer für 0 Qubits bei '{state.get('label', '?')}'"); q_params_np = None
             except Exception as e: print(f"FEHLER Konvertierung QNS Params für '{state.get('label', '?')}': {e}"); q_params_np = None
        if is_q and num_qbits > 0:
             try: self.q_system = QuantumNodeSystem(num_qubits=num_qbits, initial_params=q_params_np)
             except Exception as e: print(f"FEHLER Restore QNS für '{state.get('label', '?')}': {e}"); state['is_quantum'] = False; self.num_qubits = 0
        self.connections_serializable_temp = state.pop('connections_serializable', {})
        self.connections: Dict[str, Optional[Connection]] = {} # Standard Dictionary
        self.__dict__.update(state)
        if not hasattr(self, 'uuid') or not self.uuid: self.uuid = str(uuid_module.uuid4())

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

        if self.rag_enabled:
            # Gemini Initialisierung wird in generate_response gemacht, braucht API Key
            print(f"INFO: RAG aktiviert. Gemini Modell '{self.config.get('generator_model_name', 'gemini-1.5-flash-latest')}' wird bei Bedarf initialisiert.")
        else: print(f"INFO: RAG {'deaktiviert (Config)' if not self.config.get('enable_rag') else 'deaktiviert (google-generativeai fehlt)'}.")
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
        if not os.path.exists(file_path): print(f"FEHLER: Datei nicht gefunden: {file_path}"); return
        effective_source_name = source_name if source_name else os.path.basename(file_path)
        if effective_source_name in self.sources_processed: return
        print(f"\n📄 Verarbeite Datei: {file_path} (Quelle: {effective_source_name})")
        try:
            chunks = self._load_chunks_from_file(file_path, effective_source_name)
            if not chunks: print(f"WARNUNG: Keine Chunks aus {file_path} geladen."); return
            print(f"   -> {len(chunks)} Chunks erstellt. Beginne initiale Verarbeitung...")
            newly_added_chunk_ids = []
            chunk_iterator = tqdm(chunks, desc=f"Initiale Verarbeitung {effective_source_name}", leave=False) if TQDM_AVAILABLE else chunks
            for chunk in chunk_iterator:
                if chunk.uuid not in self.chunks:
                     self.chunks[chunk.uuid] = chunk; self.process_chunk(chunk); newly_added_chunk_ids.append(chunk.uuid)
            self.sources_processed.add(effective_source_name)
            print(f"   -> Initiale Verarbeitung von {effective_source_name} abgeschlossen ({len(newly_added_chunk_ids)} neu). Gesamt Chunks: {len(self.chunks)}.")
            if newly_added_chunk_ids: self.update_tfidf_index()
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
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            if chunk_text: chunks.append(TextChunk(text=chunk_text, source=source, index=chunk_index))
            next_start = start_index + chunk_size - overlap; start_index = next_start if next_start > start_index else start_index + 1; chunk_index += 1
        return chunks

    def process_chunk(self, chunk: TextChunk):
        activated_nodes_in_chunk: List[Node] = []; semantic_node_definitions = self.config.get("semantic_nodes", {})
        chunk_text_lower = chunk.text.lower()
        for node_label, keywords in semantic_node_definitions.items():
            if node_label not in self.nodes: continue
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', chunk_text_lower) for kw in keywords):
                node = self.nodes[node_label]; activated_nodes_in_chunk.append(node)
                if node.label not in chunk.activated_node_labels: chunk.activated_node_labels.append(node.label)
        if len(activated_nodes_in_chunk) >= 2:
            learning_signal = self.config.get("connection_strengthening_signal", 0.1); lr = self.config.get("connection_learning_rate", 0.05)
            for i in range(len(activated_nodes_in_chunk)):
                for j in range(i + 1, len(activated_nodes_in_chunk)):
                    node_a = activated_nodes_in_chunk[i]; node_b = activated_nodes_in_chunk[j]
                    conn_ab = node_a.add_connection(node_b); conn_ba = node_b.add_connection(node_a)
                    if conn_ab: node_a.strengthen_connection(node_b, learning_signal=learning_signal, learning_rate=lr)
                    if conn_ba: node_b.strengthen_connection(node_a, learning_signal=learning_signal, learning_rate=lr)

    def update_tfidf_index(self):
        if not self.chunks: print("WARNUNG: Keine Chunks für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        print("🔄 Aktualisiere TF-IDF Index...")
        current_chunk_ids = list(self.chunks.keys()); chunk_texts = [self.chunks[cid].text for cid in current_chunk_ids if cid in self.chunks]
        if not chunk_texts: print("WARNUNG: Keine Texte für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        try:
            max_features = self.config.get("tfidf_max_features", 5000)
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            self.chunk_id_list_for_tfidf = current_chunk_ids[:]
            print(f"   -> TF-IDF Index aktualisiert. Shape: {self.tfidf_matrix.shape}")
        except Exception as e: print(f"FEHLER TF-IDF Update: {e}"); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []

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
                 source_output = source_node.get_smoothed_activation()
                 if hasattr(source_node, 'connections') and isinstance(source_node.connections, dict):
                     for target_uuid, connection in source_node.connections.items():
                          if connection is None: continue
                          target_node = self.nodes.get(target_uuid)
                          if target_node: next_activation_sums[target_node.uuid] += connection.transmit(source_output)
        for node_uuid, new_sum in next_activation_sums.items():
             if node_uuid in self.nodes: self.nodes[node_uuid].activation_sum = new_sum

    def respond_to_prompt(self, prompt: str) -> List[TextChunk]:
        max_results = self.config.get("max_prompt_results", 3); relevance_threshold = self.config.get("relevance_threshold", 0.1)
        variance_penalty_factor = self.config.get("quantum_effect_variance_penalty", 0.5)
        activation_boost_factor = self.config.get("quantum_effect_activation_boost", 0.3)
        variance_penalty_factor = float(variance_penalty_factor) if isinstance(variance_penalty_factor, (int, float)) else 0.5
        activation_boost_factor = float(activation_boost_factor) if isinstance(activation_boost_factor, (int, float)) else 0.3
        prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
        # print(f"\n🔍 [Retriever] Prompt: '{prompt}' (Quantum Effects: VarPenalty={variance_penalty_factor:.2f}, ActBoost={activation_boost_factor:.2f})") # Weniger verbose
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
                 strong_connections = sorted([conn for conn in connections_dict.values() if conn and conn.weight > 0.1], key=lambda c: c.weight, reverse=True)[:5]
                 for conn in strong_connections:
                      target_node = self.nodes.get(getattr(conn, 'target_node_uuid', None))
                      if target_node and target_node not in related_nodes: related_nodes.add(target_node)
        relevant_node_labels = {node.label for node in related_nodes}; # print(f"   -> Relevante Knoten: {list(relevant_node_labels)}") # Weniger verbose
        candidate_chunks: List[TextChunk] = []
        if relevant_node_labels:
             candidate_chunks = [chunk for chunk in self.chunks.values() if any(label in chunk.activated_node_labels for label in relevant_node_labels)]
             # print(f"   -> {len(candidate_chunks)} Kandidaten-Chunks (via Knoten).") # Weniger verbose
        else: candidate_chunks = list(self.chunks.values()); # print("   -> Keine relevanten Knoten, nutze alle Chunks für TF-IDF.") # Weniger verbose
        if not candidate_chunks: print("   -> Keine Kandidaten-Chunks gefunden."); return []
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunk_id_list_for_tfidf: print("WARNUNG: TF-IDF Index nicht verfügbar."); return candidate_chunks[:max_results]
        try:
             prompt_vector = self.vectorizer.transform([prompt])
             candidate_indices_in_matrix = []; valid_candidate_chunks = []
             for c in candidate_chunks:
                  try: idx = self.chunk_id_list_for_tfidf.index(c.uuid); candidate_indices_in_matrix.append(idx); valid_candidate_chunks.append(c)
                  except ValueError: pass
             if not candidate_indices_in_matrix: print("WARNUNG: Keiner der Kandidaten im TF-IDF Index."); return candidate_chunks[:max_results]
             candidate_matrix = self.tfidf_matrix[candidate_indices_in_matrix, :]; similarities = cosine_similarity(prompt_vector, candidate_matrix).flatten()
             scored_candidates = []
             # print("   -> Applying Quantum Effects to Ranking:") # Weniger verbose
             for i, chunk in enumerate(valid_candidate_chunks):
                 base_score = similarities[i]
                 quantum_adjustment = 0.0; num_quantum_nodes_in_chunk = 0; sum_variance = 0.0; sum_activation = 0.0
                 for node_label in chunk.activated_node_labels:
                     node = self.nodes.get(node_label)
                     if node and node.is_quantum and node.q_system:
                         num_quantum_nodes_in_chunk += 1; node_activation = node.activation
                         analysis = node.analyze_jumps(node.last_measurement_log); variance = analysis.get("state_variance", 0.0)
                         sum_activation += node_activation; sum_variance += variance
                 avg_activation = (sum_activation / num_quantum_nodes_in_chunk) if num_quantum_nodes_in_chunk > 0 else 0.0
                 avg_variance = (sum_variance / num_quantum_nodes_in_chunk) if num_quantum_nodes_in_chunk > 0 else 0.0
                 variance_penalty = avg_variance * variance_penalty_factor; activation_boost = avg_activation * activation_boost_factor
                 quantum_adjustment = activation_boost - variance_penalty
                 # if num_quantum_nodes_in_chunk > 0: print(f"      - Chunk {chunk.index} ({chunk.source}): Base={base_score:.3f}, QAdj={quantum_adjustment:+.3f}") # Weniger verbose
                 final_score = np.clip(base_score + quantum_adjustment, 0.0, 1.0)
                 scored_candidates.append({"chunk": chunk, "score": final_score, "base_score": base_score, "q_adjust": quantum_adjustment})
             ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True); final_results = []
             # print("   -> Final Ranking (Quantum Adjusted):"); # Weniger verbose
             scores_printed = 0
             for item in ranked_candidates:
                 if item["score"] >= relevance_threshold and len(final_results) < max_results:
                     final_results.append(item["chunk"])
                     # print(f"      - Score: {item['score']:.4f} (Base: {item['base_score']:.4f}, QAdj: {item['q_adjust']:+.4f}), Chunk: {item['chunk'].source} ({item['chunk'].index})") # Weniger verbose
                     scores_printed +=1
                 elif len(final_results) >= max_results: break
             if not final_results and ranked_candidates:
                 # print(f"   -> Fallback zum besten Treffer (unter Schwelle {relevance_threshold}).") # Weniger verbose
                 best_fallback = ranked_candidates[0]['chunk']; final_results = [best_fallback]
                 # print(f"         - Score: {ranked_candidates[0]['score']:.4f} (Base: {ranked_candidates[0]['base_score']:.4f}, QAdj: {ranked_candidates[0]['q_adjust']:+.4f}), Chunk: {best_fallback.source} ({best_fallback.index})") # Weniger verbose
                 scores_printed += 1
             elif not final_results: print(f"   -> Keine Chunks über Schwelle {relevance_threshold} gefunden."); return []
             # if scores_printed == 0: print("   -> Keine Scores über Schwelle gefunden.") # Weniger verbose
             # print(f"   -> {len(final_results)} Chunks für Kontext ausgewählt (Quantum Ranked).") # Weniger verbose
             return final_results
        except Exception as e: print(f"FEHLER TF-IDF/Quantum Ranking: {e}"); traceback.print_exc(limit=1); return candidate_chunks[:max_results]

    # --- *** NEUE generate_response mit Persona-Prompt für Gemini *** ---
    def generate_response(self, prompt: str) -> str:
        """Generiert Antwort mit Gemini API (neue SDK), Quanteneffekten und Persona-Prompt."""

        # Prüfe globale Abhängigkeiten und RAG-Schalter
        if not GEMINI_AVAILABLE:
             return "Fehler: Google Generative AI SDK (google-generativeai) nicht installiert oder importierbar."
        if not self.rag_enabled:
            return "Fehler: RAG (Gemini) ist in der Konfiguration deaktiviert."

        # Prüfe API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
             try:
                 import streamlit as st
                 api_key = st.secrets.get("GEMINI_API_KEY")
             except (ImportError, AttributeError): pass
             except Exception as e: print(f"WARNUNG: Fehler beim Lesen von Streamlit Secrets: {e}"); pass
        if not api_key:
            return "Fehler: Gemini API Key nicht gefunden (Setze GEMINI_API_KEY Umgebungsvariable oder Streamlit Secret)."

        # --- Konfiguriere API und initialisiere Modell (falls nötig) mit NEUER SDK ---
        try:
            genai.configure(api_key=api_key)
            config_model_name = self.config.get("generator_model_name", "models/gemini-1.5-flash-latest")
            model_name_to_use = config_model_name if config_model_name.startswith("models/") else "models/gemini-1.5-flash-latest"
            if not hasattr(self, 'gemini_model') or self.gemini_model is None or self.gemini_model.model_name != model_name_to_use:
                print(f"INFO: Initialisiere Gemini Modell '{model_name_to_use}'...")
                self.gemini_model = genai.GenerativeModel(model_name_to_use)
                print(f"INFO: Gemini Modell '{model_name_to_use}' initialisiert.")
        except NameError: return "Fehler: Google Generative AI SDK (genai) nicht verfügbar."
        except GoogleAPIError as api_err: return f"Fehler: Problem bei der Google API Initialisierung ({api_err.reason if hasattr(api_err, 'reason') else 'Unbekannt'}). Prüfen Sie Key/Modellnamen."
        except Exception as e: return f"Fehler bei der Konfiguration der Gemini API: {e}"
        if not hasattr(self, 'gemini_model') or self.gemini_model is None: return "Fehler: Gemini-Modellobjekt konnte nicht initialisiert werden."
        # --- Ende Konfiguration/Initialisierung ---

        print(f"\n💬 [Generator] RAG für: '{prompt}'")
        # 1. Pre-Retrieval Simulation
        print("   -> Führe Pre-Retrieval Netzwerk-Simulation durch...")
        self.simulate_network_step(decay_connections=False)

        # 2. Sprunganalyse (wie vorher)
        jump_trigger_active = False; significant_jump_nodes = []
        if self.config.get("quantum_effect_jump_llm_trigger", True):
            # ... (Logik zur Sprunganalyse bleibt gleich - findet direkt aktivierte Q-Knoten und prüft Jumps) ...
            prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
            directly_activated_q_nodes: List[Node] = [
                node for node_label, keywords in semantic_node_definitions.items()
                if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords)
                if (node := self.nodes.get(node_label)) and node.is_quantum
            ]
            if directly_activated_q_nodes:
                for node in directly_activated_q_nodes:
                    if hasattr(node, 'last_measurement_log'):
                        analysis = node.analyze_jumps(node.last_measurement_log)
                        if analysis.get("jump_detected", False):
                            jump_trigger_active = True
                            jump_info_str = f"{node.label} (MaxJump: {analysis.get('max_jump_abs', 0)})"
                            if jump_info_str not in significant_jump_nodes: significant_jump_nodes.append(jump_info_str)
                            print(f"      -> Signifikanter Sprung in Knoten '{node.label}' detektiert!")

        # 3. Retrieval (mit Quanten-Ranking)
        retrieved_chunks = self.respond_to_prompt(prompt) # Holt die Chunks

        # --- 4. Baue den "arona_context" für den neuen Prompt ---
        arona_context_parts = []

        # Finde die relevanten Knoten erneut (oder idealerweise aus respond_to_prompt übernehmen)
        relevant_node_labels_for_context = set()
        prompt_lower = prompt.lower(); semantic_node_definitions = self.config.get("semantic_nodes", {})
        directly_activated_nodes_for_context: List[Node] = [ # Neu berechnen für Kontext-String
            node for node_label, keywords in semantic_node_definitions.items()
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords)
            if (node := self.nodes.get(node_label))
        ]
        related_nodes_for_context: set[Node] = set(directly_activated_nodes_for_context)
        if directly_activated_nodes_for_context:
             for start_node in directly_activated_nodes_for_context:
                 connections_dict = getattr(start_node, 'connections', {})
                 if not isinstance(connections_dict, dict): continue
                 strong_connections = sorted([conn for conn in connections_dict.values() if conn and conn.weight > 0.1], key=lambda c: c.weight, reverse=True)[:3] # Nimm Top 3
                 for conn in strong_connections:
                      target_node = self.nodes.get(getattr(conn, 'target_node_uuid', None))
                      if target_node and target_node not in related_nodes_for_context: related_nodes_for_context.add(target_node)
        relevant_node_labels_for_context = {node.label for node in related_nodes_for_context}

        if relevant_node_labels_for_context:
             arona_context_parts.append(f"Identifizierte relevante Kernkonzepte: {', '.join(sorted(list(relevant_node_labels_for_context)))}.")
        if jump_trigger_active:
             arona_context_parts.append(f"Zusätzlicher Hinweis: Ein Quantensprung in {', '.join(significant_jump_nodes)} deutet auf einen möglichen Perspektivwechsel hin.")

        if not retrieved_chunks:
            arona_context_parts.append("Keine spezifischen Text-Kontexte gefunden.")
            print("   -> Keine Chunks gefunden, Generierung ohne spezifischen Text-Kontext.")
        else:
            context_text = "\n---\n".join([f"[Chunk {idx+1} - Quelle: {chunk.source}]:\n{chunk.text}" for idx, chunk in enumerate(retrieved_chunks)])
            arona_context_parts.append(f"Folgende Textpassagen wurden als relevanter Kontext abgerufen ({len(retrieved_chunks)} Chunks):")
            arona_context_parts.append(context_text)
            print(f"   -> Kontext aus {len(retrieved_chunks)} Chunks für LLM vorbereitet.")

        # Füge alle Teile zum finalen Kontext-String zusammen
        arona_context_string = "\n".join(arona_context_parts)
        # --- Ende Kontext-Aufbau ---


        # 5. Baue den finalen Prompt für Gemini mit der neuen Vorlage
        input_prompt_for_gemini = f"""
Du bist ein fortgeschrittener KI-Assistent, der die Ausgabe eines experimentellen, quanten-inspirierten KI-Modells namens Quantum-NeuroPersona interpretiert und darauf aufbaut.

Der Benutzer hat folgenden ursprünglichen Prompt eingegeben:
"{prompt}"

Quantum-NeuroPersona hat diesen Prompt intern verarbeitet und liefert folgenden Kontext über seinen "Denkprozess":
{arona_context_string}

Deine Aufgabe ist es:
1.  Antworte direkt und präzise auf den *ursprünglichen Prompt* des Benutzers.
2.  Nutze den von Quantum-NeuroPersona gelieferten Kontext (Kernkonzepte, mögliche Sprünge, abgerufene Textpassagen), um die *Perspektive, den Ton oder die thematische Gewichtung* deiner Antwort zu beeinflussen. Interpretiere NeuroPersonas Kontext kreativ.
3.  Formuliere eine kohärente, gut lesbare und hilfreiche Antwort in natürlicher Sprache. Erkläre NeuroPersonas internen Prozess NICHT, sondern nutze ihn als Inspiration.
4.  Wenn der Kontext keine sinnvolle Antwort auf den ursprünglichen Prompt ermöglicht, gib eine kurze, höfliche Antwort, die dies widerspiegelt (z.B. "Basierend auf dem Kontext kann ich dazu keine spezifische Antwort geben.").

Antworte jetzt auf den ursprünglichen Prompt unter Berücksichtigung des NeuroPersona-Kontexts:
"""
        # print(f"   -> Finaler Prompt für Gemini (gekürzt):\n{input_prompt_for_gemini[:600]}...\n") # Optional für Debugging

        # 6. Generation mit Gemini API (wie vorher, aber mit neuem Prompt)
        try:
            start_gen_time = time.time()
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.get("generator_temperature", 0.7),
                # max_output_tokens=self.config.get("generator_max_length", 8192) # Kann ggf. angepasst werden
            )
            safety_settings=[ # Korrekte Definition
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = self.gemini_model.generate_content(
                input_prompt_for_gemini, # Der neue, ausführliche Prompt
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if not response.candidates:
                 block_reason = "Unbekannt"; safety_ratings_str = ""
                 # ... (Fehlerbehandlung für Blockierung wie vorher) ...
                 if response.prompt_feedback:
                     if response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
                     if response.prompt_feedback.safety_ratings: safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in response.prompt_feedback.safety_ratings])
                 print(f"WARNUNG: Gemini-Antwort blockiert. Grund: {block_reason}. Ratings: [{safety_ratings_str}]")
                 return f"[Antwort blockiert durch Sicherheitsfilter: {block_reason}]"

            generated_text = response.text
            end_gen_time = time.time()
            print(f"   -> Generierung mit Gemini in {end_gen_time - start_gen_time:.2f}s.")
            print(f"   -> Generierte Antwort: {generated_text}")
            return generated_text.strip()

        except GoogleAPIError as api_err:
            print(f"FEHLER bei der Gemini API Anfrage: {api_err}")
            return f"Fehler: Problem bei der Kommunikation mit der Gemini API ({api_err.reason if hasattr(api_err, 'reason') else 'Unbekannt'}). Prüfen Sie Key/Modellnamen."
        except Exception as e:
            print(f"FEHLER während der Textgenerierung mit Gemini: {e}")
            traceback.print_exc(limit=2)
            return "Fehler: Entschuldigung, ich konnte keine Antwort generieren."
    # --- *** ENDE generate_response mit Persona-Prompt für Gemini *** ---

    def get_network_state_summary(self) -> Dict[str, Any]:
         summary = {"num_nodes": len(self.nodes), "num_quantum_nodes": sum(1 for n in self.nodes.values() if n.is_quantum), "num_chunks": len(self.chunks),
                    "sources_processed": list(self.sources_processed), "total_connections": sum(len(node.connections) for node in self.nodes.values() if hasattr(node, 'connections')),
                    "tfidf_index_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None, "rag_enabled": self.rag_enabled,
                    "generator_model": self.config.get("generator_model_name") if self.rag_enabled else None}
         activations = [n.activation for n in self.nodes.values() if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)]
         summary["average_node_activation"] = round(np.mean(activations), 4) if activations else 0.0
         all_connections_found = []
         for source_node in self.nodes.values():
             if not hasattr(source_node, 'connections'): continue
             for target_uuid, conn in source_node.connections.items():
                  if conn is None: continue
                  target_node_obj = getattr(conn, 'target_node', None); weight = getattr(conn, 'weight', None)
                  if (target_node_obj is not None and hasattr(target_node_obj, 'label') and weight is not None and isinstance(weight, (float, np.number)) and np.isfinite(weight)):
                     all_connections_found.append({"source": source_node.label, "target": target_node_obj.label, "weight": weight})
         all_connections_found.sort(key=lambda x: x["weight"], reverse=True)
         summary["top_connections"] = all_connections_found[:10]
         return summary

    def save_state(self, filepath: str):
        print(f"💾 Speichere Zustand nach {filepath}...")
        try:
            chunks_to_save = { c.uuid: {"uuid": c.uuid, "text": c.text, "source": c.source, "index": c.index, "activated_node_labels": c.activated_node_labels} for c in self.chunks.values()}
            state_data = {"config": self.config, "nodes": {label: node.__getstate__() for label, node in self.nodes.items()}, "chunks": chunks_to_save,
                          "sources_processed": list(self.sources_processed), "chunk_id_list_for_tfidf": self.chunk_id_list_for_tfidf}
            with open(filepath, 'w', encoding='utf-8') as f:
                def default_serializer(obj):
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (datetime, deque)): return str(obj)
                    if isinstance(obj, set): return list(obj)
                    if isinstance(obj, (np.int32, np.int64)): return int(obj)
                    if isinstance(obj, (np.float32, np.float64)): return float(obj)
                    try:
                        if hasattr(obj, '__getstate__'): return obj.__getstate__()
                        return repr(obj)
                    except Exception as ser_err: return f"<SerializationError: {type(obj)}>"
                json.dump(state_data, f, indent=2, ensure_ascii=False, default=default_serializer)
            print("   -> Zustand erfolgreich gespeichert.")
        except Exception as e: print(f"FEHLER Speichern Zustand: {e}"); traceback.print_exc(limit=2)

    @classmethod
    def load_state(cls, filepath: str) -> Optional['QuantumEnhancedTextProcessor']:
        print(f"📂 Lade Zustand von {filepath}...")
        if not os.path.exists(filepath): print(f"FEHLER: Zustandsdatei {filepath} nicht gefunden."); return None
        instance = None
        try:
            with open(filepath, 'r', encoding='utf-8') as f: state_data = json.load(f)
            instance = cls(config_dict=state_data.get("config"))
            if not instance: print("FEHLER: Instanzerstellung fehlgeschlagen."); return None
            if not instance.rag_enabled and instance.config.get("enable_rag"): print("INFO: RAG in Config aktiv, aber Modell-Laden fehlgeschlagen/Libs fehlen.")
            elif instance.rag_enabled: print("INFO: RAG ist im geladenen Prozessor aktiv.")
            loaded_chunks = {}; raw_chunk_data = state_data.get("chunks", {})
            for uuid_key, chunk_data_dict in raw_chunk_data.items():
                if not isinstance(chunk_data_dict, dict): continue
                loaded_chunk_uuid = chunk_data_dict.get('uuid', uuid_key)
                init_args = {'text': chunk_data_dict.get('text', ''), 'source': chunk_data_dict.get('source', 'Unknown'), 'index': chunk_data_dict.get('index', -1)}
                try:
                    new_chunk = TextChunk(chunk_uuid=loaded_chunk_uuid, **init_args)
                    new_chunk.activated_node_labels = chunk_data_dict.get('activated_node_labels', [])
                    loaded_chunks[new_chunk.uuid] = new_chunk
                except Exception as e: print(f"FEHLER Erstellen Chunk UUID {loaded_chunk_uuid}: {e}")
            instance.chunks = loaded_chunks
            print("\n--- Lade/Aktualisiere Knoten ---")
            loaded_node_states = state_data.get("nodes", {}); print(f"Gespeicherte Knotenzustände: {len(loaded_node_states)} ({list(loaded_node_states.keys())})")
            final_nodes_dict = {}; node_uuid_map = {}
            for node_label, node_state_dict in loaded_node_states.items():
                 if not isinstance(node_state_dict, dict): continue
                 original_uuid = node_state_dict.get("uuid")
                 if not original_uuid: continue
                 node_type_name = node_state_dict.get('type', 'Node'); node_class = globals().get(node_type_name, Node)
                 try:
                     node = node_class.__new__(node_class); node.__setstate__(node_state_dict)
                     if node.uuid != original_uuid: node.uuid = original_uuid
                     # print(f"  -> Knoten '{node.label}' (UUID: {node.uuid}) aus Zustand wiederhergestellt.") # Weniger verbose
                     final_nodes_dict[node.label] = node; node_uuid_map[node.uuid] = node
                 except Exception as e: print(f"FEHLER Restore Knoten '{node_label}': {e}"); traceback.print_exc(limit=1)
            for label_in_config in instance.config.get("semantic_nodes", {}).keys():
                 if label_in_config not in final_nodes_dict:
                      print(f"  -> Knoten '{label_in_config}' neu erstellen..."); new_node = instance._get_or_create_node(label_in_config)
                      if new_node: final_nodes_dict[new_node.label] = new_node; node_uuid_map[new_node.uuid] = new_node
                      else: print(f"     FEHLER Neuerstellen von '{label_in_config}'.")
            instance.nodes = final_nodes_dict; # print(f"Endgültige Knoten: {list(instance.nodes.keys())}") # Weniger verbose
            # print(f"UUID Map ({len(node_uuid_map)}): {list(node_uuid_map.keys())}") # Weniger verbose
            # --- Ende Knoten Laden ---
            # print(f"\n--- Verbindungen wiederherstellen ---") # Weniger verbose
            total_connections_restored = 0
            for node in instance.nodes.values():
                if hasattr(node, 'connections_serializable_temp') and isinstance(node.connections_serializable_temp, dict):
                     for target_uuid, conn_dict in node.connections_serializable_temp.items():
                          target_node = node_uuid_map.get(target_uuid)
                          if target_node:
                               try:
                                   source_label = conn_dict.get('source_node_label'); conn = Connection.__new__(Connection)
                                   conn_attrs = {k:v for k,v in conn_dict.items() if k != 'target_node'}
                                   created_at_str = conn_attrs.get('created_at'); last_update_at_str = conn_attrs.get('last_update_at')
                                   try: conn_attrs['created_at'] = datetime.fromisoformat(created_at_str) if isinstance(created_at_str, str) else datetime.now()
                                   except: conn_attrs['created_at'] = datetime.now()
                                   try: conn_attrs['last_update_at'] = datetime.fromisoformat(last_update_at_str) if isinstance(last_update_at_str, str) else datetime.now()
                                   except: conn_attrs['last_update_at'] = datetime.now()
                                   conn_attrs.setdefault('target_node_uuid', target_uuid); conn_attrs.setdefault('source_node_label', source_label)
                                   conn.__dict__.update(conn_attrs); conn.target_node = target_node # Target Node explizit setzen
                                   node.connections[target_uuid] = conn; total_connections_restored += 1
                               except Exception as conn_e: print(f"FEHLER Restore Conn '{node.label}'->'{target_uuid}': {conn_e}")
                     if hasattr(node, 'connections_serializable_temp'): del node.connections_serializable_temp
            # print(f"   -> {total_connections_restored} Verbindungen wiederhergestellt.") # Weniger verbose
            instance.sources_processed = set(state_data.get("sources_processed", [])); instance.chunk_id_list_for_tfidf = state_data.get("chunk_id_list_for_tfidf", [])
            instance.update_tfidf_index()
            print(f"\n✅ Zustand geladen ({len(instance.nodes)} Knoten, {len(instance.chunks)} Chunks, {total_connections_restored} Verbindungen).")
            return instance
        except json.JSONDecodeError as json_err: print(f"FEHLER: JSON Decode Error in {filepath}: {json_err}"); return None
        except Exception as e: print(f"FEHLER Laden Zustand {filepath}: {e}"); traceback.print_exc(limit=2); return None

# --- Beispielnutzung __main__ ---
if __name__ == "__main__":
    print("="*50 + "\n Starte Quantum-Enhanced Text Processor Demo \n" + "="*50)
    CONFIG_FILE = "config_qllm.json"; STATE_FILE = "qetp_state.json"
    processor = QuantumEnhancedTextProcessor.load_state(STATE_FILE)
    if processor is None:
        print(f"\nInitialisiere neu mit '{CONFIG_FILE}'.")
        processor = QuantumEnhancedTextProcessor(config_path=CONFIG_FILE)
        if processor is None or not hasattr(processor, 'config'): print("\nFATALER FEHLER: Init fehlgeschlagen."); exit()
        training_files = processor.config.get("training_files", [])
        if not training_files: print("\nWARNUNG: Keine Trainingsdateien in Config.")
        else:
             print("\n--- Initiale Datenverarbeitung ---")
             for file in training_files: processor.load_and_process_file(file)
             print("--- Initiale Verarbeitung abgeschlossen ---"); processor.save_state(STATE_FILE)
    else: print(f"\nZustand aus '{STATE_FILE}' geladen.")

    # ---- Interaktive Schleife ----
    print("\n--- Aktueller Netzwerkstatus ---"); summary = processor.get_network_state_summary(); print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n--- Interaktive Abfrage (Typ 'exit' zum Beenden) ---")

    # Prüfe, ob API Key für Gemini benötigt wird und vorhanden ist
    if processor.rag_enabled and not os.environ.get("GEMINI_API_KEY"):
         print("\nWARNUNG: RAG ist aktiviert, aber die Umgebungsvariable GEMINI_API_KEY ist nicht gesetzt.")
         print("         Die Generierung wird wahrscheinlich fehlschlagen.")
         print("         Setzen Sie die Variable oder verwenden Sie Streamlit Secrets.")

    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.lower() == 'exit': break
            if not prompt: continue
            generated_response = processor.generate_response(prompt)
            print("\n--- Generierte Antwort ---"); print(generated_response); print("-" * 25)
        except KeyboardInterrupt: print("\nUnterbrochen."); break
        except Exception as e: print(f"\nFehler: {e}"); traceback.print_exc(limit=1)

    # ---- Ende interaktive Schleife ----

    print("\n--- Speichere finalen Zustand ---"); processor.save_state(STATE_FILE)
    print("\n" + "="*50 + "\n Demo beendet. \n" + "="*50)