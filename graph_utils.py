
# THIS FILE CONTAINS THE FUNCTIN USED TO EXTRACT FROM A CIRCUIT 
# TO BUILD THE GRAPH OBJECTS NEEDED IN THE TENSORQC FRAMEWORK
#       
#       -> Qiskit Circuit -> Dag -> CustomGraph -> cuts -> Subcircuits 




from itertools import product
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SdgGate
from .QuantumGraph import CircuitGraph, SubCircuitGraph
from .Tensors import SubCircuitTensor

# Constants for the cutting configurations
# one for initializing qubits and one for measuring
STATES = ["0", "1", "+", "i"]   # initialization of an incoming cut
BASES  = ["I", "Z", "X", "Y"]   # measurement basis of an outgoing cut


def initialize_input_cut(qc: QuantumCircuit, qubit_id: int, state: str):
    '''
    Initialize in the QuantumCircuit an incoming cut qubit in one of the states {0, 1, +, i}
    '''
    init_dict = {
        "0": [1, 0],
        "1": [0, 1],
        "+": [1 / np.sqrt(2), 1 / np.sqrt(2)],
        "i": [1 / np.sqrt(2), 1j / np.sqrt(2)],
    }

    if state not in init_dict:
        raise KeyError(f"Invalid initial state: {state}")

    qc.initialize(init_dict[state], qubit_id)


def initialize_output_cut(qc: QuantumCircuit, qubit_id: int, base: str):
    '''
    Apply a basis-change gates before measuring an outgoing cut qubit in the QuantumCircuit.

    I/Z basis: no rotation
    X basis:   H
    Y basis:   Sdg + H
    '''
    base_map = {
        "I": [],
        "Z": [],
        "X": [HGate()],
        "Y": [SdgGate(), HGate()],
    }

    if base not in base_map:
        raise KeyError(f"Invalid measurement basis: {base}")

    for gate in base_map[base]:
        qc.append(gate, [qubit_id])


def build_qiskit_subcircuit(

    CG:                  CircuitGraph,     # circuit graph
    sub:                 SubCircuitGraph,  # specific subcircuit
    input_cut_states:    dict = None,      # incoming cut edge_id -> state
    output_cut_bases:    dict = None,      # outgoing cut edge_id -> basis
    measure_output_cuts: bool = True,      # measure outgoing cuts
    raw:                 bool = False,     # if True: no cut init, no basis rotations, no cut measurements

):
    '''
    Build the qiskit QuantumCircuit of a subcircuit for a given cut configuration 
    (initialization input + measure output). Assumption: each qubit wire crosses the subcircuit at most once
    (one incoming and/or one outgoing cut per local qubit).
    '''
    # configuration dictionaries
    if input_cut_states is None:
        input_cut_states = {}
    if output_cut_bases is None:
        output_cut_bases = {}

    # if raw=False, all cut configurations must be specified
    # otherwise the circuit is given without measurement and initializations (raw cut)
    if not raw:
        missing_input = [
            edge_id for edge_id in sub.incoming_cut_edge_ids    # check for the income cuts
            if edge_id not in input_cut_states
        ]
        if missing_input:
            raise ValueError(f"Missing input cut states for edge ids: {missing_input}")

        missing_output = [
            edge_id for edge_id in sub.outgoing_cut_edge_ids    # check for the outgoing cuts
            if edge_id not in output_cut_bases
        ]
        if missing_output:
            raise ValueError(f"Missing output cut bases for edge ids: {missing_output}")

    # local qubits and classical bits: global bit_id -> local index
    qubit_ids = [b for b in sub.bit_ids if CG.bits[b].nature == "qubit"]
    clbit_ids = [b for b in sub.bit_ids if CG.bits[b].nature == "clbit"]
    # mapping global to local ids (global: full graph, local: subgraph)
    q_id_map = {bit_id: local_id for local_id, bit_id in enumerate(qubit_ids)}
    c_id_map = {bit_id: local_id for local_id, bit_id in enumerate(clbit_ids)}

    n_cut_clbits = 0
    if (not raw) and measure_output_cuts:
        n_cut_clbits = len(sub.outgoing_cut_edge_ids)

    # building the cutted subcircuit 
    qc = QuantumCircuit(len(qubit_ids), len(clbit_ids) + n_cut_clbits)

    # 1. incoming cuts initialization
    if not raw:
        for edge_id in sub.incoming_cut_edge_ids:                          # initialization of the qubit from 
            q_local = q_id_map[CG.edges[edge_id].bit_id]                   # the provided dictionary 
            initialize_input_cut(qc, q_local, input_cut_states[edge_id])    
                                                                            
    # 2. internal operations (node_ids are in topological order)
    # retreaving all the other gates operation 
    for node_id in sub.node_ids:
        node = CG.nodes[node_id]

        if node.nature != "op":
            continue

        qargs_mapped = [qc.qubits[q_id_map[b]] for b in node.qargs] # wich qubit is the operation on
        cargs_mapped = [qc.clbits[c_id_map[b]] for b in node.cargs]

        qc.append(node.qiskit_obj.op, qargs_mapped, cargs_mapped)   # adding to the circuit the gate on the specific bits

    # 3. outgoing cuts basis rotations and optional measurements
    cut_measure_map = {}

    if not raw:
        for local_cut_id, edge_id in enumerate(sub.outgoing_cut_edge_ids):  # looping on outgoing cuts
            q_local = q_id_map[CG.edges[edge_id].bit_id]
            initialize_output_cut(qc, q_local, output_cut_bases[edge_id])   # changing bases on the cut qubit 

            if measure_output_cuts:
                c_local = len(clbit_ids) + local_cut_id
                qc.measure(q_local, c_local)
                cut_measure_map[edge_id] = c_local

    sub.qiskit_circuit = qc

    return qc, q_id_map, c_id_map, cut_measure_map


def generate_cut_configurations(sub: SubCircuitGraph):
    '''
    generate the 4^(n_cuts) local configurations of a subcircuit
    '''
    in_edge_ids  = sub.incoming_cut_edge_ids
    out_edge_ids = sub.outgoing_cut_edge_ids
    cut_edge_ids = in_edge_ids + out_edge_ids

    configurations = {}

    for config_id, config_values in enumerate(
        product(range(4), repeat=len(cut_edge_ids))
    ):
        cut_config = dict(zip(cut_edge_ids, config_values))

        configurations[config_id] = {
            "cut_config": cut_config,
            "input_cut_states": {e: STATES[cut_config[e]] for e in in_edge_ids},
            "output_cut_bases": {e: BASES[cut_config[e]] for e in out_edge_ids},
        }

    return configurations



def build_GHZ_circ(n_qubit: int, measure: bool = False) -> QuantumCircuit:

    qc = QuantumCircuit(n_qubit)
    qc.h(0)
    for i in range(1, n_qubit):
        qc.cx(i - 1, i)
    if measure:
        qc.measure_all()

    return qc
