
# THIS FILE CONTAINS THE CLASSES TO HANDLE THE DAG GRPH ELEMENTS
# THE OBJECTS ARE BASED ON THE QISKIT DAG AND QUANTUM CIRCUIT
# THE OBJECTS ARE THE FOLLOWING:
#
#      CircuitBit      - quantum and classical 
#      CircuitNode     - bits, gates and measures
#      CircuitEdge     - wires   
#      CircuitGraph    - complete graph of the circuit made by CircuitBit, CircuitNode and CircuitEdge
#      SubCircuitGraph - complete graph of subcircuit obtained by cutting the original CircuitGraph
#
# the file also contain some helper function to build the above objects


from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode


# ------------------------------------------------------------
# Data structures: bit - edges - nodes - subgraph


@dataclass
class CircuitBit:
    '''
    Bit object can represent classical or quantum bits
    '''
    global_id:    int             # object global identifier
    bit_id:       int             # bit identifier
    nature:       str             # "qubit" or "clbit"
    qiskit_obj:   object          # original object


@dataclass
class CircuitNode:
    '''
    Node object based on topological node of DAG in Qiskit library
    can represent bits, gates and measures
    '''
    global_id:    int             # object global identifier
    node_id:      int             # node identifier
    nature:       str             # "input_qubit", "output_qubit", "input_cbit", "output_cbit", "op"
    qiskit_obj:   object          # original object
    name:         str = None
    qargs:        list = field(default_factory=list)  # qubit bit_ids involved
    cargs:        list = field(default_factory=list)  # clbit bit_ids involved


@dataclass
class CircuitEdge:
    '''
    Edge object: encode the connection between nodes, 
    they are the original wires of the circuit
    '''
    global_id:    int             # object global identifier
    edge_id:      int             # edge identifier
    src_node_id:  int
    dest_node_id: int
    bit_id:       int
    is_cut:       bool = False    # boolean flag to check if is a cut edge or not


@dataclass
class SubCircuitGraph:
    '''
    Graph object that encode the informations of the cutted subcurcuit 
    '''
    subcg_id:               int             # subcircuit identifier
    node_ids:               list            # original node ids (nodes in CG)
    internal_edge_ids:      list            # original edges inside the subgraph
    bit_ids:                list            # original bit ids
    incoming_cut_edge_ids:  list            # cut edges entering the subgraph: initialization
    outgoing_cut_edge_ids:  list            # cut edges leaving the subgraph: measurement basis

    qiskit_circuit:         QuantumCircuit = None
    cut_tensor:             object = None   # SubCircuitTensor filled from the circuit runs



# ------------------------------------------------------------
# Circuit graph


class CircuitGraph:
    '''
    Class to handle the DAG of a qiskit circuit
    '''

    def __init__(self,
                 bits:  Iterable,
                 nodes: Iterable,
                 edges: Iterable):

        # dictionaries of bits node and edges 
        self.bits  = {bit.bit_id: bit for bit in bits}      
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = {edge.edge_id: edge for edge in edges}

    def get_edgelist(self, with_ids=False):
        '''
        function to give the edgelist of the graph
        '''
        edge_pairs = [(e.src_node_id, e.dest_node_id) for e in self.edges.values()]  

        if with_ids:
            edge_ids = list(self.edges.keys())
            return edge_pairs, edge_ids

        return edge_pairs

    def build_SubCG_FromCuts(self, cut_edge_ids):
        '''
        function that builds the subgraphs from a list of cut edges

        --> take the edges cut ids
        --> cut it from the graph 
        --> compute the connected components

        finally build the SubCircuitGraph object from the connected components
        '''

        cut_edge_ids = set(cut_edge_ids)

        # graph without the cut edges
        G_kept = nx.Graph()
        G_kept.add_nodes_from(self.nodes.keys())
        G_kept.add_edges_from(
            (e.src_node_id, e.dest_node_id)
            for e in self.edges.values()
            if e.edge_id not in cut_edge_ids
        )

        # connected components of the cut graph
        comp_list = [sorted(comp) for comp in nx.connected_components(G_kept)]

        # building subgraphs from the connected components
        subcg_list = []
        for sub_id, comp in enumerate(comp_list):

            # nodes in the selected connected subgraph
            comp = set(comp)

            # we want to mantain information about edges role
            internal_edge_ids = []
            incoming_cut_edge_ids = []
            outgoing_cut_edge_ids = []
            bit_ids = set()

            for edge in self.edges.values():

                # boolean flag to check if the edges are inside the subgraph
                src_inside  = edge.src_node_id in comp
                dest_inside = edge.dest_node_id in comp

                # the selected edges is a cut edge
                if edge.edge_id in cut_edge_ids:
                    edge.is_cut = True

                    # update the outgoing cut edges set 
                    if src_inside:
                        outgoing_cut_edge_ids.append(edge.edge_id)
                        bit_ids.add(edge.bit_id)
                    # update the income cut edges set 
                    elif dest_inside:
                        incoming_cut_edge_ids.append(edge.edge_id)
                        bit_ids.add(edge.bit_id)

                # update the internal edges set 
                elif src_inside and dest_inside:
                    internal_edge_ids.append(edge.edge_id)
                    bit_ids.add(edge.bit_id)

            # finally we build the subgraph with all the informations
            subcg = SubCircuitGraph(
                subcg_id=sub_id,
                node_ids=sorted(comp),
                internal_edge_ids=internal_edge_ids,
                bit_ids=sorted(bit_ids),
                incoming_cut_edge_ids=incoming_cut_edge_ids,
                outgoing_cut_edge_ids=outgoing_cut_edge_ids,
            )

            subcg_list.append(subcg)

        return subcg_list


# ------------------------------------------------------------
# DAG -> CircuitGraph conversion (helper finctions)

def build_CG_fromDAG(dag: DAGCircuit) -> CircuitGraph:
    '''
    Build a CircuitGraph from a Qiskit DAGCircuit.

    Conversion pipeline:
        1. Qiskit wires  -> CircuitBit
        2. Qiskit nodes  -> CircuitNode
        3. Qiskit edges  -> CircuitEdge
    '''

    bits = []
    nodes = []
    edges = []

    # dictionaries to map qiskit objects to internal ids
    qiskit_wire_to_bit_id = {}
    qiskit_node_to_node_id = {}

    # global index ( good for all objects )
    global_id_counter = 0

    # 1. CircuitBit objects (qubits first, then clbits)
    for wire, nature in (
        # looping on the list of all bits in the dag
        [(q, "qubit") for q in dag.qubits] + [(c, "clbit") for c in dag.clbits] 

    ):
        
        bit_id = len(bits)
        # building the bit object
        bit = CircuitBit(
            global_id=global_id_counter,
            bit_id=bit_id,
            nature=nature,
            qiskit_obj=wire,
        )

        bits.append(bit)
        qiskit_wire_to_bit_id[wire] = bit_id
        global_id_counter += 1

    # helper to build a CircuitNode from a qiskit dag node
    def make_node(qiskit_node, node_id, global_id):
        
        # check if the topological qiskit node is an operational node (measure or gates)
        if isinstance(qiskit_node, DAGOpNode):
            qargs = [qiskit_wire_to_bit_id[q] for q in qiskit_node.qargs]
            cargs = [qiskit_wire_to_bit_id[c] for c in qiskit_node.cargs]

            return CircuitNode(
                global_id=global_id,
                node_id=node_id,
                nature="op",
                qiskit_obj=qiskit_node,
                name=qiskit_node.name,
                qargs=qargs,
                cargs=cargs,
            )

        # output node (classical or quantum bit)
        if isinstance(qiskit_node, (DAGInNode, DAGOutNode)):
            bit_id = qiskit_wire_to_bit_id[qiskit_node.wire]
            bit = bits[bit_id]

            side = "input" if isinstance(qiskit_node, DAGInNode) else "output"
            kind = "qubit" if bit.nature == "qubit" else "cbit"

            return CircuitNode(
                global_id=global_id,
                node_id=node_id,
                nature=f"{side}_{kind}",
                qiskit_obj=qiskit_node,
                name=None,
                qargs=[bit_id],
                cargs=[bit_id],
            )

        raise TypeError(f"Unsupported Qiskit node type: {type(qiskit_node)}")

    # 2. CircuitNode objects
    for qiskit_node in dag.topological_nodes():

        node_id = len(nodes)
        node = make_node(qiskit_node, node_id, global_id_counter)
        nodes.append(node)
        qiskit_node_to_node_id[qiskit_node] = node_id
        global_id_counter += 1

    # 3. CircuitEdge objects
    for src_node, dest_node, wire in dag.edges():

        edge = CircuitEdge(
            global_id=global_id_counter,
            edge_id=len(edges),
            src_node_id=qiskit_node_to_node_id[src_node],
            dest_node_id=qiskit_node_to_node_id[dest_node],
            bit_id=qiskit_wire_to_bit_id[wire],
        )

        edges.append(edge)
        global_id_counter += 1

    return CircuitGraph(bits, nodes, edges)


def find_candidate_cut_edges(CG: CircuitGraph):
    '''
    Find raw candidate cut edges: set of all "meaningful" edges in the DAG graph

    Candidate rule:
        - edge is quantum, not classical
        - edge touches at least one multi-qubit operation
        - edge does not start from a global input qubit
        - edge does not end at a global output qubit
        - edge does not end at a measurement
    '''

    # dictionary of more than one qubit gates
    multi_qubit_node_ids = {
        node.node_id
        for node in CG.nodes.values()
        if node.nature == "op" and len(node.qargs) > 1
    }

    # initialization of all possible cuts id
    candidate_edge_ids = []
    for edge in CG.edges.values():

        src_node  = CG.nodes[edge.src_node_id]
        dest_node = CG.nodes[edge.dest_node_id]

        touches_multi_qubit_node = (
            edge.src_node_id in multi_qubit_node_ids
            or edge.dest_node_id in multi_qubit_node_ids
        )

        # avoid useless cuts
        if not touches_multi_qubit_node:            # not between multi-qubit ops
            continue
        if CG.bits[edge.bit_id].nature != "qubit":  # not a qubit edge
            continue
        if src_node.nature == "input_qubit":        # not an input edge
            continue
        if dest_node.nature == "output_qubit":      # not an output edge
            continue
        if dest_node.name == "measure":             # not a measure edge
            continue

        candidate_edge_ids.append(edge.edge_id)

    return candidate_edge_ids
