# IN THIS FILE THERE ARE THE OBJECTS FOR THE TENSOR MAN



from .QuantumGraph import CircuitGraph, SubCircuitGraph
from .Tensors import SubCircuitTensor
from .graph_utils import generate_cut_configurations, build_qiskit_subcircuit
from qiskit.quantum_info import Statevector
import numpy as np
from itertools import product


def get_true_output_qubits(CG: CircuitGraph, sub: SubCircuitGraph):
    '''
    Split local qubits into the "true" output qubits and the outgoing-cut
    qubits. Outgoing-cut qubits are measured, so they must NOT be in the state axis: 
    only the true quantum output qubits do.
    '''
    all_qubit_bit_ids = [b for b in sub.bit_ids if CG.bits[b].nature == "qubit"]

    # bit_id of the measured qubit on each outgoing cut
    outgoing_cut_bit_ids = {CG.edges[e].bit_id for e in sub.outgoing_cut_edge_ids}

    true_qubit_bit_ids = [b for b in all_qubit_bit_ids if b not in outgoing_cut_bit_ids]

    return all_qubit_bit_ids, true_qubit_bit_ids, outgoing_cut_bit_ids


def initialize_subcircuit_tensor(CG: CircuitGraph, sub: SubCircuitGraph) -> SubCircuitTensor:
    '''
    Allocate the empty subcircuit tensor.

    Axis layout:
        [4] per incoming cut   -> initial state {0,1,+,i}
        [8] per outgoing cut   -> (4 bases {I,Z,X,Y}) x (2 outcomes {0,1})
        [2^n_true] state axis  -> true output qubits only
    '''
    in_edge_ids = sub.incoming_cut_edge_ids
    out_edge_ids = sub.outgoing_cut_edge_ids

    cut_edge_ids = in_edge_ids + out_edge_ids

    edge_roles = {e: "incoming" for e in in_edge_ids}
    edge_roles.update({e: "outgoing" for e in out_edge_ids})

    _, true_qubit_bit_ids, _ = get_true_output_qubits(CG, sub)

    # 4 for incoming cuts, 8 for outgoing cuts (base x outcome)
    cut_axis_dims = [4] * len(in_edge_ids) + [8] * len(out_edge_ids)

    tensor_shape = tuple(cut_axis_dims + [2 ** len(true_qubit_bit_ids)])

    SCT = SubCircuitTensor(
        tensor_array=np.zeros(tensor_shape),
        cut_edge_ids=cut_edge_ids,
        edge_roles=edge_roles,
        qubit_bit_ids=true_qubit_bit_ids,
    )

    sub.cut_tensor = SCT
    return SCT


def split_cut_outcome(probability_vector, n_local_qubits, cut_local_ids):
    '''
    Separate the outcome of the outgoing-cut qubits from the 'true' qubit states.

    Returns an array of shape [2]*len(cut_local_ids) + [2^n_true], where the
    leading axes are the cut outcomes (0/1) and the last axis are the true
    states.
    '''

    per_qubit = probability_vector.reshape([2] * n_local_qubits)

    # numpy axis of each outgoing cut qubit
    cut_axes = [n_local_qubits - 1 - L for L in cut_local_ids]

    # move cut axes to the front, in cut_local_ids order
    per_qubit = np.moveaxis(per_qubit, cut_axes, range(len(cut_axes)))

    # flatten the remaining (true qubit) axes into a single state axis
    n_true = n_local_qubits - len(cut_local_ids)
    new_shape = [2] * len(cut_local_ids) + [2 ** n_true]
    return per_qubit.reshape(new_shape)


def buildTensor_fromCircuit(CG: CircuitGraph, sub: SubCircuitGraph, mode: str = "th") -> SubCircuitTensor:
    '''
    Fill the subcircuit tensor by computing state vector probability of all configurations
    '''
    if mode not in ["th", "shot"]:
        raise ValueError("mode should be in ['th', 'shot']")
    if mode == "shot":
        raise NotImplementedError("shot mode not implemented yet")

    SCT = initialize_subcircuit_tensor(CG, sub)
    configurations = generate_cut_configurations(sub)

    # local_id of every local qubit, to locate the cut qubits
    all_qubit_bit_ids, _, _ = get_true_output_qubits(CG, sub)
    n_local_qubits = len(all_qubit_bit_ids)
    local_id_of = {bit_id: i for i, bit_id in enumerate(all_qubit_bit_ids)}

    # local_id of the outgoing-cut qubits, in edge order
    outgoing_cut_local_ids = [
        local_id_of[CG.edges[e].bit_id] for e in sub.outgoing_cut_edge_ids
    ]

    for configuration in configurations.values():

        qc, _, _, _ = build_qiskit_subcircuit(
            CG,
            sub,
            input_cut_states=configuration["input_cut_states"],
            output_cut_bases=configuration["output_cut_bases"],
            measure_output_cuts=False,   # we separate the outcome via reshape
        )

        probability_vector = Statevector.from_instruction(qc).probabilities()

        # shape [2]*n_out_cut + [2^n_true]
        split = split_cut_outcome(probability_vector, n_local_qubits, outgoing_cut_local_ids)

        write_config_to_tensor(SCT, sub, configuration["cut_config"], split)

    return SCT


def write_config_to_tensor(subcircuit_tensor, sub, cut_config, probabilities_by_outcome):
    '''
    Write the probabilities of one cut configuration into the tensor.

    Index packing:
      - incoming axes (dim 4) : initial-state index (0..3)
      - outgoing axes (dim 8) : base index (0..3) + measured outcome (0/1)
                                packed together as  base * 2 + outcome
    '''

    in_edge_ids = sub.incoming_cut_edge_ids
    out_edge_ids = sub.outgoing_cut_edge_ids

    # ----- index on the incoming axes -----
    # one entry per incoming cut:
    incoming_index = []
    for edge_id in in_edge_ids:
        initial_state = cut_config[edge_id]      # 0..3 -> |0>, |1>, |+>, |i>
        incoming_index.append(initial_state)
    incoming_index = tuple(incoming_index)

    # ----- index on the outgoing axes, one slice per outcome combination -----
    # each outgoing cut is measured and gives outcome 0 or 1, so with n_out
    # cuts there are 2^n_out combinations.
    # e.g. with 2 cuts -> (0,0), (0,1), (1,0), (1,1)
    n_out = len(out_edge_ids)
    all_outcome_combinations = product([0, 1], repeat=n_out)

    for possible_outcome in all_outcome_combinations:

        # build the outgoing index: for every outgoing cut, pack its base
        # with its measured outcome
        outgoing_index = []
        for position in range(n_out):
            edge_id = out_edge_ids[position]
            base = cut_config[edge_id]               # which Pauli base, 0..3
            outcome = possible_outcome[position]     # measured 0 or 1
            packed = base * 2 + outcome              # single index, 0..7
            outgoing_index.append(packed)
        outgoing_index = tuple(outgoing_index)

        # the probabilities of the true (non-cut) qubits for this outcome
        state_slice = probabilities_by_outcome[possible_outcome]   # shape [2^n_true]

        # full position in the tensor: incoming axes first, then outgoing axes
        full_index = incoming_index + outgoing_index
        subcircuit_tensor.tensor_array[full_index] = state_slice


def hss_global(handlers, s_max):
    '''
    Heavy State Selection process.
    Shared budget: prod_j |x_j| <= s_max.

    Phase 1: keep the strongest state of every subcircuit.
    Phase 2: global ranking of all remaining states; add the strongest one
             while the product stays within budget.
    '''
    n_sub = len(handlers)
    norms = [TH.compute_state_norms() for TH in handlers]

    kept = [set() for _ in range(n_sub)]

    # strongest state per subcircuit
    for j in range(n_sub):
        top = int(np.argmax(norms[j]))
        kept[j].add(top)

    def product_size():
        size = 1
        for j in range(n_sub):
            size *= len(kept[j])
        return size

    # ranking of all remaining states (globally)
    remaining = []
    for j in range(n_sub):
        for i in range(len(norms[j])):
            if i not in kept[j]:    
                remaining.append((norms[j][i], j, i))

    # ordered for importance
    remaining.sort(key=lambda t: t[0], reverse=True)

    # add states one by one while the product stays within budget
    for norm_value, j, i in remaining:
        kept[j].add(i)
        if product_size() > s_max:
            kept[j].remove(i)

    return [handlers[j].prune_to(sorted(kept[j])) for j in range(n_sub)]


def contract_all(handlers):
    '''
    Contract a list of TensorHandler to a single one.

    naive implementation: by looping on tensors find a pair that shares at least one open cut edge,
    contract them over all their shared edges at once (outgoing side as A, incoming side as B),
    then replace the two with the result. Repeat until one tensor handler is left.
    '''
    remaining = list(handlers)

    while len(remaining) > 1:

        # find one pair that shares at least one open cut edge
        found = None
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                shared = set(remaining[i].cut_edge_ids) & set(remaining[j].cut_edge_ids)
                if shared:
                    found = (i, j, shared)
                    break
            if found is not None:
                break

        if found is None:
            raise RuntimeError("no shared cut edge left, but more than one handler remains")

        i, j, shared = found
        H1, H2 = remaining[i], remaining[j]

        # decide who is A (outgoing) and who is B (incoming) using any shared edge
        any_edge = next(iter(shared))
        if H1.edge_roles[any_edge] == "outgoing":
            A, B = H1, H2
        else:
            A, B = H2, H1

        # contract over ALL shared edges at once (contract_with auto-detects them)
        merged = A.contract_with(B)

        # rebuild the list: keep everyone except the two we contracted, add the result
        remaining = [h for k, h in enumerate(remaining) if k not in (i, j)]
        remaining.append(merged)

    return remaining[0]
