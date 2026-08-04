# TensorQC

A Python implementation of circuit cutting for quantum circuits. A large circuit
is partitioned into smaller subcircuits that can be simulated independently; each
subcircuit is evaluated over a fixed set of prepare-and-measure configurations,
and the full result is reconstructed classically through tensor contraction in
the Pauli basis. An optional Heavy State Selection (HSS) step keeps the
reconstruction tractable by discarding low-weight states under a global budget.

The framework is built on top of Qiskit and NumPy and follows the approach
described in *TensorQC: Towards Scalable Distributed Quantum Computing via Tensor
Networks* by Wei Tang and Margaret Martonosi (arXiv:2502.03445).


## Background

Running a large circuit on hardware that does not have enough qubits is often
infeasible. Circuit cutting sidesteps this by severing selected wires: the
circuit splits into subcircuits, each small enough to simulate, and the original
result is recovered by post-processing.

When a wire is cut, the state crossing it is expanded in the Pauli basis,

    rho = 1/2 ( <I> I + <X> X + <Y> Y + <Z> Z ),

which turns a single cut into a sum over local prepare-and-measure settings. The
outgoing end of a cut is measured in the four Pauli bases {I, Z, X, Y}, and the
incoming end is prepared in the four states {|0>, |1>, |+>, |i>}. Each subcircuit
is therefore run across all 4^(n_cuts) local configurations. The resulting
probabilities are stored in a tensor per subcircuit, and contracting these
tensors over their shared cut edges reconstructs the uncut result. Because the
number of retained states can still grow quickly, Heavy State Selection keeps
only the highest-weight states, bounded by a shared budget.

The end-to-end flow is:

    Qiskit circuit -> DAG -> CircuitGraph -> cuts -> SubCircuitGraph
                   -> SubCircuitTensor -> TensorHandler -> (HSS) -> contraction


## Installation

    git clone https://github.com/GiacomoLongaroni/TensorQC_Implementation
    cd TensorQC_Implementation
    pip install qiskit numpy networkx

The modules use relative imports, so they are meant to be used as a package
(add an `__init__.py` and import from the package root).


## Usage

    from qiskit.converters import circuit_to_dag
    from .graph_utils import build_GHZ_circ
    from .QuantumGraph import build_CG_fromDAG, find_candidate_cut_edges
    from .tensors_utils import buildTensor_fromCircuit, hss_global, contract_all

    # 1. build a circuit and convert it into a graph
    qc = build_GHZ_circ(4)
    CG = build_CG_fromDAG(circuit_to_dag(qc))

    # 2. choose cut edges and split into subcircuits
    cuts = find_candidate_cut_edges(CG)[:1]
    subs = CG.build_SubCG_fromCuts(cuts)

    # 3. build one tensor per subcircuit
    handlers = [buildTensor_fromCircuit(CG, s).to_TensorHandler() for s in subs]

    # 4. (optional) prune low-weight states, then contract to the full result
    handlers = hss_global(handlers, s_max=1000)
    result = contract_all(handlers)


## Repository layout

The codebase is organised into four modules, following the pipeline above.

`QuantumGraph.py`
Defines the graph model of a circuit, derived from a Qiskit `DAGCircuit`. It
provides the dataclasses `CircuitBit` (a quantum or classical bit), `CircuitNode`
(an input/output terminal or an operation), `CircuitEdge` (a wire, flagged when
it is a cut), and `SubCircuitGraph` (a fragment produced by cutting). The
`CircuitGraph` class holds the full graph and exposes `build_SubCG_fromCuts`,
which removes the chosen cut edges, computes the connected components, and builds
one `SubCircuitGraph` per component while tracking which cuts are incoming and
which are outgoing. The helper `build_CG_fromDAG` performs the Qiskit-to-graph
conversion, and `find_candidate_cut_edges` proposes valid cut locations (quantum
edges touching a multi-qubit gate, excluding global inputs, outputs, and
measurements).

`graph_utils.py`
Turns a graph fragment back into a runnable Qiskit circuit for a given cut
configuration. It initialises incoming-cut qubits in the required state,
applies the basis rotations needed to measure outgoing-cut qubits, and assembles
the subcircuit with `build_qiskit_subcircuit`. `generate_cut_configurations`
enumerates all 4^(n_cuts) prepare-and-measure settings for a subcircuit, and
`build_GHZ_circ` is a small helper for producing GHZ test circuits.

`Tensors.py`
Contains the tensor layer. Two constant matrices, `PAULI_SIGN_MATRIX` and
`PAULI_TRANSITION_MATRIX`, convert raw measurement outcomes and prepared states
into Pauli-basis coefficients on the outgoing and incoming sides of a cut. The
`SubCircuitTensor` dataclass stores the raw probability tensor for one
subcircuit, with one axis per cut edge (dimension 4 for incoming, dimension 8 for
outgoing basis-and-outcome pairs) and a final axis over the true output qubits.
The `TensorHandler` class manages a tensor at any stage of the pipeline (raw,
pruned, or contracted): it computes per-state L2 norms, prunes to a chosen set of
states with `prune_to`, and contracts two handlers over their shared cuts with
`contract_with`, applying the Pauli matrices and a factor of 1/2 per contracted
edge.

`tensors_utils.py`
Drives the higher-level workflow. `buildTensor_fromCircuit` simulates every
configuration of a subcircuit with Qiskit's statevector engine and fills the
corresponding `SubCircuitTensor`, separating measured cut outcomes from the true
output states. `hss_global` implements Heavy State Selection: it keeps the
strongest state of every subcircuit, then greedily adds the globally strongest
remaining states while the product of kept states stays within the budget
`s_max`. `contract_all` repeatedly contracts pairs of handlers that share an open
cut edge until a single tensor, the reconstructed result, remains.


## Status and limitations

Only theoretical statevector evaluation (`mode="th"`) is implemented; the
shot-based mode is currently a stub. Each qubit wire is assumed to cross a
subcircuit at most once (at most one incoming and one outgoing cut per local
qubit). `contract_all` uses a naive pairing order rather than an optimised
contraction path.


## References

This implementation follows the TensorQC framework and builds on the CutQC
circuit-cutting method that preceded it.

Wei Tang and Margaret Martonosi, "TensorQC: Towards Scalable Distributed Quantum
Computing via Tensor Networks," 2025. arXiv:2502.03445.
https://arxiv.org/abs/2502.03445

Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi,
"CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations,"
ASPLOS '21: Proceedings of the 26th ACM International Conference on Architectural
Support for Programming Languages and Operating Systems, 2021, pp. 473-486.
DOI: 10.1145/3445814.3446758. arXiv:2012.02333.

## License

No license is currently specified. Add one (for example MIT) if the code is
intended to be reused.
