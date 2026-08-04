

# THIS FILE CONTAINS THE CLASSES TO HANDLE THE TENSORS OBJECTS NEEDED IN THE TENSORQC FRAMEWORK
# THE OBJECTS ARE THE FOLLOWING:
#
#      SubCircuitTensor  -  class to manipulate the probability tensor for the different configuration of the subcircuit
#      TensorHandler     -  main class to manage contraction operation between tensors:
#                           it can describe original subcircuit, merged tensors, pruned tensors 
#
# the file also contain some helper function to build the above objects



from dataclasses import dataclass
import numpy as np


# ------------------------------------------------------------
# Constant matrices for the tensor contraction of cut wires.
#
# When a wire is cut, that wire is expanded in the Pauli
# basis:  rho = 1/2 ( <I> I + <X> X + <Y> Y + <Z> Z ).
# The outgoing cut is MEASURED in the 3 Pauli bases, the incoming
# cut is PREPARED in a set of eigenstates. 
# 
# These two matrices convert the raw measurement and preparation data 
# into the Pauli-basis coefficients that are actually contracted.
# ------------------------------------------------------------

# PAULI_SIGN_MATRIX  (4 x 8)   -- OUTGOING side of a cut.
# Turns the 8 measured outcome probabilities into the 4 Pauli expectation values
#
# Input packing (columns):  index = base*2 + outcome,  base order [I, Z, X, Y]
#   col 0,1 -> I : outcomes 0,1     col 2,3 -> Z : outcomes 0,1
#   col 4,5 -> X : outcomes 0,1     col 6,7 -> Y : outcomes 0,1
#
# Output (rows): the coefficient of each Pauli operator.
#   row I : p(0) + p(1)  -> total probability I
#   row Z : p(0) - p(1)  -> <Z>   
#   row X : p(0) - p(1)  -> <X>
#   row Y : p(0) - p(1)  -> <Y>
PAULI_SIGN_MATRIX = np.array([
    [1,  1,  0,  0,  0,  0,  0,  0],   # I : p0 + p1   
    [0,  0,  1, -1,  0,  0,  0,  0],   # Z : p0 - p1
    [0,  0,  0,  0,  1, -1,  0,  0],   # X : p0 - p1
    [0,  0,  0,  0,  0,  0,  1, -1],   # Y : p0 - p1
])

# PAULI_TRANSITION_MATRIX  (4 x 4)   -- INCOMING side of a cut.
#
# Input (columns): the 4 prepared eigenstates, order [ |0>, |1>, |+>, |i> ].
# Output (rows): the coefficient of each Pauli operator [I, Z, X, Y].
#   I :  |0> + |1>              
#   Z :  |0> - |1>
#   X :  2|+> - |0> - |1>       
#   Y :  2|i> - |0> - |1>      
PAULI_TRANSITION_MATRIX = np.array([
    [ 1,  1,  0,  0],   # I : b0 + b1
    [ 1, -1,  0,  0],   # Z : b0 - b1
    [-1, -1,  2,  0],   # X : 2 b+ - b0 - b1
    [-1, -1,  0,  2],   # Y : 2 bi - b0 - b1
])


# ------------------------------------------------------------
#  tensor operations for the contraction

def apply_signs_to_outgoing(tensor_array, outgoing_axis):
    '''
    apply the pauli sign matrix on the out cut
    '''
    contraction = np.tensordot(PAULI_SIGN_MATRIX, tensor_array, axes=[1, outgoing_axis])
    contraction = np.moveaxis(contraction, 0, outgoing_axis)
    return contraction


def apply_transition_to_incoming(tensor_array, incoming_axis):
    '''
    apply the pauli transition matrix on the in cut
    '''
    contraction = np.tensordot(PAULI_TRANSITION_MATRIX, tensor_array, axes=[1, incoming_axis])
    contraction = np.moveaxis(contraction, 0, incoming_axis)
    return contraction


# ------------------------------------------------------------
# Tensor classes

@dataclass
class SubCircuitTensor:
    '''
    Raw subcircuit tensor: one axis per cut edge (incoming first, then
    outgoing) plus one last axis with the probability vector over output qubits.

    Incoming axis: dim 4  -> initial state {0,1,+,i}
    Outgoing axis: dim 8  -> (4 bases {I,Z,X,Y}) x (2 outcomes {0,1}),
    State axis:    dim 2^k -> true output qubits only (cut qubits excluded)

    TENSOR DESCRIPTION:

    indexing physical meaning ---> [in_cut_0, .. , in_cut_n, out_cut_0,.., out_cut_m, true_qubit_probability_vector ]
    vector dimensions         ---> [4       , .. , 4       , 8        ,.., 8        , 2^k ]
    '''

    tensor_array:   np.ndarray
    cut_edge_ids:   list                # axis order of the cut edges
    edge_roles:     dict                # edge_id -> "incoming" or "outgoing"
    qubit_bit_ids:  list                # global bit_ids of the local qubits (sorted)

    @property
    def shape(self):
        return self.tensor_array.shape

    def to_TensorHandler(self):
        '''
        Build the tensor handle object of the subcircuit
        '''
        n_qubits = len(self.qubit_bit_ids)
        n_states = self.tensor_array.shape[-1]

        # decode every local state index into its binary representation.
        # state_bits[k] = the bit configuration of state k, one entry per qubit
        state_bits = np.zeros((n_states, n_qubits), dtype=np.uint8)
        powers_of_two = 2 ** np.arange(n_qubits)      

        for k in range(n_states):
            # all bits of state k (binary representation of k)
            state_bits[k, :] = (k // powers_of_two) % 2

        return TensorHandler(
            tensor_array=self.tensor_array,
            state_bits=state_bits,
            cut_edge_ids=list(self.cut_edge_ids),
            edge_roles=dict(self.edge_roles),
            global_qubit_ids=list(self.qubit_bit_ids),
        )


@dataclass
class TensorHandler:
    '''
    class to handle subcircuit tensor at any stage: raw, pruned, or contracted.

    tensor_array shape = [cut axes..., n_states]
        one axis per open cut edge (dim 4 incoming, dim 8 outgoing),
        last axis is the binary state axis.

    state_bits shape = [n_states, n_qubits]
        state_bits[k, c] = bit value of qubit c in the state k.
    '''

    tensor_array:      np.ndarray
    state_bits:        np.ndarray       # [n_states, n_qubits]
    cut_edge_ids:      list             # open cut axes, in axis order
    edge_roles:        dict             # edge_id -> "incoming" | "outgoing"
    global_qubit_ids:  list             

    # ------------------------------------------------------------
    # helpers

    @property
    def n_states(self):
        return self.tensor_array.shape[-1]

    @property
    def is_fully_contracted(self):
        # no cut axis left: the tensor is the final reconstructed output
        return len(self.cut_edge_ids) == 0

    def state_dict(self, k):
        # label of state k: {global_bit_id: bit_value}
        state_row = self.state_bits[k]
        label = {}
        for column in range(len(self.global_qubit_ids)):
            global_bit_id = self.global_qubit_ids[column]
            label[global_bit_id] = int(state_row[column])
        return label




    # ------------------------------------------------------------
    # HSS pruning (local part)

    def compute_state_norms(self):
        '''
        computing the L2 norm of every state across all cut-edge configurations:

            ||p_k|| = sqrt( sum over all cut configs of p[..., k]^2 )
            eg: L2 norm of state [0,0,0...,0] = sqrt( sum( pr([0,0,0....,0]**2))) over all configurations

        Returns a 1D array of length n_states representing the L2 norm of the states.
        '''
        # the tensor has shape (cut_axis_1, ..., cut_axis_m, n_states):
        # all axes are cut configurations, EXCEPT the last one which holds the states. 
        n_axes = self.tensor_array.ndim
        cut_axes = tuple(range(n_axes - 1))     # every axis but the last

        # square, sum over all cut configurations, take the square root:
        # what is left is one norm per state
        squared = self.tensor_array ** 2
        summed_over_cuts = np.sum(squared, axis=cut_axes)
        state_norms = np.sqrt(summed_over_cuts)

        return state_norms

    def prune_to(self, kept_states):
        '''
        for the hss procedure we keep only some specific states 
        so the given state indices (a list/array of original slots).
        '''
        kept_states = np.sort(np.asarray(kept_states))

        # Keep only the selected states 
        state_axis = self.tensor_array.ndim - 1
        pruned_array = np.take(self.tensor_array, kept_states, axis=state_axis)
        pruned_bits = self.state_bits[kept_states, :]

        return TensorHandler(
            tensor_array=pruned_array,
            state_bits=pruned_bits,
            cut_edge_ids=list(self.cut_edge_ids),
            edge_roles=dict(self.edge_roles),
            global_qubit_ids=list(self.global_qubit_ids),
        )



    # ------------------------------------------------------------
    # contraction

    def contract_with(self, other):
        '''
        Contract self (A) with other (B) over ALL shared cut edges.
        For each shared edge, A must be the outgoing side and B the incoming
        side. Returns a new TensorHandler.
        '''

        # find all edges shared between the two tensors
        shared_edges = [e for e in self.cut_edge_ids if e in other.cut_edge_ids]
        if len(shared_edges) == 0:
            raise ValueError("no shared cut edge between the two tensors")

        # check roles: every shared edge must be outgoing in A, incoming in B
        for edge in shared_edges:
            assert self.edge_roles[edge] == "outgoing"
            assert other.edge_roles[edge] == "incoming"

        # merged labels: A first, then B (same order used everywhere below)
        global_qubit_ids = self.global_qubit_ids + other.global_qubit_ids   # qubits involved
        global_cut_edge_ids = [e for e in self.cut_edge_ids if e not in shared_edges] \
                            + [e for e in other.cut_edge_ids if e not in shared_edges]
        global_edge_roles = self.edge_roles | other.edge_roles
        for edge in shared_edges:
            del global_edge_roles[edge]

        # contraction axis
        # axes of the shared edges, in the same edge order for A and B
        axes_A = [self.cut_edge_ids.index(e) for e in shared_edges]     
        axes_B = [other.cut_edge_ids.index(e) for e in shared_edges]

        # bring every shared axis to the Pauli basis:
        #   A side: apply M to each outgoing axis
        #   B side: apply T to each incoming axis
        # one factor 1/2 per contracted edge
        n_shared = len(shared_edges)
        half_factor = 0.5 ** n_shared

        tensor_A_ready = self.tensor_array
        for axis in axes_A:
            tensor_A_ready = apply_signs_to_outgoing(tensor_A_ready, axis)

        tensor_B_ready = half_factor * other.tensor_array
        for axis in axes_B:
            tensor_B_ready = apply_transition_to_incoming(tensor_B_ready, axis)

        # contract shared axes 
        contraction_AB = np.tensordot(tensor_A_ready, tensor_B_ready, axes=[axes_A, axes_B])

        # after tensordot the axis order is:
        #   [ A kept cut axes, A state axis, B kept cut axes, B state axis ]
        n_cut_A = len(self.cut_edge_ids) - n_shared    # A cuts minus the contracted ones
        n_cut_B = len(other.cut_edge_ids) - n_shared   # B cuts minus the contracted ones

        # new positions of the two state axes in contraction_AB
        state_axis_A = n_cut_A
        state_axis_B = n_cut_A + 1 + n_cut_B

        # move both state axes to the end so they become adjacent (A then B)
        contraction_AB = np.moveaxis(contraction_AB, [state_axis_A, state_axis_B], [-2, -1])
        # now: [ A cut axes, B cut axes, A state axis, B state axis ]

        # merge the two state axes into one (cartesian product)
        n_states_A = self.n_states
        n_states_B = other.n_states
        kept_cut_shape = contraction_AB.shape[:-2]
        merged_array = contraction_AB.reshape(kept_cut_shape + (n_states_A * n_states_B,))

        # merged state_bits: cartesian product of the two label tables
        A_rep = np.repeat(self.state_bits, n_states_B, axis=0)
        B_tile = np.tile(other.state_bits, (n_states_A, 1))
        global_state_bits = np.concatenate([A_rep, B_tile], axis=1)

        return TensorHandler(
            tensor_array=merged_array,
            state_bits=global_state_bits,
            cut_edge_ids=global_cut_edge_ids,
            edge_roles=global_edge_roles,
            global_qubit_ids=global_qubit_ids,
        )
