# Equivariant AllSetTransformer: Mathematical Formulation

This document outlines the mathematical principles and architecture of the `EquivariantAllSetTransformerLayer`, which extends the hypergraph message-passing of the AllSetTransformer to be perfectly E(3) or SE(3) equivariant. This is critical for 3D molecular generation (like docking via continuous normalizing flows), ensuring that the model's velocity predictions rotate seamlessly with the input 3D coordinates.

## 1. Core Paradigm: Separation of Features

To guarantee equivariance, the architecture strictly separates the node representation into two distinct components:
- $x_i \in \mathbb{R}^3$: The 3D spatial coordinates of node $i$. This is an **equivariant** vector (if the molecule rotates, $x_i$ rotates).
- $h_i \in \mathbb{R}^d$: The hidden scalar features of node $i$. This is an **invariant** vector (if the molecule rotates, $h_i$ remains constant).

Let $E$ denote the set of hyperedges. For each hyperedge $e \in E$, let $V_e$ denote the set of nodes belonging to that hyperedge.

---

## 2. Vertex to Hyperedge ($V \to E$) Message Passing

The goal of this phase is to aggregate node information to create representations for the hyperedge. We must create an invariant feature vector $h_e$ and a spatial coordinate center $x_e$ for each hyperedge.

### 2.1 Hyperedge Coordinate Center ($x_e$)
The spatial location of a hyperedge is defined as an attention-weighted centroid of its constituent nodes. To ensure equivariance, the attention weights $\alpha_{i \to e}$ are computed *exclusively* from the invariant node features $h_i$:

$$ \alpha_{i \to e} = \text{Softmax}_{i \in V_e} \big( \text{MLP}_{attn}(h_i) \big) $$

$$ x_e = \sum_{i \in V_e} \alpha_{i \to e} x_i $$

*Because $\sum \alpha_{i \to e} = 1$, any translation or rotation $T(x)$ applied to $x_i$ applies linearly to $x_e$, making $x_e$ strictly equivariant.*

### 2.2 Hyperedge Invariant Features ($h_e$)
Now that we have a spatial center $x_e$, we can compute the squared Euclidean distance between each node and the hyperedge center: $d_{ie} = ||x_i - x_e||^2$. 

Since distance is invariant to 3D rotation and translation, we safely inject $d_{ie}$ into the feature aggregation to give the model structural awareness:

$$ m_{i \to e} = \text{MLP}_{msg} \big( h_i \parallel d_{ie} \big) $$

$$ h_e = \sum_{i \in V_e} \alpha_{i \to e} m_{i \to e} $$

---

## 3. Hyperedge to Vertex ($E \to V$) Message Passing

The second phase routes information from the hyperedges back to the individual nodes. Here, the network updates the invariant features $h_i$ and the equivariant coordinates $x_i$.

### 3.1 Node Feature Update ($h_i^{(l+1)}$)
Similar to the $V \to E$ step, we compute attention weights $\beta_{e \to i}$ based on the invariant features. A message is constructed using the hyperedge feature and the invariant distance:

$$ \beta_{e \to i} = \text{Softmax}_{e \in E_i} \big( \text{MLP}_{attn}(h_i \parallel h_e) \big) $$

$$ m_{e \to i} = \text{MLP}_{msg} \big( h_e \parallel d_{ie} \big) $$

$$ h_i^{(l+1)} = h_i^{(l)} + \text{MLP}_{update} \Big( h_i^{(l)} \parallel \sum_{e \in E_i} \beta_{e \to i} m_{e \to i} \Big) $$

### 3.2 Node Coordinate Update ($x_i^{(l+1)}$)
To update the 3D coordinates without breaking equivariance, the network predicts a scalar shift multiplier $\phi_x$ using an MLP acting purely on invariant features. The final coordinate update is then the sum of the relative vectors $(x_i - x_e)$ scaled by $\phi_x$:

$$ \phi_x = \text{MLP}_{coord}\big(h_i^{(l)} \parallel h_e \parallel d_{ie}\big) \in \mathbb{R} $$

$$ \Delta x_{e \to i} = (x_i^{(l)} - x_e) \cdot \phi_x $$

$$ x_i^{(l+1)} = x_i^{(l)} + \sum_{e \in E_i} \Delta x_{e \to i} $$

*Because the update direction is strictly bound to the relative vector $(x_i - x_e)$, the network cannot output arbitrary (non-equivariant) directions. If the entire complex is rotated by matrix $R$, then $(x_i - x_e)$ rotates by $R$, and the resultant $\Delta x$ rotates perfectly by $R$.*

---

## 4. Connection to Continuous Normalizing Flows

In `topoflow`, the vector field model must predict the velocity of the atoms $v(x, t)$. 

With the `EquivariantAllSetTransformer`, the predicted spatial velocity for a node is identically the total coordinate shift applied by the network:

$$ v_{spatial}(x_i) = x_i^{(L)} - x_i^{(0)} $$

Because the network guarantees that $x_i^{(L)}$ rotates with $x_i^{(0)}$, the predicted velocity vector $v_{spatial}$ inherently obeys E(3) equivariance. This removes the burden from the neural network of having to "memorize" 3D orientations through data augmentation (like `RandomRotateTranslate`), leading to massively improved alignment precision during molecular docking.
