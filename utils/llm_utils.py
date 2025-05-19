import re
import torch
from utils.lattice_utils import classify_nodes_with_geometry
import numpy as np


def parse_graph(text: str):
    """
    Parse a node/edge description like the prompt example and return
    coords  – Tensor[N, 3]  (float32)
    edges   – Tensor[M, 2]  (long)
    """
    # --- split the text into its three sections -----------------------------
    #     1) header with node number (optional for parsing)
    #     2) node coordinates lines
    #     3) edge lines
    #
    # find where the coordinates section starts and where the edges start
    coord_start = text.index("Node coordinates:") + len("Node coordinates:")
    edge_start  = text.index("Edges:")
    
    coord_block = text[coord_start:edge_start].strip().splitlines()
    edge_block  = text[edge_start + len("Edges:"):].strip().splitlines()

    # --- helper: grab all numbers in a line ---------------------------------
    num_pat = re.compile(r'[-+]?\d*\.\d+|\d+')   # floats or ints

    # Parse coordinates -------------------------------------------------------
    coords = [
        [float(x) for x in num_pat.findall(line)]
        for line in coord_block if num_pat.search(line)
    ]
    coords = torch.tensor(coords, dtype=torch.float32)  # → [N, 3]

    # Parse edges -------------------------------------------------------------
    edges = [
        [int(float(x)) for x in num_pat.findall(line)]
        for line in edge_block if num_pat.search(line)
    ]
    edges = torch.tensor(edges, dtype=torch.long)       # → [M, 2]
    

    # Construct other elements:z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms,
    num_atoms = torch.LongTensor([coords.shape[0]])
    edge_index = edges.T
    try:
        node_labels = classify_nodes_with_geometry(coords, edge_index)
        z = torch.LongTensor(torch.argmax(node_labels,dim=-1)+1)

    except:
        Warning("No geometry information, using default node labels")
        z = torch.ones(coords.shape[0], dtype=torch.long)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    lengths = torch.FloatTensor([[1,1,1]])
    angles = torch.FloatTensor([[90,90,90]])

    return z, coords, edge_index, batch, lengths, angles, num_atoms





def graph_to_text(coords: torch.Tensor, edges: torch.Tensor) -> str:
    """
    Transforms node coordinates and edge list tensors into formatted text.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 3) for node coordinates.
        edges (torch.Tensor): Tensor of shape (2, m) for edges (start and end indices).

    Returns:
        str: Formatted text representation.
    """
    n = coords.size(0)
    lines = []
    lines.append(f"Node number: {n}")
    lines.append("Node coordinates:")
    for x, y, z in coords.tolist():
        lines.append(f"({x}, {y}, {z})")
    lines.append("")  # Blank line before edges
    lines.append("Edges:")
    # Assume edges is of shape (2, m)
    start_nodes, end_nodes = edges.tolist()
    for u, v in zip(start_nodes, end_nodes):
        lines.append(f"({u}, {v})")
    return "\n".join(lines)


def construct_supervisor_input(prompt, structure, lattice_lengths=None, lattice_angles=None, properties=None):
    """
    Constructs the input for the supervisor model by combining the prompt,
    structure, and optional properties.

    Args:
        prompt (str): The prompt text.
        structure (str): The structure text.
        properties (str, optional): The properties text. Defaults to None.

    Returns:
        str: The combined input string.
    """
    input_text = f"Prompt: {prompt}\n\nStructure: {structure}\n"
    if lattice_lengths is None and lattice_angles is None:
        input_text += "\nLattice lengths: [1.0, 1.0, 1.0]\nLattice angles: [90.0, 90.0, 90.0]\n"
    else:
        input_text += f"\nLattice lengths: {lattice_lengths}\nLattice angles: {lattice_angles}"
    if properties is not None:
        input_text += "\nProperties:"
        for key, value in properties.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            input_text += f"\n{key}: {value}"
    return input_text




def find_closest_structure(
    dataset,
    improved_properties: torch.Tensor,
    batch_size: int = 10000,   # tune for your GPU / RAM budget
    p: int = 2,                 # 2 = Euclidean, 1 = Manhattan, etc.
):
    query = torch.as_tensor(improved_properties, dtype=torch.float32).view(1, -1)

    # Make sure we run on the same device as the data (or stick to CPU)
    query = query.to(next(iter(dataset)).y.device)

    best_dist = np.inf
    best_idx  = -1

    selected_idx = []

    # Stream through the dataset in chunks
    batch = []
    start_idx = 0
    for i, data in enumerate(dataset):
        batch.append(data.y.view(-1))           # flatten to (12,)
        if len(batch) == batch_size or i == len(dataset) - 1:
            Ys = torch.stack(batch)             # (B, 12) on same device
            diff = Ys - query                   # broadcast (B, 12)
            dists = torch.linalg.vector_norm(diff, ord=p, dim=1)   # (B,)
            min_val, min_pos = torch.min(dists, dim=0)

            if min_val.item() < best_dist:
                best_dist = min_val.item()
                best_idx  = start_idx + min_pos.item()

            # reset for next chunk
            batch.clear()
            start_idx = i + 1

    return [best_idx], best_dist




