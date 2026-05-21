"""Visualization utilities for PyMOL."""

import pymol
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import topoflow.utils.constants as constants
from topoflow.utils.molecules import make_mol_openbabel, mol_to_data


def visualize_data_matplotlib(
    data_object: Data,
    atom_decoder: dict = None,
    dataset_name: str = "crossdock_full",
    save_path: str = None,
    figsize: tuple = (8, 6),
    elev: int = 30,
    azim: int = 45,
    show: bool = True,
):
    """Visualizes ligand and pocket atoms in 3D using matplotlib instead of PyMOL.

    Args:
        data_object (Data): PyTorch Geometric Data object with x_lig, x_pocket, h_lig, h_pocket.
        atom_decoder (dict, optional): Mapping from atom index to element symbol. If None, loaded from dataset_name.
        dataset_name (str): Dataset name used to fetch atom_decoder from constants.
        save_path (str, optional): Path to save the figure. If None, the image is not saved.
        figsize (tuple): Size of the figure.
        elev (int): Elevation angle for the 3D plot.
        azim (int): Azimuth angle for the 3D plot.
        show (bool): Whether to call plt.show().
    """
    if atom_decoder is None and dataset_name is not None:
        atom_decoder = constants.dataset_params[dataset_name]["atom_decoder"]
    elif atom_decoder is None:
        atom_decoder = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]

    # Get elements
    lig_elements = _get_elements_from_x_data(data_object.h_lig, atom_decoder)
    pocket_elements = _get_elements_from_x_data(data_object.h_pocket, atom_decoder)

    # Color map per element
    color_map = {
        "C": "black",
        "N": "blue",
        "O": "red",
        "S": "yellow",
        "F": "green",
        "P": "orange",
        "Cl": "lime",
        "Br": "brown",
        "I": "purple",
        "B": "magenta",
        "Si": "gray",
        "Se": "cyan",
    }

    # 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Ligand and Pocket Visualization (matplotlib)")

    # Plot pocket atoms
    for pos, elem in zip(data_object.x_pocket, pocket_elements):
        x, y, z = pos.cpu().numpy()
        ax.scatter(
            x,
            y,
            z,
            c=color_map.get(elem, "gray"),
            s=20,
            label=f"Pocket-{elem}",
            alpha=0.5,
            edgecolors="k",
        )

    # Plot ligand atoms
    for pos, elem in zip(data_object.x_lig, lig_elements):
        x, y, z = pos.cpu().numpy()
        ax.scatter(
            x,
            y,
            z,
            c=color_map.get(elem, "gray"),
            s=60,
            label=f"Ligand-{elem}",
            edgecolors="k",
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("off")

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(), loc="upper right", fontsize=7, frameon=False
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Matplotlib image saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# 1. Start PyMOL (usually in headless mode for scripting)
pymol.cmd.reinitialize()  # Clear previous states
pymol.cmd.feedback("disable", "all", "everything")  # Suppress verbose output
pymol.cmd.set("internal_gui", 0)  # Disable GUI if running standalone
pymol.cmd.set("ray_trace_frames", 1)  # Ensure ray tracing for quality


# 2. Define your plotting function
def plot_molecular_complex(
    pdb_path: str,
    ligand_path: str,
    output_image_path: str = None,
    session_file_path: str = None,
    dataset_param: dict = None,
    verbose: bool = False,
    pymol_camera: list = None,
    width: int = 1200,
    height: int = 1200,
    scaffold_mask: torch.Tensor = None,
    ligand_mask: torch.Tensor = None,
):
    """Plot a protein-ligand complex in PyMOL.

    Plots a protein-ligand complex in PyMOL programmatically, with custom
    coloring for ligand atoms based on dataset_param, and can save an image or session.

    Args:
        pdb_path (str): Path to the protein PDB file (e.g., the pocket PDB file).
        ligand_path (str): Path to the ligand SDF/MOL2 file.
        output_image_path (str, optional): Path to save the output image (e.g., 'output.png').
                                           If None, no image is saved.
        session_file_path (str, optional): Path to save the PyMOL session (e.g., 'output.pse').
                                           If None, no session is saved.
        dataset_param (dict, optional): Dictionary containing dataset parameters,
                                        specifically 'idx_to_atom_map' and 'colors_dic'
                                        for ligand atom coloring.
        verbose (bool, optional): If True, prints additional information during processing.
        pymol_camera (list, optional): 18-element list representing the PyMOL view matrix.
        width (int): Width of the output image in pixels.
        height (int): Height of the output image in pixels.
        scaffold_mask (torch.Tensor, optional): Boolean mask for scaffold atoms.
        ligand_mask (torch.Tensor, optional): Boolean mask for which ligand atoms to show.
    """
    # Initialize PyMOL and clear previous objects for a fresh start
    pymol.finish_launching()  # Ensure PyMOL's environment is ready
    pymol.cmd.reinitialize()  # Clear previous objects in the PyMOL session

    # Load protein (pocket)
    # The object name in PyMOL will be "protein"
    if verbose:
        print(f"Loading protein from: {pdb_path} as 'protein'")
    pymol.cmd.load(pdb_path, "protein")

    # Load ligand
    # The object name in PyMOL will be "ligand"
    if ligand_path is not None:
        if verbose:
            print(f"Loading ligand from: {ligand_path} as 'ligand'")
        if ligand_path.endswith(".sdf"):
            pymol.cmd.load(ligand_path, "ligand", format="sdf")
        elif ligand_path.endswith(".mol2"):
            pymol.cmd.load(ligand_path, "ligand", format="mol2")
        else:
            # Fallback if extension is not .sdf or .mol2
            pymol.cmd.load(ligand_path, "ligand")

    # Basic Visualization Commands
    pymol.cmd.hide("all")  # Hide everything initially to control what's shown

    # Protein (Pocket) Visualization
    pymol.cmd.show("surface", "protein")  # Show protein as surface
    pymol.cmd.set("transparency", 0.5, "protein")  # Make surface semi-transparent
    pymol.cmd.color("lightblue", "protein")  # Color protein surface

    # Ligand Visualization - Now with custom coloring
    if ligand_path is not None:
        pymol.cmd.show("spheres", "ligand")

        if (
            dataset_param
            and "idx_to_atom_map" in dataset_param
            and "colors_dic" in dataset_param
            and scaffold_mask is None
        ):
            idx_to_atom_map = dataset_param["idx_to_atom_map"]
            colors_dic = dataset_param["colors_dic_pymol"]
            sizes = dataset_param.get("radius_dic", None)  # Optional sizes for spheres
            if verbose:
                print("Applying custom ligand coloring based on dataset_param...")
            for idx, atom_symbol in idx_to_atom_map.items():
                if idx < len(colors_dic):
                    color_hex = colors_dic[idx]
                    pymol.cmd.color(color_hex, f"ligand and elem {atom_symbol}")
                else:
                    pymol.cmd.color("gray", f"ligand and elem {atom_symbol}")
                if idx < len(sizes) and sizes is not None:
                    pymol.cmd.set(
                        "sphere_scale", sizes[idx], f"ligand and elem {atom_symbol}"
                    )
        elif scaffold_mask is not None:
            if verbose:
                print("Applying scaffold/new atom coloring...")
            for i, is_scaffold in enumerate(scaffold_mask):
                color = "0x3333ff" if is_scaffold else "0xffff00"
                pymol.cmd.color(color, f"ligand and index {i+1}")
            sizes = (
                dataset_param.get("radius_dic", [0.3] * 11)
                if dataset_param
                else [0.3] * 11
            )
            pymol.cmd.set("sphere_scale", sizes[0], "ligand")
        else:
            if verbose:
                print("Using default CPK coloring for ligand.")
            pymol.cmd.util.cbag("ligand")

        # Apply ligand mask if provided
        if ligand_mask is not None:
            if verbose:
                print("Applying ligand mask...")
            for i, show in enumerate(ligand_mask):
                if not show:
                    pymol.cmd.hide("everything", f"ligand and index {i+1}")

    # Plot bonds if available
    if ligand_path is not None:
        pymol.cmd.show("sticks", "ligand")
        # Re-apply mask to sticks
        if ligand_mask is not None:
            for i, show in enumerate(ligand_mask):
                if not show:
                    pymol.cmd.hide("sticks", f"ligand and index {i+1}")

        sizes = (
            dataset_param.get("radius_dic", [0.3] * 11) if dataset_param else [0.3] * 11
        )
        pymol.cmd.set("stick_radius", 0.4 * sizes[0], "ligand")

    # Center and zoom on the complex
    if pymol_camera is not None:
        pymol.cmd.set_view(pymol_camera)
    else:
        pymol.cmd.zoom("all", 5.0)  # Zoom to all objects with a padding
        pymol.cmd.center("all")  # Ensure the center of rotation is correct

    pymol.cmd.bg_color("white")

    # Save the image if output_image_path is provided
    if output_image_path:
        pymol.cmd.png(
            output_image_path, width=width, height=height, dpi=300, ray=1
        )  # Ray trace for PNG
        if verbose:
            print(f"Image saved to: {output_image_path}")

    # Save the session if session_file_path is provided
    if session_file_path:
        pymol.cmd.save(session_file_path)
        if verbose:
            print(f"Session saved to: {session_file_path}")

    if verbose:
        print(
            "\nPyMOL visualization setup complete. You can now interact with the window."
        )


def _get_elements_from_x_data(x_data, atom_decoder):
    """Helper function to convert one-hot encoded (or index-based) atom types to element symbols using the provided mapping."""
    if x_data.ndim == 2:
        # If 2D, assume it's one-hot encoded; find the index of the '1'
        atom_indices = np.argmax(x_data, axis=1)
    else:
        # If 1D, assume it's already an array of indices
        atom_indices = x_data
    # Map indices to element symbols, defaulting to 'UNK' for unmapped indices
    return [atom_decoder[idx.item()] for idx in atom_indices]


def save_ligand_to_sdf(
    data_object, file_path: str, atom_decoder: dict = None, verbose: bool = False
):
    """Saves the ligand information from a Data object to an SDF file.

    If the data_object is a Chem.Mol, it saves directly to the file.
    If the data_object is a PyTorch Geometric Data object, it constructs the SDF
    from the ligand's atom positions and types, reconstructing bonds using Open Babel.

    Args:
        data_object (Data or Chem.Mol): A PyTorch Geometric Data object or an RDKit Chem.Mol object containing ligand information.
        file_path (str): Path to save the SDF file.
        atom_decoder (dict, optional): A list that maps atom indices to their chemical symbols.
        verbose (bool, optional): If True, prints additional information during processing.
    """
    if isinstance(data_object, Chem.Mol):
        writer = Chem.SDWriter(file_path)
        try:
            writer.write(data_object)
        except Exception as e:
            if verbose:
                print(
                    f"Warning: RDKit failed to write molecule to SDF: {e}. Attempting fallback..."
                )
            try:
                # Fallback: remove aromaticity and try again
                mol_fallback = Chem.Mol(data_object)
                for bond in mol_fallback.GetBonds():
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        bond.SetBondType(Chem.BondType.SINGLE)
                mol_fallback.UpdatePropertyCache()
                writer.write(mol_fallback)
            except Exception as e2:
                if verbose:
                    print(f"Fallback failed: {e2}. Saving without bonds.")
                # Final fallback: writer.write fails, we could try to construct a simple 0-bond mol but it's hard here
                pass
        writer.close()
        if verbose:
            print(f"Successfully saved ligand to: {file_path}")
        return

    if atom_decoder is None:
        # Default atom map - match crossdock_full
        atom_decoder = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "OTHR"]

    # --- New Logic: Use Open Babel to reconstruct bonds for GNINA ---
    try:
        # data_object is a PyG Data object
        rdmol = make_mol_openbabel(
            data_object.x_lig.cpu().numpy(), data_object.h_lig.cpu(), atom_decoder
        )
        if rdmol is not None:
            # Recursively call with the reconstructed RDKit molecule to save properly
            return save_ligand_to_sdf(rdmol, file_path, atom_decoder, verbose)
    except Exception as e:
        if verbose:
            print(
                f"Warning: Bond reconstruction with Open Babel failed: {e}. Falling back to bond-less SDF."
            )

    # --- Fallback: Bond-less SDF (Legacy) ---
    lig_elements = _get_elements_from_x_data(data_object.h_lig, atom_decoder)
    lig_coords = data_object.x_lig
    num_atoms = len(lig_elements)
    num_bonds = 0  # We don't have bond information from your Data object

    molfile_lines = []
    molfile_lines.append(f"{os.path.basename(file_path).split('.')[0]}_ligand")
    molfile_lines.append(
        f"  Generated by topoflow.utils.visualization.save_ligand_to_sdf"
    )
    molfile_lines.append("")  # Comment line

    molfile_lines.append(
        f"{num_atoms:3d}{num_bonds:3d}  0  0  0  0  0  0  0  0999 V2000"
    )

    # Atom Block
    for i in range(num_atoms):
        x, y, z = lig_coords[i]
        element = lig_elements[i]
        # Map OTHR to Carbon placeholder for format validity
        element_sdf = "C" if element == "OTHR" else element[:3]
        molfile_lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} {element_sdf:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )

    molfile_lines.append("M END")
    molfile_lines.append("$$$$")

    try:
        with open(file_path, "w") as f:
            f.write("\n".join(molfile_lines))
        if verbose:
            print(f"Successfully saved ligand to: {file_path}")
    except IOError as e:
        print(f"Error saving ligand file {file_path}: {e}")


def save_pocket_to_pdb(
    data_object: Data, file_path: str, atom_decoder: dict = None, verbose: bool = False
):
    """Saves the pocket information from a Data object to a PDB file.

    Each atom is treated as a separate ATOM record.
    """
    if atom_decoder is None:
        # Default atom map - match crossdock_full
        atom_decoder = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F", "OTHR"]

    pocket_elements = _get_elements_from_x_data(data_object.h_pocket, atom_decoder)
    pocket_coords = data_object.x_pocket

    pdb_lines = []
    atom_serial = 1  # Start atom numbering from 1

    for i in range(len(pocket_elements)):
        x, y, z = pocket_coords[i]
        element = pocket_elements[i]

        # PDB FIX: Map OTHR to C and ensure element is max 2 chars for columns 77-78
        element_pdb = "C" if element == "OTHR" else element[:2]

        # Atom name can be same as element symbol for simplicity in PDB context
        atom_name = f"{element_pdb:<4s}"
        res_name = "PKT"  # Generic residue name for pocket atoms
        chain_id = "A"
        res_seq = i + 1  # Each atom can be its own "residue" for unique ID
        occupancy = 1.00
        temp_factor = 0.00  # Or some relevant value if available

        pdb_line = (
            f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:3s} {chain_id:1s}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
            f"{element_pdb:>2s}  "
        )
        pdb_lines.append(pdb_line)
        atom_serial += 1

    pdb_lines.append("END")  # Standard PDB file terminator

    try:
        with open(file_path, "w") as f:
            f.write("\n".join(pdb_lines))
        if verbose:
            print(f"Successfully saved pocket to: {file_path}")
    except IOError as e:
        print(f"Error saving pocket file {file_path}: {e}")


def visualize_data(
    data_object: Data,
    image_path: str = None,
    session_path: str = None,
    dataset_name: str = "crossdock_full",
    add_ligand_bonds: bool = True,
    norm_params: dict = None,
    verbose: bool = False,
    pymol_camera: list = None,
    width: int = 1200,
    height: int = 1200,
    scaffold_mask: torch.Tensor = None,
    ligand_mask: torch.Tensor = None,
    explicit_bonds: list = None,
):
    """Visualization function for a Data object containing ligand and pocket information.

    Visualizes a Data object containing ligand and pocket information using PyMOL.
    Saves the visualization as an image and optionally as a PyMOL session.

    Args:
        data_object (Data): A PyTorch Geometric Data object containing ligand and pocket information.
        image_path (str, optional): Path to save the output image (e.g., 'output.png').
                                    If None, no image is saved.
        session_path (str, optional): Path to save the PyMOL session (e.g, 'output.pse').
                                      If None, no session is saved.
        dataset_name (str, optional): Name of the dataset to retrieve parameters from constants.dataset_params.
                                      If None, default parameters are used.
        add_ligand_bonds (bool, optional): If True, adds bonds to the ligand visualization.
                                           If False, only the ligand atoms are shown without bonds.
                                           Defaults to True.
        norm_params (dict, optional): Normalization parameters used for the data object. They are used to unnormalize
                                      the ligand and pocket coordinates before visualization.
        verbose (bool, optional): If True, prints additional information during processing.
        pymol_camera (list, optional): 18-element list representing the PyMOL view matrix.
        width (int): Width of the output image in pixels.
        height (int): Height of the output image in pixels.
        scaffold_mask (torch.Tensor, optional): Boolean mask for scaffold atoms.
        ligand_mask (torch.Tensor, optional): Boolean mask for which ligand atoms to show.
        explicit_bonds (list, optional): List of explicit bonds for the ligand.
    """
    data_object = data_object.to(torch.device("cpu"))  # Ensure data is on CPU for PyMOL

    # Create local copies to avoid in-place modification of original data_object
    x_lig = data_object.x_lig.clone()
    h_lig = data_object.h_lig.clone()
    x_pocket = data_object.x_pocket.clone()
    h_pocket = data_object.h_pocket.clone()

    if norm_params is not None:
        x_lig = x_lig * norm_params["std_pocket"] + norm_params["mean_pocket"]
        x_pocket = x_pocket * norm_params["std_pocket"] + norm_params["mean_pocket"]

    # Apply ligand mask filtering if provided
    if ligand_mask is not None:
        if isinstance(ligand_mask, torch.Tensor):
            ligand_mask = ligand_mask.to("cpu")
        x_lig = x_lig[ligand_mask]
        h_lig = h_lig[ligand_mask]
        if scaffold_mask is not None:
            if isinstance(scaffold_mask, torch.Tensor):
                scaffold_mask = scaffold_mask.to("cpu")
            scaffold_mask = scaffold_mask[ligand_mask]
        # Since we already filtered the atoms, we don't want plot_molecular_complex
        # to try to filter again with original indices
        plot_ligand_mask = None
    else:
        if isinstance(scaffold_mask, torch.Tensor):
            scaffold_mask = scaffold_mask.to("cpu")
        plot_ligand_mask = None

    dataset_params = constants.dataset_params[dataset_name] if dataset_name else None
    # Save ligand and pocket to files
    ligand_file = "temp_ligand.sdf"
    pocket_file = "temp_pocket.pdb"

    if dataset_params is not None:
        atom_decoder = dataset_params.get("atom_decoder", None)

    if add_ligand_bonds:
        rdmol = make_mol_openbabel(
            x_lig,
            h_lig,
            atom_decoder=atom_decoder,
            explicit_bonds=explicit_bonds,
        )
        save_ligand_to_sdf(rdmol, ligand_file, atom_decoder, verbose)
    else:
        # Construct a dummy Data object for save_ligand_to_sdf
        temp_data = Data(x_lig=x_lig, h_lig=h_lig)
        save_ligand_to_sdf(temp_data, ligand_file, atom_decoder, verbose)

    # Construct a dummy Data object for save_pocket_to_pdb
    temp_pocket_data = Data(x_pocket=x_pocket, h_pocket=h_pocket)
    save_pocket_to_pdb(temp_pocket_data, pocket_file, atom_decoder, verbose)

    # Plot the molecular complex
    if image_path and not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if session_path and not os.path.isdir(os.path.dirname(session_path)):
        os.makedirs(os.path.dirname(session_path))

    plot_molecular_complex(
        pocket_file,
        ligand_file,
        image_path,
        session_path,
        dataset_params,
        verbose,
        pymol_camera,
        width=width,
        height=height,
        scaffold_mask=scaffold_mask,
        ligand_mask=plot_ligand_mask,
    )

    # Clean up temporary files
    os.remove(ligand_file)
    os.remove(pocket_file)


def visualize_data_rdkit(
    batched_data,
    image_path=None,
    dataset_name="crossdock_full",
    mols_per_row=4,
    sub_img_size=(300, 300),
):
    """Visualization function for a PyTorch Geometric batch Data object containing ligand information using RDKit."""
    if dataset_name is not None:
        dataset_params = constants.dataset_params[dataset_name]
        atom_decoder = dataset_params.get("atom_decoder", None)
    else:
        raise ValueError("dataset_name must be provided to retrieve parameters.")

    # Each entry in `mol_list` is one RDKit molecule built from batched data using index mask
    mols = []
    unique_ids = torch.unique(batched_data.batch_lig)
    for i in unique_ids:
        mask = batched_data.batch_lig == i
        x_lig = batched_data.x_lig[mask]
        h_lig = batched_data.h_lig[mask]
        mol = make_mol_openbabel(x_lig, h_lig, atom_decoder)
        AllChem.Compute2DCoords(mol)  # Compute 2D coordinates for better visualization
        mols.append(mol)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=[f"Mol {i}" for i in range(len(mols))],
        useSVG=False,
        returnPNG=False,
    )

    if image_path:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img.save(image_path)
        print(f"Image saved to: {image_path}")
    img.show()


def visualize_trajectory(
    data_list: list,
    image_path: str = None,
    dataset_name: str = "crossdock_full",
    add_ligand_bonds: bool = True,
    gif_path: str = None,
    plot_every: int = 1,
    verbose: bool = False,
    pymol_camera: list = None,
    width: int = 1200,
    height: int = 1200,
):
    """Visualizes a list of Data objects representing a trajectory of molecular complexes.

    The model creates a series of images showing the evolution of the ligand and pocket
    over the trajectory by using PyMOL.

    Args:
        data_list (list): A list of PyTorch Geometric Data objects, each containing ligand and pocket information.
        image_path (str, optional): Path to save the output images (e.g., 'output.png').
                                    If None, no images are saved.
        dataset_name (str, optional): Name of the dataset to retrieve parameters from constants.dataset_params.
                                      If None, default parameters are used.
        add_ligand_bonds (bool, optional): If True, adds bonds to the ligand visualization.
                                           If False, only the ligand atoms are shown without bonds.
                                           Defaults to True.
        gif_path (str, optional): Path to save the output GIF file (e.g., 'output.gif').
                                   If None, no GIF is saved.
        plot_every (int, optional): Interval at which to plot the trajectory frames.
                                    Defaults to 1 (plot every frame).
        verbose (bool, optional): If True, prints additional information during processing.
        pymol_camera (list, optional): 18-element list representing the PyMOL view matrix.
        width (int): Width of the output image in pixels.
        height (int): Height of the output image in pixels.
    """
    if not isinstance(data_list, list) or not all(
        isinstance(d, Data) for d in data_list
    ):
        raise ValueError("data_list must be a list of PyTorch Geometric Data objects.")
    if len(data_list) == 0:
        raise ValueError("data_list is empty.")
    if image_path and not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if image_path and image_path.endswith(".png"):
        image_path = image_path[:-4]
    for idx, data_object in tqdm(
        enumerate(data_list), total=len(data_list), desc="Visualizing trajectory"
    ):
        if idx % plot_every != 0 and idx != len(data_list) - 1:
            continue
        data_object = data_object.to(
            torch.device("cpu")
        )  # Ensure data is on CPU for PyMOL
        # Save ligand and pocket to files
        visualize_data(
            data_object,
            image_path=f"{image_path}_{idx}.png",
            session_path=f"{image_path}_{idx}.pse",
            dataset_name=dataset_name,
            add_ligand_bonds=add_ligand_bonds,
            verbose=verbose,
            pymol_camera=pymol_camera,
            width=width,
            height=height,
        )

    if gif_path is not None:
        # Create a GIF from the saved images
        import imageio
        from PIL import Image

        images = []
        for idx in range(len(data_list)):
            if idx % plot_every != 0 and idx != len(data_list) - 1:
                continue
            img_path = f"{image_path}_{idx}.png"
            img_data = imageio.imread(img_path)
            pil_img = Image.fromarray(img_data)
            images.append(pil_img.convert("RGB"))
        imageio.mimsave(gif_path, images, duration=1, disposal=1, loop=1, optimize=True)


if __name__ == "__main__":
    # pdb_file = "pymol_temp_plots/complex_1_pocket.pdb"  # Path to your protein PDB file
    # ligand_file = (
    #     "pymol_temp_plots/complex_1_ligand.sdf"  # Path to your ligand SDF file
    # )
    # output_image = "pymol_temp_plots/complex_1_plot.png"
    # output_session = "pymol_temp_plots/complex_1_session.pse"

    # print("Generating plot...")
    # plot_molecular_complex(pdb_file, ligand_file, output_image, output_session)
    # print("Plotting complete.")

    from rdkit.Chem import AllChem

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccncc1"]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]

    all_coords = []
    all_feats = []
    all_batch = []

    for batch_id, mol in enumerate(mols):
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

        coords, features, batch = mol_to_data(mol, atom_encoder, batch_id)
        all_coords.append(coords)
        all_feats.append(features)
        all_batch.append(batch)

    data = Data(
        x_lig=torch.cat(all_coords, dim=0),  # 3D coords
        h_lig=torch.cat(all_feats, dim=0),  # one-hot features
        batch_lig=torch.cat(all_batch, dim=0),  # batch idx
    )

    visualize_data_rdkit(data)
