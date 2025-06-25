"""Visualization utilities for PyMOL."""

import pymol
import os
import numpy as np
from torch_geometric.data import Data
import rdkit.Chem as Chem

import topoflow.utils.constants as constants
from topoflow.utils.molecules import make_mol_openbabel

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
    """
    # Initialize PyMOL and clear previous objects for a fresh start
    pymol.finish_launching()  # Ensure PyMOL's environment is ready
    pymol.cmd.reinitialize()  # Clear previous objects in the PyMOL session

    # Load protein (pocket)
    # The object name in PyMOL will be "protein"
    print(f"Loading protein from: {pdb_path} as 'protein'")
    pymol.cmd.load(pdb_path, "protein")

    # Load ligand
    # The object name in PyMOL will be "ligand"
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
    pymol.cmd.show(
        "spheres", "ligand"
    )  # Show ligand as spheres (or 'sticks' if preferred)

    if (
        dataset_param
        and "idx_to_atom_map" in dataset_param
        and "colors_dic" in dataset_param
    ):
        idx_to_atom_map = dataset_param["idx_to_atom_map"]
        colors_dic = dataset_param["colors_dic_pymol"]
        sizes = dataset_param.get("radius_dic", None)  # Optional sizes for spheres

        print("Applying custom ligand coloring based on dataset_param...")
        for idx, atom_symbol in idx_to_atom_map.items():
            # Ensure the index exists in colors_dic to prevent IndexError
            if idx < len(colors_dic):
                color_hex = colors_dic[idx]
                # PyMOL selection string: "ligand and elem <atom_symbol>"
                # For example: "ligand and elem C"
                pymol.cmd.color(color_hex, f"ligand and elem {atom_symbol}")
            else:
                print(
                    f"Warning: No color defined for index {idx} ('{atom_symbol}') in colors_dic."
                )
                # Fallback: Use a default color for unmapped elements in ligand if necessary
                pymol.cmd.color("gray", f"ligand and elem {atom_symbol}")
            if idx < len(sizes) and sizes is not None:
                pymol.cmd.set(
                    "sphere_scale", sizes[idx], f"ligand and elem {atom_symbol}"
                )
    else:
        print(
            "dataset_param not provided or missing 'idx_to_atom_map'/'colors_dic'. Using default CPK coloring for ligand."
        )
        # Fallback to PyMOL's default coloring if custom parameters are not available
        pymol.cmd.util.cbag("ligand")  # Color ligand by atom type (CPK)

    # Plot bonds if available
    pymol.cmd.show("sticks", "ligand")  # Show ligand bonds as sticks
    pymol.cmd.set(
        "stick_radius", 0.4 * sizes[0], "ligand"
    )  # Set stick radius for better visibility

    # Center and zoom on the complex
    pymol.cmd.zoom("all", 5.0)  # Zoom to all objects with a padding
    pymol.cmd.center("all")  # Ensure the center of rotation is correct

    # Optional: Set a white background
    pymol.cmd.bg_color("white")

    # Optional: Ray trace for higher quality image (can be slow for complex scenes)
    # Uncomment and adjust dimensions if you need a high-res image for publications
    # pymol.cmd.ray(400, 400) # Ray trace with specified width and height

    # Save the image if output_image_path is provided
    if output_image_path:
        pymol.cmd.png(output_image_path, dpi=300, ray=1)  # Ray trace for PNG
        print(f"Image saved to: {output_image_path}")

    # Save the session if session_file_path is provided
    if session_file_path:
        pymol.cmd.save(session_file_path)
        print(f"Session saved to: {session_file_path}")

    print("\nPyMOL visualization setup complete. You can now interact with the window.")


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


def save_ligand_to_sdf(data_object, file_path: str, atom_decoder: dict = None):
    """Saves the ligand information from a Data object to an SDF file.

    If the data_object is a Chem.Mol, it saves directly to the file.
    If the data_object is a PyTorch Geometric Data object, it constructs the SDF
    from the ligand's atom positions and types.

    Args:
        data_object (Data or Chem.Mol): A PyTorch Geometric Data object or an RDKit Chem.Mol object containing ligand information.
        file_path (str): Path to save the SDF file.
        atom_decoder (dict, optional): A list that maps atom indices to their chemical symbols.
    """
    if isinstance(data_object, Chem.Mol):
        writer = Chem.SDWriter(file_path)
        writer.write(data_object)
        writer.close()
        print(f"Successfully saved ligand to: {file_path}")
        return

    if atom_decoder is None:
        # Default atom map - crucial that this matches your data's encoding!
        atom_decoder = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]

    lig_elements = _get_elements_from_x_data(data_object.x_lig, atom_decoder)
    lig_coords = data_object.pos_lig
    num_atoms = len(lig_elements)
    num_bonds = 0  # We don't have bond information from your Data object

    molfile_lines = []
    molfile_lines.append(f"{os.path.basename(file_path).split('.')[0]}_ligand")
    molfile_lines.append(
        f"  Generated by topoflow.utils.visualization.save_ligand_to_sdf"
    )
    molfile_lines.append("")  # Comment line

    # Counts Line: aaabbblllfcccsssrrrPVVvvv
    molfile_lines.append(
        f"{num_atoms:3d}{num_bonds:3d}  0  0  0  0  0  0  0  0999 V2000"
    )

    # Atom Block
    for i in range(num_atoms):
        x, y, z = lig_coords[i]
        element = lig_elements[i]
        molfile_lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} {element:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )

    molfile_lines.append("M END")
    molfile_lines.append("$$$$")

    try:
        with open(file_path, "w") as f:
            f.write("\n".join(molfile_lines))
        print(f"Successfully saved ligand to: {file_path}")
    except IOError as e:
        print(f"Error saving ligand file {file_path}: {e}")


def save_pocket_to_pdb(data_object: Data, file_path: str, atom_decoder: dict = None):
    """Saves the pocket information from a Data object to a PDB file.

    Each atom is treated as a separate ATOM record.
    """
    if atom_decoder is None:
        # Default atom map - crucial that this matches your data's encoding!
        atom_decoder = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]

    pocket_elements = _get_elements_from_x_data(data_object.x_pocket, atom_decoder)
    pocket_coords = data_object.pos_pocket

    pdb_lines = []
    atom_serial = 1  # Start atom numbering from 1

    for i in range(len(pocket_elements)):
        x, y, z = pocket_coords[i]
        element = pocket_elements[i]
        # Atom name can be same as element symbol for simplicity in PDB context
        atom_name = f"{element:<4s}"
        res_name = "PKT"  # Generic residue name for pocket atoms
        chain_id = "A"
        res_seq = i + 1  # Each atom can be its own "residue" for unique ID
        occupancy = 1.00
        temp_factor = 0.00  # Or some relevant value if available

        pdb_line = (
            f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:3s} {chain_id:1s}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
            f"{element:>2s}  "
        )
        pdb_lines.append(pdb_line)
        atom_serial += 1

    pdb_lines.append("END")  # Standard PDB file terminator

    try:
        with open(file_path, "w") as f:
            f.write("\n".join(pdb_lines))
        print(f"Successfully saved pocket to: {file_path}")
    except IOError as e:
        print(f"Error saving pocket file {file_path}: {e}")


def visualize_data(
    data_object: Data,
    image_path: str = None,
    session_path: str = None,
    dataset_name: str = None,
    add_ligand_bonds: bool = True,
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
    """
    dataset_params = constants.dataset_params[dataset_name] if dataset_name else None
    # Save ligand and pocket to files
    ligand_file = "temp_ligand.sdf"
    pocket_file = "temp_pocket.pdb"

    if dataset_params is not None:
        atom_decoder = dataset_params.get("atom_decoder", None)
    if add_ligand_bonds:
        rdmol = make_mol_openbabel(
            data_object.pos_lig, data_object.x_lig, atom_decoder=atom_decoder
        )
        save_ligand_to_sdf(rdmol, ligand_file, atom_decoder)
    else:
        save_ligand_to_sdf(data_object, ligand_file, atom_decoder)
    save_pocket_to_pdb(data_object, pocket_file, atom_decoder)

    # Plot the molecular complex
    if image_path and not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if session_path and not os.path.isdir(os.path.dirname(session_path)):
        os.makedirs(os.path.dirname(session_path))

    plot_molecular_complex(
        pocket_file, ligand_file, image_path, session_path, dataset_params
    )

    # Clean up temporary files
    # os.remove(ligand_file)
    # os.remove(pocket_file)


# 3. Example Usage:
if __name__ == "__main__":
    pdb_file = "pymol_temp_plots/complex_1_pocket.pdb"  # Path to your protein PDB file
    ligand_file = (
        "pymol_temp_plots/complex_1_ligand.sdf"  # Path to your ligand SDF file
    )
    output_image = "pymol_temp_plots/complex_1_plot.png"
    output_session = "pymol_temp_plots/complex_1_session.pse"

    print("Generating plot...")
    plot_molecular_complex(pdb_file, ligand_file, output_image, output_session)
    print("Plotting complete.")

    # Always call pymol.cmd.quit() when you are entirely done with all PyMOL operations
    # to release resources.
    # pymol.cmd.quit() # Uncomment this if you are done after this single plot

    # Clean up dummy files
    # import shutil
    # shutil.rmtree(temp_dir)
