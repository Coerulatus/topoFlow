import torch
import numpy as np
import math
from topoflow.flow.vector_field_models.allsettransformer import AllSetTransformerVectorField

def get_rotation_matrix(angle_x, angle_y, angle_z):
    """Create a 3D rotation matrix."""
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)
    
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def check_equivariance():
    torch.manual_seed(42)
    device = "cpu"
    
    model = AllSetTransformerVectorField(h_dim=32, num_layers=1, use_gat=True, use_geometric_ast=True)
    model.eval()
    
    # 1. Create a random mock molecule and pocket
    N_lig = 5
    N_poc = 10
    x_lig = torch.randn(N_lig, 14)
    x_pocket = torch.randn(N_poc, 14)
    batch_lig = torch.zeros(N_lig, dtype=torch.long)
    batch_pocket = torch.zeros(N_poc, dtype=torch.long)
    t = torch.tensor([0.5])
    
    # 2. Get original prediction
    with torch.no_grad():
        v_orig = model(x_lig, t, x_pocket, batch_lig, batch_pocket)
    
    # 3. Rotate inputs
    R = get_rotation_matrix(math.pi/4, math.pi/3, math.pi/6)
    
    x_lig_rot = x_lig.clone()
    x_lig_rot[:, :3] = x_lig[:, :3] @ R.T
    
    x_pocket_rot = x_pocket.clone()
    x_pocket_rot[:, :3] = x_pocket[:, :3] @ R.T
    
    # 4. Get prediction on rotated inputs
    with torch.no_grad():
        v_rot = model(x_lig_rot, t, x_pocket_rot, batch_lig, batch_pocket)
        
    # 5. Rotate the original prediction to see what it SHOULD be
    v_orig_rotated = v_orig.clone()
    v_orig_rotated[:, :3] = v_orig[:, :3] @ R.T
    
    # 6. Compare
    print("=== Equivariance Test for AST+GAT ===")
    
    # Spatial velocity error
    spatial_error = torch.mean(torch.norm(v_orig_rotated[:, :3] - v_rot[:, :3], dim=-1)).item()
    print(f"Mean spatial velocity difference (L2 error): {spatial_error:.6f}")
    
    # Cosine similarity between expected and actual velocity vectors
    cos_sim = torch.nn.functional.cosine_similarity(v_orig_rotated[:, :3], v_rot[:, :3], dim=-1).mean().item()
    print(f"Mean cosine similarity between expected and actual: {cos_sim:.6f} (1.0 is perfect, 0.0 is orthogonal)")
    
    # Print some actual vectors
    print("\nExpected Velocity (Atom 0):", v_orig_rotated[0, :3].tolist())
    print("Actual Velocity   (Atom 0):", v_rot[0, :3].tolist())
    
    print("\nConclusion: If cosine similarity is near 0 or random, the model is completely blind to rotation.")
    print("This means the model has to memorize all possible rotations during training, which makes docking (relative geometry) extremely hard to learn.")

if __name__ == "__main__":
    check_equivariance()
