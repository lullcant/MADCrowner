import torch


import trimesh
import numpy as np
import torch
import sys
import os
import open3d as o3d

# Add project root to path to resolve package imports
# The script is in .../models/SAP/, so we go up 2 levels to get to DCrownFormer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.SAP.src.dpsr import DPSR


def _load_points_normals(ply_path: str):
    """
    Robustly load points and normals from a PLY file.
    Tries PyVista -> Open3D -> Trimesh to maximize compatibility with
    PLY files saved via PyVista (point-only polydata with point_data normals).
    Returns (points: np.ndarray[N,3], normals: np.ndarray[N,3]).
    Raises ValueError on failure.
    """
    # Try PyVista (best match for files saved by PyVista)
    try:
        import pyvista as pv
        mesh = pv.read(ply_path)
        pts = mesh.points
        if pts is not None and pts.size > 0:
            # Normals may be stored under 'Normals' or lowercase depending on writer
            if 'Normals' in mesh.point_data:
                nrm = mesh.point_data['Normals']
            elif 'normals' in mesh.point_data:
                nrm = mesh.point_data['normals']
            else:
                nrm = None
            if nrm is not None and nrm.shape[0] == pts.shape[0]:
                return pts.astype(np.float32), np.asarray(nrm, dtype=np.float32)
            else:
                # If normals missing, still return points and let caller error
                return pts.astype(np.float32), None
    except Exception:
        pass

    # Try Open3D
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals) if pcd.has_normals() else None
        if pts is not None and pts.size > 0:
            return pts.astype(np.float32), (nrms.astype(np.float32) if nrms is not None else None)
    except Exception:
        pass

    # Try Trimesh
    try:
        obj = trimesh.load(ply_path, process=False)
        if isinstance(obj, trimesh.Scene):
            obj = obj.dump(concatenate=True)
        # Handle PointCloud or Trimesh
        if hasattr(obj, 'vertices') and obj.vertices is not None and len(obj.vertices) > 0:
            pts = np.asarray(obj.vertices)
            # vertex_normals exists for Trimesh; for PointCloud may be absent
            nrms = getattr(obj, 'vertex_normals', None)
            if nrms is not None and len(nrms) == len(pts):
                return pts.astype(np.float32), np.asarray(nrms, dtype=np.float32)
            return pts.astype(np.float32), None
    except Exception:
        pass

    raise ValueError(f"Failed to load points/normals from {ply_path}")


def pointnorm_to_psr_npz(ply_file, npz_file, resolution=128):
    """
    Reads a PLY file with point normals, performs Differentiable Poisson
    Surface Reconstruction, and saves the resulting grid as an NPZ file.
    """
    print(f"Loading point cloud from {ply_file}...")
    try:
        vertices, normals = _load_points_normals(ply_file)
    except Exception as e:
        print(f"Error reading PLY: {e}")
        return

    if vertices is None or len(vertices) == 0:
        print(f"Error: No points found in {ply_file}")
        return
    if normals is None or len(normals) != len(vertices):
        print("Error: Normals are missing or mismatched with points.")
        return

    print("Point cloud with normals loaded successfully.")

    # Convert to tensors and add batch dimension
    vertices_tensor = torch.from_numpy(vertices.astype(np.float32))[None]
    print(vertices_tensor.shape)
    vertices_tensor = vertices_tensor/2/1.28+0.5
    normals_tensor = torch.from_numpy(normals.astype(np.float32))[None]

    # Initialize DPSR
    print(f"Initializing DPSR with resolution {resolution}x{resolution}x{resolution}...")
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)

    print("Performing reconstruction...")
    psr_grid = dpsr(vertices_tensor, normals_tensor).squeeze().cpu().numpy()

    print(f"Saving PSR grid to {npz_file}...")
    out_dir = os.path.dirname(npz_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(npz_file, psr=psr_grid)
    print("PSR grid saved successfully.")


if __name__ == '__main__':
    # --- 直接在这里修改文件路径 ---
    input_ply_path = "/mnt/disk1/linda/DCrownFormer/models/SAP/src/original_with_normals_tooth.ply"
    output_npz_path = "./output.npz"
    reconstruction_resolution = 128
    # --------------------------

    if not os.path.exists(input_ply_path):
        print(f"错误: 输入文件不存在，请检查路径: {input_ply_path}")
    else:
        pointnorm_to_psr_npz(input_ply_path, output_npz_path, reconstruction_resolution)
