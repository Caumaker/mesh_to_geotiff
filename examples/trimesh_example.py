""" Example of using the MeshToGeotiff class with surfaces.

    Takes a selection of surfaces from trimesh and
    converts them into a geotiff DEM

    Date: 02-Jun-2021
    Author: Jeremy Butler
    Copyright: Maptek Pty Ltd 2021
"""
from mesh_to_geotiff import MeshObject, MeshToGeotiff
import trimesh
import os

if __name__ == "__main__":
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/simple_mesh.obj"))
    out_file = f"{os.path.splitext(test_file)[0]}_heightmap.tif"

    mesh = trimesh.load_mesh(test_file, "obj")
    # Note: not providing colours so everything will default to green
    mesh_object = MeshObject(points=mesh.vertices, facets=mesh.faces)
    grid_size = 1.0
    try:
        grid_size = float(input("Specify grid size: (default=1.0)"))
    except:
        print(f"Invalid input, using {grid_size}m")

    combination_type = 'first'
    combination_types = ["first", "lowest", "highest"]
    temp_input = input("Specify way to combine multiple surfaces (or intersections): (default='first', options: 'first', 'lowest', 'highest')")
    if temp_input in combination_types:
        combination_type = temp_input
    
    #####################################################################
    ### Run the main computation process 
    #####################################################################
    print("Running computation")
    grid_surface = MeshToGeotiff(verbose=True)
    # Note: First time running requires numba to compile and cache - may add 15sec overhead once
    grid_surface.compute_grid(mesh_object, grid_size, combination_type, use_point_in_line_tests=False)

    # Note:
    # xyz grid available from grid_surface.grid_points
    # xyz, without null data used for raster:  xyz = xyz[grid_surface.null_mask] 

    #####################################################################            
    ### Example to save Geotiff files:
    #####################################################################
    print("Creating tif output")
    saved_as_heights, _ = grid_surface.save_geotiff(out_file)
    print(f"Saved DEM to: {saved_as_heights}")
    #####################################################################

### Output example:
# Specify grid size: (default=1.0)
# Invalid input, using 1.0m
# Specify way to combine multiple surfaces (or intersections): (default='first', options: 'first', 'lowest', 'highest')
# Running computation
# Begin gridding algorithm
# Computing triangle pre-calculations and grid references took 0.01695847511291504s
# Building coordinate grid and output arrays took 0.001127481460571289s
# Total grid points: 32218
# Valid intersection grid points: 29357
# Total gridding process took 0.38840675354003906s
# Creating tif output
# Geotiff warning: No CRS provided. Setting to WGS84, epsg:4978
# Building Geotiffs
# Setting Geotiff heightmap save path to: data\simple_mesh_heightmap.tif
# Writing rasters: w=181 h=178 total pixels=32218
# Wrote height raster to: data\simple_mesh_heightmap.tif
# Writing height raster took: 0.011513948440551758s
# Saved DEM to: f:\Dropbox\Dev\mesh_to_geotiff\data\simple_mesh_heightmap.tif