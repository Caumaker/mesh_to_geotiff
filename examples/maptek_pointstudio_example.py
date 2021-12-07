""" Example of using the MeshToGeotiff class with surfaces.

    Takes a selection of surfaces from one of the below applications and
    converts them into (a) a regular grid point cloud, and (b) a geotiff DEM

    Compatible with:
    - Maptek PointStudio, PointViewer
    - Maptek Vulcan GeologyCore
    - Maptek BlastLogic

    For more information see https://help.maptek.com/mapteksdk/ 
    and https://pypi.org/project/mapteksdk/ 

    Date: 02-Jun-2021
    Author: Jeremy Butler
    Copyright: Maptek Pty Ltd 2021
"""
from mesh_to_geotiff import MeshObject, MeshToGeotiff
from mapteksdk.project import Project
from mapteksdk.data import Surface, PointSet
from contextlib import ExitStack
from rasterio.crs import CRS
import os

if __name__ == "__main__":
    proj = Project()
    grid_size = 1.0
    try:
        grid_size = float(input("Specify grid size: (default=1.0)"))
    except:
        print(f"Invalid input, using {grid_size}m")

    combination_type = 'first'
    combination_types = ["first", "lowest", "highest"]
    temp_input = input("Specify way to combine multiple surfaces: (default='first', options: 'first', 'lowest', 'highest')")
    if temp_input in combination_types:
        combination_type = temp_input
    
    selection = proj.get_selected()
    with ExitStack() as stack:
        surfaces = [stack.enter_context(proj.read(item)) for item in selection if item.is_a(Surface)]
        mesh_objects = []
        for surface in surfaces:
            mesh_objects.append(MeshObject(surface.points, surface.point_colours, surface.facets))
        if len(mesh_objects) > 0:
            mesh_object = MeshToGeotiff.combine_surfaces(surfaces=mesh_objects)
            coord_sys = None
            # Test each surface for first with a coordinate system defined
            for surface in surfaces:
                try:
                    # Needs mapteksdk 2021.1 or above for reading coordinate system def
                    # Localisation is ignored too - expected use with standard coordinate systems
                    coord_sys = CRS.from_epsg(4978) if surfaces[0].coordinate_system is None else surfaces[0].coordinate_system.crs
                    break
                except:
                    coord_sys = CRS.from_epsg(4978)
                    print(f"Geotiff warning: Object didn't have a CRS. Setting to WGS84, epsg:4978")
            
            long_name = f"Combination {combination_type} of"
            for surface in surfaces:
                long_name += ", " + surface.id.name
            
            #####################################################################
            ### Run the main computation process 
            #####################################################################
            print("Running computation")
            grid_surface = MeshToGeotiff(verbose=True)
            grid_surface.compute_grid(mesh_object, grid_size, combination_type)

            #####################################################################
            ### Example to make PointSets of regular x,y,z grids in PointStudio:
            #####################################################################
            print("Creating pointsets")
            xyz = grid_surface.grid_points
            rgba = grid_surface.grid_colours
            # Get any potential error data raised:
            err = xyz[grid_surface.grid_errors]
            # Only keep points that weren't given a null value, as we're not interested in those in this context:
            xyz = xyz[grid_surface.null_mask] 
            rgba = rgba[grid_surface.null_mask]
            if xyz.any():
                with proj.new_or_edit(f"scrapbook/{long_name}", PointSet, overwrite=True) as pts:
                    pts.points = xyz
                    pts.point_colours = rgba
            if err.any():
                with proj.new_or_edit(f"scrapbook/{long_name}_errors", PointSet, overwrite=True) as pts:
                    pts.points = err
                    pts.point_colours[:] = [255, 0, 0, 255]
             
            #####################################################################            
            ### Example to save Geotiff files:
            #####################################################################
            print("Creating tif outputs")
            save_height_map_as = os.path.join(os.path.dirname(__file__), f"{long_name}_heightmap.tif")
            save_rgba_map_as = os.path.join(os.path.dirname(__file__), f"{long_name}_rgbamap.tif")
            saved_as_heights, saved_as_raster = grid_surface.save_geotiff(save_height_map_as, save_rgba_map_as, coord_sys)
            print(f"Saved DEM to: {saved_as_heights}")
            print(f"Saved raster to: {saved_as_raster}")
            #####################################################################
        else:
            input("Invalid selection. Please select a surface (or surfaces) to run script.")