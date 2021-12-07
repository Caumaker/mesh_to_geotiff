""" MeshToGeotiff - A fast Python algorithm to convert a 3D mesh into a geotiff

    Takes a selection of surfaces (one or multiple) and converts them into
    (a) a regular grid point cloud, and 
    (b) a geotiff DEM

    Date: 02-Jun-2021
    Author: Jeremy Butler
    Copyright: Maptek Pty Ltd 2021
"""
import numpy as np
import math
import numba as nb
from numba import njit, prange
import rasterio # needs numpy >1.20
from rasterio.crs import CRS
import os
from typing import List
import time
import logging
from dataclasses import dataclass
import sys

@dataclass
class MeshObject:
    """ A simple class to store surface/mesh arrays.
    
    Properties:
    points (np.ndarray): Array of 3D points N*3 (x,y,z), np.float64
    point_colours (np.ndarray): Array of RGBA matching the points (may be empty) N*4 (r,g,b,a), np.uint8
    points (np.ndarray): Array of indices for mesh/surface N*3 (i1,i2,i3), np.uint32
    """
    points: np.ndarray = np.empty((0,3),dtype=np.float64)
    point_colours: np.ndarray = np.empty((0,4),dtype=np.uint8)
    facets: np.ndarray = np.empty((0,3),dtype=np.uint32)

class MeshToGeotiff():
    """ Fast mesh to geotiff and regular point grid converter.
        Converts arrays of vertices(points) and faces(facets) to:
            a) a regular point grid
            b) a geotiff raster DEM

        Args
        ----
        null_val (float64): Default -32767. What to populate array with then a triangle isn't intersected
                        Note: -32767 is used by common software packages for this purpose.
        epsilon (float64): Default 0.008. Error tolerance on point-in-triangle test 
                           (recommend 0.001-0.008 to capture near-line or near-vertex intersections)
        verbose (bool): Output processing logs to console.

        Properties
        ----------
        grid_points (np.ndarray): N*3 (x,y,z) array of regular points as a result of compute_grid()
        grid_colours (np.ndarray): N*4 (r,g,b,a) array colours relating to points as a result of compute_grid()
        null_mask (np.ndarray): N*1 (bool) array of grid points that intersected surface from compute_grid()
        grid_rows (int): count of rows in grid from compute_grid() 
        grid_columns (int): count of columns in grid from compute_grid()
        grid_errors (np.ndarray): mask of points where there were potential computational issues in compute_grid()

        Functions
        ---------
        compute_grid(points, point_colours, facets, grid_distance, combination_strategy): 
                    Calculate grid from input points, facets and combination strategy.
        combine_surfaces(List[Tuple(points, point_colours, facets)]): (staticmethod)
                    Combines multiple surfaces/meshes into single set of arrays
        save_geotiff(save_path_height_map, save_path_rgba_map, coord_sys):
                    Converts current grid to a .tif file
    """
    def __init__(self, null_val:nb.float64=-32767., epsilon:nb.float64=0.008, verbose:bool=False):
        self.null_val = null_val
        self.eps = epsilon
        self.grid_spacing = None
        self.grid_points = None
        self.grid_colours = None
        self.null_mask = None
        self.grid_rows = None
        self.grid_columns = None
        self.grid_errors = None
        if (verbose):                                                        
            self.log_level=logging.DEBUG                                        
        else:                                                                   
            self.log_level=logging.INFO  
        log_format = logging.Formatter('%(message)s')
        self.log = logging.getLogger(__name__)                                  
        self.log.setLevel(self.log_level)                                                                                           
        handler = logging.StreamHandler(sys.stdout)                             
        handler.setLevel(self.log_level)                                        
        handler.setFormatter(log_format)                                        
        self.log.addHandler(handler)                                            

    def compute_grid(self, surface:MeshObject, grid_spacing:np.float64=1., combination_strategy:str='first', use_point_in_line_tests:bool=True):
        """ Computes array of points representing the grid interpolations of the surface.
        
            This version does spatial indexing of triangles within grid coordinates to 
            return iterations computed for point to triangle intersections.
            
            Args
            ----
            surface (MeshObject): MeshObject representing surface with points, faces, optional point colours
            grid_spacing (float64): Default 1.0. Spacing for geotiff (or point) grid
            
            combination_strategy (str): one of 'first', 'lowest', 'highest',
                    how to treat multiple facet intersections over grid point. 'first' is fastest.

            use_point_in_line_tests (bool): Whether to perform additional calculations for safety against points (very) close to edges,
                    or points on edges. This is on by default for added accuracy but slows the process down considerably.
                    When disbaled, the process will be much faster but there is some potential for rare erroneous output points to occur
                    or not be included in the resulting grid.
        
            Returns
            ---
            Results are stored in properties: grid_points, grid_colours, null_mask, grid_rows, grid_columns
        """
        self.log.debug("Begin gridding algorithm")
        start_time = time.time()
        self.grid_spacing = grid_spacing
        # Get combination strategy integer
        combination_strategy = combination_strategy.lower()
        combinations = {'first':0, 'lowest':1, 'highest':-1}
        if combination_strategy not in combinations.keys():
            combination_strategy = 'first'
        combination_id = combinations[combination_strategy]
        temp_time = time.time()
        # Calculate corner origins, rows, columns
        lowerX, lowerY = np.min(surface.points[:,0]), np.min(surface.points[:,1])
        upperX, upperY = np.max(surface.points[:,0]), np.max(surface.points[:,1])
        zmin, zmax = np.min(surface.points[:,2])-self.eps, np.max(surface.points[:,2])+self.eps
        rows = math.ceil((upperY-lowerY)/grid_spacing)+1
        columns = math.ceil((upperX-lowerX)/grid_spacing)+1
        p1, p2, p3 = surface.points[surface.facets[:,0]], surface.points[surface.facets[:,1]], surface.points[surface.facets[:,2]]
        # Get colour averages per facet
        if surface.point_colours is None or surface.point_colours.shape[0] != surface.points.shape[0]:
            # no colours provided, set all to green
            surface.point_colours = np.full((surface.points.shape[0], 4), fill_value=(0,255,0,255), dtype=np.uint8)
        tri_colours = surface.point_colours[surface.facets] # [[r1,g1,b1,a1],[r2,g2,b2,a3],[r3,g3,b3,a3]]
        tri_avg_col = np.floor(np.mean(tri_colours, axis=1)).astype(np.uint8) # [[r,g,b,a],] matching index of facets
        # Preparation math
        v1 = p3 - p1 
        v2 = p2 - p1 
        cp = np.cross(v1, v2)
    
        a, b, c = cp[:, 0], cp[:, 1], cp[:, 2]
        d =  np.einsum("ij,ij->i", cp, p3)
        Area = 0.5 *(-p2[:,1]*p3[:,0] + p1[:,1]*(-p2[:,0] + p3[:,0]) + p1[:,0]*(p2[:,1] - p3[:,1]) + p2[:,0]*p3[:,1])

        # Remove any vertical triangles from further calcs
        non_vertical = np.where(((np.abs(c) > 0.0001) & (np.abs(Area) > 0.001)))
        p1, p2, p3 = p1[non_vertical], p2[non_vertical], p3[non_vertical]
        a, b, c, d = a[non_vertical], b[non_vertical], c[non_vertical], d[non_vertical]
        tri_avg_col = tri_avg_col[non_vertical]
        Area = Area[non_vertical]
        
        # Determine the grid bounds that each triangle could potentially fall inside.
        # This is the _main_ algorithm optimisation pathway to allow a single iteration over triangles.
        # Essentially for each triangle, we know in advance what potential point indices to test for intersecting it.
        # axis=0 row by row comparison (max/min x/y of triangle)
        # (- lowerX/Y) brings into grid index space
        tri_extent_buffer = self.eps
        tri_extent_min_x = np.min((p1[:,0], p2[:,0], p3[:,0]), axis=0) - lowerX - tri_extent_buffer
        tri_extent_min_y = np.min((p1[:,1], p2[:,1], p3[:,1]), axis=0) - lowerY - tri_extent_buffer
        tri_extent_max_x = np.max((p1[:,0], p2[:,0], p3[:,0]), axis=0) - lowerX + tri_extent_buffer
        tri_extent_max_y = np.max((p1[:,1], p2[:,1], p3[:,1]), axis=0) - lowerY + tri_extent_buffer
        # Divide the floats by the spacing and convert to integer to match grid indices
        tri_grid_x_start = np.clip(np.floor((tri_extent_min_x)/grid_spacing).astype(dtype=np.int64), 0, columns)
        tri_grid_x_end =  np.clip(np.ceil((tri_extent_max_x)/grid_spacing).astype(dtype=np.int64), 0, columns)
        tri_grid_y_start = np.clip(np.floor((tri_extent_min_y)/grid_spacing).astype(dtype=np.int64), 0, rows)
        tri_grid_y_end =  np.clip(np.ceil((tri_extent_max_y)/grid_spacing).astype(dtype=np.int64), 0, rows) 
        # Add the real-world offset back to extents for feeding in bounding boxes for skipping non-intersects
        tri_extent_min_x += lowerX
        tri_extent_min_y += lowerY
        tri_extent_max_x += lowerX
        tri_extent_max_y += lowerY
        # Pre-compute things here with numpy that won't change during iterations:
        # s_prefix, t_prefix are the first (static) part of the barycentric test for point in triangle
        # signed parallel distance components
        s_prefix = (p1[:,1]*p3[:,0] - p1[:,0]*p3[:,1])
        t_prefix = (p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0])
        # s_x, t_x are the (static) part of the barycentric test for point in triangle for the x coordinates
        s_x = (p3[:,1] - p1[:,1])
        t_x = (p1[:,1] - p2[:,1])
        # s_y, t_y are the (static) part of the barycentric test for point in triangle for the y coordinates
        s_y = (p1[:,0] - p3[:,0])
        t_y = (p2[:,0] - p1[:,0])   
        # one_over_two_area is a pre-calculation of 1/2*Area which is a static part for the barycentric test for point in tri    
        one_over_two_area = 1/(2*Area)
        self.log.debug(f"Computing triangle pre-calculations and grid references took {time.time() - temp_time}s")
        temp_time = time.time()
        # done: pre-allocated bool array for allowing early-exit optimisation on already computed coordinate indices
        # using pre-allocated arrays allows numba to parallelise without (many) issues combining the results
        done = np.full((rows * columns), False, np.bool_)
        tried = np.full((rows * columns), False, np.bool_)
        
        # Compute the grid array to populate. Doing this now will allow parallel optimisation with numba.
        # Populate in the order to support index = y*cols+x
        x_grid = np.arange(columns, dtype=np.float64)
        y_grid = np.arange(rows, dtype=np.float64)
        x_coords, y_coords = np.meshgrid(x_grid,y_grid, sparse=False, indexing='xy')
        x_coords = (x_coords.flatten() * grid_spacing) + lowerX
        y_coords = (y_coords.flatten() * grid_spacing) + lowerY
        
        # Prepopulate 'null value' array
        z = np.full((x_coords.shape[0]), self.null_val, dtype=np.float64)
        # Create xyz array to be fed to numba for population.
        # Z defaults to our null value so that if the point doesn't fall in a triangle - no further computation needed.
        xyz = np.column_stack((x_coords, y_coords, z))
        
        # Create an array for grid colours, with default value black/transparent (0,0,0,0)
        rgba = np.full((x_coords.shape[0], 4), fill_value=0, dtype=np.uint8)
        self.log.debug(f"Building coordinate grid and output arrays took {time.time() - temp_time}s")
        
        # This function is piped into numba for automatic parallel loop support.
        # The xyz, rgba and done arrays will be populated with new heights, colours and null markers.
        # combination_id = 0 - get first result only
        # combination_id = 1 - get lowest result
        # combination_id = -1 - get highest result
        @njit(cache=True, parallel=True)
        def process(a:nb.float64[:], b:nb.float64[:], c:nb.float64[:], d:nb.float64[:],
                s_prefix:nb.float64[:], s_x:nb.float64[:], s_y:nb.float64[:],
                t_prefix:nb.float64[:], t_x:nb.float64[:], t_y:nb.float64[:], z_min:nb.float64, z_max:nb.float64,
                one_over_two_area:nb.float64[:], rows:nb.int64, cols: nb.int64, xyz:nb.float64[:,:],
                x_start:nb.int64[:], y_start:nb.int64[:], x_end:nb.int64[:], y_end:nb.int64[:], done:nb.boolean[:], eps:nb.float64,
                rgba:nb.uint8[:,:], tri_avg_col:nb.uint8[:,:], combination_id:nb.float64, tried:nb.boolean[:], 
                p1:nb.float64[:,:], p3:nb.float64[:,:], p2:nb.float64[:,:],
                tri_x_coord_min:nb.float64[:], tri_x_coord_max:nb.float64[:], tri_y_coord_min:nb.float64[:], tri_y_coord_max:nb.float64[:],
                null_val:nb.float64, use_point_in_line_tests:nb.bool_):
            
            def point_line_dist(p1, p2, point):
                p1_p2_squareLength = (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1])
                dotProduct = ((point[0] - p1[0])*(p2[0] - p1[0]) + (point[1] - p1[1])*(p2[1] - p1[1])) / p1_p2_squareLength
                if dotProduct <= 1 and dotProduct >= 0:  
                    p_p1_squareLength = (p1[0] - point[0])*(p1[0] - point[0]) + (p1[1] - point[1])*(p1[1] - point[1])
                    return math.sqrt(p_p1_squareLength - dotProduct * dotProduct * p1_p2_squareLength)
                elif dotProduct < 0:
                    return  math.sqrt((point[0] - p1[0])*(point[0] - p1[0]) + (point[1] - p1[1])*(point[1] - p1[1]))
                else:
                    return math.sqrt((point[0] - p2[0])*(point[0] - p2[0]) + (point[1] - p2[1])*(point[1] - p2[1]))

            tolerance = 1+eps
            
            # Iterating the triangle array only once is key to performance here.
            # We can do this because we know in advance which grid may points might fall into which triangles.
            for tri in prange(a.shape[0]):
                for x in prange(x_start[tri], x_end[tri]):
                    for y in prange(y_start[tri], y_end[tri]):
                        # Using grids we can infer the id of the point of interest by y*cols+x
                        idx = y*cols+x
                        point = xyz[idx]
                        # Bounding box test
                        if (point[0] < tri_x_coord_min[tri] or point[0] > tri_x_coord_max[tri] or \
                            point[1] < tri_y_coord_min[tri] or point[1] > tri_y_coord_max[tri]):
                                continue
                            
                        # Since multiple triangles may have been in the boundary zone,
                        # don't reprocess a point if it's already been done (and intersection found).
                        if combination_id == 0 and done[idx] == True:
                            # Have visited, and looking for 'first'. Don't re-process
                            continue
                        
                        z = null_val
                        # Tracking 'tried' and 'done' to mark failures for debugging with tried==1 & done==0;
                        # Excluding all the ones known in advance not to intersect at all (tried==0 & done==0).
                        tried[idx] = True
                        this_matched = False
                        # Does the point actually intersect with this triangle?
                        s = (one_over_two_area[tri] * (s_prefix[tri] + (s_x[tri] * point[0]) + (s_y[tri] * point[1])))
                        if (s<=1) and (s>=0):
                            t = (one_over_two_area[tri] * (t_prefix[tri] + (t_x[tri] * point[0]) + (t_y[tri] * point[1])))
                            if (t>=0) and (t<=1) and s+t <= tolerance:
                                # s+1 < 1 = ignore points on line (or very very close)
                                # I found 1.001 to be a good compromise.
                                # In barycentric: s 0-1 = dist along vector AB, t 0-1 = dist along vector AC
                                # When s+t > 1 it means you've walked past both and crossed vector BC.
                                tempz = (d[tri] - a[tri] * point[0] - b[tri] * point[1]) / c[tri]
                                # Even if we got a match, every so often it's rubbish.
                                # If it's not within the z extents of the triangulation, then it's dismissed.
                                if tempz >= z_min and tempz <= z_max:
                                    z = tempz
                                    this_matched = True
                                    
                        if not this_matched and use_point_in_line_tests:
                            # As the barycentric approach to point-in-triangle can be sensitive
                            # to points on (or extremely close to) edges, 
                            # this slower approach (100% overhead) will test point distance to edges along the triangle
                            # and mark it as sufficient if any edge is <eps.
                            # May cause odd results if a facet is going at a 89.9999d 
                            # slope and the point is ~3mm onto a different facet (or poor surface with overlapping facets).
                            if point_line_dist(p1[tri], p2[tri], point) <= eps:
                                this_matched = True
                            elif point_line_dist(p2[tri], p3[tri], point) <= eps:
                                this_matched = True
                            elif point_line_dist(p3[tri], p1[tri], point) <= eps:
                                this_matched = True
                            if this_matched:
                                z = (d[tri] - a[tri] * point[0] - b[tri] * point[1]) / c[tri]
                                
                        if this_matched:
                            if done[idx] == False or combination_id == 0:
                                # Rarely we get a bad computation - sanity test here if it's within the z range of the input points
                                xyz[idx][2] = z
                                rgba[idx] = tri_avg_col[tri]
                                done[idx] = True
                            else:
                                # We only compare if done, otherwise we'll be comparing with null_val
                                # lowest_multiplier just changes from 'highest'=-1 or 'lowest'=1, where
                                # 1 < 2 = 1 lower than 2.  2*-1 < 1*-1 = 2 is higher than 1.
                                if (z*combination_id) < (xyz[idx][2]*combination_id):
                                    xyz[idx][2] = z
                                    rgba[idx] = tri_avg_col[tri]
                                # Mark this index as done so we don't get back here (if combination_id is 0).
                                done[idx] = True
            # Return populated values
            return xyz, rgba, done, tried   
        
        temp_time = time.time()
        result_xyz, result_rgba, null_mask, result_tried = process(a=a, b=b, c=c, d=d, s_prefix=s_prefix, s_x=s_x, s_y=s_y,
                            t_prefix=t_prefix, t_x=t_x, t_y=t_y, z_min=zmin, z_max=zmax,
                            one_over_two_area=one_over_two_area, rows=rows, cols=columns, xyz=xyz,
                            x_start=tri_grid_x_start, y_start=tri_grid_y_start, x_end=tri_grid_x_end, y_end=tri_grid_y_end,
                            done=done, eps=self.eps, rgba=rgba, tri_avg_col=tri_avg_col, combination_id=combination_id, tried=tried,
                            p1=p1, p2=p2, p3=p3, 
                            tri_x_coord_min=tri_extent_min_x, tri_x_coord_max=tri_extent_max_x, 
                            tri_y_coord_min=tri_extent_min_y, tri_y_coord_max=tri_extent_max_y,
                            null_val=self.null_val, use_point_in_line_tests=use_point_in_line_tests)
        self.log.debug(f"Actual grid processing took {time.time() - temp_time}s")
        self.log.debug(f"Total grid points: {xyz.shape[0]}")
        self.log.debug(f"Valid intersection grid points: {np.count_nonzero(done)}")
        self.log.debug(f"Total gridding process took {time.time() - start_time}s")
        self.grid_points = result_xyz
        self.grid_colours = result_rgba
        self.null_mask = null_mask
        self.grid_rows = rows
        self.grid_columns = columns
        self.grid_errors = np.where(((result_tried == 1) & (null_mask == 0)))

    @staticmethod
    def combine_surfaces(surfaces: List[MeshObject]) -> MeshObject:
        """Takes multiple surfaces and combines them into a single object.

        Args
        ----
        surfaces (List[MeshObject]):
                A list of MeshObject (with points, facets, point_colours)

        Returns
        ---
        MeshObject (combined mesh)
        """
        result = MeshObject()
        for surface in surfaces:
            if surface.point_colours is None or surface.point_colours.shape[0] != surface.points.shape[0]:
                surface.point_colours = np.full((surface.points.shape[0], 4), fill_value=(0,255,0,255), dtype=np.uint8)
        for i in range(len(surfaces)):
            surface = surfaces[i]
            # Append index totals
            temp_facets = surface.facets.copy()
            temp_facets += result.points.shape[0]
            # Append to total arrays
            result.points = np.vstack((result.points, surface.points))
            result.facets = np.vstack((result.facets, temp_facets))
            result.point_colours = np.vstack((result.point_colours, surface.point_colours))
        return result

    def save_geotiff(self, save_path_height_map:str, save_path_rgba_map:str=None, coord_sys:CRS=None):
        """Converts computed grid into a rasterised geotiff.

        NOTE: Requires compute_grid to be run first.
        
        If save paths provided are not writeable, a safe name (1),(2),etc will be tried.
        Refer to the returned save paths rather than the input ones.
        
        Args:
        -----
            save_path_height_map (str): Path to save heightmap tif.
                    If path exists and file can't be deleted, it will be appended with (x) with final path returned.
            save_path_rgba_map (str): (optional) Path to save colour raster tif, based on point colours from computation stage
                    If path exists and file can't be deleted, it will be appended with (x) with final path returned.
            coord_sys (CRS): (optional) CRS representation of coordinate system to write to tif.
                        If none, will be set to WGS84, epsg:4978.

        Returns:
        --------
            Tuple[str,str]: Saved path for heightmap tif, Saved path for coloured rgba tif (None of not provided)

        Raises:
        -------
            ValueError: if compute_grid() has not yet been run
        """
        if not isinstance(self.grid_points, np.ndarray):
            raise ValueError("Must run compute_grid() to generate data to export Geotiff before running save_geotiff()")
        if coord_sys is None:
            coord_sys = CRS.from_epsg(4978)
            self.log.debug(f"Geotiff warning: No CRS provided. Setting to WGS84, epsg:4978")
        self.log.debug("Building Geotiffs")
        try:
            if os.path.exists(save_path_height_map):
                os.remove(save_path_height_map)
        except:
            count = 0
            path_no_ext = os.path.splitext(save_path_height_map)[0]
            path_ext = os.path.splitext(save_path_height_map)[1]
            while os.path.exists(f"{path_no_ext} ({count}){path_ext}"):
                count += 1
            save_path_height_map = f"{path_no_ext} ({count}){path_ext}"
        self.log.debug(f"Setting Geotiff heightmap save path to: {save_path_height_map}")

        start_time = time.time()
        # Origin is top-left, with half-grid offset around the points such that they fall into the centre of each raster square.
        lowerX, upperY = np.min(self.grid_points[:,0]), np.max(self.grid_points[:,1])
        rasterXOrigin = lowerX - self.grid_spacing/2
        rasterYOrigin = upperY + self.grid_spacing/2

        # Raster coordinate system goes from Top left instead of Bottom left,
        # So we need to rotate the matrix (created from reshape).
        # The final result should be a shape of (z, rows, columns) with x,y being
        # relative to the top left as 0,0, moving outward from there.
        # https://stackoverflow.com/questions/63830874/geotiff-raster-mirrored-on-python-basemap
        # https://stackoverflow.com/questions/47930428/how-to-rotate-an-array-by-%C2%B1-180-in-an-efficient-way
        # https://github.com/mapbox/rasterio/issues/1683
        raster_z = np.reshape(self.grid_points[:,2], (self.grid_rows, self.grid_columns))
        # Flip the z ordering around to allow ENU -> ESU, raster matrix coords
        raster_z = np.flip(raster_z, 0)
        
        self.log.debug(f"Writing rasters: w={self.grid_columns} h={self.grid_rows} total pixels={self.grid_rows*self.grid_columns}")
        with rasterio.open(
            save_path_height_map, 
            'w',
            driver='GTiff',
            height=self.grid_rows,
            width=self.grid_columns,
            count=1, # number of bands
            dtype=rasterio.float32, # Must be 32bit max for tif
            crs=coord_sys,
            nodata=self.null_val,
            compress='lzw',
            predictor=3, # https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
            # The transform sets the origin coordinates and the unit per pixel (scale)
            # https://github.com/sgillies/affine
            transform = rasterio.Affine.translation(rasterXOrigin, rasterYOrigin) * rasterio.Affine.scale(self.grid_spacing, -self.grid_spacing)
        ) as dst:
            dst.write(raster_z.astype(rasterio.float32), 1) # Write z-values
            dst.set_band_unit(1, 'meters')
            
        self.log.debug(f"Wrote height raster to: {save_path_height_map}")
        self.log.debug(f"Writing height raster took: {time.time() - start_time}s")

        if save_path_rgba_map is not None:
            try:
                if os.path.exists(save_path_rgba_map):
                    os.remove(save_path_rgba_map)
            except:
                count = 0
                path_no_ext = os.path.splitext(save_path_rgba_map)[0]
                path_ext = os.path.splitext(save_path_rgba_map)[1]
                while os.path.exists(f"{path_no_ext} ({count}){path_ext}"):
                    count += 1
                save_path_rgba_map = f"{path_no_ext} ({count}){path_ext}"
            self.log.debug(f"Setting Geotiff colour map save path to: {save_path_rgba_map}")
            start_time = time.time()

            # Build an RGBA raster, and flip ordering
            red = np.flip(np.reshape(self.grid_colours[:,0], (self.grid_rows, self.grid_columns)), 0)
            green = np.flip(np.reshape(self.grid_colours[:,1], (self.grid_rows, self.grid_columns)), 0)
            blue = np.flip(np.reshape(self.grid_colours[:,2], (self.grid_rows, self.grid_columns)), 0)
            alpha = np.flip(np.reshape(self.grid_colours[:,3], (self.grid_rows, self.grid_columns)), 0)
            with rasterio.open(
                save_path_rgba_map, 
                'w',
                driver='GTiff',
                height=self.grid_rows,
                width=self.grid_columns,
                count=4, # number of bands
                dtype=rasterio.uint8,
                crs=coord_sys,
                compress='lzw', # zstd and lzma were slow to compress and didn't give much better compression.
                predictor=2, # https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
                transform = rasterio.Affine.translation(rasterXOrigin, rasterYOrigin) * rasterio.Affine.scale(self.grid_spacing, -self.grid_spacing)
            ) as dst:
                dst.write(red.astype(rasterio.uint8), 1) # Write red values
                dst.write(green.astype(rasterio.uint8), 2) # Write green values
                dst.write(blue.astype(rasterio.uint8), 3) # Write blue values
                dst.write(alpha.astype(rasterio.uint8), 4) # Write alpha values (0 alpha used to infer null areas)
                dst.set_band_unit(1, 'meters')
                dst.set_band_unit(2, 'meters')
                dst.set_band_unit(3, 'meters')
                dst.set_band_unit(4, 'meters')
            self.log.debug(f"Wrote rgba raster to: {save_path_rgba_map}")
            self.log.debug(f"Writing rgba raster took: {time.time() - start_time}s")
        return (save_path_height_map, save_path_rgba_map)

