# MeshToGeotiff - A fast Python algorithm to convert a 3D mesh into a GeoTIFF
Python class for converting (very fast) 3D Meshes/Surfaces to Raster DEMs (as Geotiff) or regular point cloud grids.
- Supports gridding overlapping surfaces (e.g. highest, lowest, or first result)
- Supports output to regular x,y,z grid
- Supports output to Geotiff DEMs
- Supports point colour averaging (and outputting colour raster to TIF with heightmap)

![Example of output](/img/example.jpg "Example of output, combining multiple meshes")

## Motivation
I worked on a project that used polygonal meshes and wanted to integrate parts that relied on raster computations.
There was a need to shift between the two paradigms without having to wait too long for conversion.

I couldn't find anything fast enough in Python to seamlessly transition between mesh and rasters.
This uses Numba for parallel loops and has been heavily optimised for computation speed (intended to compete with c++ benchmarks for similar computations).

The benchmarks below indicate speeds expected on an average PC (at least for 3D processing purposes).

# Installation Instructions
### With `pip`
**Requires rasterio (which also needs gdal). These libraries are easier installed from pre-compiled wheels.**

You will need [rasterio](https://github.com/rasterio/rasterio) and [gdal](https://github.com/OSGeo/gdal).
The easiest way to install these will be a pre-compiled versions for your platform from:
 - Gdal: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
 - Rasterio: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
 - Install the gdal wheel first, then rastio wheel, as shown below
 
Python 3.7, Windows, amd64:
```
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/GDAL-3.3.3-cp37-cp37m-win_amd64.whl
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/rasterio-1.2.10-cp37-cp37m-win_amd64.whl
```

Python 3.8, Windows, amd64:
```
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/GDAL-3.3.3-cp38-cp38-win_amd64.whl
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/rasterio-1.2.10-cp38-cp38-win_amd64.whl
```

Python 3.9, Windows, amd64:
```
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/GDAL-3.3.3-cp39-cp39-win_amd64.whl
python -m pip install https://download.lfd.uci.edu/pythonlibs/w6tyco5e/rasterio-1.2.10-cp39-cp39-win_amd64.whl
```

With those satisfied it should be fine to pip install this:
```
python -m pip install git+https://github.com/jeremybutlermaptek/mesh_to_geotiff
```
or
```
python -m pip install https://github.com/jeremybutlermaptek/mesh_to_geotiff/raw/main/dist/mesh_to_geotiff-0.1.0-py3-none-any.whl
```

**When running examples, if you see this error:**
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

Update numpy to 1.2 and/or reinstall:
```
python -m pip install --upgrade --force-reinstall numpy
```

**To create a wheel for this:**
```
python setup.py bdist_wheel
```

# Usage
See examples:
 - [Trimesh](/examples/trimesh_example.py)
 - [Maptek PointStudio](/examples/maptek_pointstudio_example.py)

```python
from mesh_to_geotiff import MeshObject, MeshToGeotiff
import trimesh
mesh = trimesh.load_mesh("input_mesh.obj", "obj")
# Not providing colours will default to green.. colours are optional
mesh_object = MeshObject(points=mesh.vertices, point_colours=None, facets=mesh.faces)
grid_surface = MeshToGeotiff(verbose=True)

print("Calculating grid")
print("Note: First time running requires numba to compile and cache - may add 15sec overhead once")
grid_surface.compute_grid(mesh_object, grid_spacing=1.0)

print("Creating tif outputs")
# Exporting RGB map is optional, shown for example purposes
saved_as_heights, save_as_rgba = grid_surface.save_geotiff("export_heightmap.tif", "export_rgbmap.tif")
print(f"Saved DEM to: {saved_as_heights}")
print(f"Saved RGB to: {save_as_rgba}")

valid_xyzs = grid_surface.grid_points[grid_surface.null_mask]
print(valid_xyzs)

```

# Benchmarks
**Note:** Upon the first run, Numba must compile/cache. This can add 15~ seconds to the first-time run that will disappear after that.
Benchmarks are after this has happened once.
Test surface:
 - Points: 194,114
 - Facets: 384,874
 - Surface area: 728,550m² (Bounding: approx 1,240m x 950m)
 - CPU: AMD Ryzen 5 3600; RAM: 64gb
 
![Initial surface](/img/initial_suface.png "The initial surface")

![DEM tif output](/img/dem.png "The DEM tif output (looks the same at this zoom level regardless of density)")

**Gridding to 1m:**
 - Total time to grid: 0.82sec
 - Time to save geotiff: 0.06sec
 - Tif size: 1.6mb
 - Total raster cells: 1,185,532 

![1m grid](/img/1m_grid.png "1m grid")

**Gridding to 0.5m:**
 - Total time to grid: 0.97sec
 - Time to save geotiff: 0.13sec
 - Tif size: 5.76mb
 - Total raster cells: 4,735,248

![0.5m grid](/img/0.5m_grid.png "0.5m grid")

**Gridding to 0.1m:**
 - Total time to grid: 4.93sec
 - Time to save geotiff: 6.3sec
 - Tif size: 75mb
 - Total raster cells: 118,259,020

![0.1m grid](/img/0.1m_grid.png "0.1m grid")

**Gridding to 0.05m:**
 - Total time to grid: 19.99sec
 - Time to save geotiff: 24.5sec
 - Tif size: 223mb
 - Total raster cells: 472,973,166

![0.05m grid](/img/0.05m_grid.png "0.05m grid")
