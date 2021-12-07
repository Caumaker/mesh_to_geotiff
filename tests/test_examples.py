from __future__ import print_function
import unittest
import os
import time
class TestClass(unittest.TestCase): 
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "data")

    def load_mesh(self):
        import numpy as np
        points_path = os.path.join(self.test_path, "points.npy")
        facets_path = os.path.join(self.test_path, "facets.npy")
        colours_path = os.path.join(self.test_path, "point_colours.npy")
        return np.load(points_path), np.load(colours_path), np.load(facets_path)
         
    def test_compute(self):
        from mesh_to_geotiff import MeshObject, MeshToGeotiff
        import numpy as np
        mesh = MeshObject(*self.load_mesh())
        gridder = MeshToGeotiff()
        gridder.compute_grid(mesh)

        compare_points = np.load(os.path.join(self.test_path, "test_compare_grid.npy"))
        compare_null_mask = np.load(os.path.join(self.test_path, "test_compare_null_mask.npy"))
        compare_error_mask = np.load(os.path.join(self.test_path, "test_compare_grid_errors.npy"))
        self.assertTrue(np.allclose(gridder.grid_points, compare_points))
        self.assertTrue(np.allclose(gridder.null_mask, compare_null_mask))
        self.assertTrue(np.allclose(gridder.grid_errors, compare_error_mask))

        self.assertEqual(gridder.grid_points.shape[0], 32218)
        self.assertAlmostEquals(np.count_nonzero(gridder.null_mask), 29357, delta=5)
        self.assertEqual(gridder.grid_columns, 181)
        self.assertEqual(gridder.grid_rows, 178)

    def test_compute_none_col(self):
        from mesh_to_geotiff import MeshObject, MeshToGeotiff
        import numpy as np
        pts, _, faces = self.load_mesh()
        mesh = MeshObject(pts, None, faces)
        gridder = MeshToGeotiff()
        gridder.compute_grid(mesh)

        compare_points = np.load(os.path.join(self.test_path, "test_compare_grid.npy"))
        compare_null_mask = np.load(os.path.join(self.test_path, "test_compare_null_mask.npy"))
        compare_error_mask = np.load(os.path.join(self.test_path, "test_compare_grid_errors.npy"))
        self.assertTrue(np.allclose(gridder.grid_points, compare_points))
        self.assertTrue(np.allclose(gridder.null_mask, compare_null_mask))
        self.assertTrue(np.allclose(gridder.grid_errors, compare_error_mask))

        expected_colours = np.full((compare_points.shape[0], 4), fill_value=(0,0,0,0), dtype=np.uint8)
        expected_colours[gridder.null_mask] = np.array([0,255,0,255])
        self.assertTrue(np.array_equiv(gridder.grid_colours, expected_colours))

        self.assertEqual(gridder.grid_points.shape[0], 32218)
        self.assertEqual(gridder.grid_columns, 181)
        self.assertEqual(gridder.grid_rows, 178)

    def test_geotiff_geotiff_and_rgbmap_saves(self):
        from mesh_to_geotiff import MeshObject, MeshToGeotiff
        mesh = MeshObject(*self.load_mesh())
        gridder = MeshToGeotiff()
        gridder.compute_grid(mesh)
        time.sleep(1.5)
        file_dem, file_rgb = gridder.save_geotiff(os.path.join(self.test_path, "temp_dem.tif"), 
                             os.path.join(self.test_path, "temp_rgb.tif"))
        self.assertTrue(os.path.isfile(file_dem))
        self.assertTrue(os.path.isfile(file_rgb))
        self.assertAlmostEqual(os.path.getsize(file_dem), 78215, delta=50)
        self.assertAlmostEqual(os.path.getsize(file_rgb), 35165, delta=50)


    def test_geotiff_geotiff_no_rgbmap_saves(self):
        from mesh_to_geotiff import MeshObject, MeshToGeotiff
        pts, _, faces = self.load_mesh()
        mesh = MeshObject(pts, None, faces)
        gridder = MeshToGeotiff()
        gridder.compute_grid(mesh)
        time.sleep(1)
        file_dem, _ = gridder.save_geotiff(os.path.join(self.test_path, "temp_dem.tif"))
        self.assertTrue(os.path.isfile(file_dem))
        self.assertAlmostEqual(os.path.getsize(file_dem), 78215, delta=50)




if __name__ == '__main__':
    unittest.main()
