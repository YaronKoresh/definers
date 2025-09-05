import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from definers import kmeans_k_suggestions

class TestKmeansKSuggestions(unittest.TestCase):

    def setUp(self):
        self.X_np = np.random.rand(100, 2)

    @patch('definers.shutil.which', return_value=True)
    @patch('definers.KMeans')
    @patch('definers.silhouette_score')
    @patch('definers.davies_bouldin_score')
    @patch('definers.calinski_harabasz_score')
    def test_kmeans_k_suggestions_cpu(self, mock_ch_score, mock_db_score, mock_silhouette_score, mock_kmeans_class, mock_which):
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.inertia_ = 100
        mock_kmeans_class.return_value = mock_kmeans_instance
        mock_silhouette_score.return_value = 0.8
        mock_db_score.return_value = 0.5
        mock_ch_score.return_value = 500

        with patch('builtins.print') as mock_print:
            results = kmeans_k_suggestions(self.X_np, k_range=range(2, 5))
            mock_print.assert_called_with("Warning: CuPy (cuML) is unavailable, falling back to CPU with scikit-learn KMeans.")

        self.assertIn('wcss', results)
        self.assertIn('silhouette_scores', results)
        self.assertIn('davies_bouldin_indices', results)
        self.assertIn('calinski_harabasz_indices', results)
        self.assertIn('final_suggestion', results)
        self.assertEqual(len(results['wcss']), 3)
        self.assertEqual(mock_kmeans_class.call_count, 3)

    @patch('definers.np', new_callable=MagicMock)
    @patch('definers.KMeans')
    def test_kmeans_k_suggestions_gpu(self, mock_kmeans_class, mock_cupy):
        mock_cupy.asarray.return_value = self.X_np
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.inertia_ = 100
        mock_kmeans_class.return_value = mock_kmeans_instance

        with patch('definers.silhouette_score', return_value=0.8), \
             patch('definers.davies_bouldin_score', return_value=0.5), \
             patch('definers.calinski_harabasz_score', return_value=500), \
             patch('builtins.print') as mock_print:
            
            try:
                import cupy
                from cuml.cluster import KMeans
                with patch('importlib.import_module') as mock_import:
                    mock_import.side_effect = [cupy, KMeans]
                    results = kmeans_k_suggestions(self.X_np, k_range=range(2, 5))
                    mock_print.assert_called_with("GPU acceleration with CuPy (cuML) is available and will be used.")
            except ImportError:
                 self.skipTest("CuPy or cuML not available for GPU test.")
        
        self.assertIn('wcss', results)


    def test_small_k_range(self):
        results = kmeans_k_suggestions(self.X_np, k_range=range(2, 3))
        self.assertIn('notes', results)
        self.assertIn("K-range too small", results['notes'])

    def test_logic_for_suggestions(self):
         with patch('definers.KMeans') as mock_kmeans_class, \
             patch('definers.silhouette_score', side_effect=[0.5, 0.8, 0.6]), \
             patch('definers.davies_bouldin_score', side_effect=[0.9, 0.5, 0.7]), \
             patch('definers.calinski_harabasz_score', side_effect=[200, 300, 250]):
            
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.inertia_ = 100
            mock_kmeans_class.return_value = mock_kmeans_instance
            
            results = kmeans_k_suggestions(self.X_np, k_range=range(2, 5))

            self.assertEqual(results['suggested_k_silhouette'], 3)
            self.assertEqual(results['suggested_k_davies_bouldin'], 3)
            self.assertEqual(results['suggested_k_calinski_harabasz'], 3)


if __name__ == '__main__':
    unittest.main()
