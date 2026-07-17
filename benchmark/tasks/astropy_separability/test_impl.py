import unittest
from impl import separability_matrix


class TestSeparabilityMatrix(unittest.TestCase):

    def test_simple_separable(self):
        """Two independent 1-output models stay separable."""
        left = [[1, 0]]
        right = [[0, 1]]
        result = separability_matrix(left, right)
        expected = [[1, 0, 0, 0],
                    [0, 0, 0, 1]]
        self.assertEqual(result, expected)

    def test_coupled_right(self):
        """Right model that mixes two inputs should preserve that coupling."""
        left = [[1, 0]]
        right = [[1, 1]]  # mixes both inputs
        result = separability_matrix(left, right)
        expected = [[1, 0, 0, 0],
                    [0, 0, 1, 1]]
        self.assertEqual(result, expected)

    def test_multi_output_right(self):
        """Right block with 2 outputs and 2 inputs — separability preserved."""
        left = [[1, 0]]
        right = [[1, 0],
                 [0, 1]]
        result = separability_matrix(left, right)
        expected = [[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        self.assertEqual(result, expected)

    def test_nested_compound_model(self):
        """Exact case from the SWE-bench issue: nested compound preserves structure."""
        left = [[1, 0]]          # 1 output, 2 inputs; only uses input 0
        right = [[1, 0], [0, 1]] # 2 outputs, 2 inputs; separable
        result = separability_matrix(left, right)
        # right[0] = [1,0] → result[1][2]=1, result[1][3]=0
        self.assertEqual(result[1][2], 1)
        self.assertEqual(result[1][3], 0)
        # right[1] = [0,1] → result[2][2]=0, result[2][3]=1
        self.assertEqual(result[2][2], 0)
        self.assertEqual(result[2][3], 1)

    def test_identity_left_right(self):
        """Two identity blocks stacked column-wise."""
        left = [[1]]
        right = [[1]]
        result = separability_matrix(left, right)
        self.assertEqual(result, [[1, 0],
                                   [0, 1]])


if __name__ == "__main__":
    unittest.main()
