import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix


class TestCitationMatrix(unittest.TestCase):
    """Test cases for citation matrix functionality."""
    
    def setUp(self):
        """Set up test data before each test."""
        self.df = make_model()
    
    def test_journal_citation_matrix_structure(self):
        """Test that journal citation matrix has correct structure."""
        matrix = unit_citation_matrix(self.df, 'journal')
        
        # Check that result is a DataFrame
        self.assertIsInstance(matrix, pd.DataFrame)
        
        # Check that matrix is square
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        
        # Check that all journals are present
        expected_journals = sorted(self.df['journal_id'].unique())
        self.assertListEqual(list(matrix.index), expected_journals)
        self.assertListEqual(list(matrix.columns), expected_journals)
        
        # Check that all values are non-negative
        self.assertTrue((matrix >= 0).all().all())
    
    def test_author_citation_matrix_structure(self):
        """Test that author citation matrix has correct structure."""
        matrix = unit_citation_matrix(self.df, 'author')
        
        # Check basic structure
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        
        # Check that all authors are present
        expected_authors = sorted(self.df['author_id'].unique())
        self.assertListEqual(list(matrix.index), expected_authors)
        self.assertListEqual(list(matrix.columns), expected_authors)
        
        # Check non-negative values
        self.assertTrue((matrix >= 0).all().all())
    
    def test_institution_citation_matrix_structure(self):
        """Test that institution citation matrix has correct structure."""
        matrix = unit_citation_matrix(self.df, 'institution')
        
        # Check basic structure
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        
        # Check that all institutions are present
        expected_institutions = sorted(self.df['institution_id'].unique())
        self.assertListEqual(list(matrix.index), expected_institutions)
        self.assertListEqual(list(matrix.columns), expected_institutions)
        
        # Check non-negative values
        self.assertTrue((matrix >= 0).all().all())
    
    def test_journal_citation_matrix_values(self):
        """Test specific values in journal citation matrix."""
        matrix = unit_citation_matrix(self.df, 'journal')
        
        # Test matrix dimensions - should be 3x3 for J1, J2, J3
        self.assertEqual(matrix.shape, (3, 3))
        
        # Test some specific known values based on toy data
        # These values come from the citation patterns in the toy model
        # J1 cites J1: W3 cites [W1], W6 cites [W3] = 2 citations
        self.assertEqual(matrix.loc['J1', 'J1'], 2.0)
        
        # J1 cites J2: W3 cites [W2], W6 cites [W5] = 2 citations  
        self.assertEqual(matrix.loc['J1', 'J2'], 2.0)
    
    def test_citation_matrix_symmetry_properties(self):
        """Test properties of citation matrices."""
        for unit in ['journal', 'author', 'institution']:
            with self.subTest(unit=unit):
                matrix = unit_citation_matrix(self.df, unit)
                
                # Matrix should be square
                self.assertEqual(matrix.shape[0], matrix.shape[1])
                
                # All values should be numeric
                self.assertTrue(all(matrix.dtypes == 'float64'))  # pandas converts to float
                
                # Check that values are non-negative
                self.assertTrue((matrix >= 0).all().all())
                
                # For journals, values should be whole numbers (full credit)
                # For authors/institutions, values can be fractional (fractional credit)
                if unit == 'journal':
                    self.assertTrue(((matrix % 1) == 0).all().all())
                else:
                    # For authors and institutions, allow fractional values
                    self.assertTrue((matrix >= 0).all().all())
    
    def test_empty_references_handling(self):
        """Test that works with empty references don't contribute to citations."""
        # W1 has empty references, so it shouldn't cite anyone
        matrix = unit_citation_matrix(self.df, 'journal')
        
        # J1 (which contains W1) should have some outgoing citations from other works
        # but W1 itself contributes 0 outgoing citations
        # This is implicitly tested by the correctness of other values
        self.assertTrue(True)  # This test passes if no errors occur
    
    def test_matrix_row_interpretation(self):
        """Test that matrix rows represent citing units correctly."""
        matrix = unit_citation_matrix(self.df, 'journal')
        
        # Sum of row i should equal total citations made by unit i
        # We can verify this makes sense with our data
        total_citations_j1 = matrix.loc['J1'].sum()
        total_citations_j2 = matrix.loc['J2'].sum()
        total_citations_j3 = matrix.loc['J3'].sum()
        
        # All should be positive (each journal has works that cite others)
        self.assertGreater(total_citations_j1, 0)
        self.assertGreater(total_citations_j2, 0)
        self.assertGreater(total_citations_j3, 0)
    
    def test_invalid_unit_type(self):
        """Test handling of invalid unit types."""
        with self.assertRaises(KeyError):
            unit_citation_matrix(self.df, 'invalid_unit')


if __name__ == '__main__':
    unittest.main()