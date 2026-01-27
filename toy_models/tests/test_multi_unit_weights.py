#!/usr/bin/env python3

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configuration.make_toy_model import make_model
from rankers.multi_unit_ranks import multi_unit_citation_matrix, multi_unit_driver


class TestMultiUnitWeights(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.df = make_model(verbose=False)
    
    def test_default_weights(self):
        """Test that default weights are 1/3 each."""
        result = multi_unit_citation_matrix(self.df)
        weights = result['weights']
        
        expected_weights = {'journal': 1/3, 'author': 1/3, 'institution': 1/3}
        for unit_type in expected_weights:
            self.assertAlmostEqual(weights[unit_type], expected_weights[unit_type], places=10)
        
        # Verify weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
    
    def test_custom_weights_valid(self):
        """Test that custom weights are properly applied."""
        custom_weights = {'journal': 0.5, 'author': 0.3, 'institution': 0.2}
        result = multi_unit_citation_matrix(self.df, weights=custom_weights)
        
        weights = result['weights']
        for unit_type in custom_weights:
            self.assertAlmostEqual(weights[unit_type], custom_weights[unit_type], places=10)
    
    def test_weights_sum_validation(self):
        """Test that weights must sum to 1.0."""
        invalid_weights = {'journal': 0.5, 'author': 0.3, 'institution': 0.3}  # sums to 1.1
        
        with self.assertRaises(ValueError) as context:
            multi_unit_citation_matrix(self.df, weights=invalid_weights)
        
        self.assertIn("must sum to 1.0", str(context.exception))
    
    def test_weights_keys_validation(self):
        """Test that weights must contain exactly the required keys."""
        invalid_weights = {'journal': 0.5, 'author': 0.5}  # missing institution
        
        with self.assertRaises(ValueError) as context:
            multi_unit_citation_matrix(self.df, weights=invalid_weights)
        
        self.assertIn("must contain exactly", str(context.exception))
    
    def test_weight_effect_on_citation_counts(self):
        """Test that different weights produce different citation matrices."""
        # Get matrix with default weights
        result_default = multi_unit_citation_matrix(self.df)
        matrix_default = result_default['matrix']
        
        # Get matrix with custom weights
        custom_weights = {'journal': 0.6, 'author': 0.2, 'institution': 0.2}
        result_custom = multi_unit_citation_matrix(self.df, weights=custom_weights)
        matrix_custom = result_custom['matrix']
        
        # Matrices should be different
        self.assertFalse(matrix_default.equals(matrix_custom))
        
        # Total citations should be the same (conservation principle)
        self.assertAlmostEqual(matrix_default.sum().sum(), matrix_custom.sum().sum(), places=6)
        
        # But the distribution within citation types should be different
        # Get the journal, author, and institution blocks
        journals = [col for col in matrix_default.columns if col.startswith('J_')]
        authors = [col for col in matrix_default.columns if col.startswith('A_')]
        institutions = [col for col in matrix_default.columns if col.startswith('I_')]
        
        journal_default = matrix_default.loc[journals, journals].sum().sum()
        journal_custom = matrix_custom.loc[journals, journals].sum().sum()
        
        # Journal citations should be different between default and custom weights
        self.assertNotAlmostEqual(journal_default, journal_custom, places=6)
    
    def test_driver_weights_parameter(self):
        """Test that multi_unit_driver properly passes weights parameter."""
        custom_weights = {'journal': 0.7, 'author': 0.15, 'institution': 0.15}
        
        result = multi_unit_driver(self.df, weights=custom_weights, verbose=False)
        
        # Check that weights are preserved in result
        self.assertIn('weights', result)
        weights = result['weights']
        
        for unit_type in custom_weights:
            self.assertAlmostEqual(weights[unit_type], custom_weights[unit_type], places=10)
    
    def test_extreme_weights(self):
        """Test edge case with extreme weight distributions."""
        # Give all weight to one type
        extreme_weights = {'journal': 1.0, 'author': 0.0, 'institution': 0.0}
        
        result = multi_unit_citation_matrix(self.df, weights=extreme_weights)
        matrix = result['matrix']
        
        # Only journal-to-journal citations should be non-zero
        # (within the journal block of the matrix)
        journals = [col for col in matrix.columns if col.startswith('J_')]
        authors = [col for col in matrix.columns if col.startswith('A_')]
        institutions = [col for col in matrix.columns if col.startswith('I_')]
        
        # Author-to-author and institution-to-institution blocks should be zero
        author_block = matrix.loc[authors, authors]
        institution_block = matrix.loc[institutions, institutions]
        
        self.assertAlmostEqual(author_block.sum().sum(), 0.0, places=10)
        self.assertAlmostEqual(institution_block.sum().sum(), 0.0, places=10)
        
        # Journal block should be non-zero
        journal_block = matrix.loc[journals, journals]
        self.assertGreater(journal_block.sum().sum(), 0.0)
    
    def test_citation_conservation(self):
        """Test that total citations are conserved regardless of weights."""
        # Get single-unit totals for comparison
        from rankers.single_unit_ranks import unit_citation_matrix
        
        journal_matrix = unit_citation_matrix(self.df, 'journal')
        author_matrix = unit_citation_matrix(self.df, 'author') 
        institution_matrix = unit_citation_matrix(self.df, 'institution')
        
        expected_total = journal_matrix.sum().sum()  # Should be same for all unit types
        
        # Test different weight combinations
        weight_combinations = [
            {'journal': 1/3, 'author': 1/3, 'institution': 1/3},  # default
            {'journal': 0.5, 'author': 0.3, 'institution': 0.2},
            {'journal': 0.8, 'author': 0.1, 'institution': 0.1},
            {'journal': 1.0, 'author': 0.0, 'institution': 0.0},  # extreme
        ]
        
        for weights in weight_combinations:
            with self.subTest(weights=weights):
                result = multi_unit_citation_matrix(self.df, weights=weights)
                matrix = result['matrix']
                total_citations = matrix.sum().sum()
                
                # Total should equal the expected total (within floating point precision)
                self.assertAlmostEqual(total_citations, expected_total, places=10)


if __name__ == '__main__':
    unittest.main()