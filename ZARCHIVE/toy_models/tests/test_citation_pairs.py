import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configuration.make_toy_model import make_model
from configuration.citation_pairs import create_citer_cited_df


class TestCitationPairs(unittest.TestCase):
    """Test cases for citation pairs extraction functionality."""
    
    def setUp(self):
        """Set up test data before each test."""
        self.df = make_model()
        self.original_df = self.df[['work_id', 'referenced_works']].drop_duplicates(subset=['work_id'])
    
    def test_create_citer_cited_df_structure(self):
        """Test that create_citer_cited_df returns correct structure."""
        result = create_citer_cited_df(self.original_df)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that it has the correct columns
        expected_columns = ['citer_id', 'cited_id']
        self.assertListEqual(list(result.columns), expected_columns)
        
        # Check that all values are strings (work IDs)
        self.assertTrue(all(isinstance(val, str) for val in result['citer_id']))
        self.assertTrue(all(isinstance(val, str) for val in result['cited_id']))
    
    def test_create_citer_cited_df_content(self):
        """Test that create_citer_cited_df produces correct citation pairs."""
        result = create_citer_cited_df(self.original_df)
        
        # Expected number of citation pairs based on toy data
        # W2->W1, W3->[W1,W2], W4->[W1,W3], W5->W2, W6->[W3,W4,W5], W7->[W4,W6], W8->[W1,W5]
        # Total: 1 + 2 + 2 + 1 + 3 + 2 + 2 = 13 pairs
        expected_count = 13
        self.assertEqual(len(result), expected_count)
        
        # Test some specific citation pairs
        citation_pairs = set(zip(result['citer_id'], result['cited_id']))
        
        # Check some known citations from the toy data
        self.assertIn(('W2', 'W1'), citation_pairs)
        self.assertIn(('W3', 'W1'), citation_pairs)
        self.assertIn(('W3', 'W2'), citation_pairs)
        self.assertIn(('W4', 'W1'), citation_pairs)
        self.assertIn(('W4', 'W3'), citation_pairs)
        self.assertIn(('W6', 'W3'), citation_pairs)
        self.assertIn(('W6', 'W4'), citation_pairs)
        self.assertIn(('W6', 'W5'), citation_pairs)
    
    def test_create_citer_cited_df_no_self_citations(self):
        """Test that works don't cite themselves."""
        result = create_citer_cited_df(self.original_df)
        
        # Check that no work cites itself
        self_citations = result[result['citer_id'] == result['cited_id']]
        self.assertEqual(len(self_citations), 0)
    
    def test_create_citer_cited_df_empty_references(self):
        """Test handling of works with empty reference lists."""
        # Create test data with empty references
        test_data = pd.DataFrame({
            'work_id': ['W1', 'W2', 'W3'],
            'referenced_works': [[], ['W1'], []]
        })
        
        result = create_citer_cited_df(test_data)
        
        # Should only have one citation pair: W2 -> W1
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['citer_id'], 'W2')
        self.assertEqual(result.iloc[0]['cited_id'], 'W1')
    
    def test_create_citer_cited_df_single_citation(self):
        """Test handling of a single citation."""
        test_data = pd.DataFrame({
            'work_id': ['W1'],
            'referenced_works': [['W0']]
        })
        
        result = create_citer_cited_df(test_data)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['citer_id'], 'W1')
        self.assertEqual(result.iloc[0]['cited_id'], 'W0')


if __name__ == '__main__':
    unittest.main()