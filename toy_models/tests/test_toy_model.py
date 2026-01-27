import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configuration.make_toy_model import make_model


class TestToyModel(unittest.TestCase):
    """Test cases for the toy model data generation."""
    
    def setUp(self):
        """Set up test data before each test."""
        self.df = make_model()
    
    def test_model_structure(self):
        """Test that the toy model has the expected structure."""
        # Check that DataFrame is created
        self.assertIsInstance(self.df, pd.DataFrame)
        
        # Check expected columns
        expected_columns = {'work_id', 'journal_id', 'publication_year', 
                          'referenced_works', 'author_id', 'institution_id'}
        self.assertEqual(set(self.df.columns), expected_columns)
        
        # Check that we have data
        self.assertGreater(len(self.df), 0)
    
    def test_model_data_types(self):
        """Test that data types are as expected."""
        # Check column types
        self.assertTrue(self.df['work_id'].dtype == 'object')
        self.assertTrue(self.df['journal_id'].dtype == 'object')
        self.assertTrue(self.df['publication_year'].dtype == 'int64')
        self.assertTrue(self.df['author_id'].dtype == 'object')
        self.assertTrue(self.df['institution_id'].dtype == 'object')
    
    def test_work_ids(self):
        """Test that work IDs are as expected."""
        unique_works = set(self.df['work_id'].unique())
        expected_works = {'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8'}
        self.assertEqual(unique_works, expected_works)
    
    def test_journal_ids(self):
        """Test that journal IDs are as expected."""
        unique_journals = set(self.df['journal_id'].unique())
        expected_journals = {'J1', 'J2', 'J3'}
        self.assertEqual(unique_journals, expected_journals)
    
    def test_author_ids(self):
        """Test that author IDs are as expected."""
        unique_authors = set(self.df['author_id'].unique())
        expected_authors = {'A1', 'A2', 'A3', 'A4'}
        self.assertEqual(unique_authors, expected_authors)
    
    def test_institution_ids(self):
        """Test that institution IDs are as expected."""
        unique_institutions = set(self.df['institution_id'].unique())
        expected_institutions = {'I1', 'I2', 'I3'}
        self.assertEqual(unique_institutions, expected_institutions)
    
    def test_publication_years(self):
        """Test that publication years are reasonable."""
        years = self.df['publication_year'].unique()
        # Should have years 1, 2, 3
        expected_years = {1, 2, 3}
        self.assertEqual(set(years), expected_years)
    
    def test_unnested_structure(self):
        """Test that authorships are properly unnested."""
        # Check that we have multiple rows for works with multiple authors/institutions
        work_counts = self.df['work_id'].value_counts()
        
        # W1 has 2 authors (A1, A2) so should appear 2 times
        self.assertEqual(work_counts['W1'], 2)
        
        # W4 has 2 authors with A4 having 2 institutions, so should appear 3 times total
        self.assertEqual(work_counts['W4'], 3)
    
    def test_referenced_works_structure(self):
        """Test that referenced_works maintains list structure."""
        # Get unique work-reference combinations
        work_refs = self.df[['work_id', 'referenced_works']].drop_duplicates(subset=['work_id'])
        
        # All referenced_works should be lists
        for _, row in work_refs.iterrows():
            self.assertIsInstance(row['referenced_works'], list)
        
        # Check specific reference patterns
        w6_refs = work_refs[work_refs['work_id'] == 'W6']['referenced_works'].iloc[0]
        expected_w6_refs = ['W3', 'W4', 'W5']
        self.assertEqual(w6_refs, expected_w6_refs)
    
    def test_data_consistency(self):
        """Test that the data is internally consistent."""
        # Check that cited works exist in the dataset
        all_work_ids = set(self.df['work_id'].unique())
        
        for _, row in self.df.iterrows():
            if row['referenced_works']:  # If not empty
                for cited_work in row['referenced_works']:
                    self.assertIn(cited_work, all_work_ids, 
                                f"Work {row['work_id']} cites non-existent work {cited_work}")
    
    def test_model_reproducibility(self):
        """Test that the model produces the same results each time."""
        df1 = make_model()
        df2 = make_model()
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    unittest.main()