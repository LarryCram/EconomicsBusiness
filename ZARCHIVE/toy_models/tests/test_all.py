import unittest
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
from tests.test_toy_model import TestToyModel
from tests.test_citation_pairs import TestCitationPairs
from tests.test_citation_matrix import TestCitationMatrix


def create_test_suite():
    """Create a test suite with all tests."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(TestToyModel))
    suite.addTest(unittest.makeSuite(TestCitationPairs))
    suite.addTest(unittest.makeSuite(TestCitationMatrix))
    
    return suite


if __name__ == '__main__':
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)