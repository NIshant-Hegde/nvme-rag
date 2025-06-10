#!/usr/bin/env python3
"""
Test runner for the RAG pipeline components
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all test suites"""
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    
    # Load test suites
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success

def run_specific_test(test_name):
    """Run a specific test module"""
    if test_name.endswith('.py'):
        test_name = test_name[:-3]
    
    try:
        module = __import__(test_name)
        suite = unittest.TestLoader().loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    except ImportError as e:
        print(f"Could not import test module {test_name}: {e}")
        return False

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()