# Unit Tests for Toy Models

This directory contains comprehensive unit tests for the toy models citation analysis functionality.

## Test Structure

### Test Files

- **`test_toy_model.py`** - Tests for the toy model data generation
- **`test_citation_pairs.py`** - Tests for citation pairs extraction
- **`test_citation_matrix.py`** - Tests for citation matrix creation
- **`test_all.py`** - Test runner for all tests

### Test Coverage

#### Toy Model Tests (`TestToyModel`)
- ✅ Model structure validation
- ✅ Data type checking
- ✅ Entity ID validation (works, journals, authors, institutions)
- ✅ Publication year consistency
- ✅ Authorship unnesting verification
- ✅ Reference structure validation
- ✅ Data consistency and integrity
- ✅ Model reproducibility

#### Citation Pairs Tests (`TestCitationPairs`)
- ✅ Correct structure of citer-cited pairs
- ✅ Accurate citation pair extraction
- ✅ No self-citations validation
- ✅ Empty references handling
- ✅ Single citation edge cases

#### Citation Matrix Tests (`TestCitationMatrix`)
- ✅ Matrix structure for all units (journal, author, institution)
- ✅ Matrix dimensions and labeling
- ✅ Non-negative citation counts
- ✅ Specific value verification
- ✅ Matrix interpretation (rows = citing units, cols = cited units)
- ✅ Error handling for invalid unit types

## Running Tests

### Run All Tests
```bash
cd /home/lc/Projects/INFLUENCE/toy_models
python tests/test_all.py
```

### Run Individual Test Files
```bash
# Test toy model
python -m unittest tests.test_toy_model -v

# Test citation pairs
python -m unittest tests.test_citation_pairs -v

# Test citation matrix
python -m unittest tests.test_citation_matrix -v
```

## Test Data

The tests use the toy model data which includes:
- **8 works** (W1-W8) with realistic citation patterns
- **3 journals** (J1-J3)
- **4 authors** (A1-A4) 
- **3 institutions** (I1-I3)
- **Multiple authorship and affiliation scenarios**

## Key Test Scenarios

### Citation Patterns Tested
- Works with no references (W1)
- Single citations (W2 → W1)
- Multiple citations (W3 → [W1, W2])
- Complex citation networks (W6 → [W3, W4, W5])

### Multi-Unit Scenarios
- Authors with multiple affiliations (A4 → [I1, I3])
- Works with multiple authors
- Cross-institutional collaborations

### Edge Cases
- Empty reference lists
- Missing unit mappings
- Invalid unit types
- Data consistency checks

## Test Results

All **24 tests** pass successfully, ensuring:
- ✅ Data integrity
- ✅ Function correctness
- ✅ Edge case handling
- ✅ Matrix mathematical properties
- ✅ Citation relationship accuracy

## Design Philosophy

The tests are designed to:
1. **Validate the toy model** serves as a reliable test fixture
2. **Ensure citation extraction** correctly handles all data patterns
3. **Verify citation matrices** accurately represent unit-level relationships
4. **Catch regressions** when code changes are made
5. **Document expected behavior** through test assertions

The toy model data is specifically crafted to include various citation patterns and multi-unit scenarios that are common in real bibliometric data, making these tests robust and comprehensive.