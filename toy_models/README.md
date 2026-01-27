# Toy Models for Citation Analysis

This package provides toy models and citation analysis functionality for testing influence ranking algorithms.

## Project Structure

```
toy_models/
├── main.py                     # Main entry point
├── configuration/              # Data generation and processing
│   ├── __init__.py
│   ├── make_toy_model.py      # Toy data generation
│   └── citation_pairs.py      # Citation pair extraction
├── rankers/                   # Ranking algorithms
│   └── single_unit_ranks.py   # Unit-level citation analysis
└── tests/                     # Comprehensive unit tests
    ├── test_toy_model.py
    ├── test_citation_pairs.py
    ├── test_citation_matrix.py
    ├── test_all.py
    └── README.md
```

## Features

### Toy Model Data Generation
- **JSON Lines format** with realistic citation patterns
- **Multi-unit scenarios**: authors with multiple affiliations, collaborative works
- **8 works, 3 journals, 4 authors, 3 institutions**
- **Automatic unnesting** of authorships and institutions

### Citation Analysis
- **Citation pair extraction** using pandas explode for efficiency
- **Unit-level citation matrices** for journals, authors, and institutions
- **Flexible matrix computation** where element (i,j) = citations from unit i to unit j
- **Support for complex multi-unit relationships**

### Quality Assurance
- **24 comprehensive unit tests** with 100% pass rate
- **Edge case coverage**: empty references, multi-affiliations, data consistency
- **Regression protection** for future development

## Usage

### Basic Usage
```python
from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix, unit_driver

# Generate toy data
data = make_model()

# Create citation matrix for a specific unit
journal_matrix = unit_citation_matrix(data, 'journal')

# Run full analysis pipeline
results = unit_driver(data)
```

### With Verbose Output
```python
# Enable detailed output
data = make_model(verbose=True)
results = unit_driver(data, verbose=True)
```

### Running the Demo
```bash
cd toy_models
python main.py
```

## Testing

### Run All Tests
```bash
python tests/test_all.py
```

### Run Individual Test Suites
```bash
python -m unittest tests.test_toy_model -v
python -m unittest tests.test_citation_pairs -v  
python -m unittest tests.test_citation_matrix -v
```

## Data Model

### Input Format (JSON Lines)
```json
{"work_id": "W1", "journal_id": "J1", "publication_year": 1, "referenced_works": [], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}, {"author_id": "A2", "institution_id": ["I2"]}]}
{"work_id": "W2", "journal_id": "J2", "publication_year": 1, "referenced_works": ["W1"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}]}
```

### Output Format (Unnested DataFrame)
```
  work_id journal_id  publication_year referenced_works author_id institution_id
0      W1         J1                 1               []        A1             I1
1      W1         J1                 1               []        A2             I2
2      W2         J2                 1             [W1]        A3             I3
```

## Citation Matrix Interpretation

For a citation matrix M where M[i,j] represents citations from unit i to unit j:

- **Rows** = citing units (who is doing the citing)
- **Columns** = cited units (who is being cited)  
- **Values** = number of citation relationships

Example journal citation matrix:
```
     J1   J2   J3
J1  2.0  2.0  1.0    # Journal J1 works cite: 2 J1 works, 2 J2 works, 1 J3 work
J2  2.0  2.0  0.0    # Journal J2 works cite: 2 J1 works, 2 J2 works, 0 J3 works  
J3  3.0  0.0  1.0    # Journal J3 works cite: 3 J1 works, 0 J2 works, 1 J3 work
```

## Future Extensions

- **Ranking algorithms**: Pinski-Narin, PageRank, HITS
- **Temporal analysis**: Time-windowed citation matrices
- **Network metrics**: Centrality measures, clustering coefficients
- **Real data integration**: OpenAlex, Web of Science connectors

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- json (standard library)

## Design Principles

1. **Modular architecture** - Clear separation of data generation, processing, and analysis
2. **Test-driven development** - Comprehensive test coverage for reliability
3. **Realistic toy data** - Complex scenarios that mirror real bibliometric patterns
4. **Efficient processing** - Vectorized operations using pandas
5. **Clean interfaces** - Optional verbosity, clear function signatures