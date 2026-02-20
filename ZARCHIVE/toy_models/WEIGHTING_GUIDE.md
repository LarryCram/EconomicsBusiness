# Multi-Unit Citation Analysis with Weighting Factors and Conservation

## Overview

The multi-unit citation analysis now supports configurable weighting factors for different citation types (journal-to-journal, author-to-author, institution-to-institution) while maintaining proper citation conservation through fractional credit allocation. This ensures that the total number of citations is invariant between single-unit and multi-unit analyses.

## Key Features

### 1. Citation Conservation
- **Total citations remain constant**: Single-unit and multi-unit analyses produce the same total citation count
- **Fractional credit allocation**: Multi-author and multi-institution works share credit proportionally
- **Mathematical invariance**: Different weight combinations maintain the same total citations

### 2. Configurable Weights
- Weights control the relative importance of different citation types
- Must sum to 1.0 (unity constraint)
- Default weights: `{'journal': 1/3, 'author': 1/3, 'institution': 1/3}`

### 3. Combined Fractional Credit and Weighting
- **Journals**: Full credit (assumes no multi-journal works) × weight
- **Authors**: Fractional credit (`1/ni × 1/nj`) × weight  
- **Institutions**: Fractional credit (`1/ni × 1/nj`) × weight
- **Result**: Total citations = original count regardless of weight distribution

## Usage Examples

### Default Equal Weighting
```python
from rankers.multi_unit_ranks import multi_unit_driver

# Uses default weights: 1/3 each
results = multi_unit_driver(data, verbose=True)
```

### Custom Weighting
```python
# Example: Emphasize journal citations
custom_weights = {'journal': 0.6, 'author': 0.2, 'institution': 0.2}
results = multi_unit_driver(data, weights=custom_weights, verbose=True)

# Example: Focus on author networks
author_focused = {'journal': 0.1, 'author': 0.8, 'institution': 0.1}
results = multi_unit_driver(data, weights=author_focused, verbose=True)
```

### Extreme Cases
```python
# Pure journal analysis
journal_only = {'journal': 1.0, 'author': 0.0, 'institution': 0.0}
results = multi_unit_driver(data, weights=journal_only, verbose=True)
```

## Mathematical Formulation

For each citation from work `w_i` to work `w_j`:

1. **Base Citation**: Each citation contributes 1.0 total
2. **Unit Assignment**: Citation is assigned to all units (journal, authors, institutions) of each work
3. **Fractional Credit**: 
   - Journals: Full credit (1.0)
   - Authors: `1/ni × 1/nj` where `ni` = citing authors, `nj` = cited authors
   - Institutions: `1/ni × 1/nj` where `ni` = citing institutions, `nj` = cited institutions
4. **Weight Application**: Fractional credit is multiplied by the weight for each unit type
5. **Matrix Construction**: Weighted fractional values are summed in the combined citation matrix

### Conservation Principle
The total citations across all unit types equals the original citation count:
```
Total = Σ(weighted_journal_citations) + Σ(weighted_author_citations) + Σ(weighted_institution_citations) = Original_Count
```

### Example Calculation
If work W1 (journal J1, authors [A1, A2], institution I1) cites work W2 (journal J2, author A3, institution I2):

With weights `{'journal': 0.5, 'author': 0.3, 'institution': 0.2}`:
- **Journal**: J1 → J2 receives `0.5 × 1.0 = 0.5`
- **Authors**: 
  - A1 → A3 receives `0.3 × (1/2) × (1/1) = 0.15`
  - A2 → A3 receives `0.3 × (1/2) × (1/1) = 0.15`
- **Institution**: I1 → I2 receives `0.2 × 1.0 = 0.2`
- **Total**: `0.5 + 0.15 + 0.15 + 0.2 = 1.0` ✓

## Impact on Rankings

Different weight configurations can significantly affect the relative rankings:

1. **High journal weights**: Emphasizes venue prestige effects
2. **High author weights**: Focuses on individual researcher networks
3. **High institution weights**: Highlights institutional collaboration patterns

## Testing

Comprehensive test suite validates:
- Weight validation logic
- Conservation properties
- Ranking sensitivity to weight changes
- Edge cases and extreme configurations

## Applications

This weighting system enables:
- **Domain-specific analysis**: Different fields may value different citation types
- **Policy analysis**: Studying effects of emphasizing different evaluation criteria  
- **Sensitivity analysis**: Understanding robustness of rankings to different assumptions
- **Comparative studies**: Analyzing how different weighting schemes affect conclusions