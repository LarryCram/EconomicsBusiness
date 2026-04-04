# WRAPUP.md — Project Status Summary (April 2026)

## Project Status: LARGELY COMPLETE

All major pipeline components have been successfully implemented and completed:

**Data preparation pipeline** (Stages 1-6): Complete
- Source list construction 
- OpenAlex matching and topic filtering
- Corpus extraction  
- Institution retention diagnostics
- Edge list construction with SCC filtering
- Paper tables generation

**Spectral ranking pipeline** (Stage 7): Complete  
- CSR assembly and block construction
- Katz and bipartite resolvent iteration
- Parameter space exploration (all planned runs)
- Community structure analysis with second eigenpair computation

**Paper writing**: Substantially complete
- Sections 1-3: Complete (Introduction, Methods, Data)
- Section 4: In progress (Results - structure defined, content being developed)  
- Sections 5-6: Placeholder (Discussion, Prospects)

**Outstanding work**:
- Bootstrap uncertainty analysis (spec in BOOTSTRAP.md; fig_7.py not yet written)
- Final ranking tables (Tables 6-8)
- Results section completion
- Discussion and prospects sections

**Figure status**:
- fig_2: created (F field comparison; legend on both panels)
- fig_6: created (time-series comparison t1–t4 vs baseline 2020–24)
- fig_7: planned (bootstrap uncertainty)

All documentation files have been updated to reflect current completion status.