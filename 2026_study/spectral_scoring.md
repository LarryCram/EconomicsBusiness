# Summary.

This text describes a project about the use of spectral scoring in bibliometrics. We want to compare different scoring approaches (weighting) and different bibliometric networks (units).

# Bibliometric data.

The bibliometric data are provided as citation edge lists in a long parquet file at FILE NAME. A row is the category of the matrix, citer id, the cited id and the weight of the reference from row i to column j. Separate lists are provided for the category units: sources s, authors a and institutions I. There are also the larger edge lists si for the square citation matrix between the concatenation of units: e.g. sources and institutions - these are block matrices. The block edge lists are si, sa and ai. The nodes are labelled by unique openalex entity id strings. The parquet file has a category column to pick out the various citation matrices. 

The scale of the data is roughly 300000 articles, 600 sources, 600 institutions and 1500 authors.

# CSR format.

The edge lists of each citation matrix need to be prepared in CSR format. We need dictionaries for the mapping from the CSR index to the openalex id. I want to persist these CSR matrices perhaps in parquet although they potentially contain long lists. The CSR build should be the initial step and each CSR should have the unit code as its ID. 

# Spectral scoring.

I will use power iteration to find the spectral scores (principal eigenvector, roughly). Because the matrix might not be primitive I need to use the Katz form of the iteration with an attenuation factor \alpha. I need a function that accepts a matrix and uses Katz iteration to return the stable solution. This may also need the normalisation to be specified per iteration.

# Matrix normalisation.

Let s(I) be the row sums of the matrix (s_i = \sum_j c_ij). Define D1 = diag(s_j) and D2 = diag(1/s_i). You will also be given the vector a(I) of the number of articles in unit I. Define D3 = diag(1/a_i). Define D4 as the matrix product C^TxC. We normalise D1, D2 and D3 (M = C D) and use M = D4. We call the four cases PN (Pinski-Narin), G (Geller), S (Scimago) and H (HITS). We need a factory to select the normalisations. The normalized matrices are the basis for the Katz solution.

# Organisation.

Use a factory template for the CSR and for the Normalisations. Plan to run and save over all unit sets and all normalisations. Start with the s CSR and the PN normalisation (this is journal spectral ranking).

# Approach.

Use Python, Pandas, Numpy and duckdb as appropriate. Persist to parquet. Store working data in /home/lc/m/working. Decide whether to divide the project into a few separate .py files linked by persisted data or just build one. Build the code in the 2026_study of the Project folder EconomicsBusiness. There is a .venv there that you will use. Use test-driven approaches if you feel that will help. I prefer minimal docstrings and compact readable code without typing or extensive try-except QA. 