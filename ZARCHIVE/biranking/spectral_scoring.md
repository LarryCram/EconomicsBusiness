# Summary.

This text describes a project about the use of ordinal spectral ranking in bibliometrics. We want to compare different ranking approaches (weighting) over networks beteen different units, namely  sources and institutions. The results will be used in a paper about dual ordinal spectral ranking over sources and institutions. The topic has not been studies very much.

# Bibliometric data.

The bibliometric data are provided as citation edge lists in a long parquet file at PARQUET_NAME. A row comprises (1) a category, citer id of a citing work, the cited id of a reference, and the weight of the reference from row i to column j. Separate lists are provided for the category units: sources s, authors a and institutions i. There are also the larger edge lists si for the citation matrix between the concatenation of sources and institutions - this coresponds to block matrices. The block edge lists are ss, ii, and si. The nodes are labelled by unique openalex entity id strings. The parquet file has a category column to pick out the various citation matrices. 

The scale of the data is roughly 500000 articles, 600 sources and 600 institutions.

# CSR format.

The edge lists of each citation matrix need to be converted in CSR format. We need dictionaries for the mapping from the compacted CSR index to the original openalex id. I want to persist these CSR matrices perhaps in parquet. The parquet columns will have long lists. The build of the CSR matrix Cij from the edge list should be the initial step and each CSR should have its category as its ID. 

# Spectral ranking and citation matrix normalisation.

I want to consider two spectral rankings, one due to Pinski and Narin (PN ranking) and one due to Geller, Bollen and West (Eigenvector - or E - ranking). The rankings differ only in the way that the citation matrix is normalised before finding its principal eigenvector. Define the row sums of Cij as Si = \sum_k Cik. Define two diagonal matrices, D1 = diag(Sj) and D2 = diag(Si). From PN, trhe nomralised matrix is M1 = D1 C. For E, the normalised matrix is M2 = D2 C. M2 is a row stochastic matrix. 

# Finding the Eigenvector. 

The eigenvector problem is w = M w where w is the principal eigenvector. Since M may not be primitive, we adopt the Katz power iteration w = \alpha Mw + (1-\alpha)v where \alpha <= 1 and v = 1/N, N is the size of M. For the PN method it is necessary to scale w so that \sum_i w_i Si = \sum_i Si. For the E method, the sum of w should be 1 sonce thery are probabilities.  

# Finding the influence per publication

The PN gives influence per reference. The influence per publication \pi_i is w_i * \sum_i S_i / a_i. For E, the influence per publication is \pi_i = w_i * \sum_j a_j /a_j. 

# Organisation.

Use a factory template for the CSR - there are cases for jj, ii, and for ji-ij with diagonal blocks and without blocks (jj=ii=0). Use a factory template for PN and E that yields bot the influence vcetore and the unfluence per article.

# Approach.

Use Python, Pandas, Numpy and duckdb as appropriate. Persist to parquet. Store working data in /home/lc/m/working/econ_bus. Decide whether to divide the project into a few separate .py files linked by persisted data or just build a workflow. Build the code in the biranking folder of the Project folder EconomicsBusiness. There is a .venv there that you will use. Use test-driven approaches if you feel that will help. I prefer minimal docstrings and compact readable code without typing or extensive try-except QA. Think carefully before coding and code like an advanced developer. If you have concerns ask me.