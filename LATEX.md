# LaTeX

# Latex folder is spectral_ranking_latex
- read main.pdf

# Update to section 02
- We need a new section that compactly sets out the situation when we consider source-institution and institution-source only so that C becomes a block matrix with zero blocks on the diagonal. 
- Here is a Gemini discussion that you can use but it is not complete and I find that Claude can often do better.

The One-Mode Projection: $W = C^T C$ (Institutions to Institutions via Sources). If $W$ is primitive, the principal eigenvector is the ranking of institutions.The Bipartite Eigenvector Mapping: If $\mathbf{s}$ (Sources) is found, then $\mathbf{i} = \frac{1}{\lambda} \mathbf{C}_{IS} \mathbf{s}$.The Tripartite "Real World" Coupling: Using "Work" ($W$) as the central axis:$$s = \mathbf{B}^T w$$$$i = \mathbf{H}^T w$$$$w = \beta \mathbf{B}s + \gamma \mathbf{H}i$$This preserves the fractional counting in $\mathbf{H}$ (ARC grants) and the journal metadata in $\mathbf{B}$ (OpenAlex).
