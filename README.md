# EWFD Score

# Expression Weighted Functional Divergence (EWFD) score combines the FAS score and 
# transcript expression to assess the functional landscape of a proteome.

# Functions in this repository help to explore the effect of applying the EWFD score
# comparing to gene expression analysis.

# Two data sets are involved: the human endoderm cell lines and cardiomyocytes.
# .JSON files or raw data could be generated from pipeline 
# chrisbluemel/grand-trumpet.

# Four packages are involved to allow the analyzing the effect from different 
# perspectives of data modification.

# The functions in package Mix_Mean consider two data sets together and analyze
# with mean values over replicates.

# The functions in package Mix_Replicats consider also two data sets together 
# but apply directly the expression or movement values for replicates.

# The functions in package Single_Mean consider only one data set and only
# active genes to undergo analyzing with mean values.

# The functions in package Single_Replicates then consider only one data set
# and only active genes with replicate values directly.

