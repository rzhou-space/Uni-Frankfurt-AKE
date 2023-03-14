#!/usr/bin/env python
# coding: utf-8

import glob,os
import pandas as pd
import numpy as np
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'widget')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

##########################################################################################

def read_expression_data(direction):
    """
    This function read in all .JSON files along the given direction. 
    Involved conditions, expression data under each condition,
    replicates under each condition, and the set of genes will be extracted. 
    They will be returned as lists.
    
    direction: String. Path to the folder where all .JSON files are stored.
    
    """
    # Read the expression .json files based on the direction path. 
    os.chdir(direction)
    files = []
    expr_files = []  # Full data for all conditions.

    for file in glob.glob("*.json"):
        files.append(file)
    for file in files:
        f = open(file)
        data = json.load(f)
        expr_files.append(data) 
        
    # Extract the conditions, expression data and replicates.
    conditions = []
    expression = []
    replicates = []
    for data in expr_files:
        conditions.append(data["condition"])
        expression.append(data["expression"])
        replicates.append(data["replicates"])
        
    # Extract all appeared genes. And because under each condition, the 
    # number of genes is(should) be the same. Only take the gene name 
    # set from the first condition.
    genes = list(expression[0].keys())
    
    return conditions, expression, replicates, genes



def read_mov_data(direction):
    """
    This function extracts the movement information, conditions, replicates,
    and (GeneID, ProtID) from .JSON files containing movement results.
    Movements are applied for calculations of final ewfd score.
    Lists of conditions, replicates, mean_movement, and IDs will be returned.
    
    direction: String. Path to the folder where all .JSON files are stored.
    """
    # Read in all_movement data 
    os.chdir(direction)
    cond = []
    mov_files = []
    # Find all .json files along the direction.
    for file in glob.glob("*.json"):
        cond.append(file)
    # Open all files and gather them together.
    for c in cond:
        f = open(c)
        data = json.load(f)
        mov_files.append(data)
        
        
    # Extract condition, id and mean_mov information.
    conditions = []
    all_mean_mov = [] 
    all_gene_id = []
    all_prot_id = []
    for data in mov_files:
        conditions.append(data["condition"])
        movement = data["movement"]
        gene_id = []
        prot_id = []
        mean_mov = []
        for gene in movement.keys():
            protlist = movement[gene]["prot_ids"]
            mean_movlist = movement[gene]["mean_mov"]
            for i in range(len(protlist)):
                gene_id.append(gene)
                prot_id.append(protlist[i])
                mean_mov.append(mean_movlist[i])
        all_mean_mov.append(mean_mov)
        all_gene_id.append(gene_id)
        all_prot_id.append(prot_id)
        
    return all_mean_mov, all_gene_id, all_prot_id, conditions




def active_genes(direction):
    """
    This function extracts the active genes, i.e., the genes that are 
    represented in all replicates (total gene expression!= 0). 
    A list of the genes will be returned. 
    It will be recommended that applying the function for 
    once and reuse the returned list for multiple functions below.
    The final data frame with expression values and a list of active
    genes will be returned. 
    
    direction: The path (string) to the folder where all gene expression 
    data are.
    
    """
    # Read the expression .json files based on the direction path. 
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]

    DF = pd.DataFrame(genes, columns = ["gene_id"])
    
    # Extract the "total" gene expression over all replicates. Merge them
    # together to form a general data frame (could be returned to view the
    # original data). The index of the data frame are renamed replicates.
    # The number n after the condition indicates the n-th replicate. 
    for i in range(len(conditions)):
        expr = expression[i]
        repl = replicates[i]
        cond = conditions[i]
        total_expr = np.zeros((len(genes),len(repl)))
        for j in range(len(genes)):
            gene = genes[j]
            for k in range(len(repl)):
                repl_name = repl[k]
                total_expr[j, k] = expr[gene][repl_name]["total"]
                
        repl_names = []
        for n in range(len(repl)):
            repl_names.append(str(cond)+"_"+str(n+1))
            
        temp_DF = pd.DataFrame(total_expr, columns = repl_names)
        DF = pd.concat([DF, temp_DF], axis = 1)
        
    # Only keep those genes that are presented in all replicates.
    non_zero = DF[(DF.T != 0).all()]
    active_gene = list(non_zero["gene_id"])
    
    return non_zero, active_gene




def PCA_tool(DataFrame, delete_column):
    """
    This function applies PCA method on the final expression DataFrame to 
    generate a DataFrame containing first three PCs for all conditions. 
    The DataFrame containing PC and information content vector will be 
    returned.
    
    DataFrame: The final DataFrame containing expression data for all 
    conditions.
    delete_column: String. The name of the column in the DataFrame 
    containing geneID or (geneID, protID).
    
    """
        
    # Apply the 3-dim PCA.
    # Coding PCA with the help of:
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["conditions"])
    targets = list(y["conditions"])
    x = StandardScaler().fit_transform(x) # Turn x into normalized scale.
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['PC1', \
                              'PC2', \
                              'PC3'])
    # FinalDF contains the 3 components and the conditions as columns.
    finalDf = pd.concat([principalDf, y], axis = 1)  
    # The information contain of 3 components.
    info_contain = pca.explained_variance_ratio_
    
    return finalDf, info_contain




def PCA_plot(conditions, info_content, plot_name, principalDf):
    """
    This function visualize the results of PCA with the first three PCs.     
    The first Subplot is a dynamic 3D plot. Three projections with two 
    PCs will be also generated. This function could only be appliey with
    expression data since the data structure differs from movement data(
    applied for EWFD score). 
    
    conditions: the list of involved conditions.
    replicates: the list of replicates for all conditions.
    info_content: the list of information content generated from PCA.
    plot_name: String. The name of the plot. 
    principalDf: data frame containing PCA results. 
    
    """
        
    # Construct the plots:

    fig = plt.figure()
    fig.suptitle(plot_name, fontsize = 15)

    gs = GridSpec(18, 22)
    
    x = principalDf["PC1"]
    y = principalDf["PC2"]
    z = principalDf["PC3"]
    name = principalDf["conditions"]

    # First subplot.
    ax1 = fig.add_subplot(gs[3:16,0:13], projection='3d')
    ax1.set_xlabel('PC1', fontsize = 10)
    ax1.set_ylabel('PC2', fontsize = 10)
    ax1.set_zlabel('PC3', fontsize = 10)
    ax1.set_title('information content:'+                     str(np.round(sum(info_content), 2)*100)+"%"
                     ,fontsize=10)
    ax1.scatter(x, y ,z)
    for i, txt in enumerate(name):
        ax1.text(x[i], y[i], z[i], '%s'%txt) 


    # Second subplot.
    ax2 = fig.add_subplot(gs[0:4,16:22])
    ax2.set_xlabel('PC1', fontsize = 8)
    ax2.set_ylabel('PC2', fontsize = 8)
    ax2.scatter(x, y)
    ax2.set_title('information content:'+                    str(np.round(sum(info_content[0:2]), 2)*100)+"%",                   fontsize=8)
    for i, txt in enumerate(name):
        ax2.text(x[i], y[i], '%s'%txt, size=8) 

    # Third subplot.
    ax3 = fig.add_subplot(gs[7:11,16:22])
    ax3.set_xlabel('PC1', fontsize = 8)
    ax3.set_ylabel('PC3', fontsize = 8)
    ax3.scatter(x, z)
    ax3.set_title('information content:'+                    str(np.round(info_content[0]+info_content[2], 2)*100)+"%"
                  , fontsize=8)
    for i, txt in enumerate(name):
        ax3.text(x[i], z[i], '%s'%txt, size=8) 

    # Forth subplot.
    ax4 = fig.add_subplot(gs[14:18,16:22])
    ax4.set_xlabel('PC2', fontsize = 8)
    ax4.set_ylabel('PC3', fontsize = 8)
    ax4.scatter(y, z)
    ax4.set_title('information content:'+                    str(np.round(sum(info_content[1:]), 2)*100)+"%"
                  , fontsize=8)
    for i, txt in enumerate(name):
        ax4.text(y[i], z[i], '%s'%txt, size=8) 

    plt.show()


    
##########################################################################################

def active_gene_mean(direction):
    """
    This function generates firstly a DataFrame containing all mean values
    for gene expression and keep those only for active genes. 
    The mean values for each condition are calculated through averaging over
    all replicates.
    PCA is applied for dimensional reduction and allows visualization.
    
    direction: String. Path to the folder where all expression files are 
    stored.
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Extract active genes
    gene_set = active_genes(direction)
    active = gene_set[1]
    
    # Calculate the mean(gene expression i.e. "total") for each gene over
    # all replicates. Do this for each condition. The dataframe could be 
    # returned to virsulize the data/vectors as table. 
    all_mean_total_expr = []
    for i in range(len(conditions)):
        cond = expression[i]
        repl = replicates[i]
        total_expr = np.zeros(len(genes))
        for j in range(len(genes)):
            gene = genes[j]
            repl_expr = np.zeros(len(repl))
            for k in range(len(repl)):
                repl_name = repl[k]
                repl_expr[k] = cond[gene][repl_name]["total"]
            total_expr[j] = np.mean(repl_expr)
        all_mean_total_expr.append(total_expr)
    # The dataframe to present mean gene expression for all conditions.
    all_gene_expr_DF = pd.DataFrame(genes, columns = ["gene_id"])
    for i in range(len(conditions)):
        all_gene_expr_DF[conditions[i]] = all_mean_total_expr[i]
        
    # Only keep genes that are represented in the gene expression data.
    expr_DF = all_gene_expr_DF[all_gene_expr_DF["gene_id"].isin(active)]
    
    # Apply PCA on expression DataFrame
    results = PCA_tool(expr_DF, "gene_id")
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of PCA results.
    plot_name = "Gene Expression"
    PCA_plot(conditions, info_content, plot_name, principalDf)


    

def active_transcript_mean(direction):
    """
    The function plots the PCA of mean splice variant expreesion (transcript 
    expression) for each condition but only for active genes, i.e., those
    are present in gene expression by all replicates.
    Mean values are calculated through averaging over all replicates.
    
    dierction: the path (string) to the folder where all expression data
    are.
    
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Extract active genes
    gene_set = active_genes(direction)
    active = gene_set[1]
    
        
    # Extract all combinations (gene_id, prot_id). Because all .json files
    # have(should have) the same number of combinations. Could only take
    # the set of all combinations from data under the first condition. 
    all_gene = []
    all_prot = []
    data_cond1 = expression[0]
    repl = replicates[0][0]  # Replicates of the first condition.
    for i in range(len(genes)):
        gene = genes[i]
        keys = list(data_cond1[gene][repl].keys())
        for prot in keys[1:]:  # In every key list, the first element is "total"
            all_gene.append(gene)
            all_prot.append(prot)
    
    
    # Extract all expression values for all combinations (gene_id, prot_id)
    # for all conditions.
    total_prot_expr = []
    for i in range(len(conditions)):
        test = expression[i]
        repl = replicates[i]
        all_prot_expr = np.zeros(len(all_prot))
        for j in range(len(all_prot)):
            gene_id = all_gene[j]
            prot_id = all_prot[j]
            expr = np.zeros(len(repl))
            for k in range(len(repl)):
                expr[k] = test[gene_id][repl[k]][prot_id]
            all_prot_expr[j] = np.mean(expr)
        total_prot_expr.append(all_prot_expr)
    # Construct the data frame from the data/vectors above. 
    # The dataframe could be returned to visualize the data.
    prot_expr_DF = pd.DataFrame(list(zip(all_gene, all_prot)),                                 columns = ["gene_id", "prot_id"])
    for i in range(len(conditions)):
        prot_expr_DF[conditions[i]] = total_prot_expr[i]
        
    # Only keep genes that are active in gene expression data (in all replicates)
    transcript_DF = prot_expr_DF[prot_expr_DF["gene_id"].isin(active)]
    
    # Apply PCA on the final expression DataFrame.
    results = PCA_tool(transcript_DF, ["gene_id", "prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "Transcript Expression"
    PCA_plot(conditions, info_content, plot_name, principalDf)





def active_rel_transcript_mean(direction):
    """
    The function plots PCA results of mean relative splice variant 
    expression (transcript expression). The mean is calculated with:
    (sum_{replicates} isoform expression/gene expression) / #replicates.
    This is also the values that are applied for calculating EWFD values.
    Only active genes (from gene expression) will be considered in 
    the analysis.
    
    direction:  the path (string) to the folder where all expression data
    are.
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Extract active genes
    gene_set = active_genes(direction)
    active = gene_set[1]
    
    # Extract all combinations (gene_id, prot_id). Because all .json files
    # have(should have) the same number of combinations. Could only take
    # the set of all combinations from data under the first condition. 
    all_gene = []
    all_prot = []
    data_cond1 = expression[0]
    repl = replicates[0][0]  # Replicates of the first condition.
    for i in range(len(genes)):
        gene = genes[i]
        keys = list(data_cond1[gene][repl].keys())
        for prot in keys[1:]:  # In every key list, the first element is "total"
            all_gene.append(gene)
            all_prot.append(prot)
    
    # Calculate the vectors with the mean of fractions 
    # sum(isof/gene)/#replicates
    total_rel_prot_expr = []
    for i in range(len(conditions)):
        test = expression[i]
        repl = replicates[i]
        all_prot_expr = np.zeros(len(all_prot))
        for j in range(len(all_prot)):
            gene_id = all_gene[j]
            prot_id = all_prot[j]
            expr = np.zeros(len(repl))
            for k in range(len(repl)):
                single_gene_expr = test[gene_id][repl[k]]["total"]
                if single_gene_expr != 0:
                    expr[k] = test[gene_id][repl[k]][prot_id]/single_gene_expr
                # else: if the single gene is not expressed, then the isoforms
                # are also not expressed. In this case, the "score" will be 0.
            all_prot_expr[j] = np.mean(expr)
        total_rel_prot_expr.append(all_prot_expr)
    # Construct the dataframe from the calculation.
    rel_prot_expr_DF = pd.DataFrame(list(zip(all_gene, all_prot)),                                     columns = ["gene_id", "prot_id"])
    for i in range(len(conditions)):
        rel_prot_expr_DF[conditions[i]] = total_rel_prot_expr[i]
        
    # Only keep active genes from gene expression data (in all replicates).
    rel_transcript_DF = rel_prot_expr_DF[rel_prot_expr_DF["gene_id"].isin(active)]
    
    # Apply PCA.
    results = PCA_tool(rel_transcript_DF, ["gene_id", "prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "Relative Transcript Expression"
    PCA_plot(conditions, info_content, plot_name, principalDf)





def active_ewfd_mean(mov_direction, expr_direction):
    """
    This function applies mean values of ewfd scores calculated from
    movement values. Only active genes are involved. Visualizing the 
    data through dimensional reduction with PCA returned as a panel image.
    
    mov_direction: String. The path to the folder where all movement data 
    files are stored.
    
    expr_direction: String. The path to the folder where all expression data 
    files are stored.
    
    """
    # Read in movement data.
    mov_results = read_mov_data(mov_direction)
    
    all_mean_mov = mov_results[0]
    all_gene_id = mov_results[1]
    all_prot_id = mov_results[2]
    conditions = mov_results[3]
    
    # Extract active genes.
    gene_set = active_genes(expr_direction)
    active = gene_set[1]
    
    
    # Construct the dataframe containing all information.
    # First with two columns:
    EWFD_table = pd.DataFrame(list(zip(all_gene_id[0]
                                       , all_prot_id[0]
                                       , 1-np.asarray(all_mean_mov[0])))
                            ,columns = ["gene_id", "prot_id"
                                        , conditions[0]])
        
    # Iterativly add other condition-mean_mov columns:
    for i in range(1,len(conditions)):
        EWFD_table[conditions[i]] = 1-np.asarray(all_mean_mov[i])
    
    # Only keep genes that are represented in the gene expression data.
    ewfd_DF = EWFD_table[EWFD_table["gene_id"].isin(active)]
    
    
    # Apply PCA.
    results = PCA_tool(ewfd_DF, ["gene_id", "prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "EWFD Score"
    PCA_plot(conditions, info_content, plot_name, principalDf)

