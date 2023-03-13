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

###########################################################################################

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
    # Read in data:
    os.chdir(direction)
    files = []
    mov_files = []
    # Find all .json files along the direction.
    for file in glob.glob("*.json"):
        files.append(file)
    # Open all files and gather them together.
    for c in files:
        f = open(c)
        data = json.load(f)
        mov_files.append(data) 
    
    # Extract condition, id and mean_mov information.
    conditions = []
    replicates = []
    all_mean_mov = []
    all_id = []
    for data in mov_files:
        conditions.append(data["condition"])
        replicates.append(data["replicates"])
        movement = data["movement"]
        merge_id = []
        mean_mov = []
        for gene in movement.keys():
            protlist = movement[gene]["prot_ids"]
            mean_movlist = movement[gene]["mean_mov"]
            for i in range(len(protlist)):
                merge_id.append((gene, protlist[i]))
                mean_mov.append(mean_movlist[i])
        all_mean_mov.append(mean_mov)
        all_id.append(merge_id)
    
    return conditions, replicates, all_mean_mov, all_id, mov_files



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
    # Apply PCA approach on the merged data frame.
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["repl"])
    targets = list(y["repl"])
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
            , columns = ['PC1', 'PC2', 'PC3']
                               , index = y["repl"])
    # Modification of the Principal data frame to make plot easier.
    principalDf["class"] = 0
    principalDf["label"] = "N"
    
    
    # Information content of the PCA.
    info_content = pca.explained_variance_ratio_
    
    return principalDf, info_content



def PCA_plot_expression(conditions, replicates, info_content
                        , plot_name, principalDf):
    
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
    
    # Iterativly add replicates under the same condition to the 3D plot. 
    temp_position = 0
    fig = plt.figure()
    fig.suptitle(plot_name, fontsize = 15)
    
    gs = GridSpec(18, 22)
    
    # First subplot.
    ax1 = fig.add_subplot(gs[3:16,0:13], projection = "3d")
    for i in range(len(conditions)):
        repl = replicates[i]
        for j in range(temp_position, temp_position+len(repl)):
            principalDf["class"].values[j]= i
            principalDf["label"].values[j] = conditions[i]

        ax1.scatter(principalDf["PC1"].values[temp_position:temp_position+len(repl)]
                   , principalDf["PC2"].values[temp_position:temp_position+len(repl)]
                   , principalDf["PC3"].values[temp_position:temp_position+len(repl)]
                   , alpha = 1 # Turn off the transparency. 
                   , label = principalDf["label"][temp_position])

        temp_position += len(repl)
    # Some general set ups to the plot. 
    ax1.set_xlabel('Principal Component 1', fontsize = 10)
    ax1.set_ylabel('Principal Component 2', fontsize = 10)
    ax1.set_zlabel('Principal Component 3', fontsize = 10)
    ax1.set_title('information content:'+ str(round(sum(info_content), 3)*100)+r'%', fontsize=10)
    ax1.legend()
    
    
    # Second subplot.
    temp_position = 0
    ax2 = fig.add_subplot(gs[0:4,16:22]) 
    for i in range(len(conditions)):
        repl = replicates[i]
        ax2.scatter(principalDf["PC1"].values[temp_position:temp_position+len(repl)]
                       , principalDf["PC2"].values[temp_position:temp_position+len(repl)]
                       , label = principalDf["label"][temp_position])
        temp_position += len(repl)
        
    ax2.set_xlabel('Principal Component 1', fontsize = 10)
    ax2.set_ylabel('Principal Component 2', fontsize = 10)
    ax2.set_title('information content:'+ str(round(sum(info_content[0:2]), 2)*100)+r'%', fontsize=10)
    ax2.legend().set_visible(False)
    
    # Third Subplot.
    temp_position = 0
    ax3 = fig.add_subplot(gs[7:11,16:22])
    for i in range(len(conditions)):
        repl = replicates[i]
        ax3.scatter(principalDf["PC1"].values[temp_position:temp_position+len(repl)]
                       , principalDf["PC3"].values[temp_position:temp_position+len(repl)]
                       , label = principalDf["label"][temp_position])
        temp_position += len(repl)
        
    ax3.set_xlabel('Principal Component 1', fontsize = 10)
    ax3.set_ylabel('Principal Component 3', fontsize = 10)
    ax3.set_title('information content:'+ str(np.round(info_content[0]+info_content[2], 1)*100)+r'%', fontsize=10)
    ax3.legend().set_visible(False)
    
    # Fourth Subplot.
    temp_position = 0
    ax4 = fig.add_subplot(gs[14:18,16:22])
    for i in range(len(conditions)):
        repl = replicates[i]
        ax4.scatter(principalDf["PC2"].values[temp_position:temp_position+len(repl)]
                       , principalDf["PC3"].values[temp_position:temp_position+len(repl)]
                       , label = principalDf["label"][temp_position])
        temp_position += len(repl)
        
    ax4.set_xlabel('Principal Component 2', fontsize = 10)
    ax4.set_ylabel('Principal Component 3', fontsize = 10)
    ax4.set_title('information content:'+ str(np.round(sum(info_content[1:]), 2)*100)+r'%', fontsize=10)
    ax4.legend().set_visible(False)
    
    plt.show()
    



def PCA_plot_mov(conditions, repl_num, info_content, plot_name,
                principalDf):
    """
    This function visualize the results of PCA with the first three PCs.
    The first Subplot is a dynamic 3D plot. Three projections with two 
    PCs will be also generated. This function could only be appliey with
    movement data (used for EWFD score calculation) since the data 
    structure differs from expression data. 
    
    conditions: the list of involved conditions.
    repl_num: the list of number of replicates for each condition in the 
    order conditions listed in conditions-list.
    info_content: the list of infomation content generated from PCA.
    plot_name: String. Name of the plot.
    principalDf: the data frame containing PCA results
    """
    
    # Iterativly add replicates under the same condition to the 3D plot. 
    temp_position = 0
    fig = plt.figure()
    fig.suptitle(plot_name, fontsize = 15)
    
    gs = GridSpec(18, 22)
    
    # First Subplot
    ax1 = fig.add_subplot(gs[3:16,0:13], projection = "3d")
    for i in range(len(conditions)):
        principalDf["label"].values[i] = conditions[i]

    for i in range(len(repl_num)):
        for j in range(temp_position, temp_position+repl_num[i]):
            principalDf["class"].values[j]= i

        ax1.scatter(principalDf["PC1"].values[temp_position:temp_position+repl_num[i]]
                        , principalDf["PC2"].values[temp_position:temp_position+repl_num[i]]
                        , principalDf["PC3"].values[temp_position:temp_position+repl_num[i]]
                        , alpha = 1 # Turn off the transparency. 
                        , label = principalDf["label"][temp_position])

        temp_position += repl_num[i]
    # Some general set ups to the plot. 
    ax1.set_xlabel('Principal Component 1', fontsize = 10)
    ax1.set_ylabel('Principal Component 2', fontsize = 10)
    ax1.set_zlabel('Principal Component 3', fontsize = 10)
    ax1.set_title('information content:'+ str(round(sum(info_content), 2)*100)+r'%', fontsize=10)
    ax1.legend()
    
    # Second Subplot
    temp_position = 0
    ax2 = fig.add_subplot(gs[0:4,16:22])
    for i in range(len(repl_num)):
        ax2.scatter(principalDf["PC1"].values[temp_position:temp_position+repl_num[i]]
                        , principalDf["PC2"].values[temp_position:temp_position+repl_num[i]]
                        , label = principalDf["label"][temp_position])
        temp_position += repl_num[i]
        
    ax2.set_xlabel('Principal Component 1', fontsize = 10)
    ax2.set_ylabel('Principal Component 2', fontsize = 10)
    ax2.set_title('information content:'+ str(round(sum(info_content[0:2]), 2)*100)+r'%', fontsize=10)
    ax2.legend().set_visible(False)
    
    # Third Subplot
    temp_position = 0
    ax3 = fig.add_subplot(gs[7:11,16:22])
    for i in range(len(repl_num)):
        ax3.scatter(principalDf["PC1"].values[temp_position:temp_position+repl_num[i]]
                        , principalDf["PC3"].values[temp_position:temp_position+repl_num[i]]
                        , label = principalDf["label"][temp_position])
        temp_position += repl_num[i]
        
    ax3.set_xlabel('Principal Component 1', fontsize = 10)
    ax3.set_ylabel('Principal Component 3', fontsize = 10)
    ax3.set_title('information content:'+ str(np.round(info_content[0]+info_content[2], 2)*100)+r'%',fontsize=10)
    ax3.legend().set_visible(False)
    
    
    #Fourth Subplot
    temp_position = 0
    ax4 = fig.add_subplot(gs[14:18,16:22])
    for i in range(len(repl_num)):
        ax4.scatter(principalDf["PC2"].values[temp_position:temp_position+repl_num[i]]
                        , principalDf["PC3"].values[temp_position:temp_position+repl_num[i]]
                        , label = principalDf["label"][temp_position])
        temp_position += repl_num[i]
        
    ax4.set_xlabel('Principal Component 2', fontsize = 10)
    ax4.set_ylabel('Principal Component 3', fontsize = 10)
    ax4.set_title('information content:'+ str(np.round(sum(info_content[1:]), 2)*100)+r'%', fontsize=10)
    ax4.legend().set_visible(False)
        
    
    plt.show()
    


##########################################################################################

def active_gene_expr(direction):
    """
    This function visualize the results of PCA of gene expression values
    only for active genes, i.e.,non-zero expression values in all 
    replicates.
    A panel plot will be returned.
    
    direction: String. The path to the folder where all expression files 
    are stored.
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    
    # Sort out the active genes and the expression DataFrame of those genes.
    gene_set = active_genes(direction)
    expr_DF = gene_set[0]
    active = gene_set[1]
    
    # Apply the PCA on the DataFrame with only active genes.
    results = PCA_tool(expr_DF, "gene_id")
    principalDf = results[0]
    info_content = results[1]
    
    # Visualize the results from PCA.
    plot_name = "Gene Expression"
    PCA_plot_expression(conditions, replicates, info_content
                        , plot_name, principalDf)
    



def active_transcript_expr(direction):
    """
    This function visualize the results of PCA of transcript expression 
    values only for active genes, i.e.,non-zero gene expression values in 
    all replicates.
    A panel plot will be returned.
    
    direction: String. The path to the folder where all expression files 
    are stored.
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Sort out the active genes and the expression DataFrame of those genes.
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
    
    DF_prot = pd.DataFrame(list(zip(all_gene, all_prot))
                           , columns = ["gene_id", "prot_id"])
    
    # Extract all expression values for all combinations (gene_id, prot_id)
    # for all conditions. And build up a data frame storing all expression
    # for each (gene_id, prot_id) under every condition.
    for i in range(len(conditions)):
        expr = expression[i]
        repl = replicates[i]
        cond = conditions[i]
        all_prot_expr = np.zeros((len(all_prot), len(repl)))
        for j in range(len(all_prot)):
            gene_id = all_gene[j]
            prot_id = all_prot[j]
            for k in range(len(repl)):
                repl_name = repl[k]
                all_prot_expr[j,k] = expr[gene_id][repl_name][prot_id]

        repl_names = []
        for n in range(len(repl)):
            repl_names.append(str(cond)+"_"+str(n+1))

        temp_DF = pd.DataFrame(all_prot_expr, columns = repl_names)
        DF_prot = pd.concat([DF_prot, temp_DF], axis = 1)
        
    # Only keep the active genes in the gene expression data.
    transcript_DF = DF_prot[DF_prot["gene_id"].isin(active)]
    
    # Apply the PCA.
    results = PCA_tool(transcript_DF, ["gene_id","prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "Transcript Expression"
    PCA_plot_expression(conditions, replicates, info_content
                        , plot_name, principalDf)
    



def active_relative_transcript_expr(direction):
    """
    The function plot the PCA of splice variant expression only for active 
    genes (i.e.the total gene expression != 0 for all replicates in gene 
    expression data). The function generate for relative splice variant 
    expression firstly a data frame which contains all isoforms. Then, 
    a sub data frame will be sorted out based on the active_gene. 
    Only those isoforms will be kept whose initial gene is active.
    
    direction: String. The path to the folder where all expression files 
    are stored.
    
    """
    # Read in expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Sort out the active genes.
    active = active_genes(direction)[1]
    
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
            
    # Build up iterativly the data frame containing all relativ splice 
    # expression under each condition. 
    DF_rel_prot = pd.DataFrame(list(zip(all_gene, all_prot)), 
                                columns = ["gene_id","prot_id"])
    
    for i in range(len(conditions)):
        expr = expression[i]
        repl = replicates[i]
        cond = conditions[i]
        all_prot_expr = np.zeros((len(all_prot), len(repl)))
        for j in range(len(all_prot)):
            gene_id = all_gene[j]
            prot_id = all_prot[j]
            for k in range(len(repl)):
                repl_name = repl[k]
                single_gene_expr = expr[gene_id][repl_name]["total"]
                if single_gene_expr != 0:
                    all_prot_expr[j,k] = expr[gene_id][repl_name][prot_id]/single_gene_expr
                # else: if the single gene is not expressed, then the isoforms
                # are also not expressed. In this case, the "score" will be 0
        repl_names = []
        for n in range(len(repl)):
            repl_names.append(str(cond)+"_"+str(n+1))

        temp_DF = pd.DataFrame(all_prot_expr, columns = repl_names)
        DF_rel_prot = pd.concat([DF_rel_prot, temp_DF], axis = 1)
    
    
    # Only keep genes that are represented in the gene expression data.
    rel_transcriptDF = DF_rel_prot[DF_rel_prot["gene_id"].isin(active)]
    
    # Apply the PCA.
    results = PCA_tool(rel_transcriptDF, ["gene_id","prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "Relative Transcript Expression"
    PCA_plot_expression(conditions, replicates, info_content
                        , plot_name, principalDf)



def active_ewfd(mov_direction, expr_direction):
    """
    The function plot the PCA of EWFD scores only for active genes (i.e.
    the total gene expression != 0 for all replicates in gene expression 
    data). The function generate for EWFD scores firstly a data frame 
    which contains all isoforms. Then, a sub data frame will be sorted 
    out based on the active_gene. Only those isoforms will be
    kept whose initial gene is active.
    
    mov_direction: String. The path to the folder where all movement data 
    files are stored.
    
    expr_direction: String. The path to the folder where all expression 
    data files are stored. 
    
    """
    # Read in movement data.
    mov_data = read_mov_data(mov_direction)
    conditions = mov_data[0]
    replicates = mov_data[1]
    all_mean_mov = mov_data[2]
    all_id = mov_data[3]
    mov_files = mov_data[4]
    
    # Sort out the active genes.
    active = active_genes(expr_direction)[1]
    
    # Extract condition, id and mean_mov information.
    conditions = []
    replicates = []
    all_mean_mov = []
    all_gene_id = []
    all_prot_id = []
    for data in mov_files:
        conditions.append(data["condition"])
        replicates.append(data["replicates"])
        movement = data["movement"]
        merge_gene_id = []
        merge_prot_id = []
        mean_mov = []
        for gene in movement.keys():
            protlist = movement[gene]["prot_ids"]
            mean_movlist = movement[gene]["mean_mov"]
            for i in range(len(protlist)):
                merge_gene_id.append(gene)
                merge_prot_id.append(protlist[i])
                mean_mov.append(mean_movlist[i])
        all_mean_mov.append(mean_mov)
        all_gene_id.append(merge_gene_id)
        all_prot_id.append(merge_prot_id)
        
    # Determine the number of replicates for each condition.
    repl_num_dict = {i:conditions.count(i) for i in conditions}
    repl_num = list(repl_num_dict.values())
        
    # Construct the dataframe containing all information.
    # First with two columns:
    EWFD_table = pd.DataFrame(list(zip(all_gene_id[0], all_prot_id[0],                                       1-np.asarray(all_mean_mov[0]))), columns = ["gene_id", "prot_id"
                                   , (conditions[0], replicates[0])])
    # Iterativly add other condition-mean_EWFD columns:
    for i in range(1,len(conditions)):
        EWFD_table[(conditions[i], replicates[i])] = 1-np.asarray(all_mean_mov[i])
        
    # Only keep represented/active genes from gene expression data. 
    active_gene_ewfd = EWFD_table[EWFD_table["gene_id"].isin(active)]
    
    # Apply PCA.
    results = PCA_tool(active_gene_ewfd, ["gene_id","prot_id"])
    principalDf = results[0]
    info_content = results[1]
    
    # Visualization of results from PCA.
    plot_name = "EWFD Score"
    PCA_plot_mov(conditions, repl_num, info_content, plot_name,
                principalDf)
    

