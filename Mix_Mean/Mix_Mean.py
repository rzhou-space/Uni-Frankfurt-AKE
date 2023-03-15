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
from sklearn.cluster import KMeans

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
    Lists of conditions, mean_movement, and IDs will be returned.
    
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
    all_id = []
    for data in mov_files:
        conditions.append(data["condition"])
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
        
    return conditions, all_mean_mov, all_id





def PCA_kmeans_tool(DataFrame, delete_column, n_cluster):
    """
    This function applies PCA method on the final expression DataFrame to 
    generate a DataFrame containing first three PCs for all conditions. 
    The DataFrame containing PC and information content vector will be 
    returned.
    The results from PCA are then treated with k-means clustering algorithm.
    The number of clusters is given (should normaly equal to the number of 
    cell types or number of conditions).
    
    DataFrame: The final DataFrame containing expression data for all 
    conditions.
    delete_column: String. The name of the column in the DataFrame 
    containing geneID or (geneID, protID).
    n_cluster: The number of clusters that the data points should be 
    clustered by k-means.
    
    """
    # Apply the 3-dim PCA.
    # Coding PCA with the help of:
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    tdf = DataFrame.T
    col_index = tdf.loc[delete_column,:]
    df2 = tdf.rename(columns = col_index)
    df3 = df2.drop(delete_column)
    ids = df3.columns
    x = df3.loc[:, ids].values
    y = pd.DataFrame(df3.index.values, columns = ["conditions"])
    targets = list(y["conditions"])
    x = StandardScaler().fit_transform(x) # Turn x into normalized scale.
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', \
                              'principal component 2', \
                              'principal component 3'])
    # FinalDF contains the 3 components and the conditions as columns.
    finalDf = pd.concat([principalDf, y], axis = 1)  
    # The information contain of 3 components.
    info_contain = pca.explained_variance_ratio_
    
    
    # Apply the k-Mean approach to clustering.
    # Coding k-Mean with the help of:
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    kmeanDf = np.array(list(zip(principalDf["principal component 1"], principalDf["principal component 2"], principalDf["principal component 3"])))
    kmeans = KMeans(n_clusters=n_cluster).fit(kmeanDf)
    y_kmeans = kmeans.predict(kmeanDf)
    name = finalDf["conditions"]

    
    return kmeanDf, info_contain, y_kmeans, name




def results_plot(kmeanDf, info_content, plot_name, y_kmeans, name):
    """
    This function visualize the results of PCA after treating k-means 
    clustering algorithm. 
    Provided the k-means results data frame, the function returns
    a panel image with one dynamic 3D plot and three 2D projections.
    
    kmeanDf: the result DataFrame applied by k-means. Also the 
    DataFrame generated by PCA.
    info_content: the list of information content generated from PCA.
    plot_name: String. The name of the plot. 
    y_kmeans: the results of kmean.predict() function. 
    name: list/array of condition names in the DataFrame applied in PCA.

    """
    x = kmeanDf[:, 0]
    y = kmeanDf[:, 1]
    z = kmeanDf[:, 2]
    
    # Contruct the plots:
    fig = plt.figure()
    fig.suptitle(plot_name, fontsize = 15)

    gs = GridSpec(18, 22)


    # First subplot.
    ax1 = fig.add_subplot(gs[3:16,0:13], projection='3d')
    ax1.set_xlabel('Principal Component 1', fontsize = 10)
    ax1.set_ylabel('Principal Component 2', fontsize = 10)
    ax1.set_zlabel('Principal Component 3', fontsize = 10)
    ax1.set_title('information contain:'+                     str(np.round(sum(info_content), 2)*100)+ r'%   number of clusters:'+ str(2), fontsize=10)
    ax1.scatter(x, y, z, c=y_kmeans, s=50, cmap='viridis')
    for i, txt in enumerate(name):
        ax1.text(x[i], y[i], z[i], '%s'%txt) 


    # Second subplot.
    ax2 = fig.add_subplot(gs[0:4,16:22])
    ax2.set_xlabel('Principal Component 1', fontsize = 8)
    ax2.set_ylabel('Principal Component 2', fontsize = 8)
    ax2.scatter(x, y, c=y_kmeans, s=50, cmap='viridis')
    ax2.set_title('information contain:'+ str(np.round(sum(info_content[0:2]), 2)*100)+"%", fontsize=8)
    for i, txt in enumerate(name):
        ax2.text(x[i], y[i], '%s'%txt, size=8) 

    # Third subplot.
    ax3 = fig.add_subplot(gs[7:11,16:22])
    ax3.set_xlabel('Principal Component 1', fontsize = 8)
    ax3.set_ylabel('Principal Component 3', fontsize = 8)
    ax3.scatter(x, z, c=y_kmeans, s=50, cmap='viridis')
    ax3.set_title('information contain:'+ str(np.round(info_content[0]+info_content[2], 2)*100)+"%", fontsize=8)
    for i, txt in enumerate(name):
        ax3.text(x[i], z[i], '%s'%txt, size=8) 

    # Forth subplot.
    ax4 = fig.add_subplot(gs[14:18,16:22])
    ax4.set_xlabel('Principal Component 2', fontsize = 8)
    ax4.set_ylabel('Principal Component 3', fontsize = 8)
    ax4.scatter(y, z, c=y_kmeans, s=50, cmap='viridis')
    ax4.set_title('information contain:'+ str(np.round(sum(info_content[1:]), 2)*100)+"%", fontsize=8)
    for i, txt in enumerate(name):
        ax4.text(y[i], z[i], '%s'%txt, size=8) 

    plt.show()
    

##########################################################################################


def gene_expr_mean(direction):
    """
    This function will cluster the conditions based on the original gene 
    expression. Here, the mean gene expression value over replicates
    for each gene will be applied to construct vectors for each condition. 
    After that, the PCA approach will project the data to 3 dimensional 
    space and k-Mean will be used to cluster the conditions. 
    The output is a 3-dim interactive plot with n_cluster colored clusters.
    
    direction: the path(str) to the folder where all .json expression 
    files are involved.
    
    """
    # Read in data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
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
        
    # Apply PCA and k-means algorithm.
    # Set number of clusters equals 2.
    results = PCA_kmeans_tool(all_gene_expr_DF, "gene_id", 2)
    kmeanDf = results[0]
    info_content = results[1]
    y_kmeans = results[2]
    name = results[3]
    
    # Plot the results from PCA and k-means.
    plot_name = "Gene Expression"
    results_plot(kmeanDf, info_content, plot_name, y_kmeans, name)




def transcript_expr_mean(direction):
    """
    This function read all expression .json files along the given direction.
    Based on the protein expression vectors for each condition, the PCA 
    approach project the data to 3 dimensional space. k-Mean will be applied
    to cluster the conditions.
    The output will be a 3-dim plot with the first 3 PCA components as 
    coordinations and colored clusters.
    
    direction: the path(str) to the folder where all .json expression
    files are involved.
    
    """
    # Read in data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Extract all combinations (gene_id, prot_id). Because all .json files
    # have(should have) the same number of combinations. Could only take
    # the set of all combinations from data under the first condition. 
    all_combine = []
    data_cond1 = expression[0]
    repl = replicates[0][0]  # Replicates of the first condition.
    for i in range(len(genes)):
        gene = genes[i]
        keys = list(data_cond1[gene][repl].keys())
        for prot in keys[1:]:  # In every key list, the first element is "total"
            all_combine.append((gene, prot))
    
    
    
    # Extract all expression values for all combinations (gene_id, prot_id)
    # for all conditions.
    total_prot_expr = []
    for i in range(len(conditions)):
        test = expression[i]
        repl = replicates[i]
        all_prot_expr = np.zeros(len(all_combine))
        for j in range(len(all_combine)):
            comb = all_combine[j]
            gene_id = comb[0]
            prot_id = comb[1]
            expr = np.zeros(len(repl))
            for k in range(len(repl)):
                expr[k] = test[gene_id][repl[k]][prot_id]
            all_prot_expr[j] = np.mean(expr)
        total_prot_expr.append(all_prot_expr)
    # Construct the data frame from the data/vectors above. 
    # The dataframe could be returned to visualize the data.
    prot_expr_DF = pd.DataFrame(list(zip(all_combine)), columns = ["gene_id, prot_id"])
    for i in range(len(conditions)):
        prot_expr_DF[conditions[i]] = total_prot_expr[i]
        
    
    # Apply PCA and k-means clustering.
    # Set number of clusters equals 2.
    results = PCA_kmeans_tool(prot_expr_DF, "gene_id, prot_id", 2)
    kmeanDf = results[0]
    info_content = results[1]
    y_kmeans = results[2]
    name = results[3]
    
    # Plot the results from PCA and k-means.
    plot_name = "Transcript Expression"
    results_plot(kmeanDf, info_content, plot_name, y_kmeans, name)





def relative_transcript_expr(direction):
    """
    This function clusters the conditions based on the relative protein 
    expression: sum(isoform/gene)/#replicates for 1 isoform. 
    PCA approach will be applied to project the data to 3 dimensional space.
    k-Mean will be used to cluster the conditions. 
    The output will be a 3 dimensional plot with colored clusters.
    
    direction: the path (str) to all expression .json files. 
    """
    # Read in data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Extract all combinations (gene_id, prot_id). Because all .json files
    # have(should have) the same number of combinations. Could only take
    # the set of all combinations from data under the first condition. 
    all_combine = []
    data_cond1 = expression[0]
    repl = replicates[0][0]  # Replicates of the first condition.
    for i in range(len(genes)):
        gene = genes[i]
        keys = list(data_cond1[gene][repl].keys())
        for prot in keys[1:]:  # In every key list, the first element is "total"
            all_combine.append((gene, prot))
            
    
    # Calculate the vectors with the mean of fractions 
    # sum(isof/gene)/#replicates
    total_rel_prot_expr = []
    for i in range(len(conditions)):
        test = expression[i]
        repl = replicates[i]
        all_prot_expr = np.zeros(len(all_combine))
        for j in range(len(all_combine)):
            comb = all_combine[j]
            gene_id = comb[0]
            prot_id = comb[1]
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
    rel_prot_expr_DF = pd.DataFrame(list(zip(all_combine)),                                     columns = ["gene_id, prot_id"])
    for i in range(len(conditions)):
        rel_prot_expr_DF[conditions[i]] = total_rel_prot_expr[i]
    
    
    # Apply PCA and k-means clustering.
    # Set number of clusters equals 2.
    results = PCA_kmeans_tool(rel_prot_expr_DF, "gene_id, prot_id", 2)
    kmeanDf = results[0]
    info_content = results[1]
    y_kmeans = results[2]
    name = results[3]
    
    # Plot the results from PCA and k-means.
    plot_name = "Relative Transcript Expression"
    results_plot(kmeanDf, info_content, plot_name, y_kmeans, name)




def ewfd_mean(direction):
    """
    The function firstly read all .json files from the direction.
    EWFD values will be calculated and the DataFrame will be generated.
    After the application of pca approach on the dataset, the conditions 
    will be virsulized on a 2-dim space. The k-Mean algorithm will be used 
    to cluster the conditions.
    
    direction: the string of the path to .json movment files. 
    """
    # Read in data.
    mov_data = read_mov_data(direction)
    
    conditions = mov_data[0]
    all_mean_mov = mov_data[1]
    all_id = mov_data[2]
    
    # Construct the dataframe containing all information.
    # First with two columns:
    EWFD_table = pd.DataFrame(list(zip(all_id[0], 1-np.asarray(all_mean_mov[0]))),                            columns = ["gene_id, prot_id", conditions[0]])
        
    # Iterativly add other condition-mean_mov columns:
    for i in range(1,len(conditions)):
        EWFD_table[conditions[i]] = 1-np.asarray(all_mean_mov[i])
        
    # Apply PCA and k-means clustering.
    # Set number of clusters equals 2.
    results = PCA_kmeans_tool(EWFD_table, "gene_id, prot_id", 2)
    kmeanDf = results[0]
    info_content = results[1]
    y_kmeans = results[2]
    name = results[3]
    
    # Plot the results from PCA and k-means.
    plot_name = "Relative Transcript Expression"
    results_plot(kmeanDf, info_content, plot_name, y_kmeans, name)

