#!/usr/bin/env python
# coding: utf-8

import glob,os
import pandas as pd
import numpy as np
import scipy as sp
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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



def find_active_genes(direction):
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
    
    return active_gene




def PCA_tool(DataFrame, delete_column, conditions):
    """
    This function applies PCA on the given DataFrame containing expression
    data or EWFD scores. 
    The overall information content vector and principal components for all
    conditions will be returned.
    
    DataFrame: the DataFrame containing expression value.
    delete_column: the index of column with GeneID or (GeneID, ProtID)
    conditions: the list of involved conditions. 
    """
    # Apply the PCA approach.
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["conditions"])
    targets = list(y["conditions"])
    x = StandardScaler().fit_transform(x)
    
    n_conditions = len(conditions)
    pca = PCA(n_components=n_conditions)
    column = []
    for i in range(n_conditions):
        column.append("PC"+str(i+1))
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = column, index = conditions)
    # Variables coordinates of each condition. 
    result_principal = principalDf.to_dict("index")
    # Vector for information contains.
    info_content = pca.explained_variance_ratio_
    info_contentDf = pd.DataFrame(data = info_content
                              , columns = ["information_content"]
                              , index = column).T
    result_info = info_contentDf.to_dict("index")["information_content"]
    
    return result_info, result_principal


###########################################################################################


def PC_gene_expr(direction):
    """
    This function generates all principal components for all conditions
    from PCA of gene expression.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all expression files.
    """
    # Read expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Sort out the list of active genes.
    active_gene = find_active_genes(direction)
    
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
    only_active = all_gene_expr_DF[all_gene_expr_DF["gene_id"].isin(active_gene)]
    
    
    # Apply PCA. 
    results = PCA_tool(only_active, "gene_id", conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}
    



def PC_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    from PCA of transcript expressions. 
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all expression files.
    """
    # Read expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Sort out the list of active genes.
    active_gene = find_active_genes(direction)
    
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
    only_active_transcript = prot_expr_DF[prot_expr_DF["gene_id"].isin(active_gene)]
    
    # Apply PCA. 
    results = PCA_tool(only_active_transcript, ["gene_id", "prot_id"]
                       , conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}
    




def PC_rel_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    from PCA of relative transcript expressions. 
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all expression files.
    
    """
    # Read expression data.
    expr_data = read_expression_data(direction)
    
    conditions = expr_data[0]
    expression = expr_data[1]
    replicates = expr_data[2]
    genes = expr_data[3]
    
    # Sort out the list of active genes.
    active_gene = find_active_genes(direction)
    
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
    rel_prot_expr_DF = pd.DataFrame(list(zip(all_gene, all_prot))
                                    , columns = ["gene_id", "prot_id"])
    for i in range(len(conditions)):
        rel_prot_expr_DF[conditions[i]] = total_rel_prot_expr[i]
        
    # Only keep active genes from gene expression data (in all replicates).
    only_active_rel_transcript= rel_prot_expr_DF[rel_prot_expr_DF["gene_id"].isin(active_gene)]
    
    # Apply PCA. 
    results = PCA_tool(only_active_rel_transcript, ["gene_id", "prot_id"]
                       , conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}
    




def PC_EWFD(mov_direction, expr_direction):
    """
    This function generates all principal components for all conditions
    from PCA of EWFD scores.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    mov_direction: String. Path to the folder of all movement files (
    movements are applied for calculations of EWFD scores).
    expr_direction: String. Path to the folder of all expression files (
    gene expressions are needed for sorting out active genes). 
    """
    # Read movement data.
    mov_data = read_mov_data(mov_direction)
    
    all_mean_mov = mov_data[0]
    all_gene_id = mov_data[1]
    all_prot_id = mov_data[2]
    conditions = mov_data[3]
    
    # Sort out the list of active genes.
    active_gene = find_active_genes(expr_direction)
    
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
    only_active_ewfd = EWFD_table[EWFD_table["gene_id"].isin(active_gene)]
    
    # Apply PCA. 
    results = PCA_tool(only_active_ewfd, ["gene_id", "prot_id"]
                       , conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}


###########################################################################################

def write_JSON(expr_direction, mov_direction, writepath):
    """
    This function generate a .JSON file containing all principal components
    and information content for all four analysis levels.
    
    expr_direction: String. The path to the folder with all expression files.
    mov_direction: String. The path to the folder with all movement files.
    writepath: String. The path where the .JSON file should be generated. It
    has the format of "folder/name.json".
    """
    Data = {"gene_expr_PCA":PC_gene_expr(expr_direction)
            ,"transcript_expr_PCA":PC_transcript_expr(expr_direction)
            ,"relative_transcript_expr_PCA":PC_rel_transcript_expr(expr_direction)
            ,"EWDF_PCA": PC_EWFD(mov_direction, expr_direction)}
    pca_pc = json.dumps(Data, indent=4)  
    # If there no file with the same name exists, a new files will be 
    # generated. Otherwise data will be rewrote.
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write(pca_pc)

