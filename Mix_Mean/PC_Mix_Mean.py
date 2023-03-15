#!/usr/bin/env python
# coding: utf-8

import glob,os
import pandas as pd
import numpy as np
import scipy as sp
import json
from json import JSONEncoder
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




def PCA_tool(DataFrame, delete_column, conditions):
    """
    This function applies PCA on the given DataFrame and calculate all 
    the principal components for each condition. The over all information
    content is also returned.
    Results are returned as dictionary format. 
    
    DataFrame: the pandas DataFrame applied with PCA.
    delete_column: the index for the column with GeneID or (GeneID, ProtID).
    conditions: the list of conditions. 
    
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
        
    # Apply PCA.
    results = PCA_tool(all_gene_expr_DF, "gene_id", conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}




def PC_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    from PCA of transcript expression.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all expression files.
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
    prot_expr_DF = pd.DataFrame(list(zip(all_combine)),                                 columns = ["gene_id, prot_id"])
    for i in range(len(conditions)):
        prot_expr_DF[conditions[i]] = total_prot_expr[i]
        
        
    # Apply PCA.
    results = PCA_tool(prot_expr_DF, "gene_id, prot_id", conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}





def PC_rel_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    from PCA of relative transcript expression.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all expression files. 
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
    
    # Apply PCA.
    results = PCA_tool(rel_prot_expr_DF, "gene_id, prot_id", conditions)
    result_info = results[0]
    result_principal = results[1]
    
    return {"information_content":result_info, "conditions":result_principal}





def PC_EWFD(direction):
    """
    This function generates all principal components for all conditions
    from PCA of EWFD scores.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all movement files. 
    """
    # Read in data.
    mov_data = read_mov_data(direction)
    
    conditions = mov_data[0]
    all_mean_mov = mov_data[1]
    all_id = mov_data[2]
    
    # Check if all files have the same id-orders and number of combinations.
    for i in range(len(conditions)-1):
        if all_id[i] != all_id[i+1]:
            print("The number of combination or id-order is not the same!")
    # If id id lists are the same, then just take the first one.
    
    
    # Construct the dataframe containing all information.
    # First with two columns:
    EWFD_table = pd.DataFrame(list(zip(all_id[0], 1-np.asarray(all_mean_mov[0]))),                             columns = ["gene_id, prot_id", conditions[0]])
    # Iterativly add other condition-mean_mov columns:
    for i in range(1,len(conditions)):
        EWFD_table[conditions[i]] = 1-np.asarray(all_mean_mov[i])
    

    # Apply PCA.
    results = PCA_tool(EWFD_table, "gene_id, prot_id", conditions)
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
    Data = {"gene_expr_PCA":PC_gene_expr(expr_direction),            "transcript_expr_PCA":PC_transcript_expr(expr_direction),            "relative_transcript_expr_PCA":PC_rel_transcript_expr(expr_direction),            "EWDF_PCA": PC_EWFD(mov_direction)}
    pca_pc = json.dumps(Data, indent=4)  
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write(pca_pc)

