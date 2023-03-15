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
    # Read in data (.json files).
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
        
        
    # Extract condition, id and mean_mov information for each replicate.
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
        
        
    return conditions, replicates, all_mean_mov, all_id





def PCA_tool_expr(DataFrame, delete_column, replicates, conditions):
    """
    This function applies PCA on the given DataFrame containing expression
    data. The overall information content vector and principal components 
    for all replicates under all conditions will be returned.
    
    DataFrame: the DataFrame containing expression value.
    delete_column: the index of column with GeneID or (GeneID, ProtID)
    replicates: the list of replicates for each condition.
    conditions: the list of involved conditions. 
    
    """
    # Applying the PCA approach to generate the PC dataframe.
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["repl"])
    targets = list(y["repl"])
    x = StandardScaler().fit_transform(x)
    # Number of all replicates = number of DF columns - 1 because the first
    # column is the ids. 
    n_repl = len(DataFrame.columns)-1
    pca = PCA(n_components=n_repl)
    column = []
    for i in range(n_repl):
        column.append("PC"+str(i+1))
    principalComponents = pca.fit_transform(x)
    # Columns are the PCs and index replicates.
    principalDf = pd.DataFrame(data = principalComponents
                    , columns = column, index = y["repl"])
    
    
    # Iterativly add condition and its corresponded replicates with CP to
    # a dictionary.
    temp_pos = 0
    cond_dict = {}
    for i in range(len(conditions)):
        repl_dict = {}
        repl = principalDf.iloc[temp_pos:temp_pos+len(replicates[i])].to_dict("index")
        repl_dict["replicates"] = repl
        cond_dict[conditions[i]] = repl_dict
        temp_pos += len(replicates[i])
        
    
    # Generate the dictionary for general information content.
    info_content = pca.explained_variance_ratio_
    info_contentDf = pd.DataFrame(data = info_content
                                , columns = ["information_content"]
                                , index = column).T
    result_info = info_contentDf.to_dict("index")["information_content"]
    
    return result_info, cond_dict





def PCA_tool_mov(DataFrame, delete_column, unique_cond, repl_num
                , replicates):
    """
    This function applies PCA on the given DataFrame containing 
    EWFD scores calculated from movements and generate 
    the overall information content and principal components for all 
    replicates and conditions. 
     
    DataFrame: EWFD score DataFrame.
    delete_column: The index of the coulumn with (GeneID, ProtID) 
    unique_cond: list of the involved conditions (without repeat)
    repl_num: list of numbers of replicates for each condition.
    replicates: list of replicates for each condition. 
    
    """
    # Apply PCA approach on the EWFD data frame.
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["repl"])
    targets = list(y["repl"])
    x = StandardScaler().fit_transform(x)

    n_replicates = len(replicates)
    pca = PCA(n_components=n_replicates)
    column = []
    for i in range(n_replicates):
        column.append("PC"+str(i+1))
    principalComponents = pca.fit_transform(x)
    # Columns are the PCs and index replicates.
    principalDf = pd.DataFrame(data = principalComponents
                    , columns = column, index = y["repl"])
    
    # Information content of PCA approach. 
    info_content = pca.explained_variance_ratio_
    info_contentDf = pd.DataFrame(data = info_content
                            , columns = ["information_content"]
                            , index = column).T
    result_info = info_contentDf.to_dict("index")["information_content"]
    
    
    # Iterativly match the condition and all its replicates with 
    # corresponded PCs from principalDf. 
    temp_pos = 0
    cond_dict = {}
    for i in range(len(unique_cond)):
        repl_dict = {}
        repl = principalDf.iloc[temp_pos:temp_pos+repl_num[i]].to_dict("index")
        repl_dict["replicates"] = repl
        cond_dict[unique_cond[i]] = repl_dict
        temp_pos += repl_num[i]
        
    return result_info, cond_dict


###########################################################################################


def PC_gene_expr(direction):
    """
    This function generates all principal components for all conditions
    and all replicates from PCA of gene expression.
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
                
            
        temp_DF = pd.DataFrame(total_expr, columns = repl)
        DF = pd.concat([DF, temp_DF], axis = 1)
    
    # Apply PCA and generate the dictionary. 
    results = PCA_tool_expr(DF, "gene_id", replicates, conditions)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}





def PC_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    and all replicates from PCA of transcript expression.
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
    
    DF_prot = pd.DataFrame(list(zip(all_combine)), columns = ["gene_id,prot_id"])
    
    # Extract all expression values for all combinations (gene_id, prot_id)
    # for all conditions. And build up a data frame storing all expression
    # for each (gene_id, prot_id) under every condition.
    for i in range(len(conditions)):
        expr = expression[i]
        repl = replicates[i]
        cond = conditions[i]
        all_prot_expr = np.zeros((len(all_combine), len(repl)))
        for j in range(len(all_combine)):
            comb = all_combine[j]
            gene_id = comb[0]
            prot_id = comb[1]
            for k in range(len(repl)):
                repl_name = repl[k]
                all_prot_expr[j,k] = expr[gene_id][repl_name][prot_id]
        # Iterativly fulfill the data frame.
        temp_DF = pd.DataFrame(all_prot_expr, columns = repl)
        DF_prot = pd.concat([DF_prot, temp_DF], axis = 1)
    
    # Apply PCA and generate the dictionary. 
    results = PCA_tool_expr(DF_prot, "gene_id,prot_id", replicates, conditions)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}





def PC_rel_transcript_expr(direction):
    """
    This function generates all principal components for all conditions
    and all replicates from PCA of relative transcript expression.
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
            
    # Build up iterativly the data frame containing all relativ splice 
    # expression under each condition. 
    DF_rel_prot1 = pd.DataFrame(list(zip(all_combine)), 
                                columns = ["gene_id,prot_id"])
    
    for i in range(len(conditions)):
        expr = expression[i]
        repl = replicates[i]
        cond = conditions[i]
        all_prot_expr = np.zeros((len(all_combine), len(repl)))
        for j in range(len(all_combine)):
            comb = all_combine[j]
            gene_id = comb[0]
            prot_id = comb[1]
            for k in range(len(repl)):
                repl_name = repl[k]
                single_gene_expr = expr[gene_id][repl_name]["total"]
                if single_gene_expr != 0:
                    # Calculation relative expression. 
                    all_prot_expr[j,k] = expr[gene_id][repl_name][prot_id]/single_gene_expr
                # else: if the single gene is not expressed, then the isoforms
                # are also not expressed. In this case, the "score" will be 0

        temp_DF = pd.DataFrame(all_prot_expr, columns = repl)
        DF_rel_prot1 = pd.concat([DF_rel_prot1, temp_DF], axis = 1)
    
    # Apply PCA.
    results = PCA_tool_expr(DF_rel_prot1, "gene_id,prot_id", replicates, conditions)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}





def PC_EWFD(direction):
    """
    This function generates all principal components for all conditions
    and all replicates from PCA of EWFD scores.
    Also the overall information content vector is included.
    A dictionary of the information will be returned.
    
    direction: String. Path to the folder of all movement files. 
    """
    # Read movement data.
    mov_data = read_mov_data(direction)
    
    conditions =  mov_data[0]
    replicates = mov_data[1]
    all_mean_mov = mov_data[2]
    all_id = mov_data[3]
    
    # Get all unique conditions and the number of replicates each condition
    # has.
    repl_num_dict = {i:conditions.count(i) for i in conditions}
    # All unique conditions.
    unique_cond = list(repl_num_dict.keys())
    # The number of replicates for each condition from above list.
    repl_num = list(repl_num_dict.values())
    
    
    # Prepare the dataframe for PCA approach.
    EWFD_table = pd.DataFrame(list(zip(all_id[0], 1-np.asarray(all_mean_mov[0]))),                        columns = ["gene_id,prot_id", replicates[0]])
    for i in range(1,len(replicates)):
        EWFD_table[replicates[i]] = 1-np.asarray(all_mean_mov[i])
        
    # Apply PCA. 
    results = PCA_tool_mov(EWFD_table, "gene_id,prot_id", unique_cond, repl_num
                          , replicates)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}


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

