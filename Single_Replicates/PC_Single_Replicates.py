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
        
    return all_mean_mov, all_gene_id, all_prot_id, conditions, replicates




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
                
            
        temp_DF = pd.DataFrame(total_expr, columns = repl)
        DF = pd.concat([DF, temp_DF], axis = 1)
        
    # Only keep those genes that are presented in all replicates.
    non_zero = DF[(DF.T != 0).all()]
    active_gene = list(non_zero["gene_id"])
    
    return active_gene, non_zero




def PCA_tool_expr(DataFrame, delete_column, conditions, replicates):
    """
    This function applies PCA on the given DataFrame containing 
    expression data.
    The overall information content vector and principal components for 
    all replicates of all conditions will be returned.
    
    DataFrame: the DataFrame containing expression values.
    delete_column: the index of the column containing GeneID or (GeneID,
    ProtID).
    conditions: the list of involved conditions.
    replicates: the list of replicates for each condition. 
    """
    # Applying the PCA approach to generate the PC dataframe.
    pca_DF = DataFrame.drop(columns = delete_column)
    tdf = pca_DF.T
    ids = tdf.columns
    x = tdf.loc[:, ids].values
    y = pd.DataFrame(tdf.index.values, columns = ["repl"])
    targets = list(y["repl"])
    x = StandardScaler().fit_transform(x)
    # Number of all replicates = number of DF columns - 1 for gene expression
    # sind the first column contains gene IDs.
    if delete_column == "gene_id":
        n_repl = len(DataFrame.columns)-1
    else:
        # For transcript expressions, the first two columns are for IDs.
        n_repl = len(DataFrame.columns)-2
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
        repl = principalDf.iloc[temp_pos:temp_pos+len(replicates[i])].                to_dict("index")
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




def PCA_tool_mov(DataFrame, delete_column, replicates, repl_num
                , unique_cond):
    """
    This function applies PCA on the given DataFrame containing EWFD scores.
    The overall information content vector and principal components for 
    all replicates of all conditions will be returned.
    
    DataFrame: the DataFrame containing EWFD score.
    delete_column: the index of column with (GeneID, ProtID)
    replicates: the list of replicates for each conditions.
    repl_num: the list of number of replicates for each condition in the 
    order of conditions listed in unique_cond list.
    unique_cond: list of involved conditions.
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
    This function generates all principal components for all replicates and
    all conditions from PCA of gene expression.
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
    
    # Sort out active genes and generate the expression DataFrame only
    # containing active genes.
    active = find_active_genes(direction)
    active_gene = active[0]
    active_DF = active[1]
    
    # Apply PCA.
    results = PCA_tool_expr(active_DF, "gene_id"
                            , conditions, replicates)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}



def PC_transcript_expr(direction):
    """
    This function generates all principal components for all replicates and
    all conditions from PCA of transcript expressions. 
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
    
    # Sort out active genes.
    active = find_active_genes(direction)
    active_gene = active[0]
    
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
    
    DF_prot = pd.DataFrame(list(zip(all_gene, all_prot)), columns = ["gene_id", "prot_id"])
    
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

        temp_DF = pd.DataFrame(all_prot_expr, columns = repl)
        DF_prot = pd.concat([DF_prot, temp_DF], axis = 1)
        
    # Only keep the active genes in the gene expression data.
    active_gene_transcript= DF_prot[DF_prot["gene_id"].isin(active_gene)]
    
    # Apply PCA.
    results = PCA_tool_expr(active_gene_transcript, ["gene_id","prot_id"]
                            , conditions, replicates)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}



def PC_rel_transcript_expr(direction):
    """
    This function generates all principal components for all replicates and
    all conditions from PCA of relative transcript expressions. 
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
    
    # Sort out active genes.
    active = find_active_genes(direction)
    active_gene = active[0]
    
    # Extract all combinations (gene_id, prot_id). Because all .json files
    # have(should have) the same number of combinations. Could only take
    # the set of all combinations from data under the first condition. 
    all_gene = []
    all_prot = []
    data_cond1 = expression[0]
    repl_1 = replicates[0][0]  # Replicates of the first condition.
    for i in range(len(genes)):
        gene = genes[i]
        keys = list(data_cond1[gene][repl_1].keys())
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

        temp_DF = pd.DataFrame(all_prot_expr, columns = repl)
        DF_rel_prot = pd.concat([DF_rel_prot, temp_DF], axis = 1)
    
    
    # Only keep genes that are represented in the gene expression data.
    active_gene_rel_transcript= DF_rel_prot[DF_rel_prot["gene_id"].isin(active_gene)]
    
    # Apply PCA.
    results = PCA_tool_expr(active_gene_rel_transcript, ["gene_id","prot_id"]
                            , conditions, replicates)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"information_content":result_info, "conditions":cond_dict}




def PC_EWFD(mov_direction, expr_direction):
    """
    This function generates all principal components for all replicates and
    all conditions from PCA of EWFD scores.
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
    replicates = mov_data[4]
    
    # Sort out active genes.
    active = find_active_genes(expr_direction)
    active_gene = active[0]
    
    # Get all unique conditions and the number of replicates each condition
    # has.
    repl_num_dict = {i:conditions.count(i) for i in conditions}
    unique_cond = list(repl_num_dict.keys())
    repl_num = list(repl_num_dict.values())
        
    # Construct the dataframe containing all information.
    # First with two columns:
    EWFD_table = pd.DataFrame(list(zip(all_gene_id[0], all_prot_id[0],                                       1-np.asarray(all_mean_mov[0]))), columns = ["gene_id", "prot_id", replicates[0]])
    # Iterativly add other condition-mean_EWFD columns:
    for i in range(1,len(conditions)):
        EWFD_table[replicates[i]] = 1-np.asarray(all_mean_mov[i])
        
    # Only keep represented/active genes from gene expression data. 
    active_gene_ewfd = EWFD_table[EWFD_table["gene_id"].isin(active_gene)]
    
    # Apply PCA.
    results = PCA_tool_mov(active_gene_ewfd, ["gene_id","prot_id"]
                           , replicates, repl_num, unique_cond)
    result_info = results[0]
    cond_dict = results[1]
    
    return {"infotmation_content":result_info, "conditions":cond_dict}


##########################################################################################

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

