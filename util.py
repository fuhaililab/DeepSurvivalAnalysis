import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import ujson as json
'''
utils function file
'''

def build_gene_matrix(num_gene,num_feature):
    """
    function to bulid gene matrix
    Args:
        num_gene(int):number of gene in matrix
        num_feature(int): number of features in matrix
    """
    gene_matrix = []
    for i in range(num_gene):
        gene_index = [0 for j in range(num_feature * num_gene)]
        for j in range(num_feature):
            gene_index[i+j*num_gene] = 1
        gene_matrix.append(gene_index)

    gene_matrix = np.array(gene_matrix)
    gene_matrix = gene_matrix.T
    return gene_matrix

def build_pathway_matrix(num_gene,pathway2,geneSymbol):
    """
    function to build pathway matrix
    Args:
        num_gene(int):number of genes in matrix
        pathway2(pd.DataFrame):original pathway matrix, with every column be one pathway and
                            every row have a gene relate to that pathway.
        geneSymbol(pd.DataFrame):gene symbol that keep in the matrix.
    """
    pathway_matrix = []
    for i in range(pathway2.shape[1]) :
        p_index = [0 for k in range(num_gene)]
        p = pathway2.iloc[:, i]
        for j in range(len(p)):
            index = geneSymbol.loc[geneSymbol == p[j]].index.values
            if (len(index) != 0):
                p_index[index[0]] = 1
        pathway_matrix.append(p_index)
    pathway_matrix = np.array(pathway_matrix)
    pathway_matrix = pathway_matrix.T
    return pathway_matrix


def model_validation(model,train_x,train_cli,train_vat,train_y,train_y_normal,train_y_max,train_y_min,
                     test_x,test_cli,test_vat,test_y,test_y_normal,path):
    """
    model validation function
    """
    try:
        train_mse = model.evaluate([train_x, train_cli], train_y_normal)[1]
        train_predict_y_normal = model.predict([train_x, train_cli])
        train_predict_y = train_predict_y_normal * (train_y_max - train_y_min) + train_y_min
        train_corr = pearsonr(np.reshape(train_y, (-1)), np.reshape(train_predict_y, (-1)))[0]
        if path !="GBM":
            train_event = [False if x == "Alive" else True for x in train_vat["x_vital"]]
            test_event = [False if x == "Alive" else True for x in test_vat["x_vital"]]
        else:
            train_event = [False if x == "LIVING" else True for x in train_vat["vital_status"]]
            test_event = [False if x == "LIVING" else True for x in test_vat["vital_status"]]
        train_c_index = concordance_index(np.reshape(train_y, (-1)), np.reshape(train_predict_y, (-1)), train_event)
        test_mse = model.evaluate([test_x, test_cli], test_y_normal)[1]
        test_predict_y_normal = model.predict([test_x, test_cli])
        test_predict_y = test_predict_y_normal * (train_y_max - train_y_min) + train_y_min
        test_corr = pearsonr(np.reshape(test_y, (-1)), np.reshape(test_predict_y, (-1)))[0]
        test_c_index = concordance_index(np.reshape(test_y, (-1)), np.reshape(test_predict_y, (-1)), test_event)
    except Exception as e:
        train_c_index = 0
        test_c_index = 0
        train_corr = -1
        test_corr = -1
        train_mse = float("inf")
        test_mse = float("inf")

    return train_c_index,test_c_index,train_corr,test_corr,train_mse,test_mse

def save(filename,obj,message=None):
    """
    json file save function
    Args:
        filename(str): file save path
        obj(dict): the object that be saved
        message(str): output message
    """
    if message is not None:
        print(f"saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj,fh)

def load(path):
    """
    json file reload function
    Args:
        path(str):reload file path
    """
    with open(path,"r") as fh:
        data=json.load(fh)
    return data