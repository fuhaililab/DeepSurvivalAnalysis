import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_file import DeepSigSurvNet
from tensorflow import set_random_seed
import os
import util
'''
training file. before run the file, fill in the path variable.
'''
#argments
seed=234
num_feature=2
drop_prob=0.1
reg=0.0001
epoch=5
batch_size=32
num_pathway=46
path="GBM"  #BRCA, SKCM, GBM,etc.
save_path="GBM_result"

set_random_seed(seed)
np.random.seed(seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)


#read data from data file
pathway2=pd.read_table("data/Selected_Kegg_Pathways2.txt",sep="\t")
gene_cnv2=pd.read_table(f"data/{path}-gene_cnv_2.txt",sep="\t")
gene_exp2=pd.read_table(f"data/{path}-gene_exp_2.txt",sep="\t")
cli_info2=pd.read_table(f"data/{path}-cli_info_2.txt",sep="\t")

geneSymbol=gene_cnv2["rownames(x2)"]
num_original_gene=gene_exp2.shape[0]

if path !="GBM":
    Y=np.array(cli_info2["x_survival_time"])
else:
    Y=np.array(cli_info2["CDE_survival_time"])

# remove samples that survival time larger than 3000
time_keep_index=np.where(Y<=3000)[0]
Y=Y[time_keep_index]
gene_exp2.drop(["rownames(x1)"],axis=1,inplace=True)
gene_cnv2.drop(["rownames(x2)"],axis=1,inplace=True)
pathway2.drop(["AllGenes"],axis=1,inplace=True)
gene_exp_arr=np.array(gene_exp2)[:,time_keep_index]
gene_cnv_arr=np.array(gene_cnv2)[:,time_keep_index]

# concatenate two feature(gene expression and copy number)
X=np.vstack([gene_exp_arr,gene_cnv_arr])
X=X.T

#clinical information
cli_info2=cli_info2.iloc[time_keep_index.tolist(),:]
if path!="GBM":
    clinical_variables=cli_info2[["x_age","x_sex","x_vital","x_stage"]]
    clinical_dummy=pd.get_dummies(clinical_variables,columns=["x_sex","x_vital","x_stage"])
    clinical_variables=np.array(clinical_dummy)
else:
    clinical_variables=cli_info2[["CDE_DxAge","gender","new_neoplasm_event_type","vital_status"]]
    clinical_dummy=pd.get_dummies(clinical_variables,columns=["gender","new_neoplasm_event_type","vital_status"])
    clinical_variables=np.array(clinical_dummy)
num_cli=clinical_variables.shape[1]

#train test split
train_x,test_x,train_y,test_y,train_cli,test_cli,train_vat,test_vat=train_test_split(X,Y,clinical_variables,cli_info2,
                                                                                     test_size=0.2,random_state=seed, shuffle=True)
#normalize survival time
train_y_min=np.min(train_y)
train_y_max=np.max(train_y)
train_y_normal=(train_y-train_y_min)/(train_y_max-train_y_min)
test_y_normal=(test_y-train_y_min)/(train_y_max-train_y_min)

#normalize clinical feature
train_age_max=np.max(train_cli[:,0])
train_age_min=np.min(train_cli[:,0])
train_age_normal=(train_cli[:,0]-train_age_min)/(train_age_max-train_age_min)
test_age_normal=(test_cli[:,0]-train_age_min)/(train_age_max-train_age_min)
train_cli[:,0]=train_age_normal
test_cli[:,0]=test_age_normal

#remove genes that have zero expression across train set.
#normalize gene expression, fill abnormal expression value.
train_gene_quan=np.quantile(train_x[:,0:num_original_gene],[0.01,0.99],axis=0)
train_gene_min=train_gene_quan[0]
train_gene_max=train_gene_quan[1]
train_gene_range=train_gene_max-train_gene_min
keep_index=train_gene_range>0
train_gene_range=train_gene_range[keep_index]
train_gene_min=train_gene_min[keep_index]
geneSymbol_keep=geneSymbol[keep_index]
num_gene=len(geneSymbol_keep)

#remove zero expression genes from dataset.
geneSymbol.index=[i for i in range(num_gene)]
train_x=train_x[:,np.hstack([keep_index,keep_index])]
test_x=test_x[:,np.hstack([keep_index,keep_index])]

#renormalize gene expression to [-1,1]
train_gene_normal = (train_x[:, 0:num_gene] - train_gene_min) / train_gene_range * 2 - 1
train_gene_normal[train_gene_normal > 1] = 1
train_gene_normal[train_gene_normal < -1] = -1
train_x = np.hstack([train_gene_normal, train_x[:, num_gene:2 * num_gene]])

test_gene_normal = (test_x[:, 0:num_gene] - train_gene_min) / train_gene_range * 2 - 1
test_gene_normal[test_gene_normal > 1] = 1
test_gene_normal[test_gene_normal < -1] = -1
test_x = np.hstack([test_gene_normal, test_x[:, num_gene:2 * num_gene]])

#bulid gene matrix
gene_matrix=util.build_gene_matrix(num_gene,num_feature)
#pathway matrix
pathway_matrix=util.build_pathway_matrix(num_gene,pathway2,geneSymbol_keep)

#build model
model,inputModel,geneModel,pathwayModel=DeepSigSurvNet(gene_matrix,pathway_matrix,num_gene,num_feature,num_pathway,num_cli,drop_prob,reg)
model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              metrics=['mse'])


#train the model, validate model after each epoch. save the model with best test c-index
best_train_c_index = 0
best_test_c_index = 0
best_train_corr = -1
best_test_corr = -1
best_train_mse = float("inf")
best_test_mse = float("inf")
best_model=None
best_inputModel=None
best_geneModel=None
best_pathwayModel=None

for j in range(epoch):

    model.fit([train_x, train_cli], train_y_normal, batch_size=batch_size, epochs=1, shuffle=False)
    train_c_index,test_c_index,train_corr,test_corr,train_mse,test_mse=\
                                util.model_validation(model, train_x, train_cli, train_vat, train_y, train_y_normal, train_y_max, train_y_min,
                                                      test_x, test_cli, test_vat, test_y, test_y_normal,path)

    if best_test_c_index < test_c_index:
        best_train_c_index = train_c_index
        best_test_c_index = test_c_index
        best_train_corr = train_corr
        best_test_corr = test_corr
        best_train_mse = train_mse
        best_test_mse = test_mse
        best_model=model
        best_geneModel=geneModel
        best_inputModel=inputModel
        best_pathwayModel=pathwayModel

result = {"train_c_index": best_train_c_index,
          "test_c_index": best_test_c_index,
          "train_corr": best_train_corr,
          "test_corr": best_test_corr,
          "train_mse": best_train_mse,
          "test_mse": best_test_mse}

#save all the result

best_model.save_weights(f"{save_path}/{path}_model_weight.h5")
best_inputModel.save_weights(f"{save_path}/{path}_inputModel_weight.h5")
best_geneModel.save_weights(f"{save_path}/{path}_geneModel_weight.h5")
best_pathwayModel.save_weights(f"{save_path}/{path}_pathwayModel_weight.h5")

util.save(f"{save_path}/validation_result.json",result,f"save result for {path} cancer")

np.savez(f"{save_path}/data.npz",train_x=train_x, train_cli=train_cli, train_vat=train_vat,
         train_y=train_y, train_y_normal=train_y_normal, train_y_max=train_y_max,
         train_y_min=train_y_min,test_x=test_x, test_cli=test_cli,
         test_vat=test_vat, test_y=test_y, test_y_normal=test_y_normal,gene_matrix=gene_matrix,
         pathway_matrix=pathway_matrix,num_gene=num_gene)




