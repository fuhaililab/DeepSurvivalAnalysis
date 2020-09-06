from model_file import DeepSigSurvNet
import numpy as np
import innvestigate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
'''
analysis pathway in trained model
'''
#Argments
load_path="GBM_result" # load path
path="GBM"
num_feature=2
num_pathway=46
drop_prob=0.1
reg=0.001

#load data back
data_list=np.load(f"{load_path}/data.npz",allow_pickle=True)
train_x=data_list["train_x"]
train_cli=data_list["train_cli"]
train_vat=data_list["train_vat"]
train_y=data_list["train_y"]
train_y_normal=data_list["train_y_normal"]
train_y_max=data_list["train_y_max"]
train_y_min=data_list["train_y_min"]
test_x=data_list["test_x"]
test_cli=data_list["test_cli"]
test_vat=data_list["test_vat"]
test_y=data_list["test_y"]
test_y_normal=data_list["test_y_normal"]
gene_matrix=data_list["gene_matrix"]
pathway_matrix=data_list["pathway_matrix"]
num_gene=int(data_list["num_gene"])
num_cli=train_cli.shape[1]


#load model back
model,inputModel,geneModel,pathwayModel=DeepSigSurvNet(gene_matrix,pathway_matrix,num_gene,num_feature,num_pathway,num_cli,drop_prob,reg)
model.load_weights(f"{load_path}/{path}_model_weight.h5")
inputModel.load_weights(f"{load_path}/{path}_inputModel_weight.h5")
geneModel.load_weights(f"{load_path}/{path}_geneModel_weight.h5")
pathwayModel.load_weights(f"{load_path}/{path}_pathwayModel_weight.h5")


#investage pathway
new_X=np.vstack([train_x,test_x])
gene_x=np.array(inputModel.predict(new_X))
pathway_x=geneModel.predict(gene_x)
cli_x=np.vstack([train_cli,test_cli])
pathway_analyzer = innvestigate.create_analyzer("smoothgrad", pathwayModel,
                                                noise_scale=(np.max(pathway_x)-np.min(pathway_x))*0.1)
analysis = pathway_analyzer.analyze([pathway_x,cli_x])[0]

pdf = PdfPages(f'{load_path}/pathway_importance.pdf')
plt.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest',aspect="auto")
plt.ylabel("Sample Index")
plt.xlabel("Pathway")
plt.show()
analysis=analysis.squeeze()
np.savetxt(f"{load_path}/pathway_importance.txt",analysis)
pdf.savefig()
plt.close()
pdf.close()
