import keras.backend as K
from keras.layers import Input,Dense,BatchNormalization,Activation,Conv1D,\
    Flatten,Reshape,Dropout,concatenate,MaxPooling1D,add
from keras import Model
from keras import regularizers

"""
DeepSigSurvNet Model File
implement with Keras 2.2.4
"""

class CustomConnected(Dense):
    '''
        Bulid not-fully connected model with connection matrix
    Args:
        units(int):output_size
        connection(np.array): connection matrix with the connected edges label as 1
                              and other label as 0. The dimension sould be (input_size,output_size)
    '''



    def __init__(self, units, connections, **kwargs):
        # this is connection matrix
        self.connections = connections
        # initalize the original Dense with all the usual arguments
        super(CustomConnected, self).__init__(units, **kwargs)

    def call(self, inputs):
        kernel=self.kernel * self.connections
        output = K.dot(inputs,kernel )
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


def inception(filter_size):
    '''
    inception module
    Args:
        filter_size(int): define the filter size in model
    '''

    def f(input_layer):
        inception1=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)

        inception2=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)
        inception2=Conv1D(filter_size,3,padding="same",activation="relu")(inception2)

        inception3=Conv1D(filter_size,1,padding="same",activation="relu")(input_layer)
        inception3=Conv1D(filter_size,3,padding="same",activation="relu")(inception3)
        inception3=Conv1D(filter_size,3,padding="same",activation="relu")(inception3)

        inception4=MaxPooling1D(padding="same",strides=1)(input_layer)
        inception4=Conv1D(filter_size,3,padding="same",activation="relu")(inception4)
        concat=concatenate([inception1,inception2,inception3,inception4],axis=-1)
        concat=Conv1D(filter_size,1,padding="same",activation="relu")(concat)

        #residual connection
        output_layer=add([input_layer,concat])
        return output_layer
    return f



def DeepSigSurvNet(gene_matrix, pathway_matrix, num_gene,num_feature, num_pathway,num_cli,dropout=0.2,reg=0.001):
    """
    DeepSigSurvNet model
    Args:
        gene_matrix(np.array): connection matrix of features to each gene
        pathway_matrix(np.array): conncetion matrix of genes to each pathways
        num_gene(int): number of gene in input.
        num_feature(int): number of feature in input (gene expression, copy number,etc)
        num_pathway(int): number of pathways
        num_cli(int): number of clinical features
        dropout(int): dropout rate
        reg(int): rate of l2 weight decay regularization

    """
    #Gene information
    input_shape1=(num_gene*num_feature,)
    gene_input=Input(input_shape1)
    gene_mix=CustomConnected(num_gene,gene_matrix)(gene_input)
    if (dropout > 0):
        gene_mix=Dropout(dropout)(gene_mix)
    inputModel=Model(gene_input,gene_mix)

    #Pathway information
    gene_info=inputModel(gene_input)
    gene_info_shape=K.int_shape(gene_info)[1:]
    gene_info_input=Input(gene_info_shape)
    pathway=CustomConnected(num_pathway,pathway_matrix)(gene_info_input)
    if (dropout > 0):
        pathway=Dropout(dropout)(pathway)
    geneModel=Model(gene_info_input,pathway)

    #Inception module to capture biological interaction from pathway information
    pathway_info=geneModel(gene_info)
    predict_shape=(num_pathway,)
    predict_input=Input(predict_shape)
    pathway=Reshape((num_pathway,-1))(predict_input)
    filter_list=[16,8,4]
    for i in range(len(filter_list)):
        if i==0:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(reg))(pathway)
        else:
            conv=Conv1D(filter_list[i],3,padding="same",activation="relu",kernel_regularizer=regularizers.l2(reg))(conv)

        conv=inception(filter_list[i])(conv)
        conv=BatchNormalization()(conv)
        conv=Activation("relu")(conv)
        if(dropout>0):
            conv=Dropout(dropout)(conv)

    dense=Flatten()(conv)
    dense=Dense(32,activation="relu",kernel_regularizer=regularizers.l2(reg))(dense)

    # combine clinical information
    input_shape2=(num_cli,)
    cli_input=Input(input_shape2)
    concat=concatenate([cli_input,dense])
    cli_conv=Dense(64,activation="relu",kernel_regularizer=regularizers.l2(reg))(concat)
    cli_dense=Dense(32,activation="relu")(cli_conv)
    #predict surivial time
    output=Dense(1,activation=None)(cli_dense)

    # divide model into several part for innvestigate analysis
    pathwayModel=Model([predict_input,cli_input],output)
    output=pathwayModel([pathway_info,cli_input])
    model=Model([gene_input,cli_input],output)
    return model,inputModel,geneModel,pathwayModel

