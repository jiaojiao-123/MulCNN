# MulCNN
__MulCNN__ (Multiscale Convolutional Neural Networks) provides a deep learning method for cell type identification for scRNA-seq analysis. Using multi-scale convolution while filtering noise, the method extracts key features specific to individual cell types.

The repository contains the source code for the paper MulCNN, a novel neural network framework for single-cell RNA-Seq analysis.
Due to github's limit on uploading file size, the trained model is over 500M, so I only uploaded the core code. If you need a trained model for your research, you can contact me by email.624189018@qq.com.

The neural network model is implemented using TensorFlow 2.4.0 and the code is written in python 3.6. 

## Quick Start

MulCNN accepts scRNA-seq data format: CSV
### 1. Prepare datasets

#### CSV format

Take an example of Lawlor datasets ([GSE86469](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86469)) analyzed in the manuscript.

```shell
mkdir GSE86469
wget -P GSE86469/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE86469/suppl/GSE86469_counts.csv.gz
```

### 2. Preprocess input files
The preprocessed files generated in this step can be found in the uploaded data/example data set.csv
 
In preprocessing, parameters are used:

- **filetype** defines file type (CSV))  
- **geneSelectnum** selects a number of most variant genes. The default gene number is 2000
- **code** preprocessing.py

### 3. Run scGNN

We take an example of an analysis in GSE86469. Here we use parameters to demo purposes:

- **batch-size** defines batch-size of the cells for training.here we set as 32.
- **epoch** defines epochs in feature autoencoder, here we set as 300.
- **pca_num** defines the dimensionality of pca downscaling, here we set as 240.

If you want to reproduce results in the manuscript, please use default parameters. 


## Contributing

Souce code: [Github](https://github.com/jiaojiao-123/MulCNN)  
Author email: s20070042@s.upc.edu.cn
<br>
We are continuing adding new features. Bug reports or feature requests are welcome.
<br>
