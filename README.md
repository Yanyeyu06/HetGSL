# HetGSL: Self-supervised Heterogeneous Graph Structure Learning
## Environment Settings
> python>=3.6 \
> scipy==1.5.4 \
> torch==1.10.0 \
> dgl==0.6.1 \
> numpy==1.19.5 \


## Usage
You can use the following commend to run our model: 
> python main.py 

Here, we use "DBLP" dataset as an example. If you want to run other datasets, you can download them at "https://github.com/cynricfu/MAGNN/tree/master/data/raw".

## Base Encoder
For the Encoder module of HetGSL, we adopt the variant of HAN as its base encoder. Please refer to HAN repository: [https://github.com/Jhy1993/HAN](https://github.com/Jhy1993/HAN).


## Some tips in parameters
1. We suggest you to carefully select the *“sema_th”*  to ensure the threshold of postives for every links. This is very important to final results.
2. In main.py, except "lr" and "patience", meticulously tuning dropout and tau is applaudable.

## Contact
If you have any questions, please feel free to contact me with yanyeyu-work@foxmail.com
