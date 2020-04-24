# NLP-for-Investment-Analysis

This purpose of this project is to map the various investment styles of all the mutual funds registered with the U.S. Securities and Exchange Commission. Specifically, our approach is to represent each fund’s prospectus document as an embedding, and to measure the similarity between these embeddings. 

  todo : change below based on final report 

The data has been obtained from the government site data.gov. We have used four
separate methods for document representation, and we are in the process of determining which method is best for our purposes.

## Results
We evaluated our model by ... 

| |BERT  | TF-IDF | Doc2vec |
|--- |------------- | ------------- | --- |
| Benchmark  | 0.0812312 | 0.0912312 | 0.0875 |
| Score  | 0.11123  | 0.10123123 | 0.09333333 |

## Directory Structure

    .
    ├── data                   
        ├── monthly                                                                 # monthly return data
        ├── original                                                                # text data
        ├── CRSP Mutual Funds - Total Lookup - 2019.csv                             # lookup table
        ├── CRSP Mutual Funds - Holdings 2019-01 to 2019-12 with names, ids.csv     # stock holdings table
        ├── CRSP Mutual Funds - Fund Summary 2010-01 to 2019-12.csv                 # fund class table
    ├── plots                                                                       # visualizations 
    ├── results                                                                     # models Results
    ├── run_main.py
    └── README.md
    
## Run the model by yourself

* if you want to run the model, please run the command.:
> `pip install -r requirements.txt`
> 
> `python run_main.py --model 'model_name' --bert_version 'bert-base-nli-mean-tokens'

* model_name can be {BERT, tf-idf, doc2vec}

