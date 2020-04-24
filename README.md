# NLP-for-Investment-Analysis

This purpose of this project is to map the various investment styles of all the mutual funds registered with the U.S. Securities and Exchange Commission. Specifically, our approach is to represent each fund’s prospectus document as an embedding, and to measure the similarity between these embeddings. 

  todo : change below based on final report 

The data has been obtained from the government site data.gov. We have used four
separate methods for document representation, and we are in the process of determining which method is best for our purposes.
Unfortunately, our directory of data is too large for GitHub so it must be retrieved from: https://drive.google.com/drive/folders/1x23cEMkev1SxZ50LUlBzD-u0F8e3bbs8?usp=sharing

## Results
We evaluated our model by ... 

| |TF-IDF | Doc2vec |bert nli mean-tokens | bert-nli-stsb mean-tokens | bert-nli max-tokens | bert-nli-mean-token using risk corpus|
|--- |------------- | ------------- | --- | --- |--- | ---|
| Benchmark  | 0.0862 | 0.0875| 0.0805 |0.0916| 0.0898 |0.0938|
| Score  | 0.1062 | 0.0933 |0.1147 |0.1267 |0.1122 |0.106|

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

