# NLP-for-Investment-Analysis
## Spring 2020 Final Project: 
#### Daeil Cha, Hyun Jae Park, Yotam Ghebre

The purpose of this project is to map the various investment styles of all the mutual funds registered with the U.S. Securities and Exchange Commission. Specifically, our approach is to represent each fund’s prospectus document as an embedding, and to measure the similarity between these embeddings. 

The data has been obtained from the government site data.gov. 

## Results
The following is the model results. Description of numbers are explained in report.pdf

| |TF-IDF | Doc2vec |bert nli mean-tokens | bert-nli-stsb mean-tokens | bert-nli max-tokens | bert-nli-mean-token using risk corpus|
|--- |------------- | ------------- | --- | --- |--- | ---|
| Benchmark  | 0.0862 | 0.0875| 0.0805 |0.0916| 0.0898 |0.0938|
| Score  | 0.1062 | 0.0933 |0.1147 |0.1267 |0.1122 |0.106|

## Directory Structure

    .
    ├── data  The following data can be accessed through Google Drive. Ask through hpark@gatech.edu                  
        ├── monthly                                                                 # monthly return data
        ├── original                                                                # text data
        ├── CRSP Mutual Funds - Total Lookup - 2019.csv                             # lookup table
        ├── CRSP Mutual Funds - Holdings 2019-01 to 2019-12 with names, ids.csv     # stock holdings table
        ├── CRSP Mutual Funds - Fund Summary 2010-01 to 2019-12.csv                 # fund class table
    ├── plots                                                                       # visualizations 
        ├── monthly                                                                 # Monthly return per cluster
        ├── lipper                                                                  # Classification code per cluster 
        ├── common_stocks                                                           # Common stocks per cluster
    ├── results                                                                     # model results as txt
    ├── report.pdf
    ├── run_main.py
    └── README.md
    
## Run the model by yourself

* if you want to run the model, please run the command.:
> `pip install -r requirements.txt`
> 
> `python run_main.py --model 'model_name' --bert_version 'bert-base-nli-mean-tokens'

* model_name can be {BERT, tf-idf, doc2vec}

