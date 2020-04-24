#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict
import seaborn as sb
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import argparse




parser = argparse.ArgumentParser(description='model')

parser.add_argument('--model', default="BERT", type=str, help="BERT,tf-idf,doc2vec")

parser.add_argument('--bert_version', default="bert-base-nli-mean-tokens", type=str, help="bert version can be looked up at sentence transformer")

def run(args) :
    define_model = args.model

    bert_version = args.bert_version

    # # Import Dataset

    # In[2]:


    # Import 2019 text dataset
    # dfo_1 = pd.read_table('../data/original/2019q1_rr1/txt.tsv')
    # dfo_2 = pd.read_table('../data/original/2019q2_rr1/txt.tsv')
    # dfo_3 = pd.read_table('../data/original/2019q3_rr1/txt.tsv')
    # dfo_4 = pd.read_table('../data/original/2019q4_rr1/txt.tsv')

    # dfo = pd.concat([dfo_1,dfo_2,dfo_3,dfo_4])

    # Holdings CSV
    print("running")
    dfn2 = pd.read_csv('../data/CRSP Mutual Funds - Holdings 2019-01 to 2019-12 with names, ids.csv')


    # In[3]:

    print("getting text data...")

    # Text CSV
    dfo = pd.read_table('../data/original/2019q4_rr1/txt.tsv')

    dfn2 = dfn2[dfn2["Series ID"].isin(dfo.series.unique())]
    dfo.dropna(subset=["series"],inplace=True)
    dfo = dfo[dfo.series.isin(dfn2["Series ID"].unique())]
    dfo_a = dfo[["tag",'series','value']]
    dfo_a = dfo_a[dfo_a.tag.isin(['ObjectivePrimaryTextBlock', 'RiskNarrativeTextBlock',  'RiskReturnHeading', 'StrategyNarrativeTextBlock'])]
    dfo_a_strategy = dfo_a[dfo_a.tag == 'RiskNarrativeTextBlock'].dropna()
    df_corpus = dfo_a_strategy[["series","value"]].drop_duplicates(subset =["value"]).reset_index()
    corpus = df_corpus['value'].to_list()
    series_find = df_corpus.to_dict()['series']



    # In[4]:


    # Monthly return csv
    df_mth = pd.read_csv('../data/monthly/CRSP Mutual Funds - Monthly Returns 2010-01 to 2019-12.csv')

    df_lookup = pd.read_csv('../data/CRSP Mutual Funds - Total Lookup - 2019.csv')

    df_lu2 = df_lookup[["Series ID","crsp_fundno"]]

    df_mth2 = df_mth.merge(df_lu2,how="inner",on = 'crsp_fundno')

    df_mth2.caldt = df_mth2.caldt.astype(str).str[:-2]

    df_mth2 = df_mth2.replace("R",None)

    df_mth2.dropna(subset=["mret"],inplace=True)


    # In[5]:


    # Lipper Class name csv


    df_lip = pd.read_csv("../data/CRSP Mutual Funds - Fund Summary 2010-01 to 2019-12.csv")

    df_lip.dropna(subset= ["crsp_obj_cd"],inplace=True)

    df_lip2 = df_lip[["crsp_fundno","crsp_obj_cd"]]

    df_lip3 = df_lip2.merge(df_lu2,how="inner",on = 'crsp_fundno')


    # # Models

    # In[ ]:


    def embeddings(model,bert_ = None):
        if model == "BERT":
            if bert_ == None :
                embedder = SentenceTransformer('bert-base-nli-mean-tokens')
            else :
    #           bert_ = 'bert-base-nli-stsb-mean-tokens'
                embedder = SentenceTransformer(bert_)
            corpus_embeddings = embedder.encode(corpus)
            
        elif model == "tf-idf" :
            vectorizer = TfidfVectorizer(stop_words='english')
            corpus_embeddings = vectorizer.fit_transform(corpus)
        
        elif model == 'doc2vec' :
            labeled_corpus= [TaggedDocument(doc, [str(i)]) for i,doc in enumerate(corpus)]
            model = Doc2Vec(dm = 1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
            model.build_vocab(labeled_corpus)
            model.train(labeled_corpus,epochs=model.iter,total_examples=model.corpus_count)
            corpus_embeddings = []
            for index, row in df_corpus.iterrows():
                corpus_embeddings.append(model.docvecs[index])
        
        return corpus_embeddings

    corpus_embeddings = embeddings(define_model,bert_version)


    # In[ ]:


    # Bert pre trained models
    # bert-base-nli-mean-tokens
    # bert-base-nli-max-tokens
    # roberta-base-nli-mean-tokens
    # distilbert-base-nli-mean-tokens

    # bert-base-nli-stsb-mean-tokens
    # roberta-large-nli-stsb-mean-tokens
    # distilbert-base-nli-stsb-mean-tokens


    # # K-means Clustering

    # In[ ]:


    def elbow(vector):
        sse = {}
        for k in range(10,20):
            kmeans = KMeans(n_clusters=k, max_iter=10000).fit(vector)
            sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()
    elbow(corpus_embeddings)

    plt.savefig('./plots/elbow.png')


    # In[ ]:


    # choose number of clusters, fit the model
    num_clusters = 20
    clustering_model = KMeans(n_clusters = num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_


    # In[ ]:


    # print out the clusters
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")
    #     break


    # In[ ]:


    # Get indicies from our clustering
    cluster_index = defaultdict(list)
    for index,i in enumerate(cluster_assignment) :
        cluster_index[i].append(index)


    # # Evaluate Clustering Model

    # In[ ]:

    print("Creating Bucket")

    if 'buckets' not in locals():
        buckets = {}
        count = 0
        size_dfn = len(dfn2["Series ID"].unique())
        for i in dfn2["Series ID"].unique():
            count+=1
            if count % 100  == 0 :
                print(count,'/',size_dfn)

            buckets[i] = set(dfn2[dfn2["Series ID"] == i].cusip)


    # In[ ]:


    # Evaluate clustering based on portfolio. Good Clustering model will show that they have common stock holdings

    def metric(fund1,fund2):
        bucket1 = buckets[fund1]
        bucket2 = buckets[fund2]
        
        bucket1_size = len(bucket1)
        bucket2_size = len(bucket2)
        
        if bucket1_size == 0 or bucket2_size == 0 :
            return 'no stocks in fund'
        
        common_stocks = len(bucket1.intersection(bucket2)) 
    #   score = common_stocks
        score = (common_stocks/bucket1_size + common_stocks/bucket2_size)/2
        
        return score


    # In[ ]:


    # Get the series ID for each clusters
    cluster_seriesid = {}

    for cluster in range(num_clusters):
        cluster_size = len(cluster_index[cluster])
        print('Cluster_{} size'.format(cluster),cluster_size)
        cluster_seriesid[cluster] = [series_find[cluster_index[cluster][i]] for i in range(cluster_size)]


    # In[ ]:


    # Get within value

    within_score = 0
    count = 0

    for cluster in range(num_clusters):
        
        for fund1 in cluster_seriesid[cluster] :
            for fund2 in cluster_seriesid[cluster] :
                if fund1 == fund2 :
                    continue
                count += 1
                score = metric(fund1,fund2)
                within_score += score 
                

    final_within_score = round(within_score/count,4)

    print("within score",final_within_score)


    # In[ ]:


    # Get random sample value
    random_sample_score = 0
    count = 0

    np.random.seed(100)

    for cluster in range(num_clusters):
        size = len(cluster_seriesid[cluster])
        random_sample = np.random.choice(dfn2["Series ID"].unique(),size,replace=False)
        
        for fund1 in cluster_seriesid[cluster] :
            for fund2 in random_sample :
                count += 1
                score = metric(fund1,fund2)
                random_sample_score += score 
                

    final_random_score = round(random_sample_score/count,4)

    print("Benchmark_score",final_random_score)


    # In[ ]:




    result = pd.Series([define_model,bert_version,"Clustering Score",final_within_score,"Benchmark",final_random_score])

    result.to_csv(r'./result_{}_{}.txt'.format(define_model,bert_version), header=None, index=None)


    # # Get the Lipper Class distribution for each cluster

    # In[ ]:


    def lipper(cluster):
        
        df_lip4 = df_lip3[df_lip3["Series ID"].isin(cluster_seriesid[cluster])].copy()
        
        arr_slice = df_lip4["crsp_obj_cd"].values

        # Get unique IDs corresponding to each linear index (i.e. group) and grouped counts
        unq,unqtags,counts = np.unique(arr_slice,return_inverse=True,return_counts=True)

        count_dict = dict(zip(unq, counts))
        
        sorted_dict = sorted(count_dict.items(), key = 
                 lambda kv:(kv[1] , kv[0]),reverse=True)[:7]
        
        dfdf = pd.DataFrame({"lipper_class": [i for i,_ in sorted_dict],"count":[j for _,j in sorted_dict]})
        
        plt.figure(figsize = (15,5))
        splot = sb.barplot(x='lipper_class',y='count',data=dfdf,color="darkblue")
        plt.title("Common CRSP for cluster {} ".format(cluster),fontsize = 25)
        plt.xticks(fontsize=10)

    for i in range(num_clusters):
        lipper(i)
        plt.savefig("./plots/lipper/lipper_class_cluster {}".format(i))
        


    # # Get the Top 5 most Common Stocks for each Cluster
    # 
    # 
    # 

    # In[ ]:


    def cs_graph(cluster) :
        stock_bucket = []
        df_temp = dfn2[["Series ID","security_name"]].copy()
        df_temp = df_temp[df_temp["Series ID"].isin(cluster_seriesid[cluster])]
        
        arr_slice = df_temp["security_name"].values

        # Get unique IDs corresponding to each linear index (i.e. group) and grouped counts
        unq,unqtags,counts = np.unique(arr_slice,return_inverse=True,return_counts=True)

        count_dict = dict(zip(unq, counts))
        
        sorted_dict = sorted(count_dict.items(), key = 
                 lambda kv:(kv[1] , kv[0]),reverse=True)[:5]
        
        dfdf = pd.DataFrame({"security_name": [i for i,_ in sorted_dict],"count":[j for _,j in sorted_dict]})
        
        plt.figure(figsize = (20,5))
        splot = sb.barplot(x='security_name',y='count',data=dfdf,color="darkblue")
        plt.title("5 Most Common stocks for cluster {} ".format(cluster))
        plt.xticks(fontsize=10)
        
    for i in range(num_clusters):
        cs_graph(i)
        plt.savefig('./plots/common_stocks/ Common_stocks_cluster {}'.format(i))


    # # Get Common Words for each Cluster using bag of words

    # In[ ]:


    # Get bag of words for cluster
    def bag_of_words(cluster):
        corpus_2 = clustered_sentences[cluster]
        vec = CountVectorizer().fit(corpus_2)
        bag_of_words = vec.transform(corpus_2)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        return words_freq


    # # Examine Average Monthly/Annual Return for each Cluster 

    # In[ ]:


    # Based on our clustering get the monthly return for each cluster


    # Line/Bar Graph for all values
    def return_graph(cluster,year):
        
        df_mth_19 = df_mth2[df_mth2.caldt.str[:4] == year]
        
        cluster_series_id = cluster_seriesid[cluster]
        
        df_mth_filt = df_mth_19[df_mth_19["Series ID"].isin(cluster_series_id)].copy()
        df_mth_filt.mret = df_mth_filt.mret.astype(float)

        
        graph_df = df_mth_filt.groupby("caldt").mean().reset_index()[['caldt','mret']]
        graph_df.columns = ["Returns",'Value']
        
        plt.figure(figsize=(11,7))
        line_plot = sb.lineplot(x='Returns',y='Value',data=graph_df)
        splot = sb.barplot(x='Returns',y='Value',data=graph_df,color="darkblue")
        plt.title('Average monthly Returns for cluster {} for year {}'.format(cluster,year))
        plt.ylim(-0.2, 0.2)
        for p in splot.patches:
            if p.get_height() > 0:
                splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else :
                splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, -10), textcoords = 'offset points')

    for i in range(num_clusters):
        return_graph(i,'2019')
        plt.savefig('./plots/monthly/monthly_return cluster{}.png'.format(i))
        
        
        
def main():
    args = parser.parse_args()
    # if args.verbose is True:
    #     print('Using the specified args:')
    #     print(args)

    run(args)

if __name__ == '__main__':
    main()