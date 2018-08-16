def AreaFindings(locationArea):
    for word in locationArea:
        if "Municipality" in word and "District" not in word:
            Municipalitiyarea = word
            break
        elif "City" in word and "District" not in word:
            Municipalitiyarea = word
            break
        elif "District" in word :
            Municipalitiyarea = word
            break
        else:
            Municipalitiyarea = 'Others-SA'
    return Municipalitiyarea

def Clustering(Fields,NoofCluster,TopKeyWords):

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # Fields = _tempData['listing_title'];NoofCluster = 5
    # Fields = Fields.str.title()

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(Fields)

    model = KMeans(n_clusters=NoofCluster, init='k-means++', max_iter=100, n_init=1,random_state=12345)
    model.fit(X)

    # j = 48
    tempCluster = pd.DataFrame(columns = ['listing_title','ClusterInformation'])
    for j in range(len(Fields)):
        Y = vectorizer.transform([Fields[j]])
        prediction = model.predict(Y)
        tempCluster = tempCluster.append({'listing_title': Fields[j],'ClusterInformation': prediction[0]}, ignore_index=True)

    # Key Terms per cluster
    tempClusterWords = pd.DataFrame(columns=['ClusterInformation','KeyWords','Key'])
    centroids = model.cluster_centers_.argsort()[:, ::-1]
    _terms = vectorizer.get_feature_names()

    for NClus in range(NoofCluster):
        KeyTerms = ''
        for Terms in centroids[NClus, :TopKeyWords]:
            KeyTerms += _terms[Terms] + ' '
        tempClusterWords = tempClusterWords.append({'ClusterInformation': NClus,'KeyWords': KeyTerms}, ignore_index=True)


    return tempCluster, tempClusterWords

def recommend(prodName,numberRecommendation):

    # prodName = 'Angel care diaper bin'
    # numberRecommendation = 200
    _tempproduct = ProductClusterinformation[ProductClusterinformation['listing_title'] == prodName].reset_index(drop=True)

    # UniqueCluster = list(_tempproduct['ClusterInformation'].unique())
    # C = 0
    list(_tempproduct)
    _tempRecommendationD = ProductClusterinformation[(ProductClusterinformation['ClusterInformation'] == _tempproduct['ClusterInformation'][0]) & (ProductClusterinformation['Key'] == _tempproduct['Key'][0])].reset_index(drop=True)
    _tempRecommendationD['index'] = range(0, len(_tempRecommendationD))

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(_tempRecommendationD['listing_title'])
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    results = {}
    for idx, row in _tempRecommendationD.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-len(_tempRecommendationD):-1]
        similar_items = [(cosine_similarities[idx][i], _tempRecommendationD['listing_title'][i]) for i in
                         similar_indices]
        results[row['listing_title']] = similar_items[1:]

    recscore = pd.DataFrame(results[prodName][:len(_tempRecommendationD)],columns=['Score','listing_title'])
    ScoreWithPrice = recscore.merge(_tempRecommendationD,how='inner')
    ScoreWithPrice = ScoreWithPrice.loc[:,['listing_title','Score','listing_price']]

    from sklearn import preprocessing
    minmax_scale = preprocessing.MinMaxScaler().fit(ScoreWithPrice[['Score']])
    ScoreWithPriceScaling = pd.DataFrame(minmax_scale.transform(ScoreWithPrice[['Score']]),columns=['Score'])
    ScoreWithPriceScaling['listing_title'] = ScoreWithPrice['listing_title']
    ScoreWithPriceScaling['price'] = ScoreWithPrice['listing_price']

    Recommendation = ScoreWithPriceScaling.sort_values(['Score','price'], ascending=[0, 1]).reset_index(drop=True)

    if(numberRecommendation > len(Recommendation)):
        numberRecommendation = len(Recommendation)

    FinalRecommendation = Recommendation.loc[0:numberRecommendation - 1, ['listing_title', 'price']]

    return FinalRecommendation
