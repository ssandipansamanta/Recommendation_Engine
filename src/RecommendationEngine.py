# User Inputs
path = 'C:/Excercise/RecommendationEngine/'
specificCategory = 'Kids & Baby'


'''-->>> 
    Question-a:
            Approach    ----->
                        The recommendation Engine is build using 'za_sample_listings_incl_cat' file. It's a content based recommendation problem.
                        In the sample file, with Product and selling information, Category and location information also available. Geographical 
                        location is quite important to know as it's very unlikely to sell/buy a product to a different location. Recommendation is 
                        applied by location and for a category level
                        
                        Offline:(Asynchronous mode to run) 
                        Product description i.e. listing are classified in multiple clusters to reduce dimensions/# of products. Key words are 
                        also listed for each clusters.
                        
                        Online:(Synchronous model to run)
                        Based on product listing, recommender will search the products for that particular cluster. As the # of entries are reduced
                        so it can be done in real time. Based on number of recommendation, recommender engine will provide the nearest match based on 
                        score and also report price for that product. If user will choose a big number(100X); engine will provide all the products
                        present in the cluster and ranking would be done based on search Scores.
                        
                        The code has been written to keep the point in mind that it should be scalable to run for multiple categories at the same time
                        using docker or shell scripting.                       
                        ----------------------------------------------------------------------------------------------------------------------------
    Question-b:
            Validation  ----->
                        I have done few validation mannually, The first product is coming correct for those cases but not for more than 3 products. 
                        In the proposed methodology we need to test both the Offline and Online engine. It's very much important to understand the 
                        cluster should be created properly otherwise we will end up with totally wrong recommendation. We can simulate some more data 
                        and check the accuracy of the engine.
            
            Comparison with Naive Approach ----->
                        The proposed methodology first classify the products and after that based on TF-IDF score select the recommended product.
                        As it's two step approach, it very different from that aspect to select randomly some products. 
                        ----------------------------------------------------------------------------------------------------------------------------            
    Question-c:
            Possible Shortcomings & Extensions ----->
                        The methodology is quite simple and not robust. Definitely, the shared methodology can boost the performance. I was also 
                        thinking to use graph/network based recommendation. We can also do sentiment analysis using listing descrtiion which can also
                        be used for ranking products.
                         
                        In terms of efficiency, the code is not scalable at this point, it's taking long time to get the location, potentially multi-processing
                        would be the choice. The set up can be run parallely for multiple categories at the same time using docker or using simple shell script.
                        
            Newly Listed listing ----->
                        In current methodology, I do clustering at offline phase and find key terms for each cluster. We can use these key words for the new
                        product listing and get the cluster.
                        ----------------------------------------------------------------------------------------------------------------------------
                        
    Question-d:
            Clustering Approach shared in literature ---->
                        It's seems like difficult to implement within this short span of time. I have come up with different offline approach as a substitute
                        of the proposed methdology. I agree it needs revision to improve the recommendation 

'''

# Defining set up paths in subdirectory
Input_path = path + 'Inputs/'
Output_path = path + 'Outputs/'
Code_path = path + 'src/'

# Loading Required packages;
exec(open(Code_path+"RequiredPackages.py").read(), globals())
exec(open(Code_path+"UserDefinedFn.py").read(), globals())

data = pd.read_csv(Input_path + 'InputData.csv',encoding='Latin-1')

'''-->>> 
    STEP: 0 
        - Prepareing Ready to Model(RTM) data set from 'za_sample_listings_incl_cat' file. The Data contains Category Information and Longitute and Latitude.
        - Using Coordinates, identify the place where the seller is locating in South Africa
        - Due to Run time issue in local laptop, Data has been subset for one category and only using 500 Rows.
        - Garbage Collection made after deleting data frames from memory
'''
RTM_FDataSet = data.loc[data['category_l1_name_en'] == specificCategory].reset_index(drop=True)
RTM_DataSet = RTM_FDataSet.iloc[0:500].reset_index()

del data, RTM_FDataSet
gc.collect();gc.collect()

RTM_DataSet['area']=RTM_DataSet.apply(lambda _: '', axis=1)
RTM_DataSet['coordinates'] =  RTM_DataSet['listing_latitude'].astype(str) + ", " + RTM_DataSet['listing_longitude'].astype(str)
geolocator = Nominatim(user_agent="specify_your_app_name_here")

for i in range(len(RTM_DataSet)):
    location = pd.Series(geolocator.reverse(RTM_DataSet['coordinates'][i],timeout=None))[0]
    area = location.split(",")
    print(i)
    RTM_DataSet['area'][i] = AreaFindings(area)

RTM_DataSet.drop(['coordinates'],inplace=True, axis=1)


'''-->>>
    STEP: 0.1(intermediate steps)
        - Combine the Areas if the product count is coming less than 10.
        - Ideally we should search the distance between places and map the products accordingly.
        - This count step has been taken care only for time purpose.
'''

RTM_AREAs = RTM_DataSet.groupby(['area'],sort=True).size().reset_index(name='counts')
RTM_AREAs['Final_Area'] = np.where(RTM_AREAs['counts'] < 10,'Other-SA', RTM_AREAs['area'])
RTM_FinalDataSet = pd.merge(RTM_DataSet,RTM_AREAs).reset_index(drop=True)

del RTM_AREAs
gc.collect()

RTM_FinalDataSet.drop(['area','counts','listing_latitude', 'listing_longitude'],inplace=True, axis=1)
# RTM_FinalDataSet = pd.read_csv(Input_path + 'RTM_Dataset.csv')

'''-->>>
        STEP: 1
        - Creating Keys based on location/Coordinate and Categories information.
        - Create clusters based on # of products available in that particular location and in that category.
        - Find the Top Key Words for each Clusters.
        - Save the Cluster information with the product description for Future Use        
'''

RTM_FinalDataSet['Key'] = RTM_FinalDataSet['category_l2_name_en'].astype(str) + " : " + RTM_FinalDataSet['Final_Area'].astype(str)
ProductCounts = RTM_FinalDataSet.groupby(['Key']).size().reset_index(name='counts')

ProductClusterinformation = pd.DataFrame(columns = ['listing_title','ClusterInformation'])
Clusterinformation = pd.DataFrame(columns = ['ClusterInformation','KeyWords','Key'])

for i in range(len(ProductCounts)):
    _tempData = RTM_FinalDataSet.loc[RTM_FinalDataSet['Key'] == ProductCounts['Key'][i]].reset_index(drop=True)
    if ProductCounts['counts'][i] > 30:
        K = 5
        ProductCluster, ClusterKeyWords = Clustering(Fields=_tempData['listing_title'], NoofCluster=K,TopKeyWords=5)
    elif 15 < ProductCounts['counts'][i] < 30:
        K = 3
        ProductCluster, ClusterKeyWords = Clustering(Fields=_tempData['listing_title'], NoofCluster=K,TopKeyWords=5)
    elif 3 < ProductCounts['counts'][i] < 15:
        K = 2
        ProductCluster, ClusterKeyWords = Clustering(Fields=_tempData['listing_title'], NoofCluster=K,TopKeyWords=5)
    else:
        ProductCluster = pd.DataFrame(columns=['listing_title', 'ClusterInformation','Key'])
        ProductCluster ['listing_title'] = _tempData['listing_title']
        ProductCluster['ClusterInformation'] = 0

        ClusterKeyWords = pd.DataFrame(index=range(len(_tempData['listing_title'])),columns=['ClusterInformation', 'KeyWords','Key'])
        ClusterKeyWords['ClusterInformation'] = 0
        ClusterKeyWords['KeyWords'] = ''.join(list(_tempData['listing_title'].unique()))

    ProductCluster['Key'] = ProductCounts['Key'][i]
    ClusterKeyWords['Key'] = ProductCounts['Key'][i]
    ProductClusterinformation = ProductClusterinformation.append(ProductCluster)
    Clusterinformation = Clusterinformation.append(ClusterKeyWords)

ProductClusterinformation = pd.merge(ProductClusterinformation,RTM_FinalDataSet[["Key","listing_title",'listing_price']],how='left')

Clusterinformation.to_csv(Output_path+"ClusterInformation.csv",index=False)
ProductClusterinformation.to_csv(Output_path+ "ProductClusterinformation.csv",index=False)

'''-->>>
        STEP: 2
        - Made Recommendation based on product Names
        - # of Recommendations can be varied as per user choice.
        - Use Cluster information to get the scores and price to recommend products.
'''

RecommendedItemWithPrice = recommend(prodName = 'Angel care diaper bin',numberRecommendation=1)
