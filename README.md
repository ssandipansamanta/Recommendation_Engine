The recommendation Engine is build using 'sample_file' present in Input Folder. It's a content based recommendation problem.
In the sample file, with Product and selling information, Category and location information also available. Geographical location is quite important to know as it's very unlikely to sell/buy a product to a different location. Recommendation is applied by location and for a category level

Offline:(Asynchronous mode to run) 
Product description i.e. listing are classified in multiple clusters to reduce dimensions/# of products. Key words are also listed for each clusters.

Online:(Synchronous model to run)

Based on product listing, recommender will search the products for that particular cluster. As the # of entries are reduced so it can be done in real time. Based on number of recommendation, recommender engine will provide the nearest match based on score and also report price for that product. If user will choose a big number(10X); engine will provide all the products present in the cluster and ranking would be done based on search Scores.

The code has been written to keep the point in mind that it should be scalable to run for multiple categories at the same time using docker or shell scripting.
