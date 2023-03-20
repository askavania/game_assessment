

# Identifying Potential Hit Games by Analyzing Description of Games 

## Problem Statement
A client has tasked our team to assist them in making better decisions in investing in potential hit games by assessing the description of the games.  


To assist the client in making better investment decisions in potential hit games, we can implement a text classification model to analyze the descriptions of the games. The model can predict whether a game will be a hit or not based on the description provided by the client.

The first step in the process is to collect a large dataset of game descriptions and their reviews score from popular gaming platform A. This data will be used to train and evaluate the model.

Next, we will pre-process the description text to clean and normalize the data. This includes removing stop words, punctuation, and converting all text to lowercase. Additionally, we can perform lemmatization or stemming to reduce words to their base form and reduce the number of unique tokens in the dataset.

Once the text data has been pre-processed, we can use techniques like count vectorizing, bag-of-words or TF-IDF to convert the text data into numerical feature vectors. These feature vectors can then be used as input to the machine learning model.

In this case, a suitable machine learning algorithm for the task might be a Naive Bayes classifier or a Random Forest classifier. We will train the model on the pre-processed data and evaluate its performance on a held-out validation set. If the performance is not up to the desired standards, we can make adjustments to the pre-processing steps or the model architecture to improve performance.

Finally, the trained model can be used to predict the success of new games based on their descriptions. The client can use this information to make informed investment decisions.

Overall, by using text classification, we should be able to provide valuable insights to the client on the potential success of new games and help them make better investment decisions.


**Some additional pointers for this project:**

- This project contains multiple notebooks for different purposes;
    - In ["Webscraping"](https://git.generalassemb.ly/benedictyong/project/tree/master/project_3/webscraping) folder;
        - ["gamelist_extraction.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/gamelist_extraction.ipynb) : Used for extracting a list of games('gamesdata.csv') by chronological order from popular gaming platform A site
        - ["games1.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/games1.ipynb) : Used for extracting description, reviews & tags of rows first 10k games in list of games('gamesdata.csv')
        - ["games2.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/games2.ipynb) : Used for extracting description, reviews & tags of rows 10k - 20k games in list of games('gamesdata.csv')
        - ["games3.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/games3.ipynb) : Used for extracting description, reviews & tags of rows 20k - 30k games in list of games('gamesdata.csv')
        - ["games4.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/games4.ipynb) : Used for extracting description, reviews & tags of rows 30k - 40k games in list of games('gamesdata.csv')
        - ["games5.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/games5.ipynb) : Used for extracting description, reviews & tags of rows 40k - 50k games in list of games('gamesdata.csv')
        *To enhance processing time, webscraping of data from steam website has been split in multiple notebooks(notebooks 'games1-5') then combined into one at a later stage.*
        - ["merge_data.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/webscraping/merge_data.ipynb) : Used for merging list of games('gamesdata.csv') with corresponding games' description, reviews and tags data. 
        
    - In main repo:
        - ["GAME_assessment.ipynb"](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/GAME_assessment.ipynb) : Main notebook containing data wrangling, EDA, processing and modeling. 


# About the Datasets

There are a few datasets included in the data folder for this project;

* [List of Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/gamesdata.csv) : A list of games that has been scraped from the popular gaming platform A site in chronological order of game release date
* [First 10k Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/first_ten_k_games.csv) : Description, reviews & tags data of rows first 10k games in list of games dataset
* [10k-20k Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/10to20_k_games.csv) : Description, reviews & tags data of rows 10k - 20k games in list of games dataset
* [20k-30k Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/twenty_k_games.csv) : Description, reviews & tags data of rows 20k - 30k games in list of games dataset
* [30k-40k Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/30to40_k_games.csv) : Description, reviews & tags data of rows 30k - 40k games in list of games dataset
* [40k-50k Games Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/40to50_k_games.csv) : Description, reviews & tags data of rows 40k - 50k games in list of games dataset
* [Master Data](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/dataset/master_df3.csv) : A complete dataset that has been combined list of games data with their corresponding game description, reviews & tags
    
    
Below is some information about our cleaned datasets and their features that we will be using.
<br>

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|title|str|name of game|
|link|str|game url|
|r_date|datetime|game release date|
|price|float|price of game|
|description|str|description of game|
|reviews|str|reviews summary|
|tagged|str|users-defined tags of games|

---

### Summary of Findings

From our EDA, we have gathered the following information;

1. It seems like the gaming industry has been very active since 2017, or at least popular gaming platform A has been doing well and started to collect data since then. The number of games released per year has broken the 5000 mark in 2020. Perhaps this could be due to boost contributed by the effect of stay home period due to covid.
![Game Trend](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/games_per_year.jpg)

<br><br>
2. Discounting the years 2016 and before as well as 2023(since we're only at the 1st month), the ceiling prices of games seems to have increase year on year.
![Price of games](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/prices%20of%20games%20over%20years.jpg)

<br><br>
3. As the p-value of the game price is >0.05, it lets us safely assume that the price has no effect on the positivity of the reviews.
![OLS](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/ols%20result.jpg)

<br><br>
4. The median percentage for positive reviews is at 81% with the upper percentile at 91%. We can take this into account when determining our threshold for our measure of success for a game.

<br><br>
5. Looking at Fig 1, there are less number of games as the number of reviews per game increase. That makes sense, as from our domain understanding in gaming, there will be more reviews for more popular games due to the amount of people playing that particular game.
![fig1](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/fig1.jpg)

<br><br>
6. Followed by Fig.5, where we observe following;
There are at least 100 reviews for games that costs between \$20 to \$100, and the number of reviews falls drastically when a game costs more than \$100. It could be due to the high price that leads to less people buying the games, hence resulting in a minimal count of reviews. This could affect our accuracy metric for a successful game. 
There are most number of reviews in games that costs in the \$70 range and \$100 range(the most number of reviews).
![Fig5](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/price%20vs%20reviews.jpg)

<br><br>
7. In Fig.2, there is a sharp increase in the counts of 75% positive reviews. 
![Fig2](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/fig2.jpg)

<br><br>
8. We saw from Fig. 3a & 3b that there are usually more outliers in games that has lower number of reviews and that there are substantial amount of games that have been positively reviewed in each group.

![Fig3a](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/fig3a.jpg)
![Fig3b](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/fig%203.b.jpg)


Additionally, we also found out there are some frequent words that exists in successful games. 
![Freqwords](https://git.generalassemb.ly/benedictyong/project/blob/master/project_3/Images/freqwords.jpg)
<br><br>

---

### Conclusion & Caveats/Recommendations
In our business use case, as there is only a finite amount of money that can be invested in a number of games, we have put dominance on evaluating our model with precision as we would want to accurately capture True Positives (successful games that will return the investments) as well as lowering the False Positives (to avoid wasting money on games that turn out to a flop) based on the description text of the games. As such, we are choosing to go ahead with our Complement Naive Bayes model with the current data that we possess. <br><br>

Further exploration and improvement can be done in future, and the following are some suggestions for the roadmap;

1) We realized that there is a trend in using images and videos/gifs instead of just plain text in the game descriptions, especially in triple A a.k.a big budget games, we can further explore on how we can analyze such data on top of plain text in future as well.

2) As our problem statement is to predict a game's success purely on their game description, our model's prediction is based only on a small dimension of our game data. Perhaps we can look into other data types of the game characteristics/features such as developer/publisher/system requirements etc and combine different models to get a better prediction. 


 


