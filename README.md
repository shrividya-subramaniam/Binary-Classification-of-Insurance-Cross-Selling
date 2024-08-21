# Binary Classification of Insurance Cross Selling
 Solution to Kaggle Playground Prediction Competition (Season 4, Episode 7)


## Problem Statement
The objective of this competition is to predict which customers respond positively to an automobile insurance offer.


## Evaluation Metric
Submissions are evaluated using **area under the ROC curve** using the predicted probabilities and the ground truth targets.


## Data Dictionary

#### Train Data
Feature |
--- | 
id                    
Gender                 
Age                   
Driving_License         
Region_Code           
Previously_Insured     
Vehicle_Age          
Vehicle_Damage         
Annual_Premium        
Policy_Sales_Channel  
Vintage               
Response     

#### Test Data
Feature |
--- |
id                    
Gender                 
Age                   
Driving_License         
Region_Code           
Previously_Insured     
Vehicle_Age          
Vehicle_Damage         
Annual_Premium        
Policy_Sales_Channel  
Vintage 


## Data Preprocessing
1. Train and test dataframes are merged.
2. Region_Code, Policy_Sales_Channel, Annual_Premium columns are casted into integer type.
3. Missing Value Imputation – There are no missing values in train or test dataset.
4. Encoding categorical variables
- Gender, Vehicle_Age and Vehicle_Damage are encoding with numerical mapping.
5. Feature Generation
- Interaction variables are created by combining the Previously_Insured column with Annual_Premium, Vehicle_Age, Vehicle_Damage, and Vintage respectively using pd.factorize.
6. Since the dataset is very large, the features are downcasted into their respective data types to optimise memory usage.
7. The preprocessed dataset is then split into X, y and test dataset. 


## Prediction Approach
- As the train dataset is very large and computationally expensive, ensemble models such as LightGBM, XGBoost and CatBoost are evaluated using Stratified K-fold Cross Validation.
- CatBoost model is selected as the final model as it achieved a ROC AUC score than other models.
- The final CatBoost model is trained with 2 fold Stratified K-fold Cross Validation using NVidia P100 GPU.(Only 2 folds are used in cross validation due to memory constraints.)
- The model achieved a ROC AUC score of 0.89507 and 0.89482 on public leaderboard and private leaderboard respectively. 


## Leaderboard
- [Public Leaderboard](https://www.kaggle.com/competitions/playground-series-s4e7/leaderboard?tab=public) (username:Shrividya Subramaniam): 346th Rank (Top 15%)
- [Private Leaderboard](https://www.kaggle.com/competitions/playground-series-s4e7/leaderboard?) (username:Shrividya Subramaniam): 350th Rank (Top 15%)
