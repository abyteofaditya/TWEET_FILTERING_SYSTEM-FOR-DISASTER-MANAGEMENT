# Copyright © 2024 Aditya Sarohaa
# Licensed under CC BY-NC-ND 4.0 (No modifications or commercial use allowed)

#Import all the libraries
import pandas as pd #for data handling
import numpy as np
import pickle#for saving the model trained in the program
from textblob import TextBlob #for judging sentiments of the required tweets
from sklearn.preprocessing import LabelEncoder # To encode the string base data so that machine can understand
from sklearn.model_selection import train_test_split as tts #To split the training and testing datasets
from sklearn.linear_model import LogisticRegression # Importting to train the regression model.Used since for Binary Classification,Simplicity & Interpretability,Efficient for High-Dimensional Data,Baseline Model
from sklearn.metrics import accuracy_score # This is to check the accuracy of the model we made
from sklearn.feature_extraction.text import TfidfVectorizer# To vectorize our tweet messages.Captures Important Words,Handles Sparse Data,Doesn't Require Deep Domain Knowledge,Effectiveness in Text Classification


encoder=LabelEncoder()
def training(file_name):
    try:
        data=pd.read_csv(f'{file_name}.csv')
        #If our file cntains some missing columns or boxes in case then they can be filled by using fillna function of numpy
        data.fillna(np.nan)
        
        #encoding one column that contains string data and saving the encoded text to a new column instead
        #category is a column in the dataset that contains whether the data is INFORMATIVE or NOT
        data['category_rating']=encoder.fit_transform(data['category'])
        vectorizer=TfidfVectorizer(max_features=479)
        #Actual_unr_tweets , this column contains actual tweets which we need to extract base don relevancy, firstly we need to vectorize it
        Actual_tweets_num=vectorizer.fit_transform(data['Actual_unr_tweets']).toarray()# This function converts the spatial matrix into an array
        #defining feature columns for training
        data.to_csv('new_file_name.csv', index=False)
        X_features=data[['confidence_score']].values#the confidence _score column contains information regarding the confidency of the tweet 
        X=np.hstack((X_features,Actual_tweets_num))#Combining the numerical and Tf-id features using numpy
        
        #defining target variables
        y=data['category_rating']
        
        #Now we split dataset into train and test and would keep 20 % of the data fro testing
        X_train,X_test,y_train,y_test=tts(X,y,test_size=0.2,random_state=42)
        model = LogisticRegression(random_state=42, class_weight='balanced')

        #Now finally we got to train the regression model
        model.fit(X_train,y_train)
        
        #Making predictions
        y_pred=model.predict(X_test)
        
        #Calculating and displaying the accuracy score of our model
        accuracy_score_value=accuracy_score(y_test,y_pred)
        print('Accuracy of the dataset is :', round(accuracy_score_value,2))
        #Dumping the trained model in a file name trained_model.pkl
        with open('trained_model.pkl','ab') as f:#Appending the file so that each time we train our model with the new dataset it can get saved
            pickle.dump(model,f)
        with open('vectorizer.pkl','wb') as f1:
            pickle.dump(vectorizer,f1)
        print('The trained model appended sucessfully.')
        
    except FileNotFoundError:
        print('The required file cannot be found. Please try again with a valid file name.')
    except Exception as e:
        print(f'Error : An unexpected error occured \n {e}')
def sentiment_judgement(new_file):#defining a new function for sentiment judgement
    try:
        sentiment_list=[]#An empty list creation
        for tweet in new_file['Actual_unr_tweets']:
            analyse_tweet=TextBlob(tweet)
            sentiment=analyse_tweet.sentiment.polarity
            sentiment_label='Urgent' if sentiment<0 else 'Not_urgent' if sentiment>0  else 'Neutral'
            sentiment_list.append(sentiment_label)
        new_file.loc[:, 'Sentiment'] = sentiment_list

    except FileNotFoundError:
        print('The required file cannot be found. Please try again with a valid file name.')
    except Exception as e:
        print(f'Error : An unexpected error occured :\n{e}')
def Predicting_output_for_user(new_file):
    new_data=pd.read_csv(f'{new_file}.csv')
    try:
        
        
        #If our file cntains some missing columns or boxes in case then they can be filled by using fillna function of numpy
        new_data.fillna(np.nan)
        
        #encoding one column that contains string data and saving the encoded text to a new column instead
        #category is a column in the dataset that contains whether the data is INFORMATIVE or NOT
        new_data['category_rating']=encoder.fit_transform(new_data['category'])
        with open('vectorizer.pkl','rb') as f0:
            vectorizer=pickle.load(f0)
        #Actual_unr_tweets , this column contains actual tweets which we need to extract base don relevancy, firstly we need to vectorize it
        Actual_tweets_num=vectorizer.transform(new_data['Actual_unr_tweets']).toarray()# This function converts the spatial matrix into an array
        #defining feature columns for training
        new_data.to_csv('new_file_name.csv', index=False)
        X_features=new_data[['confidence_score']].values#the confidence _score column contains information regarding the confidency of the tweet 
        X_new=np.hstack((X_features,Actual_tweets_num))#Combining the numerical and Tf-id features using numpy
        # opening the trained model for usage
        with open('trained_model.pkl','rb') as f2:
            model=pickle.load(f2)
        new_pred=model.predict(X_new)

        #converting the array datatype into integers
        
        new_data['new_pred_list']=new_pred#Adding a new column for tracking predictions.
        print('Now printing the new data with first 14 columns predited list column added in which predictions of category_rating will be there \n\n ',new_data.head(14))
        sentiment_judgement(new_data)#calling the function
        # sentiment_ask=int(input('Enter \n1-Urgent\n2-Not urgent\n3-Neutral\n'))
        # if sentiment_ask==1:
        #     print(new_data.loc[new_data['Sentiment']=='Urgent'])
        # elif sentiment_ask==2:
        #     print(new_data.loc[new_data['Sentiment']=='Not_urgent'])
        # else:
        #     print(new_data.loc[new_data['Sentiment']=='Neutral'])
        while True:
            #printing the mapping of categories to encoded values
            category_mapping=dict(zip(encoder.classes_,range(len(encoder.classes_))))
            print('Mapping of categories to encoded values : \n')
            for category, encoding in category_mapping.items():
                print(f"{category}: {encoding}")
            ask=int(input('Which type of information do you want?'))
            if ask in category_mapping.values():
                temp_data=new_data.loc[new_data['new_pred_list']==ask]
                sentiment_judgement(temp_data)
                sentiment=int(input('Enter the sentiment filter that you want on the tweets :\n1-Urgent\t2-neutral\t3-Not_urgent\n'))
                for i in temp_data.columns:
                    print(i, end='\n')
                column_see = input('Enter the exact column that you want to see: \n')
                if sentiment==1:
                    print(temp_data[column_see].loc[temp_data['Sentiment']=='Urgent'])
                elif sentiment==2:
                    print(temp_data[column_see].loc[temp_data['Sentiment']=='Neutral'])
                elif sentiment==3:
                    print(temp_data[column_see].loc[temp_data['Sentiment']=='Not_urgent'])
                else:
                    print('Invalid input!')
                    continue
            
            else:
                break      
        assert 'confidence_score' in new_data.columns, "'confidence_score' column missing in test data"
        assert not new_data['confidence_score'].isnull().any(), "Null values found in 'confidence_score'"

    # except FileNotFoundError:
    #     print('Invalid .No file found')
    except EOFError:
        print(' Error: X has 480 features, but LogisticRegression is expecting 479 features as input.')
    except pickle.UnpicklingError:
        print('Error : The model file is not a valid pickeling file')
    except Exception as e:
        print(f'Error : An unexpected error occured : \n {e}.')
    
#Training some files for representation.These functions can be used to save more datasets in future.
# training('Cautions_and_advice')
# training('TWEETDATA 1')
# training('TWEETDATA 2')
# training('TWEETDATA 4')
# training('TWEETDATA 5')
# training('TWEETDATA 6')
# training('TWEETDATA 7')





#PREDICTING ON A TESTING FILE OR FOR REAL TIME NEW FRESH DATASETS.

Predicting_output_for_user('TWEETDATA 4')
