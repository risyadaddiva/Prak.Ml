#%%
import pandas  as pd

# %%
# train Data
trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")

# %%
# test Data
testData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")

# %%
#cek
trainData.sample(frac=1).head(5) # shuffle the df and pick first 5

# %%
#vectorizing
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])

# %%
#Creating a Linear SVM Model
import time
from sklearn import svm
from sklearn.metrics import classification_report
# %%
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['Label'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# %%
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['Label'], prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])
# %%
# Calculate F1-score for the 'pos' class
precision = report['pos']['precision']
recall = report['pos']['recall']
f1_score = 2 * ((precision * recall) / (precision + recall))
print('F1-score (pos):', f1_score)
# %%
#Test the SVM classifier on Amazon reviews
def predict_review(review):
    review_vector = vectorizer.transform([review])
    prediction = classifier_linear.predict(review_vector)
    return prediction[0]
# %%
review = """SUPERB, I AM IN LOVE IN THIS PHONE"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))
# %%
review = """Do not purchase this product. My cell phone blast when I switched the charger"""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))
# %%
review = """It's not even 5 days since i purchased this product.
I would say this a specially blended worst Phone in all formats.
ISSUE 1:
Have you ever heard of phone which gets drained even in standby mode during night?
Kindly please see the screenshot if you want to believe my statement.
My phone was in full charge at night 10:07 PM . I took this screenshot and went to sleep.
Then I woke up at morning and 6:35 AM and battery got drained by 56% in just standby condition.
If this is the case consider how many hours it will work, during day time.
It's not even 5 hours the battery is able to withstand.
ISSUE 2:
Apart from the battery, the next issue is the heating issue .I purchased a iron box recently from Bajaj in this sale.
But I realized this phone acts a very good Iron box than the Bajaj Iron box. I am using only my headphones to get connected in the call. I am not sure when this phone is will get busted due to this heating issue. It is definitely a challenge to hold this phone for even 1 minute. The heat that the phone is causing will definitely burn your hands and for man if you keep this phone in your pant pocket easily this will lead to infertility for you. Kindly please be aware about that.
Issue 3:
Even some unknown brands has a better touch sensitivity. The touch sensitivity is pathetic, if perform some operation it will easily take 1-2 minutes for the phone to response.
For your kind information my system has 73% of Memory free and the RAM is also 56% free.
Kindly please make this Review famous and lets make everyone aware of this issue with this phone.
Let's save people from buying this phone. There are people who don't even know what to do if this issue happens after 10 days from the date of purchase. So I feel at least this review will help people from purchasing this product in mere future."""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))

# %%
#pickling the model
import pickle
# %%
# pickling the vectorizer
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))
# %%
# pickling the model
pickle.dump(classifier_linear, open('classifier.sav', 'wb'))
# %%
