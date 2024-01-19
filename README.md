# CS412-Project
Term Project for CS412 Machine Learning

# Team Members
Mehmet Can Türkmen - 29544

Mehmet Barış Bozkurt - 28137

Alp Tuna Dağdanaş - 28958

Zeynep Pancar - 28303

Yiğit Kaan Tonkaz - 29154

# Overview of the repository
Code snippet1:

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

prompts_cleaned = [preprocess_text(prompt) for prompt in prompts]
questions_cleaned = [preprocess_text(question) for question in questions]

print(prompts_cleaned)
print(questions_cleaned)
Explanation: In this code snippet, we applied text preprocessing to both questions and prompts. The text preprocessing includes converting them to lowercase, removing digits and non-alphanumeric characters. Moreover, we removed stopwords and lemmatized words. The reason for text preprocessing was to have clean and standardized text which helped us to increase our model's accuracy.

Code snippet2:
sentences = [prompt.split() for prompt in prompts]

vector_size = 500
window = 5
min_count = 5

w2v_model = Word2Vec(
    sentences,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=4)


def document_vector(word2vec_model, doc):
    words = doc.split()
    word_vectors = []

    for word in words:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])

    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)

    document_vector = np.mean(word_vectors, axis=0)
    return document_vector

code2_word2vec_prompts = dict()

for code, user_prompts in code2prompts.items():
  if len(user_prompts) == 0:
      # some files have issues
      print(code+".html")
      continue

  word2_vec = pd.DataFrame([document_vector(w2v_model, doc) for doc in user_prompts],
                               columns={i: f"Q_{i}" for i in range(w2v_model.vector_size)})
  code2_word2vec_prompts[code] = word2_vec

question_word2vec = pd.DataFrame([document_vector(w2v_model, doc) for doc in questions], columns={i: f"Q_{i}" for i in range(w2v_model.vector_size)})

Explanation: In this code snippet, we created Word2Vec embeddings for each prompt and question in the given datasets. Additionally, we performed hyper parameter tuning to increase our performance.

code snippet 3:

keywords2search = ["error", "no", "thank", "next", "Entropy", "yes", "correct", "exactly", "certainly", "sure", "good", "well", "bad", "here", "how","problem","great","wrong"]

Explanatinon: We added a couple of more keywords in order to increase our model's performance. We tried to add words which might be used the most.

code snippet4: 
threshold_copy_paste = 0.8  # Set your desired threshold for copying and pasting
code2cosine = dict()
for code, user_prompts_tf_idf in code2prompts_tf_idf.items():
    code2cosine[code] = pd.DataFrame(cosine_similarity(questions_TF_IDF,user_prompts_tf_idf))

for i in range(len(questions)):
    question_mapping_scores[f"copy_paste_indicator_Q{i}"] = (
        question_mapping_scores[f"Q_{i}"] > threshold_copy_paste
    ).astype(int)

Explanation: We added a copy paste indicator column for each of the questions in homework1 and set the threshold to 0.8 since it was the value that increased the accuracy the most.

code snippet 5:

X = pd.DataFrame.from_dict(code2features, orient='index').fillna(0)
cluster_range = range(2, 11)

silhouette_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # Set n_init explicitly to 10 in order to suppress the warning
    cluster_labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, cluster_labels))

Explanation: Before performing a k-means clustering, we tried to plot the silhouette scores and aimed to find an optimal number of clusters for the given dataset by analyzing how the silhouette score varies with different cluster numbers.

code snippet 6:

df = pd.DataFrame(code2features).T

for column in df.columns:
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)

Explanation: Iterate through columns and replaced NaN values with the corresponding column's mean value

code snippet 7:

X = pd.DataFrame.from_dict(code2features, orient='index').fillna(0)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)
X['cluster_label'] = cluster_labels
cluster_label_mapping = {0: "Detailed Interactors", 1: "Concise Interactors"}

Explanation: Clustered characteristics of the students as Detailed Interactors (students who provide detailed prompts and responses) and Concise Interactors (students who use shorter prompts and responses).

code snippet 8:

regressor = RandomForestRegressor(n_estimators=200)
regressor.fit(X_train, y_train)
max_depth = max(tree.tree_.max_depth for tree in regressor.estimators_)
print(f"Maximum depth of any tree in the RandomForestRegressor: {max_depth}")
param_grid = {
    'max_depth': [5, 8, 12, 16],
    'min_samples_split': [2, 4, 10, 14, 20]
}

regressor = RandomForestRegressor(n_estimators=100)

grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')

grid_search.fit(X_train, y_train)
cols_to_include = ['param_max_depth', 'param_min_samples_split', 'mean_test_score', 'std_test_score']
results = pd.DataFrame(grid_search.cv_results_)[cols_to_include]
results.sort_values(by='mean_test_score', ascending=False)

Explanation: Created a Random Forest Regressor with regularization. Besides, performed hyperparameter tuning using grid search. (Hyperparameters:max_depth, min_samples_split)

code snippet 9:
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth = 5,
    min_samples_split = 45
)

regressor.fit(X_train, y_train)


y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

y_train_pred = np.clip(y_train_pred, 0, 100)
y_test_pred = np.clip(y_test_pred, 0, 100)

print("MSE Train:", mean_squared_error(y_train, y_train_pred))
print("MSE TEST:", mean_squared_error(y_test, y_test_pred))

print("R2 Train:", r2_score(y_train, y_train_pred))
print("R2 TEST: ",r2_score(y_test, y_test_pred)*100)

cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores = -cv_scores
print("Cross-Validation Scores:", cv_scores)

Explanation: Observed cross-validation scores provide an additional evaluation measure by assessing the model's performance on different subsets of the training data. Furthermore, mapped and bounded the score predictions between 0 and 100. 


# Methodology

To start with, we focused on text preprocessing of both questions and prompts. By applying text preprocessing, we aimed to increase the model's performance by getting rid of unnecessary characters that are used in prompts and questions. The text preprocessing consisted of converting each letter to lowercase letters, removing digits and non-alphanumeric characters, removing stopwords and lemmatized words. After that, we created a Word2vec model to transform the preprocessed text into numerical vectors to see if a pattern would be observed between words or not. Then, we performed hyperparameter tuning in our Word2vec model in order to enhance the model's accuracy, involving adjustments to vector size, window size, and minimum count. Then we continued with determining a threshold value for a "copy and paste" indicator. By using our Word2vec model and the cosine similarity scores, we set the threshold to 0.8 since that value was giving the best accuracy score for our model. Moreover, the reason why we used cosine similarity scores was because of the fact that those similarity scores were indicating how similiar the prompts and the questions are. After setting the best threshold value, we added a new column next to each question to show in which questions the threshold was exceeded. The reason why we did that was to have additional insights in each student's interaction with ChatGPT. Furhtermore, we added new keywords to search such as "yes", "correct", "exactly", "certainly", "sure", "good", "well", "bad", "here", "how","problem","great","wrong" which might be used by student's the most in their interactions with CHATGPT. Then, we moved on with using a clustering algorithm to enhance our model's score. We selected k-means clutering algorithm because it provided the highest accuracy score in our model (we tried hierarchial clustering and DSCAN clustering algoritms). However, to determine the number of clusters we first wanted to observe silhouette scores and decide the number of clusters with the highest silhouette score. Therefore, we set a cluster range (between 2 and 11) and plotted the silhouette score of each number of cluster. Based on the plot, the highest silhouette score was observed in number of clusters=2 and hence we used that value as our number of clusters in the k-means clustering algorithm. In the k-means algorithm, we clustered characteristics of the students as detailed interactors (students who provide detailed prompts and responses) and concise interactors (students who use shorter prompts and responses). As additional feature engineering feature, we tried to prevent our model from NaN values which would decrease our model's accuracy. Hence, we put mean value for each NaN (Not a Number) entry in its respective column. After that, we created a random forest regressor and additionally, applied regularization to avoid overfitting. The reason why we used random forest regressor as our model was because it is capable of capturing complex relationships and it is a a robust and accurate predictive model. Then, we used GridSearch technique to perform hyperparameter tuning. As hyperparameters, we selected maximum depth and minimum samples split. Upon completion of the GridSearch process, the chosen hyperparameter configuration was implemented in our Random Forest Regressor model. After that, we mapped and bounded the score predictions between 0 and 100 and used cross-validation in order to provide an additional evaluation measure by assessing the model's performance on different subsets of the training data.



# Results

MSE Train:

MSE Test:

R2 Train:

R2 Test:

Cross-Validation Scores:


# Contributions of Group Members

Alp Tuna Dağdanaş: 






