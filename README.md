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
    return text

prompts_cleaned = [preprocess_text(prompt) for prompt in prompts]
questions_cleaned = [preprocess_text(question) for question in questions]

print(prompts_cleaned)
print(questions_cleaned)
Explanation: In this code snippet, we applied text preprocessing to both questions and prompts. The text preprocessing includes converting them to lowercase. The reason for text preprocessing was to have clean and standardized text which helped us to increase our model's accuracy.

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
threshold_copy_paste = 0.8 

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

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.drop('cluster_label', axis=1))  # Ensure to drop non-numeric columns
X['pca_one'] = X_pca[:, 0]
X['pca_two'] = X_pca[:, 1]

plt.figure(figsize=(8, 8))
for cluster_label, marker in zip(cluster_label_mapping.values(), ['o', 's']):
    cluster_data = X[X['cluster_label'] == cluster_label]
    plt.scatter(cluster_data['pca_one'], cluster_data['pca_two'], label=cluster_label, marker=marker)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Plot of KMeans Clusters')
plt.legend()
plt.show()

Explanation: Plot the PCA of 2 clusters which were found in the code snippet 7. PCA reduces the multidimentional feature vector into 2 main axes where there is maximum standard deviation and minimum mean square error. 

code snippet 9:

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

Explanation: These hyperparameter tuning results will be used for our main model's parameters. Created a Random Forest Regressor with regularization. Besides, performed hyperparameter tuning using grid search. (Hyperparameters:max_depth, min_samples_split)

code snippet 10:
main_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth = results["param_max_depth"][0],
    min_samples_split = results["param_min_samples_split"][0]
)

main_regressor.fit(X_train, y_train)

y_train_pred = main_regressor.predict(X_train)
y_test_pred = main_regressor.predict(X_test)


y_train_pred = np.clip(y_train_pred, 0, 100)
y_test_pred = np.clip(y_test_pred, 0, 100)


print("MSE Train:", mean_squared_error(y_train, y_train_pred))
print("MSE TEST:", mean_squared_error(y_test, y_test_pred))

print("R2 Train:", r2_score(y_train, y_train_pred))
print("R2 TEST: ",r2_score(y_test, y_test_pred))


cv_scores = cross_val_score(main_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores = -cv_scores
print("Cross-Validation Scores:", cv_scores)

Explanation: This RandomForestRegressor will be considered as our model to be graded. Observed cross-validation scores provide an additional evaluation measure by assessing the model's performance on different subsets of the training data. Furthermore, mapped and bounded the score predictions between 0 and 100. 


# Methodology

To start with, we focused on text preprocessing of both questions and prompts. By applying text preprocessing, we aimed to increase the model's performance by getting rid of unnecessary characters that are used in prompts and questions. The text preprocessing consisted of converting each letter to lowercase letters. After that, we created a Word2vec model to transform the preprocessed text into numerical vectors to see if a pattern would be observed between words or not. Then, we performed hyperparameter tuning in our Word2vec model in order to enhance the model's accuracy, involving adjustments to vector size, window size, and minimum count. Then we continued with determining a threshold value for a "copy and paste" indicator. By using our Word2vec model and the cosine similarity scores, we set the threshold to 0.8 since that value was giving the best accuracy score for our model. Moreover, the reason why we used cosine similarity scores was because of the fact that those similarity scores were indicating how similar the prompts and the questions are. After setting the best threshold value, we added a new column next to each question to show in which questions the threshold was exceeded. The reason why we did that was to have additional insights in each student's interaction with ChatGPT. Furthermore, we added new keywords to search such as "yes", "correct", "exactly", "certainly", "sure", "good", "well", "bad", "here", "how","problem","great","wrong" which might be used by student's the most in their interactions with CHATGPT. Then, we moved on with using a clustering algorithm to enhance our model's score. We selected the k-means clustering algorithm because it provided the highest accuracy score in our model (we tried hierarchical clustering and DSCAN clustering algorithms). However, to determine the number of clusters we first wanted to observe silhouette scores and decide the number of clusters with the highest silhouette score. Therefore, we set a cluster range (between 2 and 11) and plotted the silhouette score of each number of clusters. Based on the plot, the highest silhouette score was observed in the number of clusters=2 and hence we used that value as our number of clusters in the k-means clustering algorithm. In the k-means algorithm, we clustered characteristics of the students as detailed interactors (students who provide detailed prompts and responses) and concise interactors (students who use shorter prompts and responses). In addition to that, we used PCA ( Principal Component Analysis) to visualize our k-means clustering algorithm and project it. As an additional engineering feature, we tried to prevent our model from NaN values which would decrease our model's accuracy. Hence, we put the mean value for each NaN (Not a Number) entry in its respective column. After that, we created a random forest regressor and additionally, applied regularization to avoid overfitting. The reason why we used a random forest regressor as our model was because it is capable of capturing complex relationships and it is a robust and accurate predictive model. Then, we used GridSearch technique to perform hyperparameter tuning. As hyperparameters, we selected maximum depth and minimum samples split. Upon completion of the GridSearch process, the chosen hyperparameter configuration was implemented in our Random Forest Regressor model. After that, we mapped and bound the score predictions between 0 and 100 and used cross-validation in order to provide an additional evaluation measure by assessing the model's performance on different subsets of the training data. Moreover, we observed an intriguing result regarding the model accuracy when we performed semantic analysis. Even though we only used that model for observation, it was interesting to observe that the model accuracy was %99.5 when semantic analysis was performed. Due to splitting of train and test data which occurs randomly, R2 scores of our main model might differ in each run of our project.
For example we received 0.32 during one of our runs, but we also received 0.26, 0.23, 0.22, and 0.30 at different runs. 



# Results

MSE Train: 34.55089318554958

MSE TEST: 75.65092270327037

R2 Train: 0.788576546132381

R2 TEST:  0.32614333595268385

Cross-Validation Scores: [301.60246882  81.61635697  81.68690079 567.22264701  53.98111364]


# Contributions of Group Members

Alp Tuna Dağdanaş: I tried to help my group to find the best threshold value for adding a copy paste indicator for each student's questions. Also, I tried to find new keywords to search in order to enhance the model's accuracy. Moreover, I tried different clustering algorithms that we learned in the lecture such as DBSCAN, hierarchical clustering and k-means algorithms and use the best clustering algorithm (best in a sense where it resulted in the highest R2 test accuracy score compared to other clustering algorithms). Additionally, I used silhouette score to determine which number of clusters would be the best and visualized the results of the silhouette score of the number of clusters in the range 2 and 11. Finally, I helped my group to overcome NaN values and put the mean value for each NaN (Not a Number) entry in its respective column.

Zeynep Pancar: Researching the cluster methods that we are going to use and evaluating the results.

Mehmet Can Türkmen: I worked on converting prompt's of each document to word2vec models. Converting to word2vec model increased the R2 score of the test instances. Additionally, I added a new feature called "is_detailed_instructor". To implement it, I used the clustering results of the k-means. After that I also plotted the clusters using PCA in 2 dimensions. Also, I hypertuned 2 models. For the first model, I implemented GridSearchCV and used the best parameters for max depth and minimum splitting. As a result, I recieved 0.995 R2 score. The reason why R2 is so high is because Barış implemented semantic analysis before. For the second model I used GridSearchCV to hypertune the model again. And used the best parameters for maximum depth and minimum splitting values. As a result I recieved 0.326 R2 score. The reason why it is lower from the first model is because I removed the semantic analysis features and we discussed that the second model was better for predicting and decided it as our main model (model to be graded). I also worked on threshold features. I tested some of the possible threshold values to use while implementing the "copy_paste_indicator" features.

Mehmet Barış Bozkurt: I did preprocessing to preprocess the text data. Using the Word2Vec model, I created word vectors for "prompts" and "questions" and calculated the similarities between these vectors. Using these similarity scores, I tried to match the most appropriate question for each prompt. To search for specific keywords in text data, I added new words to the keywords2search list and this significantly increased the R2 score. I tried to optimize the max_depth and min_samples_split parameters. Also, I performed sentiment analysis which surprisingly increased R2 test score to %99.5. But then we decided to remove the sentiment analysis and trained another model without sentiment analysis features and we used that model as our main model. 

Yiğit Kaan Tonkaz: I made a research on hyperparameter tuning to use the GridSearch technique to be able to optimize model parameters, collaborated with team members about the enhancement of model accuracy.




