import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sys import argv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

scaler = StandardScaler()

def pca(df, features, components):
  # Separate out the features
  x = df.loc[:, features].values

  # Separate out the target
  y = df.loc[:, ["target"]].values

  # Stardardize the features
  x = scaler.fit_transform(x)

  # Apply PCA to features
  pca = PCA(n_components=components)
  principal_components = pca.fit_transform(x)
  principal_df = pd.DataFrame(data=principal_components,
    columns=["principal component {}".format(i + 1) for i in range(components)])

  return principal_df

def random_color():
  return [x for x in [random.uniform(0, 1) for i in range(3)]]

def color_distance(lhs, rhs):
  return sum(abs(x[0] - x[1]) for x in zip(lhs, rhs))

def random_distinct_colors(n, max_attempts):
  already_chosen = [random_color()]
  for i in range(n):
    max_distance = None
    best_color = None
    for j in range(max_attempts - 1):
      color = random_color()
      best_distance = min([color_distance(color, c) for c in already_chosen])
      if not max_distance or best_distance > max_distance:
        max_distance = best_distance
        best_color = color
    already_chosen.append(best_color)
  return already_chosen

def knn(X_train, X_test, y_train, y_test, K):
  # Normalize features
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # Training
  classifier = KNeighborsClassifier(n_neighbors = K)
  classifier.fit(X_train, y_train)

  # Make prediction on test data
  y_pred = classifier.predict(X_test)

  return y_test, y_pred

def main():
  # Set default arguments
  n_components = 2
  test_percentage = .2
  max_K = 50

  # Analyze arguments
  for i in range(1, len(argv)):
    if i == 1: n_components = int(argv[i]) if int(argv[i]) in range(2, 4) else 2
    elif i == 2: test_percentage = float(argv[i])
    elif i == 3: max_K = int(argv[i])

  # Print chosen arguments
  print("Number of principal components: {}".format(n_components))
  print("{}%% training data, {}%% test data".format((1 - test_percentage) * 100,
    test_percentage * 100))
  print("Testing with K = [1, {})".format(max_K))

  # Load data in dataframe
  df = pd.read_csv("https://raw.githubusercontent.com/Halleluwah/datasets/master/seeds-dataset.csv", sep=";")

  # Varieties of wheat
  varieties = ["Kama", "Rosa", "Canadian"]

  # Set the target column
  df.rename(columns={df.columns[-1]:"target"}, inplace=True)

  # Apply PCA
  final_df = pca(df, df.columns[:-1], n_components)

  # Concatenate dataset along axis=1
  final_df = pd.concat([final_df, df[["target"]]], axis=1)

  # Visualize 2D projection of the final dataset
  fig = plt.figure(figsize=(12, 4))
  # If 2D scatter plot
  if n_components == 2:
    ax = fig.add_subplot(1, 2, 1)
  else:
    ax = fig.add_subplot(1, 2, 1, projection=str(n_components) + "d")
    ax.set_zlabel("principal component 3")
  ax.set_xlabel("principal component 1")
  ax.set_ylabel("principal component 2")
  ax.set_title("{} component PCA".format(n_components))
  targets = [i + 1 for i in range(len(varieties))]
  colors = random_distinct_colors(len(varieties), 50)
  for target, color in zip(targets, colors):
    indices = final_df["target"] == target
    # Adapt scatter plot to the number of principal components
    ax.scatter(*(final_df.loc[indices, "principal component {}".format(i + 1)]
      for i in range(n_components)), c=[color])
  ax.legend(varieties)
  ax.grid()

  # Split dataset into attributes (X) and labels (y)
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values

  # Divide dataset into traning and test splits
  # (default: 80% train data, 20% test data)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)

  # Apply KNN
  y_test, y_pred = knn(X_train, X_test, y_train, y_test, 5)

  # Evaluate the algorithm
  print("\nConfusion matrix for K = 5:\n{}".format(confusion_matrix(y_test, y_pred)))
  print("\nClassification report for K = 5:\n{}".format(classification_report(y_test, y_pred)))

  # Compare error rate with the K value
  error = []
  for i in range(1, max_K):
    y_test, y_pred = knn(X_train, X_test, y_train, y_test, i)
    error.append(np.mean(y_pred != y_test))

  # Plot error values and K values
  ax = fig.add_subplot(1, 2, 2)
  ax.set_xlabel("K value")
  ax.set_ylabel("Mean error")
  ax.set_title("Error rate in function of K values")
  ax.plot(range(1, max_K), error, color="b", linestyle="dashed",
    marker="o", markerfacecolor="w", markersize=5)

  # Show results
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
	main()
