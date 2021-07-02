import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

class KNN:
    def __init__(self, filepath, target):
        self.dataset = pd.read_csv(filepath)
        self.X = self.dataset.drop([target], axis=1)
        self.Y = self.dataset[target]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.4, random_state=0)

        self.best_K = 0

    def train_knn(self, K):
        if (self.best_K != 0 and K == None):
            knn = KNeighborsClassifier(self.best_K)
        else :
            knn = KNeighborsClassifier(K)

        knn.fit(self.X_train, self.Y_train)

        return knn.score(self.X_test, self.Y_test)

    def search_best_k_value(self, showPLT):
        k_range = list(range(1,26))
        scores = []
        best_score = -1
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.Y_train)
            Y_pred = knn.predict(self.X_test)

            new_score = metrics.accuracy_score(self.Y_test, Y_pred)

            if (new_score > best_score):
                best_score = new_score
                self.best_K = k
            scores.append(new_score)

        if showPLT:
            plt.plot(k_range, scores)
            plt.xlabel('Valores de K')
            plt.ylabel('Pontuação')
            plt.title('Pontuação para diferentes valores de K')
            plt.show()