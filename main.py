import json

from models.KNN import KNN
from models.refact import refact_dataset

def get_config():
    file = open('config.json', 'r')
    data = json.loads(file.read())
    
    file.close()
    return data

def main():
    cfg = get_config()
    print("STARTING - REFACTOR FILE")

    refact_dataset(
        cfg["datasetPath"] + cfg["datasetFilename"],
        cfg["datasetPath"] + cfg["refactoredFilename"]
    )

    print("ENDED - REFACTOR FILE\n")

    print("STARTING - KNN HEPATITE")
    knn = KNN(cfg["datasetPath"] + cfg["refactoredFilename"], 'Category')

    firstscore = knn.train_knn(5)
    print('score | K = 5: ', firstscore)
    print(" - close panel to continue - ")
    knn.search_best_k_value(showPLT = True)
    print("best K: ", knn.best_K)
    
    bestscore = knn.train_knn(None)
    print('score | K = ', knn.best_K, ': ', bestscore)
    print("ENDED - KNN HEPATITE\n")


if __name__ == "__main__":
    main()