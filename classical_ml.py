import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import argparse
import timeit

def load_data():
    train_data = np.load("train_data.npy")
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    test_data = np.load("test_data.npy")
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(f"Train data shape: {x_train.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def get_model(model_type="svm"):
    if model_type == "svm":
        return SVC(C=1.0, kernel="rbf", gamma="scale", random_state=123, verbose=True)
    else:
        return LogisticRegression(penalty='l2', C=1.0, random_state=123, verbose=1, n_jobs=-1, max_iter=1000)
    

def train(model_type, use_pca=False, n_components=0.95):
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_data()
    model = get_model(model_type)

    if use_pca:
        pca = PCA(n_components=n_components, random_state=123).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        print("Applied PCA")

    print("Training model...")
    model.fit(x_train, y_train)
    print(f"Accuracy on test set: {model.score(x_test, y_test)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="svm", help="Type of model to train")
    parser.add_argument("-p", "--use_pca", action="store_true", help="Whether to use PCA")
    parser.add_argument("-n", "--n_components", type=float, default=0.95, help="Number of components to keep")

    args = parser.parse_args()

    start = timeit.default_timer()
    train(args.model_type, args.use_pca, args.n_components)
    print(f"Time taken: {timeit.default_timer() - start} seconds")