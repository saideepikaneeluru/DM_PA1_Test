# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
import utils as u
# Any other imports you need


# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """


    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        
        answer = {}
        answer["nb_classes_train"] = len(np.unique(ytrain))
        answer["nb_classes_test"] = len(np.unique(ytest))
        answer["class_count_train"] = np.bincount(ytrain)
        answer["class_count_test"] = np.bincount(ytest)
        answer["length_Xtrain"] = Xtrain.shape[0]
        answer["length_Xtest"] = Xtest.shape[0]
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        
        """Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")"""
        
        

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """


    def partB(self, input_X, input_y, train_sizes=[1000, 5000, 10000], test_sizes=[200, 1000, 2000]):
    result_dict = {}

        for train_size, test_size in zip(train_sizes, test_sizes):
            # Extract a subset of the dataset
            X_train_subset = input_X[:train_size]
            y_train_subset = input_y[:train_size]
            X_test_subset = input_X[train_size:train_size + test_size]
            y_test_subset = input_y[train_size:train_size + test_size]

            # Instantiate a logistic regression model
            classifier = LogisticRegression(max_iter=300, solver='lbfgs', multi_class='multinomial', random_state=self.seed)

            # Train the model
            classifier.fit(X_train_subset, y_train_subset)

            # Cross-validation using ShuffleSplit for training data
            cv_splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
            cross_val_scores = cross_val_score(classifier, X_train_subset, y_train_subset, cv=cv_splitter)

            # Training and testing scores
            train_accuracy = classifier.score(X_train_subset, y_train_subset)
            test_accuracy = classifier.score(X_test_subset, y_test_subset)

            # Class distribution in training and testing sets
            unique_classes, train_class_counts = np.unique(y_train_subset, return_counts=True)
            _, test_class_counts = np.unique(y_test_subset, return_counts=True)

            # Populate the result dictionary
            answer[ntrain] = {
                "ntrain": train_size,
                "ntest": test_size,
                "class_count_train": train_class_counts.tolist(),
                "class_count_test": test_class_counts.tolist(),
                "mean_cv_score": np.mean(cross_val_scores),
                "std_cv_score": np.std(cross_val_scores),
                "train_score": train_accuracy,
                "test_score": test_accuracy,
            }

        return result_dict



 
