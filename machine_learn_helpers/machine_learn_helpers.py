# pylance: enable=WARNING_CODE
# pylint: disable=line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, too-many-branches, too-many-return-statements, too-many-nested-blocks

"""Helper functions for machine learning tasks.

This module provides a collection of helper functions for various machine
learning tasks, including feature selection, model fitting, and evaluation.

Functions:
    - iterate_rfe: Performs recursive feature elimination and returns the
        accuracy scores and selected feature counts.
    - find_rfe_features: Selects the top n features using recursive feature
        elimination.
    - logr_coefs: Prints the coefficients of a logistic regression model.
    - grid_searcher: Performs grid search for hyperparameter tuning.
    - model_fitter: Fits a model and returns the accuracy and confusion matrix.
    - text_model_fitter: Fits a text classification model and returns the
        accuracy and confusion matrix.
    - regression_model_fitter: Fits a regression model and returns the R^2 score.
    - scaled_split: Splits the data into train and test sets and scales the data.
    - model_barplot: Plots the accuracy of models in a bar plot.
    - model_heatmap: Plots the confusion matrices in a heatmap format.
    - make_date_features: Creates date-related features from a date column.
"""
import time

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tqdm.notebook import tqdm

np.set_printoptions(precision=2, suppress=True)
# Set options for printing pandas DataFrames
pd.set_option("display.float_format", lambda x: f"{x:.2f}")
pd.set_option("display.max_columns", None)
plt.style.use("seaborn")
np.random.seed(42)


def ml_imports():
    """Standard ML imports to copy and paste"""
    print(
        """import graphviz
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_squared_error,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.svm import SVC, SVR
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        export_graphviz,
    )
    from tqdm.notebook import tqdm

    np.set_printoptions(precision=2, suppress=True)
    # Set options for printing pandas DataFrames
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.max_columns", None)
    plt.style.use("seaborn")
    np.random.seed(42)"""
    )


# Generic iterating for testing
def plot_algorithm(accuracies, poss, k_list, marker=True) -> list:
    """
    Plots the accuracy of a given algorithm with respect to the k parameter.
    Returns a sorted list of the top 10 values, given their k.

    Parameters:
        accuracies (list): A list of accuracy scores for each value of k.
        poss (list): A list of tuples containing the accuracy score and corresponding k value.
        k_list (list): A list of k values used for the algorithm.
        marker (bool, optional): Whether to plot the data points with markers or not. Default is True.

    Returns:
        list: A sorted list of the top 10 tuples from `poss`, containing the accuracy score and corresponding k value.
    """
    plt.figure(figsize=(17, 9))
    if marker is True:
        sorts = sorted(poss, reverse=True)
        print(sorts[:10])
        plt.plot(k_list, accuracies, marker="o")
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.title("Classifier Accuracy")
        plt.show()
    elif marker is False:
        sorts = sorted(poss, reverse=True)
        print(sorts[:10])
        plt.plot(k_list, accuracies)
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.title("Classifier Accuracy")
        plt.show()
    else:
        raise ValueError("Marker must be True or False")
    return sorts


def iterate_knn(X_train, X_test, y_train, y_test, k_list):
    """
    Iterates through the k values in k_list and returns the accuracy and k value by applying KNN(K).
    Returns a list of accuracies and a list of k values.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        k_list (list): A list of k values to iterate over.

    Returns:
        tuple: A tuple containing two lists:
        - accuracies (list): A list of accuracy scores for each value of k.
        - poss (list): A list of tuples containing the accuracy score and corresponding k value.
    """
    accuracies = []
    poss = []
    for k in tqdm(k_list):
        classifier = KNeighborsClassifier(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        poss.append([accuracy_score(y_test, y_pred), k])
    return accuracies, poss


def iterate_dt(X_train, X_test, y_train, y_test, k_list, criterion="gini"):
    """
    Iterates through the k values in k_list and returns the accuracy and k value by applying Decision Tree(DT).
    Returns a list of accuracies and a list of k values.
    Model is created within, so no need to pass it.
    call accuracies, poss = iterate_dt()

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        k_list (list): A list of k values to iterate over for the maximum depth of the Decision Tree.
        criterion (str, optional): The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Default is "gini".

    Returns:
    tuple: A tuple containing two lists:
        - accuracies (list): A list of accuracy scores for each value of k.
        - poss (list): A list of tuples containing the accuracy score and corresponding k value.
    """
    accuracies = []
    poss = []
    for k in tqdm(k_list):
        classifier = DecisionTreeClassifier(
            max_depth=k, random_state=42, criterion=criterion
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        poss.append([accuracy_score(y_test, y_pred), k])
    return accuracies, poss


def plot_dt(dt, x, class_names, show=True):
    """
    Creates a plot of the given decision tree.

    Parameters:
        dt (sklearn.tree.DecisionTreeClassifier or sklearn.tree.DecisionTreeRegressor): The decision tree model to be plotted.
        x (pandas.DataFrame): The input features used to train the decision tree model.
        class_names (list or numpy.ndarray): A list or array of class names for the target variable.
        show (bool, optional): Whether to display the plot or return the graphviz object. Default is True.

    Returns:
    graphviz.Source: The graphviz object representing the decision tree plot.

    Notes:
    - This function requires the graphviz library to be installed.
    - The class names should be the unique values of the target variable (y) in string format.
    - If show is True, the function will display the plot directly. If False, it will return the graphviz object, which can be saved or displayed later.
    Example:
    >>> mlh.plot_dt(dt_model, X, df["class"].unique(), show=True)
    """
    if show is False:
        dot_data = export_graphviz(
            dt,
            out_file=None,
            feature_names=x.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data)
        return graph
    elif show is True:
        dot_data = export_graphviz(
            dt,
            out_file=None,
            feature_names=x.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
        )

        graph = graphviz.Source(dot_data)
        graph.render("decision_tree", view=True)
        return graph
    else:
        raise ValueError("Show must be True or False")


def iterate_rfe(X_train, X_test, y_train, y_test, k_list, estimator, step=1):
    """
    Implements recursive feature elimination to find optimal parameters

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        k_list (list): A list of k values to iterate over for the number of features to select.
        estimator (object): A supervised learning estimator with a `fit` method that updates the `coef_` or `feature_importances_` attribute.
        step (int, optional): If greater than or equal to 1, then `step` corresponds to the (integers) number of features to remove at each iteration. If within (0.0, 1.0), then `step` corresponds to the percentage (rounded down) of features to remove at each iteration. Default is 1.

    Returns:
    tuple: A tuple containing two lists:
        - accuracies (list): A list of accuracy scores for each value of k.
        - poss (list): A list of tuples containing the accuracy score and corresponding k value.
    """

    accuracies = []
    poss = []
    for k in tqdm(k_list):
        classifier = RFE(estimator=estimator, n_features_to_select=k, step=step)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        poss.append([accuracy_score(y_test, y_pred), k])
    return accuracies, poss


def find_rfe_features(X_train, y_train, n_features, estimator, columns):
    """
    Finds the selected features using Recursive Feature Elimination (RFE) with the given estimator and number of features.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        df (pandas.DataFrame): The original DataFrame containing the features and target variable.
        n_features (int): The number of features to select.
        estimator (object): A supervised learning estimator with a `fit` method that updates the `coef_` or `feature_importances_` attribute.
        columns (list or numpy.ndarray): A list or array of feature names corresponding to the columns in X_train (use X.columns).

    Returns:
    numpy.ndarray: A NumPy array containing the names of the selected features.
    """

    selector_new = RFE(estimator=estimator, n_features_to_select=n_features)
    selector_new.fit(X_train, y_train)
    selected_features = columns[selector_new.support_]
    print("Selected Features:")
    return selected_features


def logr_coefs(estimator, X_train, y_train, df, targ_indx):
    """
    Prints the coefficients of a logistic regression model.

    Parameters:
        estimator (object): A logistic regression estimator with a `fit` method.
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        df (pandas.DataFrame): The original DataFrame containing the features and target variable.
        targ_indx (int): The index of the target variable column in the DataFrame.

    Returns:
    pandas.DataFrame: A DataFrame containing the feature names and their corresponding coefficients, sorted in descending order by the coefficient values.
    """
    coef_logr = estimator
    coef_logr.fit(X_train, y_train)
    coeff_df = pd.DataFrame(df.columns.delete(targ_indx))
    coeff_df.columns = ["Feature"]
    coeff_df["Correlation"] = pd.Series(coef_logr.coef_[0])
    coeff_df = coeff_df.sort_values(by="Correlation", ascending=False)
    print("coeff_df")
    return coeff_df


def grid_searcher(
    X_train,
    y_train,
    C=(1, 10, 100),
    gamma=(1, 10, 100),
    kernel=("rbf"),
    verbose=0,
    model=None,
):
    """
    Performs grid search for the given hyperparameters of a Support Vector Machine (SVM) model.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        C (tuple or list, optional): A tuple of values for the regularization parameter C. Default is [1, 10, 100].
        gamma (tuple or list, optional): A tuple of values for the kernel coefficient gamma. Default is [1, 10, 100].
        kernel (tuple or list, optional): A tuple of kernel functions to use. Default is ['rbf'].
        verbose (int, optional): Controls the verbosity of the grid search output. Default is 0 (no output).
        model (object, optional): A custom SVM model object to use for grid search. If None, a default SVC model is used.


    Notes:
    - This function uses the GridSearchCV class from scikit-learn to perform grid search with 5-fold cross-validation.
    - The best hyperparameters and the corresponding cross-validation accuracy score are printed.
    - If a custom model is provided, it should implement the `fit` method and have the same parameters as the SVC model from scikit-learn.
    """
    if model is None:
        svc = SVC(random_state=42)
    else:
        svc = model
    param_grid = {"C": C, "gamma": gamma, "kernel": kernel}

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        svc,
        param_grid,
        cv=5,
        verbose=verbose,
        scoring="accuracy",
    )
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and cross-validation accuracy
    print("Best hyperparameters:", grid_search.best_params_)
    print("Cross-validation accuracy:", grid_search.best_score_)


def model_fitter(X_train, X_test, y_train, y_test, model, y_pred=False, multi=False):
    """
    Fits the given model and returns the accuracy, confusion matrix, and optionally the predictions.

    Parameters:
        X_train (numpy.ndarray or pandas.DataFrame): Training data features.
        X_test (numpy.ndarray or pandas.DataFrame): Test data features.
        y_train (numpy.ndarray or pandas.Series): Training data target labels.
        y_test (numpy.ndarray or pandas.Series): Test data target labels.
        model (object): A supervised learning model with a `fit` and `predict` method.
        y_pred (bool, optional): If True, returns the predicted values for X_test. Default is False.
        multi (bool, optional): If True, calculates precision, recall, and F1-score for multi-class classification problems using the 'macro' averaging method. Default is False.

    Returns:
    tuple: A tuple containing the following values:
        - model_acc (float): The accuracy score of the model on the test data.
        - model_cm (numpy.ndarray): The confusion matrix of the model on the test data.
        - model_pred (numpy.ndarray, optional): The predicted values for X_test if `y_pred` is True.

    Notes:
    - This function fits the given model to the training data and evaluates its performance on the test data.
    - It prints the classification report, confusion matrix, accuracy score, precision, recall, and F1-score.
    - If `multi` is True, the precision, recall, and F1-score are calculated using the 'macro' averaging method, suitable for multi-class classification problems.
    - If `y_pred` is True, the function returns the predicted values for X_test as the third element of the tuple.
    """

    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    print(classification_report(y_test, model_pred))
    model_acc = accuracy_score(y_test, model_pred)
    model_acc *= 100
    model_cm = confusion_matrix(y_test, model_pred)
    print(model_cm)
    print("accuracy", model_acc)
    if multi:
        print("precision", precision_score(y_test, model_pred, average="macro") * 100)
        print("f1_score", f1_score(y_test, model_pred, average="macro") * 100)
        print("recall_score", recall_score(y_test, model_pred, average="macro") * 100)
    else:
        print("precision", precision_score(y_test, model_pred) * 100)
        print("f1_score", f1_score(y_test, model_pred) * 100)
        print("recall_score", recall_score(y_test, model_pred) * 100)
    if y_pred is False:
        return model_acc, model_cm
    elif y_pred is True:
        return model_acc, model_cm, model_pred
    else:
        raise ValueError("y_pred must be True or False")


def text_model_fitter(
    training_vectors, test_vectors, training_labels, test_labels, model, y_pred=False
):
    """
    Fits the given model and returns the accuracy and confusion matrix

    Parameters:
        training_vectors (numpy.ndarray or scipy.sparse matrix): Training data vectors.
        test_vectors (numpy.ndarray or scipy.sparse matrix): Test data vectors.
        training_labels (numpy.ndarray or pandas.Series): Training data labels.
        test_labels (numpy.ndarray or pandas.Series): Test data labels.
        model (object): A supervised learning model with a `fit` and `predict` method.
        y_pred (bool, optional): If True, returns the predicted values for test_vectors. Default is False.

    Returns:
    tuple: A tuple containing the following values:
        - model_score (float): The accuracy score of the model on the test data.
        - model_cm (numpy.ndarray): The confusion matrix of the model on the test data.
        - model_pred (numpy.ndarray, optional): The predicted values for test_vectors if `y_pred` is True.

    Notes:
    - This function fits the given model to the training data and evaluates its performance on the test data.
    - It prints the confusion matrix, accuracy score, precision, recall, and F1-score.
    - If `y_pred` is True, the function returns the predicted values for test_vectors as the third element of the tuple.
    """

    model.fit(training_vectors, training_labels)
    model_pred = model.predict(test_vectors)
    model_score = model.score(test_vectors, test_labels)
    model_score *= 100
    model_cm = confusion_matrix(test_labels, model_pred)
    print(model_cm)
    print("accuracy", model_score)
    print("precision", precision_score(test_labels, model_pred) * 100)
    print("f1_score", f1_score(test_labels, model_pred) * 100)
    print("recall_score", recall_score(test_labels, model_pred) * 100)
    print("model_score")
    if y_pred is False:
        return model_score, model_cm
    elif y_pred is True:
        return model_score, model_cm, model_pred
    else:
        raise ValueError("y_pred must be True or False")


def regression_model_fitter(X_train, X_test, y_train, y_test, model, y_pred=False):
    """
    Fits the given regression model to the training data and evaluates it on the test data.

    Parameters:
        X_train (array-like): Features for training the model.
        X_test (array-like): Features for testing the model.
        y_train (array-like): Target values for training the model.
        y_test (array-like): Target values for testing the model.
        model (object): A regression model object that implements 'fit' and 'predict'.
        y_pred (bool, optional): If True, returns the predicted values for X_test. Defaults to False.

    Returns:
    tuple: A tuple containing the following values:
        - model_score (float): The R^2 score of the model on the test data.
        - model_pred (array-like, optional): The predicted values for X_test if `y_pred` is True. This value is only returned if `y_pred` is True.`
    """

    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    model_score *= 100
    if y_pred is False:
        return model_score
    elif y_pred is True:
        return model_score, model_pred
    else:
        raise ValueError("y_pred must be True or False")


def scaled_split(X, y, test_size=0.25, random_state=42):
    """
    Splits the data into train and test sets and scales the data

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): Input data features.
        y (numpy.ndarray or pandas.Series): Input data target labels.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.25.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Default is 42.

    Returns:
    tuple: A tuple containing the following values:
        - X_train (numpy.ndarray): Scaled training data features.
        - X_test (numpy.ndarray): Scaled test data features.
        - y_train (numpy.ndarray or pandas.Series): Training data target labels.
        - y_test (numpy.ndarray or pandas.Series): Test data target labels.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def model_barplot(
    model_accs: list,
    model_names: list,
    xlim: tuple = (80, 100),
    *,
    title: str = "Accuracy of Models",
    xlabel: str = "Accuracy",
):
    """
    Plots the accuracy of models in a bar plot

    Parameters:
        model_accs (list): A list of accuracy scores for the models.
        model_names (list): A list of names for the models, corresponding to the accuracy scores.
        xlim (tuple, optional): A tuple specifying the x-axis limits for the plot. Default is (80, 100).
        title (str, optional): The title for the plot. Default is "Accuracy of Models".
        xlabel (str, optional): The label for the x-axis. Default is "Accuracy".


    Notes:
    - This function uses the seaborn and matplotlib libraries to create a horizontal bar plot.
    - The accuracy scores and model names are sorted in descending order of accuracy.
    - The accuracy scores are displayed as text annotations on the bars.
    - The plot is displayed using the `plt.show()` function.
    """

    df_accs = pd.DataFrame({"models": model_names, "accuracies_models": model_accs})
    df_sorted = df_accs.sort_values(by="accuracies_models", ascending=False)
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(17, 9))
    plt.title(title)
    sns.barplot(
        y="models",
        x="accuracies_models",
        data=df_sorted,
        palette="Set2",
        orient="h",
    )
    plt.xlim(xlim)
    plt.ylabel("Model")
    plt.xlabel(xlabel)

    for i, v in enumerate(df_sorted["accuracies_models"]):
        plt.text(v + 0.001, i, str(round(v, 2)), color="black")

    sns.despine()
    fig.tight_layout()
    plt.show()


def model_heatmap(
    cm_arrays,
    cm_names,
    nrows=2,
    ncols=2,
    cmap="YlOrRd",
    figsize=(20, 15),
    xlabels=("Positive", "Negative"),
    ylabels=("Positive", "Negative"),
    wspace=None,
    bottom=None,
    top=None,
    left=None,
    right=None,
    tight_layout=True,
):
    """
    Plots the confusion matrices in a heatmap format

    Parameters:
        cm_arrays (list): A list of confusion matrix arrays to plot.
        cm_names (list): A list of names for the confusion matrices, corresponding to the arrays.
        nrows (int, optional): The number of rows in the subplot grid. Default is 2.
        ncols (int, optional): The number of columns in the subplot grid. Default is 2.
        cmap (str, optional): The colormap to use for the heatmap. Default is "YlOrRd".
        figsize (tuple, optional): The size of the figure in inches. Default is (20, 15).
        xlabels (list or tuple, optional): A list of labels for the x-axis. Default is ["Positive", "Negative"].
        ylabels (list or tuple, optional): A list of labels for the y-axis. Default is ["Positive", "Negative"].
        wspace (float, optional): The amount of width reserved for blank space between subplots.
        bottom (float, optional): The bottom margin of the subplots.
        top (float, optional): The top margin of the subplots.
        left (float, optional): The left margin of the subplots.
        right (float, optional): The right margin of the subplots.
        tight_layout (bool, optional): If True, adjusts the padding between and around subplots. Default is True.


    Notes:
    - This function uses the seaborn and matplotlib libraries to create a heatmap plot for the given confusion matrices.
    - The confusion matrices are plotted in a grid of subplots, with the number of rows and columns specified by `nrows` and `ncols`.
    - Each subplot is labeled with the corresponding name from `cm_names`.
    - The colormap used for the heatmap can be specified using the `cmap` parameter.
    - The plot is displayed using the `plt.show()` function.
    """

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Loop through the confusion matrix arrays and plot them on the subplots
    if nrows == 1 and ncols == 1:
        sns.heatmap(cm_arrays, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix for {cm_names}")
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        plt.subplots_adjust(
            wspace=wspace, bottom=bottom, top=top, left=left, right=right
        )
        if tight_layout is True:
            fig.tight_layout()
        plt.show()
    else:
        for i, cm_array in enumerate(cm_arrays):
            row_idx = i // ncols
            col_idx = i % ncols
            if nrows == 1:
                curr_ax = ax[col_idx]
            elif ncols == 1:
                curr_ax = ax[row_idx]
            else:
                curr_ax = ax[row_idx, col_idx]
            sns.heatmap(cm_array, annot=True, fmt="d", cmap=cmap, ax=curr_ax)
            curr_ax.set_xlabel("Predicted")
            curr_ax.set_ylabel("Actual")
            curr_ax.set_title(f"Confusion Matrix for {cm_names[i]}")
            curr_ax.set_xticklabels(xlabels)
            curr_ax.set_yticklabels(ylabels)
        plt.subplots_adjust(
            wspace=wspace, bottom=bottom, top=top, left=left, right=right
        )
        if tight_layout is True:
            fig.tight_layout()
        plt.show()


def make_date_features(
    df,
    date="Date",
    index=True,
    hour=True,
    weekday=True,
    weekend=True,
    day_of_year=True,
    quarter=True,
):
    """
    Creates date features from a dataframe and the column of date

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the date column.
        date (str, optional): The name of the column containing the date information. Default is "Date".
        index (bool, optional): If True, the date column is treated as the index of the DataFrame. Default is True.
        hour (bool, optional): If True, creates an "hour" column with the hour of the day. Default is True.
        weekday (bool, optional): If True, creates a "weekday" column with the day of the week (0-6). Default is True.
        weekend (bool, optional): If True, creates a "weekend" column with a boolean indicating if the day is a weekend. Default is True.
        day_of_year (bool, optional): If True, creates a "day_of_year" column with the day of the year (1-366). Default is True.
        quarter (bool, optional): If True, creates a "quarter" column with the quarter of the year (1-4). Default is True.

    Returns:
    pandas.DataFrame: The input DataFrame with the additional date feature columns.

    Notes:
    - This function assumes that the date column contains datetime objects or can be converted to datetime objects using `pd.to_datetime`.
    - If `index` is True, the date column is treated as the index of the DataFrame, and the new feature columns are added as new columns.
    - If `index` is False, the date column is treated as a regular column, and the new feature columns are added as new columns.
    - If any of the optional parameters (`hour`, `weekday`, `weekend`, `day_of_year`, `quarter`) is False, the corresponding feature column is not created.
    """

    if index is False:
        df[date] = pd.to_datetime(df[date])
        df["year"] = df[date].dt.year
        df["month"] = df[date].dt.month
        df["day"] = df[date].dt.day
        if hour is True:
            df["hour"] = df[date].dt.hour
        if weekday is True:
            df["weekday"] = df[date].dt.weekday
        if weekend is True:
            df["weekend"] = df[date].dt.dayofweek >= 5
        if day_of_year is True:
            df["day_of_year"] = df[date].dt.dayofyear
        if quarter is True:
            df["quarter"] = df[date].dt.quarter

    elif index is True:
        df.index = pd.to_datetime(df.index)
        df.loc[:, "year"] = df.index.year
        df.loc[:, "month"] = df.index.month
        df.loc[:, "day"] = df.index.day
        if hour is True:
            df.loc[:, "hour"] = df.index.hour
        if weekday is True:
            df.loc[:, "weekday"] = df.index.weekday
        if weekend is True:
            df.loc[:, "weekend"] = df.index.dayofweek >= 5
        if day_of_year is True:
            df.loc[:, "day_of_year"] = df.index.dayofyear
        if quarter is True:
            df.loc[:, "quarter"] = df.index.quarter

    else:
        raise ValueError("index must be True or False")
    return df


def xgb_time_series_split(
    df,
    features,
    target,
    cv_search=False,
    test_size=365,
    gap=7,
    n_splits=3,
    date="Date",
    index=True,
    hour=True,
    weekday=True,
    weekend=True,
    day_of_year=True,
    quarter=True,
    verbose=2,
    learning_rate=0.01,
):
    """
    Performs time series split on a dataframe and returns the scores and predictions

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the features and target.
        features (list): A list of column names representing the features.
        target (str): The name of the column representing the target variable.
        cv_search (bool, optional): If True, performs cross-validation search. Default is False.
        test_size (int, optional): The length of the test set. Default is 365.
        gap (int, optional): The gap between the train and test sets. Default is 7.
        n_splits (int, optional): The number of splits for cross-validation. Default is 3.
        date (str, optional): The name of the column containing the date information. Default is "Date".
        index (bool, optional): If True, the date column is treated as the index of the DataFrame. Default is True.
        hour (bool, optional): If True, creates an "hour" column with the hour of the day. Default is True.
        weekday (bool, optional): If True, creates a "weekday" column with the day of the week (0-6). Default is True.
        weekend (bool, optional): If True, creates a "weekend" column with a boolean indicating if the day is a weekend. Default is True.
        day_of_year (bool, optional): If True, creates a "day_of_year" column with the day of the year (1-366). Default is True.
        quarter (bool, optional): If True, creates a "quarter" column with the quarter of the year (1-4). Default is True.
        verbose (int, optional): The verbosity level for XGBoost. Default is 2.
        learning_rate (float, optional): The learning rate for XGBoost. Default is 0.01.

    Returns:
    If cv_search is False, returns:
        - preds (list): A list of predicted values for each test set.
        - scores (list): A list of root mean squared error (RMSE) scores for each test set.
        - params (list): A list of parameter dictionaries for each XGBoost model.
    If cv_search is True, returns:
        - cv_results (dict): A dictionary containing the cross-validation results.

    Notes:
    - This function performs time series cross-validation using the TimeSeriesSplit class from scikit-learn.
    - The input DataFrame is sorted by index before splitting.
    - Date features are created using the `make_date_features` function.
    - An XGBRegressor model is trained and evaluated on each train-test split.
    - If `cv_search` is True, cross-validation is performed to find the best hyperparameters.
    """

    if cv_search is False:
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        df = df.sort_index()

        preds = []
        scores = []
        params = []
        for train_idx, val_idx in tss.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = make_date_features(
                train,
                date=date,
                index=index,
                hour=hour,
                weekday=weekday,
                weekend=weekend,
                day_of_year=day_of_year,
                quarter=quarter,
            )
            test = make_date_features(
                test,
                date=date,
                index=index,
                hour=hour,
                weekday=weekday,
                weekend=weekend,
                day_of_year=day_of_year,
                quarter=quarter,
            )
            X_train = train.loc[:, features]
            y_train = train.loc[:, target]
            X_test = test.loc[:, features]
            y_test = test.loc[:, target]
            reg = xgb.XGBRegressor(
                base_score=0.5,
                booster="gbtree",
                n_estimators=1000,
                early_stopping_rounds=50,
                objective="reg:squarederror",
                max_depth=3,
                learning_rate=learning_rate,
            )
            reg.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100,
            )

            y_pred = reg.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)
        return scores, preds
    else:
        # Define the grid search object

        param_grid = {
            "n_estimators": [100, 500, 1000],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.5],
            "colsample_bytree": [0.3, 0.5, 0.7],
            "subsample": [0.5, 0.7, 1.0],
            "early_stopping_rounds": [50, 100, 150],
        }

        cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        xgb_grid = GridSearchCV(
            estimator=xgb.XGBRegressor(
                base_score=0.5,
                booster="gbtree",
                objective="reg:squarederror",
            ),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            verbose=verbose,
            cv=cv,
        )

        # Fit grid search object to the data
        X = df.loc[:, features]
        X = make_date_features(
            X,
            date=date,
            index=index,
            hour=hour,
            weekday=weekday,
            weekend=weekend,
            day_of_year=day_of_year,
            quarter=quarter,
        )
        y = df.loc[:, target]
        i = 0
        preds = []
        scores = []
        params = []
        for train_index, val_index in cv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            start_time = time.time()
            xgb_grid.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=verbose,
            )
            elapsed_time = time.time() - start_time
            print(
                f"Iteration {i+1}/{n_splits} completed in {elapsed_time:.2f} seconds",
                flush=True,
            )
            # Get best estimator and its corresponding parameters
            xgb_best = xgb_grid.best_estimator_
            print("Best XGBoost regressor: ", xgb_best)
            print("Cross-validation accuracy:", xgb_grid.best_score_)
            xgb_params = xgb_grid.best_params_
            print("Best parameters for XGBoost regressor: ", xgb_params)
            params.append(xgb_params)
            # Make predictions on the test data and compute the score
            y_pred = xgb_best.predict(X_val)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
            i += 1

        return scores, params, preds


def time_series_split_predict(
    df,
    model,
    features,
    target,
    cv_search=False,
    test_size=365,
    gap=7,
    n_splits=3,
    date="Date",
    index=True,
    hour=True,
    weekday=True,
    weekend=True,
    day_of_year=True,
    quarter=True,
):
    """
    Performs time series split on a dataframe and returns the root mean squared error (RMSE) scores.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the features and target.
        model (object): A supervised learning model with a `fit` and `predict` method.
        features (list): A list of column names representing the features.
        target (str): The name of the column representing the target variable.
        cv_search (bool, optional): If True, performs cross-validation search. Default is False.
        test_size (int, optional): The length of the test set. Default is 365.
        gap (int, optional): The gap between the train and test sets. Default is 7.
        n_splits (int, optional): The number of splits for cross-validation. Default is 3.
        date (str, optional): The name of the column containing the date information. Default is "Date".
        index (bool, optional): If True, the date column is treated as the index of the DataFrame. Default is True.
        hour (bool, optional): If True, creates an "hour" column with the hour of the day. Default is True.
        weekday (bool, optional): If True, creates a "weekday" column with the day of the week (0-6). Default is True.
        weekend (bool, optional): If True, creates a "weekend" column with a boolean indicating if the day is a weekend. Default is True.
        day_of_year (bool, optional): If True, creates a "day_of_year" column with the day of the year (1-366). Default is True.
        quarter (bool, optional): If True, creates a "quarter" column with the quarter of the year (1-4). Default is True.

    Returns:
    list: A list of root mean squared error (RMSE) scores for each test set.

    Notes:
    - This function performs time series cross-validation using the TimeSeriesSplit class from scikit-learn.
    - The input DataFrame is sorted by index before splitting.
    - Date features are created using the `make_date_features` function.
    - The provided model is trained on each training set and evaluated on the corresponding test set.
    - The RMSE score is calculated for each test set using the `mean_squared_error` function from scikit-learn.
    - If `cv_search` is True, the function does not perform any operation and returns `None`.
    """

    if cv_search is False:
        tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        df = df.sort_index()

        preds = []
        scores = []
        for train_idx, val_idx in tqdm(tss.split(df)):
            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            train = make_date_features(
                train,
                date=date,
                index=index,
                hour=hour,
                weekday=weekday,
                weekend=weekend,
                day_of_year=day_of_year,
                quarter=quarter,
            )
            test = make_date_features(
                test,
                date=date,
                index=index,
                hour=hour,
                weekday=weekday,
                weekend=weekend,
                day_of_year=day_of_year,
                quarter=quarter,
            )

            X_train = train.loc[:, features]
            y_train = train.loc[:, target]
            X_test = test.loc[:, features]
            y_test = test.loc[:, target]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)
        return scores
    else:
        pass


def visualize_time_series_split(
    df,
    target,
    test_size=365,
    gap=7,
    n_splits=3,
    n_rows=3,
    n_cols=1,
    figsize=(17, 7),
    sharex=True,
):
    """
    Visualizes the time series split on a dataframe.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the target variable.
        target (str): The name of the column representing the target variable.
        test_size (int, optional): The length of the test set. Default is 365.
        gap (int, optional): The gap between the train and test sets. Default is 7.
        n_splits (int, optional): The number of splits for cross-validation. Default is 3.
        n_rows (int, optional): The number of rows in the subplot grid. Default is 3.
        n_cols (int, optional): The number of columns in the subplot grid. Default is 1.
        figsize (tuple, optional): The size of the figure in inches. Default is (17, 7).
        sharex (bool, optional): If True, the x-axis is shared across all subplots. Default is True.

    Notes:
    - This function uses the TimeSeriesSplit class from scikit-learn to split the data into train and test sets.
    - The function creates a grid of subplots using `plt.subplots` from matplotlib.
    - For each train-test split, the function plots the target variable for the training and test sets in a separate subplot.
    - The function adds a vertical line at the start of the test set to visually separate the train and test data.
    - The function adjusts the layout of the subplots using `plt.tight_layout()`.
    - The function does not return any values; it only displays the plot.
    """

    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    # pylint: disable=unused-variable
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex)
    # pylint: enable=unused-variable
    fold = 0
    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        train[target].plot(
            ax=axs[fold],
            label="Training Set",
            title=f"Data Train/Test Split Fold {fold+1}",
        )
        test[target].plot(ax=axs[fold], label="Test Set")
        axs[fold].axvline(test.index.min(), color="black", ls="--")
        fold += 1
    plt.tight_layout()


def create_lags(df, target, n_lags=3):
    """
    Creates lag features for a dataframe.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the target variable.
        target (str): The name of the column representing the target variable.
        n_lags (int, optional): The number of lag features to create. Default is 3.

    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for lag features.

    Notes:
    - This function assumes that the index of the input DataFrame is a datetime-like index.
    - The lag features are created by shifting the target column values by a specified number of days (determined by `lag*364`).
    - The resulting lag columns are named `"lag1"`, `"lag2"`, `"lag3"`, and so on, based on the value of `n_lags`.
    - The lag features can be useful when working with time series data, as they incorporate past values of the target variable into the analysis or modeling process.

    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Date': pd.date_range('2022-01-01', periods=10),
    ...                    'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> df = df.set_index('Date')
    >>> df = create_lags(df, 'target', n_lags=2)
    >>> df.head()
                   target  lag1  lag2
    Date
    2022-01-01        1   NaN   NaN
    2022-01-02        2   1.0   NaN
    2022-01-03        3   2.0   1.0
    2022-01-04        4   3.0   2.0
    2022-01-05        5   4.0   3.0
    """
    target_map = df[target].to_dict()
    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = (df.index - pd.Timedelta(f"{lag*364} days")).map(target_map)
    return df


def create_future_df(
    start,
    end,
    df,
    target,
    freq="D",
    date="Date",
    index=True,
    hour=True,
    weekday=True,
    weekend=True,
    day_of_year=True,
    n_lags=3,
):
    """
    Creates a future dataframe with date features and lag features.

    This function generates a DataFrame with future dates, adds date-related features, and includes lag features
    based on the provided historical data. It also separates the future data from the historical data.

    Parameters:
        start (str or datetime-like): The start date of the future period. Should be in a format recognized by
            `pd.date_range`.
        end (str or datetime-like): The end date of the future period. Should be in a format recognized by
            `pd.date_range`.
        df (pd.DataFrame): The historical data to be used. Must contain the target column and optionally a column
            with dates.
        target (str): The name of the target column in `df` for which lag features will be created.
        freq (str, optional): Frequency of the future dates. Default is "D" for daily. Other options include "H"
            for hourly, "W" for weekly, etc.
        date (str, optional): The name of the date column in `df` used for creating date features. Default is "Date".
        index (bool, optional): Whether to include the index of the future dates in the DataFrame. Default is True.
        hour (bool, optional): Whether to include hour features. Default is True.
        weekday (bool, optional): Whether to include weekday features. Default is True.
        weekend (bool, optional): Whether to include weekend features. Default is True.
        day_of_year (bool, optional): Whether to include day-of-year features. Default is True.
        n_lags (int, optional): The number of lag features to create. Default is 3.

    Returns:
    tuple:
        -future_w_features (pd.DataFrame): DataFrame containing the future dates with date features and lag features.
        -df_and_future (pd.DataFrame): Combined DataFrame of historical data and future dates, including both date features and lag features.
    """
    df = df.copy()
    future = pd.date_range(start, end, freq=freq)
    future_df = pd.DataFrame(index=future)
    future_df["isFuture"] = True
    df["isFuture"] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = make_date_features(
        df_and_future,
        date=date,
        index=index,
        hour=hour,
        weekday=weekday,
        weekend=weekend,
        day_of_year=day_of_year,
    )
    df_and_future = create_lags(df_and_future, target, n_lags=n_lags)
    future_w_features = df_and_future.query("isFuture").copy()
    return future_w_features, df_and_future
