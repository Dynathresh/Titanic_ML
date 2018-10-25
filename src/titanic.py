"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from src.util import *
from collections import Counter

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from scipy.interpolate import spline


######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set

        counts = Counter(y).most_common()
        self.probabilities_ = counts
        return self

        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)

        classes = []
        counts = []
        for count in self.probabilities_:
            classes.append(count[0])
            counts.append(count[1])
        my_size = 0
        for c in counts:
            my_size += c
        for i in range(0, len(counts)):
            counts[i] = counts[i]/my_size
        n, d = X.shape
        y = np.random.choice(classes, size=n, p=counts)
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2):
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = 0
    test_error = 0
    for t in range(0, ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=t)
        clf.fit(X_train, y_train)  # fit training data using the classifier

        y_train_pred = clf.predict(X_train)  # take the classifier and run it on the training data
        train_error += 1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True)

        y_test_pred = clf.predict(X_test)  # take the classifier and run it on the training data
        test_error += 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)

    train_error = train_error/ntrials
    test_error = test_error/ntrials

    print('\t-- cross-validated %s training error: %.3f' % (clf.__class__.__name__, train_error))
    print('\t-- cross-validated %s testing error: %.3f' % (clf.__class__.__name__, test_error))

        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()  # create Random classifier, which includes all model parameters
    clf.fit(X, y)  # fit training data using the classifier
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X,y)
    y_pred = clf.predict(X)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph

    # save the classifier -- requires GraphViz and pydot
    '''
    import pydot
    from io import StringIO
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    '''



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    for k in [3, 5, 7]:
        print('k = {}'.format(k))
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        y_pred = clf.predict(X)  # take the classifier and run it on the training data
        train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
        print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    error(MajorityVoteClassifier(), X, y, ntrials=100, test_size=0.2)
    error(RandomClassifier(), X, y, ntrials=100, test_size=0.2)
    error(DecisionTreeClassifier(criterion='entropy'), X, y, ntrials=100, test_size=0.2)
    error(KNeighborsClassifier(n_neighbors=5), X, y, ntrials=100, test_size=0.2)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    k_list = []
    error_list = []
    for k in range(1, 50, 2):
        k_list.append(k)
        clf = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(clf, X, y, cv=10)
        avg_score = sum(cv_scores)/len(cv_scores)
        err = 1 - avg_score
        error_list.append(err)
    # xs = np.linspace(min(k_list), max(k_list), 300)  # 300 represents number of points to make between T.min and T.max
    # power_smooth = spline(k_list, error_list, xs)
    # plt.plot(xs, power_smooth)
    plt.plot(k_list, error_list) #Original plot connected with straight lines
    plt.show()

    best_k = k_list[error_list.index(min(error_list))]
    print('k for lowest error: k = {}'.format(best_k))

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    d_list = []
    errors_list = []
    train_error_list = []
    test_error_list = []
    for d in range(1,21):
        d_list.append(d)
        print('d = {}'.format(d))
        err = error(DecisionTreeClassifier(criterion='entropy', max_depth=d), X, y, ntrials=100, test_size=0.2)
        errors_list.append(err)
        train_error_list.append(err[0])
        test_error_list.append(err[1])
    '''
    # xs = np.linspace(min(d_list), max(d_list), 300)  # 300 represents number of points to make between T.min and T.max
    # power_smooth = spline(d_list, train_error_list, xs)
    # plt.plot(xs, power_smooth)
    # plt.show()
    #
    # xl = np.linspace(min(d_list), max(d_list), 300)  # 300 represents number of points to make between T.min and T.max
    # power_smooth = spline(d_list, test_error_list, xl)
    # plt.plot(xl, power_smooth)
    # plt.show()
    '''

    # plt.plot(d_list, train_error_list) #Original plot connected with straight lines
    # plt.show()
    # plt.plot(d_list, test_error_list)  #Original plot connected with straight lines
    # plt.show()

    plt.plot(d_list, errors_list)
    plt.show()

    best_train_d = d_list[train_error_list.index(min(train_error_list))]
    best_test_d = d_list[test_error_list.index(min(test_error_list))]
    print('Depth for lowest training error: d = {}'.format(best_train_d))
    print('Depth for lowest testing error: d = {}'.format(best_test_d))

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    error(DecisionTreeClassifier(criterion='entropy'), X, y, ntrials=100, test_size=0.1)


    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
    split_size_list = []
    decision_train_error_list = []
    decision_test_error_list = []
    knn_train_error_list = []
    knn_test_error_list = []
    for i in range(1, 11):
        split_size_list.append(0.1 * i)
        max_index = int(0.1 * i * X_train.shape[0])
        sub_X_train = X_train[0:max_index]
        sub_y_train = y_train[0:max_index]

        clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)
        clf.fit(sub_X_train, sub_y_train)
        sub_y_train_pred = clf.predict(sub_X_train)  # take the classifier and run it on the training data
        train_error = 1 - metrics.accuracy_score(sub_y_train, sub_y_train_pred, normalize=True)
        decision_train_error_list.append(train_error)
        y_test_pred = clf.predict(X_test)  # take the classifier and run it on the testing data
        test_error = 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)
        decision_test_error_list.append(test_error)

        clf = KNeighborsClassifier(n_neighbors=7)
        clf.fit(sub_X_train, sub_y_train)
        sub_y_train_pred = clf.predict(sub_X_train)  # take the classifier and run it on the training data
        train_error = 1 - metrics.accuracy_score(sub_y_train, sub_y_train_pred, normalize=True)
        knn_train_error_list.append(train_error)
        y_test_pred = clf.predict(X_test)  # take the classifier and run it on the testing data
        test_error = 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)
        knn_test_error_list.append(test_error)

    print(decision_train_error_list)
    print(decision_test_error_list)
    plt.plot(split_size_list, decision_train_error_list) #Original plot connected with straight lines
    plt.show()
    plt.plot(split_size_list, decision_test_error_list)  #Original plot connected with straight lines
    plt.show()

    print(knn_train_error_list)
    print(knn_test_error_list)
    plt.plot(split_size_list, knn_train_error_list)  # Original plot connected with straight lines
    plt.show()
    plt.plot(split_size_list, knn_test_error_list)  # Original plot connected with straight lines
    plt.show()
    '''
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
