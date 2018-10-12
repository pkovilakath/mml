
from django.contrib import messages
from django.shortcuts import render
from datetime import datetime
import os
import pandas

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    ('kn', KNeighborsClassifier(3)),
    ('svc', SVC(kernel="rbf", C=0.025, probability=True)),
    ('nusvc', NuSVC(probability=True)),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
    ('ab', AdaBoostClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('gn', GaussianNB()),
    ('lda', LinearDiscriminantAnalysis()),
    ('qda', QuadraticDiscriminantAnalysis())]


def home(request):
    if request.method == "POST" and request.FILES['training_file']:
        doIt = True

        # catch all possible errors
        if request.POST["fileType"] != "csv":
            messages.error(request, "Currently accepting only csv files.")
            doIt = False

        if int(float(request.POST["totFeatures"])) <= 0:
            messages.error(request, "Invalid Number of features in each tuple of the Training File.")
            doIt = False
        else:
            totFeatures = int(float(request.POST["totFeatures"]))

        if int(float(request.POST["numClassification"])) <= 0:
            messages.error(request, "Invalid Element number of classification (0 based) in each tuple.")
            doIt = False
        else:
            numClassification = int(float(request.POST["numClassification"]))

        if doIt:
            fn = handle_uploaded_file(request.FILES['training_file'], str(request.FILES['training_file']))
            dataframe = pandas.read_csv(fn)  # , names=names)
            # dataframes to maximise data for results... n sets and rotate to try different set as test with others as learning input
            array = dataframe.values
            X = array[:, 0:totFeatures]
            Y = array[:, numClassification]

            # prepare configuration for cross validation test harness
            seed = 11

            # evaluate each model in turn
            results = []
            names = []
            scoring = 'accuracy'
            for name, model in classifiers:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)

    return render(request, "app/home.html",
                  {
                      'number': 5,
                      'mainTitle': 'ML - Classification ',
                      'year': datetime.now().year,
                  })


def handle_uploaded_file(file, filename):
    if not os.path.exists('upload/'):
        os.mkdir('upload/')
    fn = 'upload/' + filename
    with open(fn, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return fn
