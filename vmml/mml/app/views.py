
from django.contrib import messages
from django.shortcuts import render
from datetime import datetime
import os
import pandas
# import matplotlib.pyplot as plt
import time

from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    ('kn', KNeighborsClassifier(3), 'K Neighbors'),
    ('svc', SVC(kernel="rbf", C=0.025, probability=True), 'Support Vector'),
    ('nusvc', NuSVC(probability=True), 'NuSupport Vector'),
    ('dt', DecisionTreeClassifier(), 'Decision Tree'),
    ('rf', RandomForestClassifier(), 'Random Forest'),
    ('ab', AdaBoostClassifier(), 'Ada Boost'),
    ('gb', GradientBoostingClassifier(), 'Gradient Boosting'),
    ('gn', GaussianNB(), 'Gaussian NB'),
    ('lda', LinearDiscriminantAnalysis(), 'Linear Discriminant Analysis'),
    ('qda', QuadraticDiscriminantAnalysis(), 'Quadratic Discriminant Analysis')]


def home(request):
    effs = ["<p>Once the Training is complete, efficacy results will be displayed here.</p>"]
    tts = ["<p>Once the Training is complete, Results will be displayed here.</p>"]

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
            selectedClassifiers=request.POST.getlist('classifier')
            for name, model, fn in classifiers:
                if name in selectedClassifiers:
                    startT = time.time()
                    kfold = model_selection.KFold(n_splits=10, random_state=seed)
                    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                    endT = time.time()
                    results.append((name, model, fn, cv_results.mean(), cv_results.std(), (endT-startT)))
                    # results.append(cv_results)
                    # names.append(name)
                    #avg=cv_results.mean()

                    #some output
                    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    # print(msg)

            #sort by mean & time
            results_eff=results[:]
            results_eff.sort(reverse=True,key=effSortFunc)
            # results_tt=results[:]
            # results_tt.sort(key=ttSortFunc)

            #set up display
            effs=["<table class='cellpadding'><tr><th>Model</th><th>Efficacy (%)</th><th>Time (ms)</th></tr>"]
            for r in results_eff:
                effs.append("<tr><td>"+r[2]+"</td><td style='text-align:right;'>"+'{0:9.2f}'.format(r[3]*100)+"</td><td style='text-align:right;'>"+'{0:9.2f}'.format(r[5]*1000)+"</td></tr>")
            effs.append("</table>")

            # # boxplot algorithm comparison
            # fig = plt.figure()
            # fig.suptitle('Algorithm Comparison')
            # ax = fig.add_subplot(111)
            # plt.boxplot(results)
            # ax.set_xticklabels(names)
            # plt.show()



    return render(request, "app/home.html",
                  {
                      'number': 5,
                      'mainTitle': 'ML - Classification ',
                      'year': datetime.now().year,
                      'effs': effs,
                      'tts': tts,
                  })

def effSortFunc(e):
    return(e[3])

def ttSortFunc(e):
    return(e[5])

def handle_uploaded_file(file, filename):
    if not os.path.exists('upload/'):
        os.mkdir('upload/')
    fn = 'upload/' + filename
    with open(fn, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return fn
