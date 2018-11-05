
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
from django.http import JsonResponse
import pickle

CONST_UPLOAD_DIR = "upload"

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

def predict(request):
    fileName=""
    totFeatures=0
    numClassification=0

    if request.method == "POST" and request.FILES['test_file']:
        doIt = True

        # catch all possible errors
        if request.POST["fileType"] != "csv":
            messages.error(request, "Currently accepting only csv files.")
            doIt = False

        if int(float(request.POST["totFeatures"])) <= 0:
            messages.error(request, "Invalid Element number of features (0 based) in each tuple.")
            doIt = False
        else:
            totFeatures = int(float(request.POST["totFeatures"]))

        if "model" in request.POST:
            fileName = request.POST['model']+'.model'
        else:
            messages.error(request, "No model selected.")
            doIt = False


        if doIt:
            #test file
            testfile=str(request.FILES['test_file'])
            fn = handle_uploaded_file(request.FILES['test_file'], str(request.FILES['test_file']))
            dataframe = pandas.read_csv(fn)  # , names=names)
            # dataframes to maximise data for results... n sets and rotate to try different set as test with others as learning input
            array = dataframe.values
            X_test = array[:, 0:totFeatures]
            Y_test = [0] * len(X_test)

            # prepare configuration for cross validation test harness
            seed = 11

            #model
            loaded_model = pickle.load(open(mkFullPath(fileName), 'rb'))
            result = loaded_model.(X_test, Y_test)
            print(result)

    # set up display
    savedModels=[]
    for file in os.listdir(CONST_UPLOAD_DIR):
        if file.endswith(".model"):
            fileN=os.path.splitext(file)[0]
            elements=fileN.split('^')
            elements.append(fileN)
            savedModels.append(elements)
            # data_html=data.to_html()

    effs = "No Models saved."
    if len(savedModels) > 0:
            #set up display
            effs="<table class='cellpadding'><tr><th>Name</th><th>Model</th><th>Training File</th><th>Efficacy (%)</th><th>Time (ms)</th><th># of Features</th><th>Use</th></tr>"
            for r in savedModels:
                effs+="<tr><td>"+r[0]+"</td><td>"+r[1]+"</td><td>"+r[2]+"</td><td style='text-align:right;'>"+r[3]+"</td><td style='text-align:right;'>"+r[4]+"</td><td style='text-align:right;'>"+r[5]+\
                      "</td><td><input type='radio' id='"+r[0]+r[1]+r[2]+"' name='model' value='"+r[6]+"' onclick='updateNumFeatures("+r[5]+")'/></td></tr>"
            effs+="</table>"

    return render(request, "app/predict.html",
                  {
                      'mainTitle': 'ML - Classification ',
                      'year': datetime.now().year,
                      'effs': effs,
                  })



def saveModel(request):
    rData={'msg':'Error','mn':'none'}
    totFeatures=int(request.POST['totFeatures'])
    numClassification=int(request.POST['numClassification'])
    mn=request.POST['mn']
    fileName=request.POST['fileName']
    userfn=request.POST["userfn"]
    ef=request.POST["ef"]
    ti=request.POST["ti"]
    classifier=()
    if request.method == "POST" and len(userfn)>0 and len(mn)>0 and len(fileName)>0 and totFeatures>0 and numClassification>0:
        doIt = True
        if doIt:
            fileName=fileName
            fn=mkFullPath(fileName)
            dataframe = pandas.read_csv(fn)  # , names=names)
            array = dataframe.values
            X = array[:, 0:totFeatures]
            Y = array[:, numClassification]
            for item in classifiers:
                if item[2]==mn:
                    rData['mn']=mn
                    classifier=item[1]
                    break

            if classifier != ():
                seed = 11
                classifier.fit(X,Y)
                import re
                userfn=re.sub('[!^$]', '-', userfn)
                savename=mkFullPath(userfn+'^'+mn+'^'+fileName+'^'+ef+'^'+ti+'^'+str(totFeatures)+'.model')
                pickle.dump(classifier, open(savename, 'wb'))

    return JsonResponse(rData)


def learn(request):
    effs = ["<p>Once the Training is complete, efficacy results will be displayed here.</p>"]
    # tts = ["<p>Once the Training is complete, Results will be displayed here.</p>"]
    fileName=""
    totFeatures=0
    numClassification=0

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
            messages.error(request, "Invalid Element number of features (0 based) in each tuple.")
            doIt = False
        else:
            numClassification = int(float(request.POST["numClassification"]))

        if doIt:
            fileName=str(request.FILES['training_file'])
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
            effs=["<table class='cellpadding'><tr><th>Model</th><th>Efficacy (%)</th><th>Time (ms)</th><th></th></tr>"]
            for r in results_eff:
                ef='{0:9.2f}'.format(r[3]*100)
                ti='{0:9.2f}'.format(r[5]*1000)
                effs.append("<tr><td>"+r[2]+"</td><td style='text-align:right;'>"+ef+"</td><td style='text-align:right;'>"+ti+"</td><td><button onclick='save_model("+'"'+r[2]+'","'+fileName+'",'+str(totFeatures)+','+str(numClassification)+',"'+ef.strip()+'","'+ti.strip()+'")'+"' style='border-radius: 5px;'>Save Model</button></td></tr>")
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
                      'fileName': fileName,
                      'fn': fileName,
                      'totFeatures':totFeatures,
                      'numClassification': numClassification,
                  })

def effSortFunc(e):
    return(e[3])

def ttSortFunc(e):
    return(e[5])

def handle_uploaded_file(file, filename):
    if not os.path.exists(CONST_UPLOAD_DIR+'/'):
        os.mkdir(CONST_UPLOAD_DIR+'/')
    fn = mkFullPath(filename)
    with open(fn, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return fn

def mkFullPath(filename):
    return CONST_UPLOAD_DIR+'/' + filename