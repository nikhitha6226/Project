from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,drink_driving_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Drink_Driving_Detection(request):
    if request.method == "POST":

        if request.method == "POST":

            idnumber= request.POST.get('idnumber')
            City_Location= request.POST.get('City_Location')
            day= request.POST.get('day')
            Sex= request.POST.get('Sex')
            Age= request.POST.get('Age')
            Time= request.POST.get('Time')
            Day_of_week= request.POST.get('Day_of_week')
            Educational_level= request.POST.get('Educational_level')
            Vehicle_driver_relation= request.POST.get('Vehicle_driver_relation')
            Driving_experience= request.POST.get('Driving_experience')
            Type_of_vehicle= request.POST.get('Type_of_vehicle')
            Owner_of_vehicle= request.POST.get('Owner_of_vehicle')
            Service_year_of_vehicle= request.POST.get('Service_year_of_vehicle')
            Lanes_or_Medians= request.POST.get('Lanes_or_Medians')
            Road_allignment= request.POST.get('Road_allignment')
            Road_surface_type= request.POST.get('Road_surface_type')
            Vehicle_movement= request.POST.get('Vehicle_movement')

        df = pd.read_csv('Driving_Datasets.csv', encoding='latin-1')

        def apply_response(Label):
            if (Label == 0):
                return 0  # Not Detected
            elif (Label == 1):
                return 1  # Detected

        df['results'] = df['Label'].apply(apply_response)

        # cv = CountVectorizer()
        x = df['idnumber'].apply(str)
        y = df['results']

        print("Review")
        print(x)
        print("Results")
        print(y)

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))

        # X = cv.fit_transform(df['Vehicle_movement'].apply(lambda x: np.str_(x)))

        X = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        idnumber1 = [idnumber]
        vector1 = cv.transform(idnumber1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Drink Driving Detection'
        elif (prediction == 1):
            val = 'Drink Driving Detection'

        print(val)
        print(pred1)

        drink_driving_detection.objects.create(idnumber=idnumber,
        City_Location=City_Location,
        day=day,
        Sex=Sex,
        Age=Age,
        Time=Time,
        Day_of_week=Day_of_week,
        Educational_level=Educational_level,
        Vehicle_driver_relation=Vehicle_driver_relation,
        Driving_experience=Driving_experience,
        Type_of_vehicle=Type_of_vehicle,
        Owner_of_vehicle=Owner_of_vehicle,
        Ser_year_of_veh=Service_year_of_vehicle,
        Lanes_or_Medians=Lanes_or_Medians,
        Road_allignment=Road_allignment,
        Road_surface_type=Road_surface_type,
        Vehicle_movement=Vehicle_movement,
        Prediction=val)

        return render(request, 'RUser/Predict_Drink_Driving_Detection.html',{'objs': val})
    return render(request, 'RUser/Predict_Drink_Driving_Detection.html')



