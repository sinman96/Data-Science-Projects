import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
    
def eu_export_information(data, datasubset, time_period):
    #Find the amount traded in exports with each country
    export_totals = []
    for i in range(0,len(datasubset)):
        export_totals.append(data[data.Partner == datasubset[i]]
    ['Trade Value (US$)'].sum())
    export_total = sum(export_totals)
    for i in range(0, len(export_totals)):
        export_totals[i] /= export_total
    #Weighted percentages of trade with each country over this time
    most_traded_countries = []
    for i in range(0, len(datasubset)):
        most_traded_countries.append([datasubset[i],export_totals[i]])
    print(most_traded_countries)
    df = pd.DataFrame({'Percentage of Russian EU exports': export_totals},
                  index= datasubset)
    plot = df.plot.pie(figsize=(20, 20), subplots = "True")
    plt.title('Russian EU export partners ' + time_period, fontsize = 20)
    plt.legend(loc ='lower left')
    return export_totals

def eu_export_changes(phase3, phase2, phase1, countries):
    percentage_changes1to2 = []
    percentage_changes2to3 = []
    eu_changes = []
    for i in range(0,len(phase1)):
        percentage_changes1to2.append(
        100 *(phase2[i] - phase1[i])/phase1[i])
        percentage_changes2to3.append(
        100 *(phase3[i] - phase2[i])/phase2[i])
        
    for i in range(0, len(countries)):
        eu_changes.append([countries[i], percentage_changes1to2[i],
                          percentage_changes2to3[i]])
    df = pd.DataFrame(eu_changes, columns=
                      ['Country',
                       'Post Georgia trade % change',
                       'Post Crimea trade % change'])
    ax = df.plot.bar()
    plt.xticks(range(len(df)),df['Country'])
    plt.legend(loc='upper left')
    plt.show()

def model_accuracy_trades(data):
    #Default predictive data split	
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.squeeze(y)
    print(X)
    #Only non numeric fields are Partner and Commodity, so those will be encoded
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct1 = ColumnTransformer(transformers=[('encoder',
    OneHotEncoder(sparse = False), [2])], remainder='passthrough')
    X = np.array(ct1.fit_transform(X))
    
    """## Splitting the dataset into the Training set and Test set"""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size = 0.25, random_state = 0)
 
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    """## Making the Confusion Matrix"""
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    sns.heatmap(cm, cmap = 'BuPu_r', annot=True)
    print("XGBoost Classification Accuracy is: {:.2f} %".format(100*accuracy_score(y_test, y_pred)))

    """## Applying K-Fold Cross Validation##"""
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("K-Fold-Cross-Validation Accuracy is: {:.2f} %".format(accuracies.mean()*100))
    print("K-Fold-Cross-Validation Standard Deviation is: {:.2f} %".format(accuracies.std()*100))
