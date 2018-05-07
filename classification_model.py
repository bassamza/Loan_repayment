"""
Run this script third

This script builds random forest and gradient boosting classifiers to model the loan repayment status. Randomizedsearch
method is used to find the best set of parameters for each classifier using the 'AUC' score. The performance of the
models is compared with respect to abase classifier, i.e most frequent class. Ranking of the most important features
using the classifier results are reported and plotted. Also, the ROC curve for two classifiers is plotted.

__author__ = "Nastaran Bassamzadeh"
__email__ = "nsbassamzadeh@gmail.com"

"""

import os
from time import time
import pandas as pd
import numpy as np
from scipy import interp
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.metrics import roc_curve, auc

# Create the directory 'figures' if it is does not exist
if not os.path.isdir("./figures/classifiers"):
    os.makedirs("./figures/classifiers")
# -----------------------------------------------------------------------------------------------------------------------

# Preparing data

# -----------------------------------------------------------------------------------------------------------------------
loan_data = pd.read_csv("./call_loan.csv")
loan_data.drop(['person_id_random'], axis=1, inplace=True)
list_of_attributes = loan_data.columns.values.tolist()[:-1]

# change the pandas dataframe to numpy array
all_inputs = loan_data.drop(['paid_first_loan'], axis=1).values
all_classes = loan_data.paid_first_loan.values

# Impute the missing values to the median value
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
all_inputs = imp.fit_transform(all_inputs)

# -----------------------------------------------------------------------------------------------------------------------

# Random forest-Randomized search

# -----------------------------------------------------------------------------------------------------------------------
random_forest_classifier = RandomForestClassifier()

parameters_rf = {'n_estimators': [5, 10, 25, 50, 70, 100],
                 'min_samples_split': range(2, 20),
                 'criterion': ['gini', 'entropy'],
                 'bootstrap': [True, False],
                 'warm_start': [True, False]}

n_iter_search = 50
random_search_rf = RandomizedSearchCV(random_forest_classifier,
                                      param_distributions=parameters_rf,
                                      scoring='roc_auc',
                                      cv=10)

start = time()
random_search_rf.fit(all_inputs, all_classes)
print("RandomizedSearchCV for random forest took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print('Best roc_auc score of the RandomForest model using Randomized search is: {}'.format(random_search_rf.best_score_))
print('Best parameters of the RandomForest model using Randomized search are: {}'.format(random_search_rf.best_params_))

# take the best classifier from the search and use that
random_forest_classifier = random_search_rf.best_estimator_

# Plot cross validated performance of the finally selected RF model
rf_scores = cross_val_score(random_forest_classifier, all_inputs, all_classes, cv=10, scoring='roc_auc')
print('The mean CV AUC score is %.2f for the RandomForest model'% (rf_scores.mean()))

sb.boxplot(rf_scores)
sb.stripplot(rf_scores, jitter=True, color='green').set_title(
    'Cross-validated AUC score of the Random forest model')
plt.savefig("./figures/classifiers/CV_AUC_score_Randomforest.pdf", dpi=150)
plt.clf()

# Finding the most important features from Randomforest model


# Print the feature ranking
print('\n')
print('Feature ranking by index:')
for f in range(all_inputs.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print('\n')
print('Feature ranking by name:')
for f in range(all_inputs.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list_of_attributes[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances by random forest classifier")
plt.bar(range(all_inputs.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(all_inputs.shape[1]), indices)
plt.xlim([-1, all_inputs.shape[1]])
plt.savefig("./figures/classifiers/feature_rankings_randomforest.pdf", dpi=150)
plt.clf()
# -----------------------------------------------------------------------------------------------------------------------

# Gradient Boosting-Randomized search

# -----------------------------------------------------------------------------------------------------------------------
gradient_boosting_classifier = GradientBoostingClassifier()

parameters_grd = {'n_estimators': [100, 150, 200],
                  'loss': ['deviance', 'exponential'],
                  'learning_rate': np.arange(0.01, 0.1, 0.03),
                  'max_depth': [2, 4, 6, 8],
                  'subsample': np.arange(0.3, 1, 0.3)}

n_iter_search = 50
random_search_grd = RandomizedSearchCV(gradient_boosting_classifier,
                                       param_distributions=parameters_grd,
                                       scoring='roc_auc',
                                       cv=10)

start = time()
random_search_grd.fit(all_inputs, all_classes)
print('\n')
print("RandomizedSearchCV for gradient boosting took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
print('Best roc_auc score of the Gradient Boosting model using Randomized search is: {}'.format(
      random_search_grd.best_score_))
print('Best parameters of the Gradient Boosting model using Randomized search are: {}'.format(
      random_search_grd.best_params_))

# take the best classifier from the search and use that
gradient_boosting_classifier = random_search_grd.best_estimator_

# Plot cross validated performance of the finally selected GB model
grd_scores = cross_val_score(gradient_boosting_classifier, all_inputs, all_classes, cv=10, scoring='roc_auc')
print('The mean CV AUC score is %.2f for the Gradient Boosting model' % (grd_scores.mean()))


sb.boxplot(grd_scores)
sb.stripplot(grd_scores, jitter=True, color='green').set_title(
            'Cross-validated AUC score of the gradient boosting model')
plt.savefig("./figures/classifiers/CV_AUC_score_gradient_boosting.pdf", dpi=150)
plt.clf()


# Finding the most important features from Gradient boosting model
importances = gradient_boosting_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print('\n')
print('Feature ranking by index:')
for f in range(all_inputs.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print('\n')
print('Feature ranking by name:')
for f in range(all_inputs.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, list_of_attributes[indices[f]], importances[indices[f]]))


# Plot the feature importances of the gradient boosting
plt.figure()
plt.title("Feature importances by gradient boosting classifier")
plt.bar(range(all_inputs.shape[1]), importances[indices],
        color="r", align="center")
plt.xticks(range(all_inputs.shape[1]), indices)
plt.xlim([-1, all_inputs.shape[1]])
plt.savefig("./figures/classifiers/feature_rankings_gradient_boosting.pdf", dpi=150)
plt.clf()
# ----------------------------------------------------------------------------------------------------------------------

# ROC curve with cross-validation for Random forest model

# ----------------------------------------------------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(all_inputs, all_classes):
    probas_ = random_forest_classifier.fit(all_inputs[train], all_classes[train]).predict_proba(all_inputs[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(all_classes[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr_rf = np.mean(tprs, axis=0)
mean_tpr_rf[-1] = 1.0
mean_auc_rf = auc(mean_fpr, mean_tpr_rf)
std_auc_rf = np.std(aucs)
plt.plot(mean_fpr, mean_tpr_rf, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_rf, std_auc_rf),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr_rf + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_rf - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve-Random forest classifier')
plt.legend(loc="lower right")
plt.savefig("./figures/classifiers/RF_ROC_curve.pdf", dpi=150)
plt.clf()

# ----------------------------------------------------------------------------------------------------------------------

# ROC curve with cross-validation for gradient boosting model

# ----------------------------------------------------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(all_inputs, all_classes):
    probas_ = gradient_boosting_classifier.fit(all_inputs[train], all_classes[train]).predict_proba(all_inputs[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(all_classes[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr_gb = np.mean(tprs, axis=0)
mean_tpr_gb[-1] = 1.0
mean_auc_gb = auc(mean_fpr, mean_tpr_gb)
std_auc_gb = np.std(aucs)
plt.plot(mean_fpr, mean_tpr_gb, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_gb, std_auc_gb),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr_gb + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr_gb - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve-Gradient boosting decision tree classifier')
plt.legend(loc="lower right")
plt.savefig("./figures/classifiers/GBDT_ROC_curve.pdf", dpi=150)
plt.clf()

# ----------------------------------------------------------------------------------------------------------------------

# ROC curve for mean AUC gradient boosting model

# ----------------------------------------------------------------------------------------------------------------------
mean_fpr = np.linspace(0, 1, 100)

plt.plot(mean_fpr, mean_tpr_rf, color='b',
         label=r'Random forest mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_rf, std_auc_rf),
         lw=2, alpha=.8)

plt.plot(mean_fpr, mean_tpr_gb, color='g',
         label=r'Gradient boosting mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_gb, std_auc_gb),
         lw=2, alpha=.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random forest vs. Gradient boosting classifier')
plt.legend(loc="lower right")
plt.savefig("./figures/classifiers/GB_RF_ROC_curve.pdf", dpi=150)
plt.clf()