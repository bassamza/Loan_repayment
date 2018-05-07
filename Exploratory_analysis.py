"""
Run this script second

This script does exploratory analysis on the features that will be used for building the classifier models(engineered
features). The monotonicity relation between attributes and the target variable (loan_repayment) is explored by plotting
the probability of repayment versus each feature. Since we are planning to use ensembles of decision tree
models, we do not need to normalize data or make any log transformations to make the distributions Gaussian like.

__author__ = "Nastaran Bassamzadeh"
__email__ = "nsbassamzadeh@gmail.com"

"""
import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

sb.set(color_codes=True)

loan_data = pd.read_csv("./call_loan.csv")
loan_data.drop(['person_id_random'], axis=1, inplace=True)
loan_data = loan_data.dropna()

# -----------------------------------------------------------------------------------------------------------------------

# Plotting distribution of the class variable

# -----------------------------------------------------------------------------------------------------------------------
ax = sb.countplot(loan_data.paid_first_loan)
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height() * 1.01))

# Create the directory 'figures' if it is does not exist
if not os.path.isdir("./figures/basic_plots"):
    os.makedirs("./figures/basic_plots")


plt.savefig("./figures/basic_plots/Countplot_loan_repayment'.pdf", dpi=150)
plt.clf()
# -----------------------------------------------------------------------------------------------------------------------

# Plotting distribuion of each attribute

# -----------------------------------------------------------------------------------------------------------------------
for column_index, column in enumerate(loan_data.columns):
    if column == 'paid_first_loan':
        continue
    plt.subplot(3, 6, column_index + 1)
    b=sb.distplot(loan_data[column])
    b.set_xlabel(xlabel=column,fontsize=5)
    b.set(yticks=[])
    b.set(xticks=[])
plt.savefig("./figures/basic_plots/Input_features_dist.pdf", dpi=150)
plt.clf()
# -----------------------------------------------------------------------------------------------------------------------

# Figure out the monotonicity relation between attributes and the target variable

# -----------------------------------------------------------------------------------------------------------------------
if not os.path.isdir("./figures/probability_of_repayment"):
    os.makedirs("./figures/probability_of_repayment")

data = loan_data.copy()
for column_index, column in enumerate(data.columns):
    if column == 'paid_first_loan':
        continue
    data[column] = pd.qcut(data[column], 10, duplicates='drop')
    g = sb.factorplot(x=column, y="paid_first_loan", data=data, size=6, kind="bar", palette="muted")
    g.set_ylabels("Repayment probability")
    plt.savefig("./figures/probability_of_repayment/Probability_loan_repayment vs."+column+".pdf", dpi=150)
    plt.clf()
# -----------------------------------------------------------------------------------------------------------------------

# Plotting boxplot of each attribute for each class

# -----------------------------------------------------------------------------------------------------------------------
if not os.path.isdir("./figures/boxplots"):
    os.makedirs("./figures/boxplots")

for column_index, column in enumerate(loan_data.columns):
    if column == 'paid_first_loan':
        continue
    sb.boxplot(x='paid_first_loan', y=column, data=loan_data)
    plt.savefig("./figures/boxplots/Boxplot_" + column + ".pdf", dpi=150)
    plt.clf()


