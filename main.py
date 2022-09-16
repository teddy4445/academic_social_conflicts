import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


# CONSTS #
SEED = 73
DF_READY = "preprocess_dataset.csv"
RESULTS_FOLDER = "results"
# CONSTS #


def os_prepare():
    try:
        os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER))
    except:
        pass


def load_data():
    if os.path.exists(DF_READY):
        return pd.read_csv(DF_READY)
    df = pd.read_csv("Research disagreements questionnaire  (Responses) - Form responses 1.csv")
    df = fix_names(df=df)
    return df


def fix_names(df: pd.DataFrame):
    df.drop(["Timestamp", "Email address"], axis=1, inplace=True)
    mapper = {
        "1) Gender": "gender",
        "2) Age": "age",
        "3) Academic degree": "degree",
        "4) Main field of research": "field",
        "5) Second field of research (if any)": "second_field",
        "6) Do you work\study only in the academia? ": "only_academia",
        "7) How many years have passed since your first academic publication (i.e., academic age)": "academic_age",
        "8) Current affiliation's country ": "country",
        "9) Based on the last 5 years, on average, how many research projects do you involved in, at the same time? ": "project_count",
        "10) How many academic papers did you publish with one or more co-author(s) other than your advisor(s)? ": "coauthorship_papers",
        "11) How many solo (i.e., you are the only author) papers did you publish? ": "solo_papers",
        "12) How many advisors did you have during your master's studies?": "master_advisors",
        "13) How many papers you published during your Masters with your advisor(s)? ": "master_papers",
        "14) How many advisors did you have during your Ph.D. studies?": "phd_advisors",
        "15) How many papers you published during your Ph.D. studies with your advisor(s)? ": "phd_papers",
        "16) How many years your (main) master's advisor is/was older than you?": "master_advisor_age",
        "17) What is/was your master's advisor academic title?": "master_advisor_title",
        "18) What is/was your (main) master's advisor's gender?": "master_advisor_gender",
        "19) Did you have a conflict about credit distribution in a paper with your Master's advisor(s)?": "master_conflict",
        "20) How many years your (main) Ph.D.'s  (or MD)   advisor is/was older than you?": "phd_advisor_age",
        "21) What is/was your (main) Ph.D. (or MD) advisor's gender?": "phd_advisor_gender",
        "22) Did you have a conflict about credit distribution in a paper with your Ph.D.  (or MD) advisor(s)?": "phd_conflict",
        "23) What is/was your Ph.D.'s (or MD) advisor academic title?": "phd_advisor_title",
        "24) Did you have a conflict about credit distribution in a paper with your peers?": "peer_conflict",
        "25) Did you add someone to a paper without him\\her contributing at all (or not enough to be named an author)? ": "add_peer",
        "26) How often do you ask authors to cite you as part of a review process?": "cite_reviewer",
        "27) How often do you cite your colleagues even when the citation is not entirely scientifically warranted?": "cite_friends",
        "28) How often a co-author raise demands or claims to get more credit in a paper, usually with a treat of delaying the paper's submission/publication?": "peer_conflict",
        "29)  How many times do you update a papers' authors list based on co-authors' demands and NOT BASED ON CHANGES IN THE CONTRIBUTION of the authors?": "peer_conflict_delay",
        "30) How often do you have to demand to get more credit for you work in a research?": "self_demand"
    }
    df = df.rename(columns=mapper)
    ord_enc = OrdinalEncoder()
    numeric_columns = df.select_dtypes(include=np.number)
    non_numeric_columns = [val for val in list(df) if val not in numeric_columns]
    for col in non_numeric_columns:
        df[col] = ord_enc.fit_transform(df[[col]])
    df.dropna(inplace=True)
    df.to_csv(DF_READY, index=False)
    return df


def test_model(df: pd.DataFrame,
               y_col: str):
    try:
        x_train, x_test, y_train, y_test = train_test_split(df.drop([y_col], axis=1),
                                                            df[y_col],
                                                            test_size=0.2,
                                                            random_state=SEED)
        model = GridSearchCV(RandomForestClassifier(),
                             {
                                 "max_depth": [3, 6],
                                 #"criterion": ["gini", "entropy"],
                                 "ccp_alpha": [0, 0.01, 0.05]
                             },
                             verbose=2)
        model.fit(x_train,
                  y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        print("y={} | train = {}, test = {}".format(y_col,
                                                    train_score,
                                                    test_score))
        forest_importances = pd.Series(model.best_estimator_.feature_importances_, index=list(x_train))
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax)
        ax.set_title("{} - Tr={:.3f},Ts={:.3f}".format(y_col, train_score, test_score))
        #ax.set_xlabel("Feature")
        ax.set_ylabel("Feature importance")
        fig.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, "feature_important_{}.png".format(y_col)), dpi=400)
        plt.close()
    except Exception as error:
        print("Cannot compute '{}' due to {}".format(y_col, error))


def run():
    os_prepare()
    df = load_data()
    [test_model(df=df, y_col=col) for col in list(df)]


if __name__ == '__main__':
    run()
