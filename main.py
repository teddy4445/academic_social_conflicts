# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# project imports
from consts import *
from hypothesis import Hypothesis
from compare_with_record import CompareWithReality


class Main:
    """
    Main class of the project, initial
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        Main.os_prepare()
        df = Main.load_data()
        CompareWithReality.run(df=df)
        Hypothesis.all_tests(df=df)

    @staticmethod
    def os_prepare():
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER))
        except:
            pass
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, FI_FOLDER))
        except:
            pass
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, DT_FOLDER))
        except:
            pass

    @staticmethod
    def load_data():
        if os.path.exists(DF_READY):
            return pd.read_csv(DF_READY)
        df = pd.read_csv("Research disagreements questionnaire  (Responses) - Form responses 1.csv")
        df = Main.fix_names(df=df)
        return df

    @staticmethod
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

        df["gender"] = df["gender"].replace({"Male": 0,
                                             "Female": 1})
        df["age"] = df["age"].replace({"26-35": 31,
                                       "18-25": 21,
                                       "46-55": 51,
                                       "36-45": 41,
                                       "56-65": 61,
                                       "66+": 71})
        df["degree"] = df["degree"].replace({"First (Bachelor)": 1,
                                             "Second (Master)": 2,
                                             "Third (Ph.D. \\ MD)": 3,
                                             "Professor": 4})
        df["field"] = df["field"].replace({})  # TODO: later
        df["only_academia"] = df["only_academia"].replace({"No": 0, "Yes": 1})
        df["academic_age"] = df["academic_age"].replace({"3-5": 4, "1-2": 1, "0": 0, "10-20": 15, "6-9": 7, "20+": 20})
        df["country"] = df["country"].replace() # TODO: later
        df["project_count"] = df["project_count"].replace({"0": 0,
                                                           "1": 1,
                                                           "2": 2,
                                                           "3": 3,
                                                           "4": 4,
                                                           "5+": 5})
        df["coauthorship_papers"] = df["coauthorship_papers"].replace()  # TODO: later
        df["solo_papers"] = df["solo_papers"].replace()  # TODO: later
        df["master_advisors"] = df["master_advisors"].replace()  # TODO: later
        df["phd_advisors"] = df["phd_advisors"].replace()  # TODO: later
        df["phd_papers"] = df["phd_papers"].replace()  # TODO: later
        df["master_advisor_age"] = df["master_advisor_age"].replace()  # TODO: later
        df["master_advisor_title"] = df["master_advisor_title"].replace()  # TODO: later
        df["master_advisor_gender"] = df["master_advisor_gender"].replace()  # TODO: later
        df["master_conflict"] = df["master_conflict"].replace()  # TODO: later TODO: later
        df["phd_advisor_age"] = df["phd_advisor_age"].replace()  # TODO: later TODO: later
        df["phd_advisor_gender"] = df["phd_advisor_gender"].replace()  # TODO: later later
        df["phd_conflict"] = df["phd_conflict"].replace()  # TODO: later TODO: later later
        df["phd_advisor_title"] = df["phd_advisor_title"].replace()  # TODO: laterer later
        df["peer_conflict"] = df["peer_conflict"].replace()  # TODO: laterO: laterer later
        df["add_peer"] = df["add_peer"].replace({"Yes": 1, "No": 0})
        df["cite_reviewer"] = df["cite_reviewer"].replace()  # TODO: laterO: laterer later
        df["cite_friends"] = df["cite_friends"].replace()  # TODO: laterO: laterer laterer
        df["peer_conflict"] = df["peer_conflict"].replace()  # TODO: laterO: laterer later
        df["peer_conflict_delay"] = df["peer_conflict_delay"].replace()  # TODO: laterO: laterer later
        df["second_field"] = df["second_field"].replace()  # TODO: laterO: laterer later

        df.dropna(inplace=True)
        df.to_csv(DF_READY, index=False)
        return df


if __name__ == '__main__':
    Main.run()
