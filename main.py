# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
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
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO))
        except:
            pass

    @staticmethod
    def load_data():
        if os.path.exists(DF_READY):
            return pd.read_csv(DF_READY)
        df = pd.read_csv("data.csv")
        df = Main.fix_names(df=df)
        return df

    @staticmethod
    def fix_names(df: pd.DataFrame):
        try:
            df.drop(["Timestamp", "Email address"], axis=1, inplace=True)
        except Exception as error:
            pass

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
            "28) How often a co-author raise demands or claims to get more credit in a paper, usually with a treat of delaying the paper's submission/publication?": "peer_conflict_raise",
            "29)  How many times do you update a papers' authors list based on co-authors' demands and NOT BASED ON CHANGES IN THE CONTRIBUTION of the authors?": "peer_conflict_delay",
            "30) How often do you have to demand to get more credit for you work in a research?": "self_demand"
        }
        df = df.rename(columns=mapper)

        df["gender"] = df["gender"].replace({"Male": 0,
                                             "Female": 1,
                                             "Prefer not to say": 0})
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
        df["only_academia"] = df["only_academia"].replace({"No": 0,
                                                           "Yes": 1})
        df["only_academia"] = [0 if val not in [0, 1] else val for val in df["only_academia"]]
        df["academic_age"] = df["academic_age"].replace({"3-5": 4,
                                                         "1-2": 1,
                                                         "0": 0,
                                                         "10-20": 15,
                                                         "6-9": 7,
                                                         "20+": 20})
        df["project_count"] = df["project_count"].replace({"0": 0,
                                                           "1": 1,
                                                           "2": 2,
                                                           "3": 3,
                                                           "4": 4,
                                                           "5+": 5})
        df["coauthorship_papers"] = df["coauthorship_papers"].replace({"0": 0,
                                                                       "1": 1,
                                                                       "2-5": 3,
                                                                       "5-10": 7,
                                                                       "10+": 10})
        df["solo_papers"] = df["solo_papers"].replace({"0": 0,
                                                       "1": 1,
                                                       "2-5": 3,
                                                       "5-10": 7,
                                                       "10+": 10})
        df["master_advisors"] = df["master_advisors"].replace({"1": 1,
                                                               "2": 2,
                                                               "3+": 3,
                                                               "I did a direct Ph.D.": 0})
        df["master_advisors"] = [val if isinstance(val, int) else 0 for val in df["master_advisors"]]
        df["phd_advisors"] = df["phd_advisors"].replace({"1": 1,
                                                         "2": 2,
                                                         "3+": 3})
        df["master_papers"] = df["master_papers"].replace({"0": 0, "1": 1, "2": 2, "3+": 3})
        df["phd_papers"] = df["phd_papers"].replace({"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5+": 5})
        df["master_advisor_age"] = df["master_advisor_age"].replace({"Younger than me": -1,
                                                                     "0-5": 3,
                                                                     "5-10": 7,
                                                                     "10-20": 15,
                                                                     "20-40": 30,
                                                                     "40+": 40})
        df["master_advisor_title"] = df["master_advisor_title"].replace({"Dr.": 0, "Prof.": 1})
        df["master_advisor_gender"] = df["master_advisor_gender"].replace({"Male": 1,
                                                                           "Female": 0})
        df["master_conflict"] = df["master_conflict"].replace({"Yes": 1,
                                                               "No": 0})
        df["phd_advisor_age"] = df["phd_advisor_age"].replace({"Younger than me": -1,
                                                               "0-5": 3,
                                                               "5-10": 7,
                                                               "10-20": 15,
                                                               "20-40": 30,
                                                               "40+": 40})
        df["phd_advisor_gender"] = df["phd_advisor_gender"].replace({"Male": 1,
                                                                     "Female": 0})
        df["phd_conflict"] = df["phd_conflict"].replace({"Yes": 1,
                                                         "No": 0})
        df["phd_conflict"] = [0 if val not in [0, 1] else val for val in df["phd_conflict"]]

        df["phd_advisor_title"] = df["phd_advisor_title"].replace({"Dr.": 0, "Prof.": 1})
        df["peer_conflict"] = [val.split(",")[0] if "," in val else val for val in df["peer_conflict"]]
        df["peer_conflict"] = df["peer_conflict"].replace({"No": 0,
                                                           "Yes - They were from the same *field of study* as I am": 1,
                                                           "Yes - They were from the same *institute* as I am": 2,
                                                           "Yes - They were from the same *country* as I am": 3,
                                                           "Yes - They were from the same *gender* as I am": 4})
        df["add_peer"] = df["add_peer"].replace({"Yes": 1, "No": 0})
        df["cite_reviewer"] = df["cite_reviewer"].replace({"Only when absolutely relevant and warranted": 0,
                                                           "When relevant but not necessarily \"a must\"": 1,
                                                           "When in the general scope of the work": 2,
                                                           "Whenever I can": 3})
        df["cite_friends"] = df["cite_friends"].replace({"Only when absolutely relevant and warranted": 0,
                                                         "When relevant but not necessarily \"a must\"": 1,
                                                         "When in the general scope of the work": 2,
                                                         "Whenever I can": 3})
        df["peer_conflict_raise"] = df["peer_conflict_raise"].replace({"All the time": 3,
                                                                       "Often": 2,
                                                                       "Rarely": 1,
                                                                       "Never": 0})
        df["peer_conflict_delay"] = df["peer_conflict_delay"].replace({"All the time": 3,
                                                                       "Often": 2,
                                                                       "Rarely": 1,
                                                                       "Never": 0})
        df["second_field"] = df["second_field"].replace({"All the time": 3,
                                                         "Often": 2,
                                                         "Rarely": 1,
                                                         "Never": 0})
        df["self_demand"] = df["self_demand"].replace({"All the time": 3,
                                                       "Often": 2,
                                                       "Rarely": 1,
                                                       "Never": 0})

        df["major_field"] = Main.reduce_field(list(df["field"]))

        df["field"] = OrdinalEncoder().fit_transform(df["field"].values.reshape(-1, 1))
        df["country"] = OrdinalEncoder().fit_transform(df["country"].values.reshape(-1, 1))
        df.dropna(axis=1,
                  how='all',
                  inplace=True)
        df.drop(["second_field"], axis=1, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)
        df.to_csv(DF_READY, index=False)
        return df

    @staticmethod
    def reduce_field(data: list):
        all_options = set(data)
        print(all_options)
        # 0 - exact, 1 - social, 2 - nature, 3 - eng, 4 - other
        mapper = {
            'Neuroscience': 0,
            'Politics': 1,
            'Applied physics': 0,
            'History': 1,
            'Mechanical engineering': 3,
            'Process/Chemical engineering': 3,
            'Oncology': 4,
            'Environmental science': 2,
            'Nano/Micro science': 0,
            'Social sciences': 1,
            'Management': 1,
            'Sociology': 1,
            'Civil engineering': 3,
            'Philosophy': 2,
            'Basic biology': 0,
            'Plant production and environmental agriculture': 3,
            'Law': 1,
            'Agricultural science in society and economy': 2,
            'Computer Science (CS)': 0,
            'Chemistry': 0,
            'Clinical internal medicine': 4,
            'Linguistics': 1,
            'Nursing': 4,
            'Genome science': 2,
            'Clinical surgery': 4,
            'Psychology': 1,
            'Physics': 0,
            'Electrical and electronic engineering': 3,
            'Basic medicine': 4,
            'Mathematics': 0,
            'Boundary medicine': 4,
            'Dentistry': 4,
            'Biological Science': 2,
            'Literature': 1,
            'Material engineering': 3,
            'Anthropology': 1,
            'Architecture and building engineering': 3,
            'Economics': 1,
            'Animal life science': 4,
            'Pharmacy': 4,
            'Informatics': 0
        }
        return [mapper[val] for val in data]


if __name__ == '__main__':
    Main.run()
