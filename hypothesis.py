# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# project imports
from consts import *


class Hypothesis:
    """
    Test several hypothesis for the research
    """

    def __init__(self):
        pass

    @staticmethod
    def all_tests(df: pd.DataFrame):
        #Hypothesis.general_stats(df=df)
        #[Hypothesis.test_model(df=df, y_col=col) for col in list(df)]
        #Hypothesis.conflict_minimal(df=df)
        Hypothesis.working_in_academic_influence(df=df)
        Hypothesis.reduced_field_on_conflicts(df=df)

    @staticmethod
    def conflict_minimal(df: pd.DataFrame):
        # just for view
        only_conflict = df[df["master_conflict"] == 1]

        new_df = df[["gender", "country", "field", "master_advisors", "master_conflict", "master_advisor_gender"]]
        new_df["same_gender"] = new_df["gender"] == new_df["master_advisor_gender"]
        new_df.drop(["master_advisor_gender"], axis=1, inplace=True)
        Hypothesis.test_model(df=new_df,
                              y_col="master_conflict",
                              prefix_same="conflict_minimal_")

        new_df = df[["gender", "country", "field", "phd_advisors", "phd_conflict", "phd_advisor_gender"]]
        new_df["same_gender"] = new_df["gender"] == new_df["phd_advisor_gender"]
        new_df.drop(["phd_advisor_gender"], axis=1, inplace=True)
        Hypothesis.test_model(df=new_df,
                              y_col="phd_conflict",
                              prefix_same="conflict_minimal_")

    @staticmethod
    def working_in_academic_influence(df: pd.DataFrame):
        for source_col in "academic_age,only_academia".split(","):
            for name in "add_peer,cite_reviewer,cite_friends,peer_conflict_raise,peer_conflict_delay,self_demand".split(","):
                plt.bar(df[source_col].unique(),
                        [(df[df[source_col] == val][name] != 0).sum() / (df[df[source_col] == val][name] != 0).count() for val in df[source_col].unique()],
                        width=0.8,
                        color="black")
                plt.xlabel(source_col)
                plt.ylabel(name)
                plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "{}_dist_{}.png".format(source_col,
                                                                                                                       name)),
                            dpi=400)
                plt.close()

        plt.bar(df["master_advisors"].unique(),
                [(df[df["master_advisors"] == val]["master_papers"]).sum() / (df[df["master_advisors"] == val]["master_papers"]).count() for val in df["master_advisors"].unique()],
                width=0.8,
                color="black")
        plt.xlabel("Number of Master advisors")
        plt.ylabel("Papers on average")
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "master_paper_on_average.png"), dpi=400)
        plt.close()

        plt.bar(df["phd_advisors"].unique(),
                [(df[df["phd_advisors"] == val]["phd_papers"]).sum() / (df[df["phd_advisors"] == val]["phd_papers"]).count() for val in df["phd_advisors"].unique()],
                width=0.8,
                color="black")
        plt.xlabel("Number of Ph.D. advisors")
        plt.ylabel("Papers on average")
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "phd_paper_on_average.png"), dpi=400)
        plt.close()

    @staticmethod
    def reduced_field_on_conflicts(df: pd.DataFrame):
        ticks = ["Exact", "Social", "Nature", "Eng.", "Medicine"]
        for col in ["master_conflict", "phd_conflict"]:
            plt.bar(df["major_field"].unique(),
                    [(df[df["major_field"] == val][col]).sum() / (df[df["major_field"] == val][col]).count() for val in df["major_field"].unique()],
                    width=0.8,
                    color="black")
            plt.xticks(range(len(ticks)), ticks)
            plt.xlabel("Field")
            plt.ylabel("Conflicts")
            plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "field_{}.png".format(col)), dpi=400)
            plt.close()

    @staticmethod
    def test_model(df: pd.DataFrame,
                   y_col: str,
                   prefix_same: str = ""):
        try:
            x_train, x_test, y_train, y_test = train_test_split(df.drop([y_col], axis=1),
                                                                df[y_col],
                                                                test_size=0.2,
                                                                random_state=SEED)
            model = GridSearchCV(DecisionTreeClassifier(),
                                 {
                                     "max_depth": [3, 6],
                                     "criterion": ["gini", "entropy"],
                                     "ccp_alpha": [0, 0.01, 0.05]
                                 },
                                 verbose=2)
            model.fit(x_train,
                      y_train)
            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)
            print("RF: y={} | train = {}, test = {}".format(y_col,
                                                            train_score,
                                                            test_score))
            forest_importances = pd.Series(model.best_estimator_.feature_importances_, index=list(x_train))
            fig, ax = plt.subplots()
            forest_importances.plot.bar(ax=ax)
            ax.set_title("{} - Tr={:.3f},Ts={:.3f}".format(y_col, train_score, test_score))
            # ax.set_xlabel("Feature")
            ax.set_ylabel("Feature importance")
            fig.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, FI_FOLDER,
                                     "feature_important_{}{}.png".format(prefix_same, y_col)),
                        dpi=400)
            plt.close()

            # save tree structure
            text_representation = tree.export_text(model.best_estimator_)
            for index, col in enumerate(list(df)):
                text_representation = text_representation.replace("feature_{}".format(index),
                                                                  col)
            with open(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, DT_FOLDER, "dt_{}.txt".format(y_col)), "w") as fout:
                fout.write(text_representation)

        except Exception as error:
            print("Cannot compute '{}' due to {}".format(y_col, error))

    @staticmethod
    def general_stats(df: pd.DataFrame):
        for method in ["pearson", "spearman"]:
            corr_metrix = df.corr(method=method)
            sn.heatmap(corr_metrix,
                       vmax=1,
                       vmin=-1,
                       annot=False)
            plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, "{}_heatmap.png".format(method)),
                        dpi=500)
            plt.close()
            sn.heatmap(corr_metrix.replace(1, 0),
                       annot=False)
            plt.savefig(
                os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, "{}_heatmap_zoomin.png".format(method)),
                dpi=500)
            plt.close()
        """
        # pair plot
        sn.pairplot(df)
        plt.savefig(os.path.join(os.path.dirname(__file__), Main.RESULTS_FOLDER, "pairplot.png"),
                    dpi=500)
        plt.close()
        """
