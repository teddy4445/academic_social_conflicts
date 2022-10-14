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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

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
        # Hypothesis.general_stats(df=df)
        # [Hypothesis.test_model(df=df, y_col=col) for col in list(df)]
        # Hypothesis.conflict_minimal(df=df)
        # Hypothesis.working_in_academic_influence(df=df)
        Hypothesis.reduced_field_on_conflicts(df=df)
        Hypothesis.reduced_countries(df=df)
        # Hypothesis.linear_socio_researcher_on_conflicts(df=df)
        # Hypothesis.tree_socio_researcher_on_conflicts(df=df)

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
            for name in "add_peer,cite_reviewer,cite_friends,peer_conflict_raise,peer_conflict_delay,self_demand".split(
                    ","):
                plt.bar(df[source_col].unique(),
                        [(df[df[source_col] == val][name] != 0).sum() / (df[df[source_col] == val][name] != 0).count()
                         for val in df[source_col].unique()],
                        width=0.8,
                        color="black")
                plt.xlabel(source_col)
                plt.ylabel(name)
                plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO,
                                         "{}_dist_{}.png".format(source_col,
                                                                 name)),
                            dpi=400)
                plt.close()

        plt.bar(df["master_advisors"].unique(),
                [(df[df["master_advisors"] == val]["master_papers"]).sum() / (
                    df[df["master_advisors"] == val]["master_papers"]).count() for val in
                 df["master_advisors"].unique()],
                width=0.8,
                color="black")
        plt.xlabel("Number of Master advisors")
        plt.ylabel("Papers on average")
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "master_paper_on_average.png"),
                    dpi=400)
        plt.close()

        plt.bar(df["phd_advisors"].unique(),
                [(df[df["phd_advisors"] == val]["phd_papers"]).sum() / (
                    df[df["phd_advisors"] == val]["phd_papers"]).count() for val in df["phd_advisors"].unique()],
                width=0.8,
                color="black")
        plt.xlabel("Number of Ph.D. advisors")
        plt.ylabel("Papers on average")
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "phd_paper_on_average.png"),
                    dpi=400)
        plt.close()

    @staticmethod
    def reduced_field_on_conflicts(df: pd.DataFrame):

        plt.bar(df["major_field"].unique(),
                [len(df[df["major_field"] == val]) for val in df["major_field"].unique()],
                width=0.8,
                color="black")
        plt.xlabel("Field", fontsize=16, weight="bold")
        plt.ylabel("Count", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "field_counts.png"), dpi=400)
        plt.close()

        ticks = ["Exact", "Social", "Nature", "Eng.", "Medicine"]
        for col in ["master_conflict", "phd_conflict", "add_peer", "cite_reviewer",
                    "cite_friends", "peer_conflict_raise", "peer_conflict_delay"]:

            plt.errorbar(df["major_field"].unique(),
                         [(df[df["major_field"] == val][col]).mean() for val in df["major_field"].unique()],
                         [(df[df["major_field"] == val][col]).std() for val in df["major_field"].unique()],
                         fmt="o",
                         capsize=3,
                         ecolor="black",
                         color="black")

            plt.xticks(range(len(ticks)), ticks)
            plt.xlabel("Field")
            plt.ylabel("Conflicts")
            plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, LIZA_HYPO, "field_{}.png".format(col)),
                        dpi=400)
            plt.close()

    @staticmethod
    def reduced_countries(df: pd.DataFrame):

        location_counts = [len(df[df["reduced_location"] == val]) for val in df["reduced_location"].unique()]
        plt.bar(df["reduced_location"].unique(),
                location_counts,
                width=0.8,
                color="black")
        plt.xlabel("Location", fontsize=16, weight="bold")
        plt.ylabel("Count", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, ARIEL_HYPO, "reduced_location_counts.png"),
                    dpi=400)
        plt.close()
        print(" and ".join(["{}:{}".format(val, location_counts[i]) for i, val in enumerate(df["reduced_location"].unique())]))

        ticks = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
        for col in ["master_conflict", "phd_conflict", "add_peer", "cite_reviewer",
                    "cite_friends", "peer_conflict_raise", "peer_conflict_delay"]:
            plt.errorbar(df["reduced_location"].unique(),
                         [(df[df["reduced_location"] == val][col]).mean() for val in df["reduced_location"].unique()],
                         [(df[df["reduced_location"] == val][col]).std() for val in df["reduced_location"].unique()],
                         fmt="o",
                         capsize=3,
                         ecolor="black",
                         color="black")
            plt.xticks(range(len(ticks)), ticks, rotation=45)
            plt.xlabel("Field", fontsize=16, weight="bold")
            plt.ylabel(col.replace("_", " "), fontsize=16, weight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, ARIEL_HYPO,
                                     "reduced_location_{}.png".format(col)), dpi=400)
            plt.close()

    @staticmethod
    def tree_socio_researcher_on_conflicts(df: pd.DataFrame):
        x_cols = "gender,age,degree,field,only_academia,academic_age,country,project_count,coauthorship_papers,solo_papers".split(
            ",")
        with open(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, ARIEL_HYPO,
                               "tree_socio_researcher_on_conflicts.csv"), "w") as ans_file:
            for y_col in ["master_conflict", "phd_conflict", "add_peer", "cite_reviewer",
                          "cite_friends", "peer_conflict_raise", "peer_conflict_delay"]:
                # print pairplot
                rf = RandomForestClassifier(max_depth=5)
                x = df[x_cols]
                y = df[y_col]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=73)
                rf.fit(x_train, y_train)
                y_train_pred = rf.predict(x_train)
                train_r2 = r2_score(y_true=y_train, y_pred=y_train_pred)
                train_mae = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
                train_mse = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
                y_test_pred = rf.predict(x_test)
                test_r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
                test_mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
                test_mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
                ans_file.write("col={}, train_r2={:.4f}, train_mae={:.4f}, train_mse={:.4f}"
                               ", test_r2={:.4f}, test_mae={:.4f}, test_mse={:.4f}\n".format(y_col,
                                                                                             train_r2,
                                                                                             train_mae,
                                                                                             train_mse,
                                                                                             test_r2,
                                                                                             test_mae,
                                                                                             test_mse))

    @staticmethod
    def linear_socio_researcher_on_conflicts(df: pd.DataFrame):
        x_cols = "gender,age,degree,field,only_academia,academic_age,country,project_count,coauthorship_papers,solo_papers".split(
            ",")
        with open(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, ARIEL_HYPO,
                               "linear_socio_researcher_on_conflicts.csv"), "w") as ans_file:
            for y_col in ["master_conflict", "phd_conflict", "add_peer", "cite_reviewer",
                          "cite_friends", "peer_conflict_raise", "peer_conflict_delay"]:
                # print pairplot
                lg = LinearRegression()
                x = df[x_cols]
                y = df[y_col]
                lg.fit(x, y)
                y_pred = lg.predict(x)
                r2 = r2_score(y_true=y, y_pred=y_pred)
                mae = mean_absolute_error(y_true=y, y_pred=y_pred)
                mse = mean_squared_error(y_true=y, y_pred=y_pred)
                eq_text = ["{:.2f} * {}".format(lg.coef_[index], col_name) for index, col_name in enumerate(list(x))]
                ans_file.write("Eq={} + {}, col={}, r2={:.4f}, mae={:.4f}, mse={:.4f}\n".format(" + ".join(eq_text),
                                                                                                lg.intercept_,
                                                                                                y_col,
                                                                                                r2,
                                                                                                mae,
                                                                                                mse))

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
            with open(os.path.join(os.path.dirname(__file__), RESULTS_FOLDER, DT_FOLDER, "dt_{}.txt".format(y_col)),
                      "w") as fout:
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
