__author__ = 'Yijun Lou, Changjie Ma, Xiaokang Feng, Ning Fan, Xiaoman Gong, Ziheng Zhou'
__copyright__ = "Copyright 2019, The Group Project of IAQF"
__credits__ = ["Yijun Lou", "Changjie Ma", "Xiaokang Feng", "Ning Fan", "Xiaoman Gong", "Ziheng Zhou"]
__license__ = "University of Illinois, Urbana Champaign"
__version__ = "1.0.0"

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from copy import deepcopy as dcp
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class CSF:

    def __init__(self):
        '''
        Initialize class with empty result dictionary.
        '''
        self.result = {}
        pass

    def parse_data(self, path = '/Users/lou/PycharmProjects/Credit_Spread_Forecast/output/combined_data.csv', label = 'label_BBB'):
        '''
        Parse data of Moody Credit Score from a certain path.
        :param path: string (default='MLF_GP1_CreditScore.csv'), path of the file
        :return:
        '''
        self.data_set = pd.read_csv(path, index_col=["DATE"], parse_dates=['DATE'])
        self.attr_table = self.data_set.iloc[:, :-4]
        # self.inv_grd = self.data_set.iloc[:, -2]
        self.label = self.data_set[label]

    def describe_data(self):
        print(self.data_set.describe())
        # fig, ax = plt.subplots(figsize=(6, 4))
        # sns.countplot(x='InvGrd', data=self.data_set)
        # plt.title("Count of InvGrd")
        # plt.show()

        n = len(self.attr_table)
        invgrd_0 = len(self.data_set[self.data_set['InvGrd'] == 0])
        invgrd_1 = len(self.data_set[self.data_set['InvGrd'] == 1])

        print("% of instrument labeled 1 in dataset: ", invgrd_1 * 100 / n)
        print("% of instrument labeled 0 in dataset: ", invgrd_0 * 100 / n)

        # cor = self.attr_table.corr(method = "pearson")
        # fig, ax = plt.subplots(figsize=(8, 6))
        # plt.title("Correlation Plot")
        # sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
        #             square=True, ax=ax)
        # plt.tight_layout()
        # plt.show()

        sns.pairplot(self.attr_table)
        plt.show()

    def make_training_test(self, test_size=0.25, random_state=None):
        '''
        Make training and test sets.
        :param test_size: float, int or None, optional (default=0.25), proportion of the dataset to include in the test split.
        :param random_state: int, RandomState instance or None, optional (default=None).
        :return:
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.attr_table, self.label, test_size=test_size, random_state=random_state, stratify=self.label)
        # self.X_train = SelectKBest(chi2, k=13).fit_transform(self.attr_table, self.inv_grd)


    def initialize(self, path = '/Users/lou/PycharmProjects/Credit_Spread_Forecast/output/combined_data.csv', label = "label_BBB", test_size=0.25, random_state=None):
        '''
        Initialize the class. First parse data from certain path and make training and test sets and initialize model settings.
        :param path: string (default='MLF_GP1_CreditScore.csv'), path of the file.
        :param test_size: float, int or None, optional (default=0.25), proportion of the dataset to include in the test split.
        :param random_state: int, RandomState instance or None, optional (default=None).
        :return:
        '''
        self.parse_data(path, label=label)
        self.make_training_test(test_size, random_state)

        # n_components = list(range(1, self.attr_table.shape[1]+1,1))
        n_components = [1,24]
        # C = np.logspace(-4, 4, 50)
        C = np.logspace(-4, 4, 10)
        penalty = ['l1', 'l2']
        kernel = ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]
        kernel = ["linear", "poly", "rbf", "sigmoid"]
        fib = [1,2,3,5,8,13,21,34,55,89]
        # max_depth = list(range(1, len(self.attr_table.shape[1]), 1))
        # no_input_neurals = len(self.data.columns - 2)
        # no_output_neurals = len(self.inv_grd.columns)
        # no_samples_training = len(self.attr_table) * 0.9
        hidden_layer_sizes = [int(a * ((len(self.data_set.columns)-1)**(0.5))) for a in range(1, 11)]
        hidden_layer_sizes = [(36, 28, 14, 7), (36, 18, 9, 3), (36, 24, 12, 6), (36, 20, 8, 4)]

        # TODO: FILL THE DICTIONARY BELOW AND RUN THIS SCRIPT
        # Initialize the process dictionary, pipeline will be generated based on this dictionary.
        self.process_dict = {
            'preprocessing': {
                # 'standard_scaler': StandardScaler(),
                'do_nothing': None,
            },
            'decomposition': {
                # 'pca': PCA(),
                #'lda': LDA(),
                #'kpca': KernelPCA(),
                'do_nothing': None,
            },
            'model': {
                # 'logistic': LogisticRegression(),
                # 'neural_net':  MLPClassifier(),
                 'random_forest': RandomForestClassifier(),
                # 'decision_tree': DecisionTreeClassifier(),
                # 'extra_tree': ExtraTreeClassifier(),
                # 'SVC': SVC(),
                # 'gaussian_nb': GaussianNB(),
                # 'knn': KNeighborsClassifier(),
                # 'ada_boost': AdaBoostClassifier(base_estimator=KNeighborsClassifier(algorithm='auto', n_neighbors=3, p=1)),
                # 'voting_clf': VotingClassifier(estimators=[('ad_boost', AdaBoostClassifier(base_estimator=RandomForestClassifier(criterion='gini', max_depth=8, max_features='auto', n_estimators=100), n_estimators=2000, algorithm="SAMME")), ('p2', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini', max_depth=7, max_features='auto', splitter='best'), n_estimators=2000, algorithm="SAMME")), ("p3", KNeighborsClassifier(algorithm='auto', n_neighbors=3, p=1))])
            }
        }

        # Initialize the parameters dictionary, parameters that pass into models of pipeline will be generated based on this dictionary.
        self.parameters_dict = {
            'pca': {
                'n_components': n_components,
                # 'svd_solver': ['full', 'arpack', 'randomized']
            },
            'lda': {
                'n_components': n_components,
            },
            'kpca': {
                'n_components': n_components,
                'kernel': kernel,
            },
            'logistic': {
                'C': C,
                'penalty': penalty,
            },
            'neural_net': {
                'hidden_layer_sizes': hidden_layer_sizes,
                'solver': ['lbfgs'],#['lbfgs', 'sgd', 'adam'],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],#['tanh'],#['identity', 'logistic', 'tanh', 'relu'],
            },
            'random_forest': {
                'n_estimators': [50,100,200],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [4, 5, 6, 7, 8],
                'criterion': ['gini', 'entropy'],
            },
            'ada_boost':{
                'n_estimators': [10, 50, 100, 500, 1000, 2000],
                'algorithm': ["SAMME"]
            },
            'decision_tree':{
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [4, 5, 6, 7, 8, None],
                'max_features': ['auto', 'sqrt', 'log2'],
            },
            'extra_tree':{
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [4, 5, 6, 7, 8, None],
                'max_features': ['auto', 'sqrt', 'log2'],
            },
            'SVC':{
                'kernel': kernel,
                'gamma': [1,2,3],
            },
            'knn':{
                'n_neighbors': [3,4,5,6,7,8],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1,2],
            },
            'voting_clf':{
                'voting': ['hard','soft'],
                'weights': [[1, 1, 1], [1, 2, 1], [1, 3, 1], [2, 1, 1], [2 , 2 , 1], [3, 1, 1], [3,2,1], [3,3,1], [3,3,2], [2,2,3], [3,2,3]],
            }
        }

    def run(self):
        '''
        Run model in single processor.
        :return:
        '''
        self.make_pipelines()
        self.show_result()

    def run_mp(self, processes=4):
        '''
        Run model in multiple processor.
        :return:
        '''
        self.do_task_mp(processes=processes)
        self.show_result(file_name=self.generate_file_name())

    def generate_file_name(self):
        ret = ""
        for k, v in self.process_dict['model'].items():
            ret += k+"_"
        return ret + ".csv"

    def make_pipelines(self):
        '''
        Core algorithm, coming soon.
        :return:
        '''
        for k1, v1 in self.process_dict['preprocessing'].items():
            steps = []
            parameters = {}
            k1_list = []
            if k1 != 'do_nothing':
                steps.append((k1, v1))
                if k1 in self.parameters_dict:
                    for k, v in self.parameters_dict[k1].items():
                        key = k1 + "__" + k
                        parameters[key] = v
                        k1_list.append(key)
            for k2, v2 in self.process_dict['decomposition'].items():
                k2_list = []
                if k2 != 'do_nothing':
                    steps.append((k2, v2))
                    if k2 in self.parameters_dict:
                        for k, v in self.parameters_dict[k2].items():
                            key = k2 + "__" + k
                            parameters[key] = v
                            k2_list.append(key)
                for k3, v3 in self.process_dict['model'].items():
                    k3_list = []
                    if k3 in self.parameters_dict:
                        for k, v in self.parameters_dict[k3].items():
                            key = k3 + "__" + k
                            parameters[key] = v
                            k3_list.append(key)
                    steps.append((k3, v3))
                    # r_key = ((k1 + "+") if k1 != 'do_nothing' else "") + ((k2 + "+") if k2 != 'do_nothing' else "")  + k3
                    r_key = k1 + "+" + k2 + "+" + k3
                    self.result[r_key] = {}
                    # print("steps", steps)
                    # print("parameters", parameters)
                    self.run_estimator(r_key, Pipeline(steps=dcp(steps)), dcp(parameters))
                    steps.remove((k3, v3))
                    for key in k3_list:
                        del parameters[key]
                if k2 != 'do_nothing':
                    steps.remove((k2, v2))
                    for key in k2_list:
                        del parameters[key]
            if k1 != 'do_nothing':
                steps.remove((k1, v1))
                for key in k1_list:
                    del parameters[key]

    def run_estimator(self, key, pipe, parameters, scoring=None, cv=10):
        '''
        Fit the current pipeline.
        :param key: string, combination of names in a pipeline.
        :param pipe: Pipeline Object, the pipeline that will be fitted.
        :param parameters: dict, parameters of each part of pipeline.
        :param scoring: string, callable, list/tuple, dict or None, default: None.
        :param cv: int, cross-validation generator or an iterable, optional.
        :return:
        '''
        print("Generating estimator %s" % key)
        self.clf = GridSearchCV(estimator=pipe, param_grid=parameters, scoring=scoring, cv=cv)
        print("Fitting model")
        self.clf.fit(self.attr_table, self.inv_grd)
        print("Saving result")
        self.record_model(key, parameters)
        print("-" * 60 + "\n")

    def make_pipelines_mp(self):
        '''
        Core algorithm, coming soon.
        :return:
        '''
        ret_list = []
        for k1, v1 in self.process_dict['preprocessing'].items():
            steps = []
            parameters = {}
            k1_list = []
            if k1 != 'do_nothing':
                steps.append((k1, v1))
                if k1 in self.parameters_dict:
                    for k, v in self.parameters_dict[k1].items():
                        key = k1 + "__" + k
                        parameters[key] = v
                        k1_list.append(key)
            for k2, v2 in self.process_dict['decomposition'].items():
                k2_list = []
                if k2 != 'do_nothing':
                    steps.append((k2, v2))
                    if k2 in self.parameters_dict:
                        for k, v in self.parameters_dict[k2].items():
                            key = k2 + "__" + k
                            parameters[key] = v
                            k2_list.append(key)
                for k3, v3 in self.process_dict['model'].items():
                    k3_list = []
                    if k3 in self.parameters_dict:
                        for k, v in self.parameters_dict[k3].items():
                            key = k3 + "__" + k
                            parameters[key] = v
                            k3_list.append(key)
                    steps.append((k3, v3))
                    # r_key = ((k1 + "+") if k1 != 'do_nothing' else "") + ((k2 + "+") if k2 != 'do_nothing' else "") + k3
                    r_key = k1 + "+" + k2 + "+" + k3
                    self.result[r_key] = {}
                    ret_list.append([r_key, Pipeline(steps=dcp(steps)), dcp(parameters)])
                    steps.remove((k3, v3))
                    for key in k3_list:
                        del parameters[key]
                if k2 != 'do_nothing':
                    steps.remove((k2, v2))
                    for key in k2_list:
                        del parameters[key]
            if k1 != 'do_nothing':
                steps.remove((k1, v1))
                for key in k1_list:
                    del parameters[key]
        return ret_list

    def run_estimator_mp(self, iter_list, scoring='accuracy', cv=5, n_jobs=1):
        '''
        TODO
        :param iter_list:
        :param scoring:
        :param cv:
        :return:
        '''
        key = iter_list[0]
        pipe = iter_list[1]
        parameters = iter_list[2]
        print("Generating estimator %s" % key)
        self.clf = GridSearchCV(estimator=pipe, param_grid=parameters, scoring=scoring, cv=cv, n_jobs=n_jobs)
        # print("Fitting model")
        self.clf.fit(self.X_train, self.y_train)
        # print("Saving result")
        ret = self.record_model(key, parameters)
        # print("-" * 60 + "\n")
        # print("Estimator %s down" % key)
        return ret

    def do_task_mp(self, processes=2):
        '''
        TODO
        :return:
        '''
        pipeline_list = self.make_pipelines_mp()
        pool = mp.Pool(processes=processes)
        ret_list = pool.map(self.run_estimator_mp, pipeline_list)
        pool.close()
        pool.join()

        for item in ret_list:
            key = item[0]
            best_score = item[1]
            in_sample_acc = item[2]
            out_of_sample_acc = item[3]
            self.result[key]['best_score'] = best_score
            self.result[key]['in_sample_accuracy'] = in_sample_acc
            self.result[key]['out_of_sample_accuracy'] = out_of_sample_acc
            self.result[key]['best_parameters_set'] = {}
            for p, v in item[4:]:
                self.result[key]['best_parameters_set'][p] = v

    def record_model(self, key, parameters):
        '''
        Save best score and parameters for current model.
        :param key: string, combination of names in a pipeline.
        :param parameters: dict, parameters of each part of pipeline.
        :return: list, with [key, score, all pairs of parameters with its best value in order...], length of this list is different based on number of parameters.
        '''
        # 输出best score
        ret = [key]
        print("%s Best score: %0.3f" % (key, self.clf.best_score_))
        self.result[key]['best_score'] = self.clf.best_score_
        ret.append(self.clf.best_score_)
        # print(self.clf.score(self.X_train, self.y_train), self.clf.score(self.X_test, self.y_test))
        self.result[key]['in_sample_accuracy'] = self.clf.score(self.X_train, self.y_train)
        ret.append(self.result[key]['in_sample_accuracy'])
        self.result[key]['out_of_sample_accuracy'] = self.clf.score(self.X_test, self.y_test)
        ret.append(self.result[key]['out_of_sample_accuracy'])
        self.result[key]['best_parameters_set'] = {}
        # 输出最佳的分类器的参数
        best_parameters = self.clf.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            self.result[key]['best_parameters_set'][param_name] = best_parameters[param_name]
            ret.append((param_name, best_parameters[param_name]))
        return ret

    def show_result(self, save=True, file_name='result.csv'):
        '''
        Print result of each pipeline.
        :param save: bool, save as csv file if the value is true.
        :param file_name: string, save file using this name.
        :return:
        '''
        best_score = 0
        best_estimator = ""
        estimator_list = []
        best_score_list = []
        in_sample_acc_list = []
        out_of_sample_acc_list = []
        params_list = []
        max_no_params = 0
        for k, v in self.result.items():

            estimator_name = k.replace('do_nothing+', '')
            # estimator_score = v['best_score']

            # print("Estimator Name: %s" % (estimator_name))
            # print("Best Score: %0.3f" % (estimator_score))

            estimator_score = v['best_score']
            estimator_score_osp = v['out_of_sample_accuracy']
            estimator_score_isp = v['in_sample_accuracy']

            print("Estimator Name: %s" % (estimator_name))
            print("Best Score: %0.3f" % (estimator_score))
            print("In of sample accuracy: %0.3f" % (estimator_score_isp))
            print("Out of sample accuracy: %0.3f" % (estimator_score_osp))

            if save:
                estimator_list.append(k)
                in_sample_acc_list.append(estimator_score_isp)
                out_of_sample_acc_list.append(estimator_score_osp)
                best_score_list.append(estimator_score)

            if estimator_score > best_score:
                best_estimator = estimator_name
                best_score = estimator_score

            print("Best parameters set:")
            params_count = 0
            estimator_params_list = []
            for p, v in self.result[k]['best_parameters_set'].items():
                params_count +=1
                print("\t%s: %r" % (p, v))
                if save:
                    estimator_params_list.append(p + ": " + str(v))
            if save:
                params_list.append(estimator_params_list)
            if params_count > max_no_params:
                max_no_params = params_count
            print("*" * 60 + "\n")

        if save:
            self.save_result(estimator_list, best_score_list, in_sample_acc_list, out_of_sample_acc_list, params_list, max_no_params, file_name)
        print("BEST ESTIMATOR IS %s WITH PREDICT SCORE %0.3f" % (best_estimator, best_score))

    def save_result(self, estimator_list, best_score_list, in_sample_acc_list, out_of_sample_acc_list, params_list, max_no_params, file_name='result.csv'):
        cwd = os.getcwd()
        full_path = cwd + "/" + file_name
        print("Save result as %s" % full_path)

        data = {
            'preprocessing': [],
            'decomposition': [],
            'classifier': [],
        }

        for i in range(0, max_no_params):
            col_name = "parameter_" + str(i+1)
            data[col_name] = []

        data['best_score'] = []
        data['in_sample_accuracy'] = []
        data['out_of_sample_accuracy'] = []

        for i in range(0, len(estimator_list)):

            curr_estimator = estimator_list[i]
            p1, p2, p3 = curr_estimator.split("+")
            data['preprocessing'].append(p1)
            data['decomposition'].append(p2)
            data['classifier'].append(p3)

            curr_in_sample_acc = in_sample_acc_list[i]
            curr_out_of_sample_acc = out_of_sample_acc_list[i]
            best_score = best_score_list[i]
            data['best_score'].append(best_score)
            data['in_sample_accuracy'].append(curr_in_sample_acc)
            data['out_of_sample_accuracy'].append(curr_out_of_sample_acc)

            curr_params_set = params_list[i]
            for j in range(0, max_no_params):
                col_name = "parameter_" + str(j + 1)
                data[col_name].append(curr_params_set[j] if j < len(curr_params_set) else '')

        df = pd.DataFrame.from_dict(data)
        df.to_csv(file_name)

    def generate_voting_classifier(self, file='result.csv'):
        logistic_clf = LogisticRegression(C=0.046415888336127774, penalty='l2')
        random_forest_clf = RandomForestClassifier(criterion='gini', max_depth=8, max_features='auto', n_estimators=100)
        neural_net_clf = MLPClassifier(hidden_layer_sizes=36, activation='tanh', solver='lbfgs')
        decision_tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=7, max_features='auto', splitter='best')
        # p1 = Pipeline([('sd', StandardScaler()), ('lg', logistic_clf)])
        # p2 = Pipeline([('rf', random_forest_clf)])
        # p3 = Pipeline([('sd', StandardScaler()), ('pca', PCA(n_components=1)), ('nn', neural_net_clf)])
        # eclf1 = VotingClassifier(estimators=[('p1', p1), ('p2', p2), ('p3', p3)], voting='hard')
        # eclf1.fit(self.X_train, self.y_train)
        # print(eclf1.score(self.X_train, self.y_train), eclf1.score(self.X_test, self.y_test))
        #
        # p1.fit(self.X_train, self.y_train)
        # print(p1.score(self.X_train, self.y_train), p1.score(self.X_test, self.y_test))
        # p2.fit(self.X_train, self.y_train)
        # print(p2.score(self.X_train, self.y_train), p2.score(self.X_test, self.y_test))
        # p3.fit(self.X_train, self.y_train)
        # print(p3.score(self.X_train, self.y_train), p3.score(self.X_test, self.y_test))

        ad_boost = AdaBoostClassifier(base_estimator=decision_tree_clf, n_estimators=2000, algorithm="SAMME")
        ad_boost.fit(self.X_train, self.y_train)
        print(ad_boost.score(self.X_train, self.y_train), ad_boost.score(self.X_test, self.y_test))
        ad_boost2 = AdaBoostClassifier(base_estimator=random_forest_clf, n_estimators=2000, algorithm="SAMME")
        ad_boost2.fit(self.X_train, self.y_train)
        print(ad_boost2.score(self.X_train, self.y_train), ad_boost2.score(self.X_test, self.y_test))

        eclf2 = VotingClassifier(estimators=[('ad_boost', ad_boost), ('p2', ad_boost2)], voting='hard', weights=[1,1])
        eclf2.fit(self.X_train, self.y_train)
        print(eclf2.score(self.X_train, self.y_train), eclf2.score(self.X_test, self.y_test))

if __name__ == '__main__':
    my_clf = CSF()
    my_clf.initialize()
    # my_clf.describe_data()
    my_clf.run_mp()
    # my_clf.generate_voting_classifier()