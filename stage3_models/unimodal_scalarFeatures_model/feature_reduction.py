# feature selection:
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel 

# pipeline:
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# evaluation:
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# used models:
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # also for feature selection of RFE

# Cross Validation:
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

# storing model: 
import pickle
import os

# rest:
import numpy as np
import pandas as pd
from collections import defaultdict

# Getting the used versions of imports:
import sys
sys.path.insert(0, '../../stage-1/overall_used_tools')
import requirements_check as rc

import sklearn, collections
rc.check(sys, [sklearn, pickle, os, np, pd, collections])


# When using MusicBERT you get an error for numpy versions >= 1.24!

class PipelinedScalarTraining:
	def __init__(self, number_cv_splits=5):
		print('start process')
		self.x = None
		self.y = None
		self.y1 = None
		self.y2 = None
		self.y3 = None

		self.gkf_inner = StratifiedShuffleSplit(n_splits=number_cv_splits) 
		self.gkf_outer = StratifiedShuffleSplit(n_splits=number_cv_splits) 

		# directory to store results:
		self.clf_path = "clf" # path were to final classifiers are stored
		try: 
			os.mkdir(self.clf_path)
		except:
			pass

	def dataset_preparation(self, datasetB_human,regression=False):
		""" DIRECT CALL """
		print("PUT IN A DATAFRAME AS ARUMENT HERE WHICH CONTAINS THE FEATURES AND IN ADDITION THE COLUMN FOR Y-LABELS 'final_label' IN FORMAT OF [0 0 1]")



		# Letting the two datasets separated:
		X_datasetB_human = datasetB_human.replace(float('nan'),0)

		# Shuffling the rows:
		X_datasetB_human = X_datasetB_human.sample(frac=1)

		y_datasetB_human = np.array([v.strip('][').split(', ') for v in X_datasetB_human['final_label'].values]) 


		
		try:
			X_datasetB_human.drop(['final_label', 'source_id', 'sample_id','Unnamed: 0'], axis=1, inplace=True) 

		except:
			X_datasetB_human.drop(['final_label'], axis=1, inplace=True) 

		# Getting out y:
		if regression == False:
			y_datasetB_human_out1 = y_datasetB_human[:,0].astype(int)
			y_datasetB_human_out2 = y_datasetB_human[:,1].astype(int)
			y_datasetB_human_out3 = y_datasetB_human[:,2].astype(int)

		else:
			y_datasetB_human_out1 = y_datasetB_human[:,0].astype(float)
			y_datasetB_human_out2 = y_datasetB_human[:,1].astype(float)
			y_datasetB_human_out3 = y_datasetB_human[:,2].astype(float)

		self.x = X_datasetB_human
		self.y = y_datasetB_human
		self.y1 = y_datasetB_human_out1
		self.y2 = y_datasetB_human_out2
		self.y3 = y_datasetB_human_out3

		return

	def load_model(self, filename, path):
		""" Loads a classifiers from the pickle file"""
		print('start loading model')
		filename = os.path.join(path, filename)
		with open(filename, 'rb') as fh:
			data = pickle.load(fh)
		return data['clf'], data['score'], data['results'], data['dscr']

	def store_model(self, filename, path, clf_to_store, score, results, description=''):
		""" Stores a model to the pickle file
		:param clf_to_store: classifier that should be stored
		:param description: Here you can add a helpful string
		:param features: a list of features that were used
		"""

		os.makedirs(path, exist_ok=True)
		filename = os.path.join(path, filename)
		storage = {'clf': clf_to_store.tolist(), 'score': score, 'results': results, 'dscr': description} 
		print('start storing model in ', filename)

		with open(filename, 'wb') as fh:
			clf_list = pickle.dump(storage, fh)

	def evaluate(self, grid, X_test= None, y_test= None):
		""" simple function to evaluate the classifier
		It plots following: 
		"""
		print('start evaluation')

		results = pd.DataFrame(grid.cv_results_)
		print('Best params: \n', results['params'][grid.best_index_])

		clf = grid.best_estimator_
		print('Accuracy cross_val:\n', grid.best_score_ )

		if not X_test is None and not y_test is None:

			y_test = np.squeeze(y_test.to_numpy())
			y_pred = clf.predict(X_test)
			print('Accuracy test_set:\n', clf.score(X_test, list(y_test)) )
			print(classification_report(list(y_test), list(y_pred)))
			print('confusion matrix:\n')
			metrics.plot_confusion_matrix(clf, X_test, list(y_test))
			plt.show()

		return results

	def select_top_features(self, feature_number_approach1=20, feature_number_approach2=20, categorical=True):
		""" Here two imported are approched and the final feature set then merged, and the dataset reduced"""
		""" DIRECT CALL """

		temp_X = self.x.to_numpy()
		temp_y = np.squeeze(self.y)

		# approach 1:
		if categorical == True:
			estimator = RandomForestClassifier()
		else:
			estimator = RandomForestRegressor()
		selector = RFE(estimator, n_features_to_select=feature_number_approach1, step=4)
		selector = selector.fit(temp_X, temp_y)

		recursive_feature_elimination = list(self.x.columns[selector.get_support()])


		# approach 2:
		if categorical:
			estimator = RandomForestClassifier()
		else:
			estimator = RandomForestRegressor()
		selector = SelectFromModel(estimator, max_features=feature_number_approach2)
		selector.fit(temp_X, temp_y)

		selectFromModel_feature_selection = list(self.x.columns[selector.get_support()])


		# merge the approaches:
		mask = list(set(recursive_feature_elimination) | set(selectFromModel_feature_selection))
		self.x = self.x[mask] 

		return
	
	def train_clf(self, pipe, tuned_parameters,regression=False):
		""" NO DIRECT CALL """
		#grids = []
		scores_datasets, estimators_datasets = [],[] 
		datasets = [(self.x, self.y1, None),(self.x, self.y2, None),(self.x, self.y3, None)]
		print('start train:')

		test_size = int(self.x.shape[0]/100*20) # 20 % of the dataset will become part of test set
		number_splits_test, number_splits_vali = int(self.x.shape[0]/test_size),int(self.x.shape[0]/test_size)


		sss_test = StratifiedShuffleSplit(n_splits=number_splits_test, random_state=42) 
		sss_vali = StratifiedShuffleSplit(n_splits=number_splits_vali, random_state=42) 


		estimator_collector_sets, estimator_score_collector_sets = None, None 

		for dataset_ind, (X, y, metadata) in enumerate(datasets):

			if regression==True:
				bin_edges_y = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit_transform(y.reshape(-1, 1)).squeeze()
				score_method = 'neg_mean_squared_error' #  sklearn.metrics.SCORERS.keys() to see possible scoring methods or https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=All%20scorer%20objects%20follow%20the,negated%20value%20of%20the%20metric.
			else:
				bin_edges_y = y
				score_method = 'accuracy' 
			print('regression',regression)
			print('training for y output round: ', dataset_ind, '/2')
			
			estimator_collector, estimator_score_collector = [],[]
			for train_ind, test_ind in sss_test.split(X, bin_edges_y):

				X_train_test = X.iloc[train_ind]
				y_train_test = y[train_ind]

				X_test_test = X.iloc[test_ind]
				y_test_test = y[test_ind]

				if regression==True:
					bin_edges_y_train_test = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit_transform(y_train_test.reshape(-1, 1)).squeeze()
				else:
					bin_edges_y_train_test = y_train_test

				gkf_iterable = sss_vali.split(X_train_test, bin_edges_y_train_test)
				grid_hyper_vali = GridSearchCV(pipe, tuned_parameters, scoring=score_method, cv=gkf_iterable, n_jobs=-1)
				# GridSearchCV() tries out all hyperparameter combinations from 'tuned_parameters'.
				# It uses Stratified Cross Validation. Because of that it is not suited for multilabel problems.

				grid_hyper_vali.fit(X_train_test, np.squeeze(y_train_test))
				#print('best vali results',grid_hyper_vali.cv_results_)
				grid_hyper_vali.cv_results_
				grid_hyper_vali.best_score_
				print('best vali score',grid_hyper_vali.best_score_)
				best_estimator_hyper_vali = grid_hyper_vali.best_estimator_

				print('best vali est', best_estimator_hyper_vali )
				#print('y_test_test',y_test_test)
				score_estimator_test_fold = best_estimator_hyper_vali.score(X_test_test, y_test_test)
				#print('found score test',score_estimator_test_fold)
				estimator_collector.append(best_estimator_hyper_vali)
				estimator_score_collector.append(score_estimator_test_fold)

			if estimator_collector_sets is None:
				estimator_collector_sets = np.empty((len(datasets),np.array(estimator_collector).shape[0],np.array(estimator_collector).shape[1]),dtype='object')
				estimator_score_collector_sets = np.empty((len(datasets),len(estimator_score_collector),1))


			estimator_collector_sets[dataset_ind] = np.array(estimator_collector)
			estimator_score_collector_sets[dataset_ind] = np.array(estimator_score_collector).reshape(-1,1)
			

		return estimator_collector_sets, estimator_score_collector_sets

			


	
	def getting_best_est_para(self, est_coll, est_score_coll, method, data_name):

		#return est_coll, est_score_coll,f
		# determining the score which performed best on average on test folds:
		dict_para_y1 = defaultdict(lambda: [])
		dict_belonging_est_y1 = defaultdict(lambda: '')
		dict_para_y2 = defaultdict(lambda: [])
		dict_belonging_est_y2 = defaultdict(lambda: '')
		dict_para_y3 = defaultdict(lambda: [])
		dict_belonging_est_y3 = defaultdict(lambda: '')

		for dataset_ind in range(3): # having 3 y-s to predict
			for est_ind in range(len(est_coll)):


				if method == 'knn':
					para1 = str(est_coll[dataset_ind][est_ind][2].n_neighbors)
					para2 = str(est_coll[dataset_ind][est_ind][2].weights)
					para_full = 'n_neighbors='+para1+'weights='+para2

				elif method == 'rf':
					para1 = str(est_coll[dataset_ind][est_ind][2].n_estimators)
					para2 = str(est_coll[dataset_ind][est_ind][2].criterion)
					para_full = 'n_estimators='+para1+'criterion='+para2

				elif method == 'svm':
					para1 = str(est_coll[dataset_ind][est_ind][2].C)
					para2 = str(est_coll[dataset_ind][est_ind][2].gamma)
					para3 = str(est_coll[dataset_ind][est_ind][2].kernel)

					para_full = 'C='+para1+'gamma='+para2+'kernel='+para3

				elif method == 'mlp':
					para1 = str(est_coll[dataset_ind][est_ind][2].activation)
					para2 = str(est_coll[dataset_ind][est_ind][2].batch_size)
					para_full = 'activation='+para1+'batch_size='+para2


				if dataset_ind==0 and len(est_score_coll[dataset_ind][est_ind]) >0:

					dict_para_y1[para_full].append(est_score_coll[dataset_ind][est_ind][0])
					dict_belonging_est_y1[para_full] = est_coll[dataset_ind][est_ind]

				elif dataset_ind==1 and len(est_score_coll[dataset_ind][est_ind]) >0:

					dict_para_y2[para_full].append(est_score_coll[dataset_ind][est_ind][0])
					dict_belonging_est_y2[para_full] = est_coll[dataset_ind][est_ind]

				elif dataset_ind==2 and len(est_score_coll[dataset_ind][est_ind]) >0:

					dict_para_y3[para_full].append(est_score_coll[dataset_ind][est_ind][0])
					dict_belonging_est_y3[para_full] = est_coll[dataset_ind][est_ind]


		dict_para_mean_y1 = dict() 
		dict_para_mean_y2 =  dict()
		dict_para_mean_y3 =  dict() 
		
		max_len = max(len(list(dict_para_y1.keys())),len(list(dict_para_y2.keys())), len(list(dict_para_y3.keys())))
		differ1 = max_len - len(list(dict_para_y1.keys())) 
		differ2 = max_len - len(list(dict_para_y2.keys())) 
		differ3 = max_len - len(list(dict_para_y3.keys())) 

		for key1,key2,key3 in zip(list(dict_para_y1.keys())+[None]*differ1,list(dict_para_y2.keys())+[None]*differ2, list(dict_para_y3.keys())+[None]*differ3):

			if key1 is not None:		

				mean_score = np.array(dict_para_y1[key1]).mean()

				dict_para_mean_y1[key1] = mean_score

			if key2 is not None:
				mean_score = np.array(dict_para_y2[key2]).mean()
				dict_para_mean_y2[key2] = mean_score

			if key3 is not None:
				mean_score = np.array(dict_para_y3[key3]).mean()
				dict_para_mean_y3[key3] = mean_score

		best_ind_y1 = np.argmax(np.array(list(dict_para_mean_y1.values()), dtype=float))
		best_ind_y2 = np.argmax(np.array(list(dict_para_mean_y2.values()), dtype=float))
		best_ind_y3 = np.argmax(np.array(list(dict_para_mean_y3.values()), dtype=float))

		# for y1:
		chosen_est_para_y1 = list(dict_para_mean_y1.keys())[best_ind_y1]
		chosen_est_y1 = dict_belonging_est_y1[chosen_est_para_y1] # the one to store
		best_score_y1 = list(dict_para_mean_y1.values())[best_ind_y1]

		# store final model:
		description = method
		temp = data_name + '_' + method + '_y1' 
		clf_path = self.clf_path + '/' + data_name
		self.store_model(temp, clf_path, chosen_est_y1, best_score_y1, dict(dict_para_y1), description)

		# for y2:
		chosen_est_para_y2 = list(dict_para_mean_y2.keys())[best_ind_y2]
		chosen_est_y2 = dict_belonging_est_y2[chosen_est_para_y2] # the one to store
		best_score_y2 = list(dict_para_mean_y2.values())[best_ind_y2]

		# store final model:
		temp = data_name + '_' + method + '_y2'
		self.store_model(temp, clf_path, chosen_est_y2, best_score_y2, dict(dict_para_y2), description)
			
		# for y3:
		chosen_est_para_y3 = list(dict_para_mean_y3.keys())[best_ind_y3]
		chosen_est_y3 = dict_belonging_est_y3[chosen_est_para_y3] # the one to store
		best_score_y3 = list(dict_para_mean_y3.values())[best_ind_y3]

		# store final model:
		temp = data_name + '_' + method + '_y3'
		self.store_model(temp, clf_path, chosen_est_y3, best_score_y3, dict(dict_para_y3), description)

		return chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3

	
	def my_knn(self, regression=False, data_name=''): 

		""" DIRECT CALL """

		print('start knn:')
		description = """
		knn 
		"""  # add here a description that will be stored with the clf

		tuned_parameters = [{'selector__features': self.x.columns.tolist(), 'clf__n_neighbors': list(range(1, 100, 10)), 'clf__weights': ['uniform', 'distance']} ]
		# Over these hyperparameters is looped by GridSearchCV() which takes the best scoring parameter combination.



		# set up pipeline
		if regression == False: 
			method = KNeighborsClassifier()
		else:
			method = KNeighborsRegressor()

		pipe = Pipeline([('selector', Selector()), ('scaler', StandardScaler()), ('clf', method)]) # StandardScaler() for the case that data was not normalized

		# train classifier
		est_coll, est_score_coll =  self.train_clf(pipe, tuned_parameters,regression=regression)

		chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3 = self.getting_best_est_para(est_coll, est_score_coll, 'knn', data_name)
		return chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3

	def my_rf(self, regression=False, data_name=''):

		""" DIRECT CALL """

		description = """
		RandomForest gridsearch on raw data
		"""  # add here a description that will be stored with the clf

		if regression == False: 
			method = RandomForestClassifier(n_jobs=-1)
			criterions = ['entropy', 'gini']
		else:
			method = RandomForestRegressor(n_jobs=-1)
			criterions = ['squared_error', 'absolute_error'] # , 'friedman_mse', 'poisson']


		tuned_parameters = [{'selector__features': self.x.columns.tolist(), 'clf__n_estimators': list(range(50, 300, 20)), 'clf__criterion': criterions}]

		# set up pipeline
		pipe = Pipeline([('selector', Selector()), ('scaler', StandardScaler()), ('clf', method)])

		# train classifier

		est_coll, est_score_coll =  self.train_clf(pipe, tuned_parameters,regression=regression)

		chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3 = self.getting_best_est_para(est_coll, est_score_coll,'rf', data_name)
		return chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3

	def my_svm(self,regression=False, data_name=''):

		""" DIRECT CALL """

		description = """
		SVM gridsearch
		"""  # add here a description that will be stored with the clf

		tuned_parameters = [{'selector__features': self.x.columns.tolist(), 'clf__C': [i/100 for i in range(10,100,40)]+[float(i) for i in range(2,100,10)], 'clf__gamma': ['scale', 'auto'], 'clf__kernel': ['linear', 'rbf', 'sigmoid']}] 
		# set up pipeline
		if regression == False: 
			method = SVC()
		else:
			method = SVR()

		pipe = Pipeline([('selector', Selector()), ('scaler', StandardScaler()), ('clf', method)])

		# train classifier
		est_coll, est_score_coll =  self.train_clf(pipe, tuned_parameters,regression=regression)

		chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3 = self.getting_best_est_para(est_coll, est_score_coll,'svm', data_name)
		return chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3

	def my_mlp(self, regression=False, data_name=''):

		""" DIRECT CALL """


		description = """
		NN gridsearch
		"""


		tuned_parameters = [{'selector__features': self.x.columns.tolist(), 'clf__batch_size': [1, 16, 32, 64], 'clf__activation': ['tanh', 'relu'] }]

		# set up pipeline:
		if regression == False: 
			method = MLPClassifier(hidden_layer_sizes=(50,50,10,8,4), shuffle=True)
			#method = MLPClassifier(hidden_layer_sizes=(64, 64, 32, 32, 16, 2), shuffle=True)
		else:
			method = MLPRegressor(hidden_layer_sizes=(50,50,10,8,4), shuffle=True)
			#method = MLPRegressor(hidden_layer_sizes=(64, 64, 32, 32, 16, 2), shuffle=True)

		pipe = Pipeline([('selector', Selector()), ('scaler', StandardScaler()), ('clf', method)])

		# train classifier
		est_coll, est_score_coll =  self.train_clf(pipe, tuned_parameters, regression)

		chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3 = self.getting_best_est_para(est_coll, est_score_coll,'mlp', data_name)
		return chosen_est_para_y1, best_score_y1, chosen_est_para_y2, best_score_y2, chosen_est_para_y3, best_score_y3


class Selector(BaseEstimator, TransformerMixin):
    """ A Feature selector for use in the pipeline
    Using this the classifier stores the relevant features itself and we don't have to select them manually if we apply the clf later.
    """
    
    def __init__(self, features=[]):
        super().__init__()
        self.features = features
        
    def fit(self, X_shuffled, y_shuffled = None):
        return self
    
    def transform(self, X_shuffled, y_shuffled = None):
       	

        X_shuffled = pd.DataFrame(X_shuffled, columns=X_shuffled.columns)

        try:
            re = X_shuffled[self.features]
            re = re.reshape(-1,1)

        except:
            re = X_shuffled[self.features].values.reshape(-1,1)

        return re





