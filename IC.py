#!/usr/bin/env python3.7


'''****************************************************************************
 * IC.py: Image Classification Using sklearn
 ******************************************************************************
 * v0.1 - 01.06.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************'''


################################################################################
# Parameters
################################################################################

dataset               = 'datasets/mnist_png'
split_into_train_test = False

if not split_into_train_test:
	split_test_size = None
else:
	split_test_size = 0.1

target_size = None # (64, 64)
runs        = 5

enable_regression_classifiers = False

disable_sklearn_warnings = True


################################################################################
# Disable sklearn warnings
################################################################################

if disable_sklearn_warnings:
	import warnings

	def warn(*args, **kwargs):
		pass

	warnings.warn = warn


################################################################################
# Imports
################################################################################

import argparse
import datetime
import os
import sklearn
import time

import numpy as np

from skimage.io        import imread
from skimage.transform import resize

from sklearn.preprocessing import normalize

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble              import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model          import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.neural_network        import MLPClassifier
from sklearn.svm                   import LinearSVC, SVC
from sklearn.tree                  import DecisionTreeClassifier

if enable_regression_classifiers:
	from sklearn.metrics import r2_score

	from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
	from sklearn.svm          import LinearSVR, SVR


################################################################################
# Misc
################################################################################

regression_classifiers = ['ElasticNet', 'Lasso', 'LinearRegression', 'Ridge', 'LinearSVR', 'SVR']

IC                  = '[IC]'
IC_separator_string = (80 - len(f'{IC} ')) * '#'

def IC_print(string, file=None):
	print(f'{IC} {string}', file=file)




################################################################################
# Load the dataset
################################################################################

def load_dataset(dataset, target_size):
	data   = []
	labels = []

	directories = [os.path.join(dataset, directory) for directory in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, directory))]

	for i, directory in enumerate(directories):
		files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

		for file in files:
			if not file.endswith('.csv'):
				image = imread(file)
			else:
				image = np.genfromtxt(file, delimiter=',')

			if target_size is not None:
				image = resize(image, target_size, order=0) # nearest-neighbor interpolation

			data.append(image.flatten())
			labels.append(i)

	data_clipping_min = np.finfo('float32').min
	data_clipping_max = np.finfo('float32').max
	data = normalize(np.clip(data, data_clipping_min, data_clipping_max))

	return sklearn.utils.Bunch(data=np.array(data), labels=np.array(labels))




################################################################################
# Print run / report info
################################################################################

def IC_print_run(run, classifier_string, log_file):
	IC_print(IC_separator_string)
	IC_print(f'{IC_separator_string[0]} [run={run}] {classifier_string}')
	IC_print(IC_separator_string)

	IC_print(IC_separator_string,                                         file=log_file)
	IC_print(f'{IC_separator_string[0]} [run={run}] {classifier_string}', file=log_file)
	IC_print(IC_separator_string,                                         file=log_file)


def IC_print_report(report, time_diff, classifier, classifier_string, log_file):
	IC_print(f'classifier={classifier}')
	IC_print(f'report={report}')
	IC_print(f'in time_diff={time_diff} seconds')

	IC_print(f'classifier={classifier}',          file=log_file)
	IC_print(f'report={report}',                  file=log_file)
	IC_print(f'in time_diff={time_diff} seconds', file=log_file)




################################################################################
# Run the selected classifier
################################################################################

def run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file):
	start_time = time.time()
	classifier.fit(x_train, y_train)
	time_diff  = time.time() - start_time

	y_prediction = classifier.predict(x_test)

	if classifier_string not in regression_classifiers:
		report = sklearn.metrics.classification_report(y_test, y_prediction, output_dict=True)
	else:
		report = r2_score(y_test, y_prediction)

	reports_dict[classifier_string] = report

	IC_print_report(report, time_diff, classifier, classifier_string, log_file)




################################################################################
# Start a new run
################################################################################

def start_new_run(run, timestamp, x_train, x_test, y_train, y_test):
	reports_dict = {}

	log_file = open(f'IC_log_{timestamp}_run{run}.log', 'w')




	############################################################################
	# LinearDiscriminantAnalysis
	############################################################################

	classifier_string = 'LinearDiscriminantAnalysis'

	IC_print_run(run, classifier_string, log_file)

	classifier = LinearDiscriminantAnalysis()

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# QuadraticDiscriminantAnalysis
	############################################################################

	classifier_string = 'QuadraticDiscriminantAnalysis'

	IC_print_run(run, classifier_string, log_file)

	classifier = QuadraticDiscriminantAnalysis()

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# ExtraTreesClassifier
	############################################################################

	classifier_string = 'ExtraTreesClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = ExtraTreesClassifier(n_jobs=-1, verbose=1)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# RandomForestClassifier
	############################################################################

	classifier_string = 'RandomForestClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = RandomForestClassifier(n_jobs=-1, verbose=1)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# LogisticRegression
	############################################################################

	classifier_string = 'LogisticRegression'

	IC_print_run(run, classifier_string, log_file)

	classifier = LogisticRegression(n_jobs=-1, verbose=1)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# RidgeClassifier
	############################################################################

	classifier_string = 'RidgeClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = RidgeClassifier()

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# GaussianNB
	############################################################################

	classifier_string = 'GaussianNB'

	IC_print_run(run, classifier_string, log_file)

	classifier = GaussianNB()

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# KNeighborsClassifier
	############################################################################

	classifier_string = 'KNeighborsClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = KNeighborsClassifier(n_jobs=-1)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# MLPClassifier
	############################################################################

	classifier_string = 'MLPClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = MLPClassifier(verbose=True)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# LinearSVC
	############################################################################

	classifier_string = 'LinearSVC'

	IC_print_run(run, classifier_string, log_file)

	classifier = LinearSVC(verbose=1)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# SVC
	############################################################################

	classifier_string = 'SVC'

	IC_print_run(run, classifier_string, log_file)

	classifier = SVC(verbose=True)

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


	############################################################################
	# DecisionTreeClassifier
	############################################################################

	classifier_string = 'DecisionTreeClassifier'

	IC_print_run(run, classifier_string, log_file)

	classifier = DecisionTreeClassifier()

	run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)




	if enable_regression_classifiers:
		########################################################################
		# ElasticNet
		########################################################################

		classifier_string = 'ElasticNet'

		IC_print_run(run, classifier_string, log_file)

		classifier = ElasticNet()

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


		########################################################################
		# Lasso
		########################################################################

		classifier_string = 'Lasso'

		IC_print_run(run, classifier_string, log_file)

		classifier = Lasso()

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


		########################################################################
		# LinearRegression
		########################################################################

		classifier_string = 'LinearRegression'

		IC_print_run(run, classifier_string, log_file)

		classifier = LinearRegression(n_jobs=-1)

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


		########################################################################
		# Ridge
		########################################################################

		classifier_string = 'Ridge'

		IC_print_run(run, classifier_string, log_file)

		classifier = Ridge()

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


		########################################################################
		# LinearSVR
		########################################################################

		classifier_string = 'LinearSVR'

		IC_print_run(run, classifier_string, log_file)

		classifier = LinearSVR(verbose=1)

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)


		########################################################################
		# SVR
		########################################################################

		classifier_string = 'SVR'

		IC_print_run(run, classifier_string, log_file)

		classifier = SVR(verbose=True)

		run_classifier(classifier, classifier_string, x_train, x_test, y_train, y_test, reports_dict, log_file)




	log_file.close()

	return reports_dict




################################################################################
# Run IC
################################################################################

def run(
	dataset               = dataset,
	split_into_train_test = split_into_train_test,
	split_test_size       = split_test_size,
	target_size           = target_size,
	runs                  = runs):

	reports_dict_list = []

	timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


	if target_size is not None:
		if type(target_size) is not tuple:
			target_size = tuple(target_size)

		if len(target_size) == 1:
			target_size *= 2


	if not split_into_train_test:
		dataset_train = load_dataset(os.path.join(dataset, 'train'), target_size)
		dataset_test  = load_dataset(os.path.join(dataset, 'test'),  target_size)

		x_train, y_train = dataset_train.data, dataset_train.labels
		x_test,  y_test  = dataset_test.data,  dataset_test.labels

		x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
	else:
		dataset_train_test = load_dataset(dataset, target_size)

		x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
			dataset_train_test.data, dataset_train_test.labels, test_size=split_test_size)


	for run in range(1, runs + 1):
		reports_dict = start_new_run(
			run       = run,
			timestamp = timestamp,
			x_train   = x_train,
			x_test    = x_test,
			y_train   = y_train,
			y_test    = y_test)

		reports_dict_list.append(reports_dict)


	IC_print(f'reports_dict_list={reports_dict_list}')

	with open(f'IC_log_{timestamp}.log', 'w') as log_file:
		IC_print(f'reports_dict_list={reports_dict_list}', file=log_file)




################################################################################
# parse_args
################################################################################

def parse_args():
	parser = argparse.ArgumentParser(description='IC - Image Classification Using sklearn')

	parser.add_argument('--dataset',                                      default=dataset)
	parser.add_argument('--split-into-train-test', type=bool,             default=split_into_train_test)
	parser.add_argument('--split-test-size',       type=float,            default=split_test_size)
	parser.add_argument('--target-size',           type=int,   nargs='+', default=target_size)
	parser.add_argument('--runs',                  type=int,              default=runs)

	return parser.parse_args()


################################################################################
# main
################################################################################

if __name__ == '__main__':
	args = parse_args()

	IC_print(f'args={args}')

	run(
		dataset               = args.dataset,
		split_into_train_test = args.split_into_train_test,
		split_test_size       = args.split_test_size,
		target_size           = args.target_size,
		runs                  = args.runs)

