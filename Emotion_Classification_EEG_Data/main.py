import os
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from pywt import wavedec
from biosppy.signals import eeg

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import mne
from mne.io import read_raw_edf
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.preprocessing.ica import corrmap
from mne.decoding import CSP

# initialization.
def init():
	print('Init...')

	# input and output directories for reading and writing files
	datasetdir = 'datasets'
	outputdir = 'outputs'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	# dataset parameters
	dataset_name = 'Amigos'
	feature_type = 'stat'
	method = 'classification'
	target = 'arousal'
	validation_method = '9fold'
	csp_components = [1,2,5,10]
	clf_names = ['svm','extratrees'] # 'knn', 'logistic', 'lda', 'svm', 'tree', 'randomforest', 'extratrees', 'gradboost', 'adaboost', 'mlp', 'ecoc'
	
	return datasetdir, outputdir, dataset_name, feature_type, method, target, validation_method, csp_components, clf_names

def asymmetry():
	print('Asymmetry...')
	
	# load the dataset from file
	dataset_path = os.path.join(datasetdir, 'sig.mat')
	annotation_path = os.path.join(datasetdir, 'toYachen.mat')
	
	dataset_mat = scipy.io.loadmat(dataset_path)
	eeg8 = dataset_mat['sig8']
	eeg14 = dataset_mat['sig14']
	eeg18 = dataset_mat['sig18']
	eeg19 = dataset_mat['sig19']
	
	annotation_mat = scipy.io.loadmat(annotation_path)
	annot = annotation_mat['labels']
	annot8 = annot[3][0]
	annot14 = annot[13][0]
	annot18 = annot[17][0]
	annot19 = annot[18][0]
	
	# convert to mne compatible format
	data = {}
	channel_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp']
	fs = 256
	channel_type = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg','ecg']
	info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=channel_type)
	montage = mne.channels.read_montage('standard_1020')
				
	# create rawarray
	raw8 = mne.io.RawArray(eeg8[:,0:eeg8.shape[1]-500],info,verbose='error')
	raw14 = mne.io.RawArray(eeg14[:,0:eeg14.shape[1]-500],info,verbose='error')
	raw18 = mne.io.RawArray(eeg18[:,0:eeg18.shape[1]-500],info,verbose='error')
	raw19 = mne.io.RawArray(eeg19[:,0:eeg19.shape[1]-500],info,verbose='error')
	raw8.set_montage(montage)
	raw14.set_montage(montage)
	raw18.set_montage(montage)
	raw19.set_montage(montage)
	
	# extract the signals
	raw = raw14.pick_types(meg=False,eeg=True,ecg=False)
	annot = np.transpose(annot14)
	#raw = raw.set_eeg_reference('average')
	
	# calculate the powers
	eeg_data = raw.get_data()
	output = eeg.get_power_features(signal=np.transpose(eeg_data), sampling_rate=256, size=2, overlap=0.5)
	theta = np.average(output['theta'],axis=0)
	alpha_low = np.average(output['alpha_low'],axis=0)
	alpha_high = np.average(output['alpha_high'],axis=0)
	beta = np.average(output['beta'],axis=0)
	gamma = np.average(output['gamma'],axis=0)
	sum = theta+alpha_low+alpha_high+beta+gamma
	np.concatenate((theta/sum,alpha_low/sum,alpha_high/sum,beta/sum,gamma/sum), axis=0)

	# plot the powers
	plt.figure()
	alpha = output['alpha_low']
	alpha_left = alpha[:,0] + alpha[:,1] + alpha[:,2]
	alpha_right = alpha[:,11] + alpha[:,12] + alpha[:,13]
	time = output['ts']
	offset = 30
	alpha_left = alpha_left[offset:alpha_left.shape[0]-offset]
	alpha_right = alpha_right[offset:alpha_right.shape[0]-offset]
	time = time[offset:time.shape[0]-offset]
	plt.plot(time, alpha_left)
	plt.plot(time, alpha_right)
	annot = signal.resample(annot,annot.shape[0]/4)
	ratio = np.ptp(alpha_left),np.ptp(alpha_right)/np.ptp(annot)
	annot = annot * ratio
	time = range(annot.shape[0])
	time = time + offset*np.ones(len(time))
	plt.plot(time, annot)
	#plt.show()
	plt.savefig(os.path.join(outputdir, 's1_14'+'.png'))
	
	return
	
# load the dataset from file
def loading_dataset():
	print('Loading Dataset...')

	# load the dataset from file
	dataset_path = os.path.join(datasetdir, dataset_name + '_Data.mat')

	dataset_mat = scipy.io.loadmat(dataset_path)
	eeg = dataset_mat['eeg']
	ecg = dataset_mat['ecg']
	n_trials = dataset_mat['number_of_trials']
	l_stimuli = dataset_mat['stimuli_length']
	affective_rating = dataset_mat['affective_rating']
	personality = dataset_mat['personality']
	
	# extract the arousal and valence
	arousal = affective_rating.copy()[:,:,0]
	valence = affective_rating.copy()[:,:,1]
	
	# convert to mne compatible format
	data = {}
	channel_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4','ECGR','ECGL']
	fs = 128
	channel_type = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','ecg','ecg']
	info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=channel_type)
	montage = mne.channels.read_montage('standard_1020')
	for subject_idx in range(n_trials.shape[0]):
		subject_data = {}
		for stimuli_idx in range(n_trials.shape[1]):
			if n_trials[subject_idx,stimuli_idx]==1:
				
				# create rawarray
				l = l_stimuli[stimuli_idx][0]
				eeg_data = eeg[subject_idx,stimuli_idx,:,0:l]
				ecg_data = ecg[subject_idx,stimuli_idx,:,0:l]
				concatenated_data = np.concatenate((eeg_data,ecg_data), axis=0)
				raw = mne.io.RawArray(concatenated_data,info,verbose='error')
				raw.set_montage(montage)
				subject_data[stimuli_idx] = raw
				
		# add the subject only if all stimulus are present
		if set(subject_data.keys())==set(range(n_trials.shape[1])):
			data[subject_idx] = subject_data
	
	# dichotomize arousal
	arousal_med = np.median(np.ravel(arousal))
	upper = []
	lower = []
	for idx in range(np.ravel(arousal).shape[0]):
		if np.ravel(arousal)[idx] > arousal_med:
			upper.append(np.ravel(arousal)[idx])
		else:
			lower.append(np.ravel(arousal)[idx])
	arousal_upper = np.median(np.asarray(upper))
	arousal_lower = np.median(np.asarray(lower))
	
	arousal_dichotomized = np.zeros(arousal.shape)
	for subject_idx in range(arousal.shape[0]):
		for stimuli_idx in range(arousal.shape[1]):
			if arousal[subject_idx,stimuli_idx] > arousal_upper:
				arousal_dichotomized[subject_idx,stimuli_idx] = 1
			elif arousal[subject_idx,stimuli_idx] < arousal_lower:
				arousal_dichotomized[subject_idx,stimuli_idx] = -1
			else:
				arousal_dichotomized[subject_idx,stimuli_idx] = 0
	
	# dichotomize valence
	valence_med = np.median(np.ravel(valence))
	upper = []
	lower = []
	for idx in range(np.ravel(valence).shape[0]):
		if np.ravel(valence)[idx] > valence_med:
			upper.append(np.ravel(valence)[idx])
		else:
			lower.append(np.ravel(valence)[idx])
	valence_upper = np.median(np.asarray(upper))
	valence_lower = np.median(np.asarray(lower))
	
	valence_dichotomized = np.zeros(valence.shape)
	for subject_idx in range(valence.shape[0]):
		for stimuli_idx in range(valence.shape[1]):
			if valence[subject_idx,stimuli_idx] > valence_upper:
				valence_dichotomized[subject_idx,stimuli_idx] = 1
			elif valence[subject_idx,stimuli_idx] < valence_lower:
				valence_dichotomized[subject_idx,stimuli_idx] = -1
			else:
				valence_dichotomized[subject_idx,stimuli_idx] = 0
				
	# dichotomize peronality
	med = np.median(personality,axis=0)
	upper = np.zeros(personality.shape[1])
	lower = np.zeros(personality.shape[1])
	for personality_idx in range(personality.shape[1]):
		personality_upper = []
		personality_lower = []
		for subject_idx in range(personality.shape[0]):
			if personality[subject_idx,personality_idx] > med[personality_idx]:
				personality_upper.append(personality[subject_idx,personality_idx])
			else:
				personality_lower.append(personality[subject_idx,personality_idx])
		upper[personality_idx] = np.median(np.asarray(personality_upper))
		lower[personality_idx] = np.median(np.asarray(personality_lower))
	
	personality_dichotomized = np.zeros(personality.shape)
	for subject_idx in range(personality.shape[0]):
		for personality_idx in range(personality.shape[1]):
			if personality[subject_idx,personality_idx] > upper[personality_idx]:
				personality_dichotomized[subject_idx,personality_idx] = 1
			elif personality[subject_idx,personality_idx] < lower[personality_idx]:
				personality_dichotomized[subject_idx,personality_idx] = -1
			else:
				personality_dichotomized[subject_idx,personality_idx] = 0
	
	return data, arousal_dichotomized, valence_dichotomized, personality_dichotomized

# artifact removal
def preprocessing():
	print('Preprocessing...')

	data_preprocessed = {}
	for subject_idx, subject_data in data.items():
		subject_data_preprocessed = {}
		for stimuli_idx, raw in subject_data.items():
			
			# visualization
			# raw.plot_psd(tmax=np.inf, fmax=250)
			# raw.plot(scalings='auto', show=True, block=True)
			# plt.plot(np.linspace(0,1*math.pi,data.shape[1]),data[0,:])
			# plt.show()
			
			# artifact removal
			# ica = mne.preprocessing.ICA(n_components=0.999, method='fastica')
			# ica.fit(raw.pick_types(meg=False,eeg=True,ecg=False,exclude='bads'))
			# ica.plot_components()
			# ecg_epochs = create_ecg_epochs(raw, tmin=-0.5, tmax=0.5)
			# ecg_average = ecg_epochs.average()
			# ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
			# ica.plot_overlay(ecg_average, exclude=ecg_inds, show=False)
			# ica.plot_scores(scores, exclude=ecg_inds)
			# ica.plot_properties(raw,picks=[1,2])
			# average_ecg = create_ecg_epochs(raw).average()
			# print('We found %i ECG events' % average_ecg.nave)
			# average_ecg.plot_joint()
			
			subject_data_preprocessed[stimuli_idx] = raw.pick_types(meg=False,eeg=True,ecg=False).get_data()
			
		data_preprocessed[subject_idx] = subject_data_preprocessed
	
	return data_preprocessed

# feature extraction
def feature_extraction():
	print('Feature Extraction...')

	subjectIdx = [1,12]
	stimuliIdx = [1,5]
	data_features = {}
	for subject_idx, subject_data in data_preprocessed.items():
		subject_data_features = {}
		for stimuli_idx, stimuli_data in subject_data.items():
			
			# power features
			output = eeg.get_power_features(signal=np.transpose(stimuli_data), sampling_rate=128, size=2, overlap=0.5)
			theta = np.average(output['theta'],axis=0)
			alpha_low = np.average(output['alpha_low'],axis=0)
			alpha_high = np.average(output['alpha_high'],axis=0)
			beta = np.average(output['beta'],axis=0)
			gamma = np.average(output['gamma'],axis=0)
			sum = theta+alpha_low+alpha_high+beta+gamma
			feature_vector = np.concatenate((theta/sum,alpha_low/sum,alpha_high/sum,beta/sum,gamma/sum), axis=0)
			
			# plot power features
			#plt.figure()
			#plt.plot(output['ts'], np.sum(output['theta'][:,:],axis=1))
			#plt.plot(output['ts'], np.sum(output['alpha_low'][:,:],axis=1))
			#plt.plot(output['ts'], np.sum(output['alpha_high'][:,:],axis=1))
			#plt.plot(output['ts'], np.sum(output['beta'][:,:],axis=1))
			#plt.plot(output['ts'], np.sum(output['gamma'][:,:],axis=1))
			
			if valence_dichotomized[subject_idx,stimuli_idx]==-1:
				plt.figure()
				alpha = output['alpha_low'] + output['alpha_low']
				alpha_left = alpha[:,0] + alpha[:,1] + alpha[:,2]
				alpha_right = alpha[:,11] + alpha[:,12] + alpha[:,13]
				# plt.plot(output['ts'], alpha_left)
				plt.plot(output['ts'], alpha_right)
				plt.show()
				# sys.exit("Error message")
				# plt.savefig(os.path.join(outputdir, '-1_'+str(subject_idx)+'_'+str(stimuli_idx)+'.png'))
				plt.close()
			
			# wavelet features
			#feature_vector = []
			#for channel_idx in range(eeg.shape[0]):
				#feature_vector.append(pywt.wavedec(eeg,'db1',level=2))
			#print(len(feature_vector))
			
			subject_data_features[stimuli_idx] = feature_vector
			
		data_features[subject_idx] = subject_data_features
	
	#sys.exit("Error message")
				
	return data_features
		
# partitioning the dataset to training and testing
def partition(training_idx,testing_idx):
	
	# convert indices to keys
	keys = data_features.keys()
	training_keys = [keys[idx] for idx in training_idx]
	testing_keys = [keys[idx] for idx in testing_idx]
	
	# print the testing subject indices
	# print('\ttesting on: ',testing_idx)
	
	# seperate the data to train and test by subjects
	training_data = [data_features[key] for key in training_keys]
	training_arousal_dichotomized = arousal_dichotomized[training_keys,:]
	training_valence_dichotomized = valence_dichotomized[training_keys,:]
	training_personality_dichotomized = personality_dichotomized[training_keys,:]
	testing_data = [data_features[key] for key in testing_keys]
	testing_arousal_dichotomized = arousal_dichotomized[testing_keys,:]
	testing_valence_dichotomized = valence_dichotomized[testing_keys,:]
	testing_personality_dichotomized = personality_dichotomized[testing_keys,:]
	
	# dimensions with size 1 have been automatically offsetd. add the offsetd dimensions if needed
	if training_arousal_dichotomized.ndim==1:
		training_arousal_dichotomized = np.expand_dims(training_arousal_dichotomized,axis=0)
		training_valence_dichotomized = np.expand_dims(training_valence_dichotomized,axis=0)
		training_personality_dichotomized = np.expand_dims(training_personality_dichotomized,axis=0)
	if testing_arousal_dichotomized.ndim==1:
		testing_arousal_dichotomized = np.expand_dims(testing_arousal_dichotomized,axis=0)
		testing_valence_dichotomized = np.expand_dims(testing_valence_dichotomized,axis=0)
		testing_personality_dichotomized = np.expand_dims(testing_personality_dichotomized,axis=0)
		
	# pack the data
	training_packed = [training_data,training_arousal_dichotomized,training_valence_dichotomized,training_personality_dichotomized]
	testing_packed = [testing_data,testing_arousal_dichotomized,testing_valence_dichotomized,testing_personality_dichotomized]
	
	return training_packed, testing_packed
	
# utility function to report best score of tuning
def report(clf_name, results, score, n_top=1):

	candidates = np.flatnonzero(results['rank_test_score'] == 1)
	# print('validation accuracy: {0:.2f}, test accuracy: {1:.2f}'.format(results['mean_test_score'][candidates[0]],score))
	
	# for candidate in candidates:
		# print('\nClassifier: ', clf_name)
		# print('Best parameters: {0}'.format(results['params'][candidate]))
		# print('Best validation accuracy: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))

# classification
def classification(training_packed, testing_packed, n_components, clf_name):
	
	# unpack the data
	training_data = training_packed[0]
	training_arousal_dichotomized = training_packed[1]
	training_valence_dichotomized = training_packed[2]
	training_personality_dichotomized = training_packed[3]
	testing_data = testing_packed[0]
	testing_arousal_dichotomized = testing_packed[1]
	testing_valence_dichotomized = testing_packed[2]
	testing_personality_dichotomized = testing_packed[3]
	
	scores = []
	n_stimuli = 16
	if target=='arousal':
	
		# extract feature vectores and labels
		training_features = {key: [] for key in range(n_stimuli)}
		testing_features = {key: [] for key in range(n_stimuli)}
		training_labels = {key: [] for key in range(n_stimuli)}
		testing_labels = {key: [] for key in range(n_stimuli)}
		for subject_idx, subject_data in enumerate(training_data):
			for stimuli_idx, feature in subject_data.items():
				if training_arousal_dichotomized[subject_idx,stimuli_idx]!=0:
					training_features[stimuli_idx].append(feature)
					training_labels[stimuli_idx].append(training_arousal_dichotomized[subject_idx,stimuli_idx])
		for subject_idx, subject_data in enumerate(testing_data):
			for stimuli_idx, feature in subject_data.items():
				if testing_arousal_dichotomized[subject_idx,stimuli_idx]!=0:
					testing_features[stimuli_idx].append(feature)
					testing_labels[stimuli_idx].append(testing_arousal_dichotomized[subject_idx,stimuli_idx])
		training_features = {key: np.array(training_features[key]) for key in range(n_stimuli)}
		testing_features = {key: np.array(testing_features[key]) for key in range(n_stimuli)}
		training_labels = {key: np.array(training_labels[key]) for key in range(n_stimuli)}
		testing_labels = {key: np.array(testing_labels[key]) for key in range(n_stimuli)}
		
		for stimuli_idx in range(n_stimuli):
		
			# skip if no train or test data
			if training_labels[stimuli_idx].shape[0]==0 or testing_labels[stimuli_idx].shape[0]==0:
				score.append(np.nan)
				continue
			
			# run the CSP
			#csp = CSP(n_components=n_components,reg='ledoit_wolf',transform_into='average_power')
			#training_features = csp.fit_transform(training_eegs[stimuli_idx],training_labels[stimuli_idx])
			#testing_features = csp.transform(testing_eegs[stimuli_idx])
			
			# create the classifier objects
			classifiers = {
				'knn':KNeighborsClassifier(),
				'logistic':LogisticRegression(),
				'lda':LinearDiscriminantAnalysis(),
				'svm':SVC(),
				'tree':DecisionTreeClassifier(),
				'randomforest':RandomForestClassifier(),
				'extratrees':ExtraTreesClassifier(),
				'gradboost':GradientBoostingClassifier(),
				'adaboost':AdaBoostClassifier(),
				'mlp':MLPClassifier(),
				'ecoc':OutputCodeClassifier(SVC(C=2,kernel='linear',shrinking=True,class_weight='balanced'), code_size=2)}

			# specify parameters of the classifiers
			param_set = {}
			if clf_name=='knn': #89.9,90.8 'n_neighbors':17, 'p':1, 'weights':'distance'
				param_set.update({'n_neighbors':[1,9,13,17,25,50], 'p':[1,2,3,5], 'weights':['distance'], 'algorithm':['auto'], 'n_jobs':[3]})
			elif clf_name=='logistic': #94.4 'C':1, 'solver':'newton-cg'
				param_set.update({'C':[1,2,3,4], 'solver':['newton-cg'], 'class_weight':['balanced'], 'max_iter':[100]})
			elif clf_name=='lda': #94.9 'solver':'lsqr'
				param_set.update({'solver':['lsqr','eigen'], 'shrinkage':['auto']})
			elif clf_name=='svm': #95.3 'C':1, 'kernel':'linear'
				param_set.update({'C':[0.1,0.5,1,1.5,2,5], 'kernel':['linear'], 'shrinking':[True], 'probability':[False], 'class_weight':['balanced'], 'decision_function_shape':['ovr']})
			elif clf_name=='tree': #82.3 'max_depth':15
				param_set.update({'min_samples_leaf':[10,50,75,100], 'class_weight':['balanced'], 'presort':[True]})
			elif clf_name=='randomforest': #91.8 'n_estimators':300, 'min_samples_leaf':None, 'max_depth':25
				param_set.update({'n_estimators':[500,1000], 'max_features':[5,10,25], 'min_samples_leaf':[1,10,25] ,'max_depth':[None], 'bootstrap':[True], 'class_weight':['balanced'], 'oob_score':[False], 'n_jobs':[3]})
			elif clf_name=='extratrees': #92.8 'n_estimators':500, 'max_depth':50
				param_set.update({'n_estimators':[100,300], 'max_features':[None], 'min_samples_leaf':[1,10,50], 'max_depth':[None], 'bootstrap':[False], 'class_weight':['balanced'], 'oob_score':[False], 'n_jobs':[3]})
			elif clf_name=='gradboost': #92.3 'n_estimators':100, 'learning_rate':0.1, 'min_samples_leaf':50
				param_set.update({'n_estimators':[100], 'max_features':['auto'], 'learning_rate':[0.1], 'min_samples_leaf':[50]})
			elif clf_name=='adaboost': #57.9 'n_estimators':100, 'learning_rate':0.1
				param_set.update({'n_estimators':[100,500], 'learning_rate':[0.01,0.1]})
			elif clf_name=='mlp': #95.0 'hidden_layer_sizes':(50,), 'alpha':10, 'solver':'lbfgs'
				param_set.update({'hidden_layer_sizes':[(25,),(50,)], 'alpha':[1,5,10], 'solver':['adam']})
			elif clf_name=='ecoc':
				param_set.update({})
				
			# run grid search or randomized search
			classifier_search = GridSearchCV(classifiers[clf_name], param_grid=param_set, cv=2, n_jobs=3)
			
			classifier_search.fit(training_features[stimuli_idx], training_labels[stimuli_idx])
			score = classifier_search.score(testing_features[stimuli_idx], testing_labels[stimuli_idx])
			report(clf_name,classifier_search.cv_results_,score)
			
			# Printing the results
			# class_balance = np.mean(labels == labels[0])
			# class_balance = max(class_balance, 1. - class_balance)
			# print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),class_balance))

			# print the recognition accuracy
			# print('\taccuracy: {0:.3f}'.format(metrics.accuracy_score(testing_labels,predicted_labels)))
			
			scores.append(score)
	
	elif target=='personality':
		for personality_idx in range(n_personality):
		
			# extract labels
			training_labels = training_personality_dichotomized[:,personality_idx]
			testing_labels = testing_personality_dichotomized[:,personality_idx]
			
			# just keep confident samples, by personality measure
			training_offset_idx = []
			for subject_idx in range(training_labels.shape[0]):
				if training_labels[subject_idx]==0:
					training_offset_idx.append(subject_idx)
			corrected_training_data = list(training_data)
			for idx in sorted(training_offset_idx,reverse=True):
				del corrected_training_data[idx]
			corrected_training_labels = np.delete(training_labels,training_offset_idx,axis=0)
			testing_offset_idx = []
			for subject_idx in range(testing_labels.shape[0]):
				if testing_labels[subject_idx]==0:
					testing_offset_idx.append(subject_idx)
			corrected_testing_data = list(testing_data)
			for idx in sorted(testing_offset_idx,reverse=True):
				del corrected_testing_data[idx]
			corrected_testing_labels = np.delete(testing_labels,testing_offset_idx,axis=0)
			
			# skip if empty
			if (corrected_training_labels.size==0) or (corrected_testing_labels.size==0):
				scores.append(np.nan)
				continue
			
			# init the feature vectores
			training_features = np.zeros((len(corrected_training_data),0))
			testing_features = np.zeros((len(corrected_testing_data),0))
			for stimuli_idx in range(n_stimuli):
		
				# extract the stimuli data
				training_eegs = []
				for subject_data in corrected_training_data:
					training_eegs.append(subject_data[stimuli_idx])
				training_eegs = np.array(training_eegs)
				testing_eegs = []
				for subject_data in corrected_testing_data:
					testing_eegs.append(subject_data[stimuli_idx])
				testing_eegs = np.array(testing_eegs)
			
				# run the CSP
				csp = CSP(n_components=n_components,reg='ledoit_wolf',transform_into='average_power')
				training_features = np.append(training_features, csp.fit_transform(training_eegs,corrected_training_labels), axis=1)
				testing_features = np.append(testing_features, csp.transform(testing_eegs), axis=1)
				
			# create the classifier objects
			classifiers = {
				'knn':KNeighborsClassifier(),
				'logistic':LogisticRegression(),
				'lda':LinearDiscriminantAnalysis(),
				'svm':SVC(),
				'tree':DecisionTreeClassifier(),
				'randomforest':RandomForestClassifier(),
				'extratrees':ExtraTreesClassifier(),
				'gradboost':GradientBoostingClassifier(),
				'adaboost':AdaBoostClassifier(),
				'mlp':MLPClassifier(),
				'ecoc':OutputCodeClassifier(SVC(C=2,kernel='linear',shrinking=True,class_weight='balanced'), code_size=2)}

			# specify parameters of the classifiers
			param_set = {}
			if clf_name=='knn': #89.9,90.8 'n_neighbors':17, 'p':1, 'weights':'distance'
				param_set.update({'n_neighbors':[1,9,13,17,25,50], 'p':[1,2,3,5], 'weights':['distance'], 'algorithm':['auto'], 'n_jobs':[3]})
			elif clf_name=='logistic': #94.4 'C':1, 'solver':'newton-cg'
				param_set.update({'C':[1,2,3,4], 'solver':['newton-cg'], 'class_weight':['balanced'], 'max_iter':[100]})
			elif clf_name=='lda': #94.9 'solver':'lsqr'
				param_set.update({'solver':['lsqr','eigen'], 'shrinkage':['auto']})
			elif clf_name=='svm': #95.3 'C':1, 'kernel':'linear'
				param_set.update({'C':[0.1,0.5,1,1.5,2,5], 'kernel':['linear'], 'shrinking':[True], 'probability':[False], 'class_weight':['balanced'], 'decision_function_shape':['ovr']})
			elif clf_name=='tree': #82.3 'max_depth':15
				param_set.update({'min_samples_leaf':[10,50,75,100], 'class_weight':['balanced'], 'presort':[True]})
			elif clf_name=='randomforest': #91.8 'n_estimators':300, 'min_samples_leaf':None, 'max_depth':25
				param_set.update({'n_estimators':[500,1000], 'max_features':[5,10,25], 'min_samples_leaf':[1,10,25] ,'max_depth':[None], 'bootstrap':[True], 'class_weight':['balanced'], 'oob_score':[False], 'n_jobs':[3]})
			elif clf_name=='extratrees': #92.8 'n_estimators':500, 'max_depth':50
				param_set.update({'n_estimators':[100,300], 'max_features':[None], 'min_samples_leaf':[1,10,50], 'max_depth':[None], 'bootstrap':[False], 'class_weight':['balanced'], 'oob_score':[False], 'n_jobs':[3]})
			elif clf_name=='gradboost': #92.3 'n_estimators':100, 'learning_rate':0.1, 'min_samples_leaf':50
				param_set.update({'n_estimators':[100], 'max_features':['auto'], 'learning_rate':[0.1], 'min_samples_leaf':[50]})
			elif clf_name=='adaboost': #57.9 'n_estimators':100, 'learning_rate':0.1
				param_set.update({'n_estimators':[100,500], 'learning_rate':[0.01,0.1]})
			elif clf_name=='mlp': #95.0 'hidden_layer_sizes':(50,), 'alpha':10, 'solver':'lbfgs'
				param_set.update({'hidden_layer_sizes':[(25,),(50,)], 'alpha':[1,5,10], 'solver':['adam']})
			elif clf_name=='ecoc':
				param_set.update({})
				
			# run grid search or randomized search
			classifier_search = GridSearchCV(classifiers[clf_name], param_grid=param_set, cv=4, n_jobs=3)
			
			classifier_search.fit(training_features, corrected_training_labels)
			score = classifier_search.score(testing_features, corrected_testing_labels)
			report(clf_name,classifier_search.cv_results_,score)
			
			# Printing the results
			# class_balance = np.mean(labels == labels[0])
			# class_balance = max(class_balance, 1. - class_balance)
			# print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),class_balance))

			# print the recognition accuracy
			# print('\taccuracy: {0:.3f}'.format(metrics.accuracy_score(testing_labels,predicted_labels)))
			
			scores.append(score)
	
	return scores

# cross validation
def cross_validation():
	print('Cross Validation...')

	# for each csp_components
	for n_components in csp_components:
		print 'n_components: ', n_components
	
		# for each classifier
		for clf_name in clf_names:
			print 'clf_name: ', clf_name
	
			n_validations = 10
			tot_scores = []
			
			# run many times
			for test_idx in range(n_validations):
				print 'test_idx: ', test_idx
				
				mean_scores = []
				n_subjects = len(data_features)
				
				# leave one out
				if validation_method=='loso':
					indices = np.random.permutation(n_subjects)
					for subject_idx in range(n_subjects):
						training_idx = indices
						training_idx = np.delete(training_idx,subject_idx)
						testing_idx = indices[subject_idx]
							
						# classification
						training_packed, testing_packed = partition(training_idx,testing_idx)
						scores = classification(training_packed, testing_packed, n_components, clf_name)
						
						# append the scores to the total scores
						mean_scores.append(scores)
				
				# k-fold
				elif validation_method[1:5]=='fold':
					K = int(validation_method[0])
					indices = np.random.permutation(n_subjects)
					for pivot in range(K):
						if pivot<K-1:
							testing_idx = indices[int(pivot*math.ceil(n_subjects/K)):int((pivot+1)*math.ceil(n_subjects/K))]
							training_idx = np.copy(indices)
							training_idx = np.delete(training_idx, range(int(pivot*math.ceil(n_subjects/K)),int((pivot+1)*math.ceil(n_subjects/K))))
						else:
							testing_idx = indices[int(pivot*math.ceil(n_subjects/K)):]
							training_idx = indices[0:int(pivot*math.ceil(n_subjects/K))]
						
						# classification
						training_packed, testing_packed = partition(training_idx,testing_idx)
						scores = classification(training_packed, testing_packed, n_components, clf_name)	
						
						# append the scores to the total scores
						mean_scores.append(scores)
				
				mean_scores = np.array(mean_scores)
				mean_scores = np.nanmean(mean_scores, axis=0)
				tot_scores.append(mean_scores)
			
				print(mean_scores)
							
			tot_scores = np.array(tot_scores)
			tot_scores = np.mean(tot_scores, axis=0)
			
			print(tot_scores)
				
	return
	
# print the result and plot the confusion matrix
def result(testing_labels, predicted_labels):

	# class names
	classes = ['']

	# print the recognition accuracy
	print('Test accuracy: {0:.3f}'.format(metrics.accuracy_score(testing_labels, predicted_labels)))

	# calculate and normalize confusion matrix
	cnf_matrix = confusion_matrix(testing_labels, predicted_labels)
	cnf_matrix = cnf_matrix.astype('int')
	norm_cnf_matrix = np.copy(cnf_matrix)
	norm_cnf_matrix = norm_cnf_matrix.astype('float')
	for row in range( cnf_matrix.shape[0]):
		s = cnf_matrix[row, :].sum()
		if s > 0:
			for col in range( cnf_matrix.shape[0]):
				norm_cnf_matrix[row, col] = np.double(cnf_matrix[row, col]) / s

	# print confusion matrix
	np.set_printoptions(precision=2)
	# print('\nConfusion Matrix=\n', cnf_matrix, '\n', '\nNormalized Confusion Matrix=\n', norm_cnf_matrix, '\n')

	# save confusion matrix as text
	np.savetxt(os.path.join(outputdir, 'Confusion Matrix.txt'), cnf_matrix, delimiter='\t', fmt='%d')

	# plot confusion matrix
	plt.figure()
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, int(cnf_matrix[i, j]), horizontalalignment="center",
				 color="white" if cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Confusion Matrix.jpg'), bbox_inches='tight', dpi=300)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()

	# plot normalized confusion matrix
	plt.figure()
	plt.imshow(norm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = norm_cnf_matrix.max() / 2.
	for i, j in itertools.product(range(norm_cnf_matrix.shape[0]), range(norm_cnf_matrix.shape[1])):
		plt.text(j, i, float("{0:.2f}".format(norm_cnf_matrix[i, j])), horizontalalignment="center",
				 color="white" if norm_cnf_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(outputdir, 'Normalized Confusion Matrix.jpg'), bbox_inches='tight', dpi=600)
	plt.get_current_fig_manager().window.showMaximized()
	plt.show()

	return

# program main. following variables will be accesible within all functions without any need for passing.
if __name__ == '__main__':

	datasetdir, outputdir, dataset_name, feature_type, method, target, validation_method, csp_components, clf_names = init()

	asymmetry()
	
	# data, arousal_dichotomized, valence_dichotomized, personality_dichotomized = loading_dataset()
	
	# data_preprocessed = preprocessing()
	
	# data_features = feature_extraction()
	
	# cross_validation()
	
	# result(testing_labels, predicted_labels)
