import pandas as pd
from collections import defaultdict
import sys

import openXBOW

import numpy as np

import os
import re


sys.path.insert(0, '../../stage-1/overall_used_tools')
import requirements_check as rc
import collections
rc.check(sys, [pd,collections,np,os,re])


class BOWConverter:
	def __init__(self, path):

		self.midi_feat_path = path

	def sorter_func(self, string):
		''' NO NEED FOR DIRECT CALL '''


		#string = str(string)
		time_finder = re.search(r'\[\d+\]',string) # for functionals which can contain [number]
		# d+ as up to multiple digits
		time_finder2 = re.search(r'\_\d+$' , string) # for df_music_ex columns which can end on _number
		# $ as mark of the end position of the string we search through

		if time_finder:
			found_ele = str(time_finder.group())
			found_number = int(found_ele[1:-1])

		elif time_finder2:
			found_number = int(string.split('_')[-1])

		return found_number


	def codebook_generator(self, functionals2, csv_name): 
		''' NO NEED FOR DIRECT CALL '''

		codebook_df = pd.DataFrame() # collects all samples in one df

		# Generate dictionary with all columns with time independent keys:
		only_func_list = []
		time_interval_dict = defaultdict(list)
		for ele in functionals2.columns.tolist():

			time_finder = re.search(r'\[\d+\]' , ele)
			time_finder2 = re.search(r'\_\d+$' , ele)

			if time_finder:
			    modified_ele = ele.replace(str(time_finder.group()),'')
			    time_interval_dict[modified_ele].append(ele)
			    
			elif time_finder2:
			    modified_ele = ele.replace(str(time_finder2.group()),'')
			    time_interval_dict[modified_ele].append(ele)
			    
			else:
			    only_func_list.append(ele)
		  
		# bring list content into right time order:
		for k,v in time_interval_dict.items():
			v.sort(key=self.sorter_func)

		    
		# go through each sample song:
		for i in range(functionals2.shape[0]):
			one_sample = functionals2.take([i]) # should be same as .iloc[i]

			# Take sample/one file and create dict with same keys and values which are sorted after the principle of 
			# the time_interval_dict:
			sample_dict = defaultdict(list)

			for key,val_list in time_interval_dict.items():
				key_list = []
				for single_val in val_list:
					belonging_value = one_sample[single_val].iloc[0] # .iloc[0] gives us the entry value
					key_list.append(belonging_value)
				sample_dict[key] = key_list

			# create dataframe for one sample file where all time steps of each features are stored in the columns:
			sample_df = pd.DataFrame.from_dict(sample_dict, orient='index').T 
			# orient='index' gives us per dict key 
			# one row. Afterwards transpose dict such that per key one column.
			# only values in it which has to be aggreagted over time
			sample_df.insert(0, 'file_name', one_sample.index[0])

			# Add sample df to big df:
			codebook_df = codebook_df.append(sample_df) # add new rows
			# the codebook will have all files which are finally represented as bag of words

		# Replace nan-values with 0:
		codebook_df.fillna(0, inplace=True)


		# save codebook_df as .csv:
		codebook_df.to_csv(csv_name, index=False)

		return codebook_df, time_interval_dict

	def df_col_cleaning(self, snippets_padded_by_breaks=True): 
		''' CALL WHEN YOU HAVE DATAFRAME WITH LIST ENTRIES WHICH SHALL BECOME ELEMENTWISE ENTRIES. 
Parameter snippets_padded_by_breaks means that the track snippet is complete but it is surrounded by breaks. This is the case for the manual trimming of MIDI files to the content of short mp3-files. Therefore, features have to be irgnored then containing information about breaks and the duration of the full MIDI file. '''

		print("pre condition: we need column 'sample_id' at first position which describes the sample names")
		print("CALL WHEN YOU HAVE DATAFRAME WITH LIST ENTRIES WHICH SHALL BECOME ELEMENTWISE ENTRIES")

		midi_df = pd.read_csv(self.midi_feat_path) # one sample is missing

		excluding_list = ['Basic Pitch Histogram', 'Pitch Class Histogram', 'Folded Fifths Pitch Class Histogram', 'Melodic Interval Histogram', 'Vertical Interval Histogram', 'Wrapped Vertical Interval Histogram', 'Chord Type Histogram', 'Initial Time Signature', 'Rhythmic Value Histogram', 'Rhythmic Value Median Run Lengths Histogram', 'Rhythmic Value Variability in Run Lengths Histogram', 'Beat Histogram Tempo Standardized', 'Beat Histogram', 'Pitched Instruments Present', 'Unpitched Instruments Present', 'Note Prevalence of Pitched Instruments', 'Note Prevalence of Unpitched Instruments', 'Time Prevalence of Pitched Instruments', 'Unnamed: 0', 'source_id']
		if snippets_padded_by_breaks:
			excluding_list2 = ['Duration in Seconds', 'Complete Rests Fraction', 'Partial Rests Fraction', 'Average Rest Fraction Across Voices', 'Longest Complete Rest', 'Longest Partial Rest', 'Mean Complete Rest Duration', 'Mean Partial Rest Duration', 'Median Complete Rest Duration', 'Median Partial Rest Duration', 'Variability of Complete Rest Durations', 'Variability of Partial Rest Durations', 'Variability Across Voices of Combined Rests']
			excluding_list.extend(excluding_list2)

		for single_col in excluding_list:
			try: 
				del midi_df[single_col]
			except:
				pass

		try:
			midi_df['sample_id'] = midi_df['sample_id'].str.replace(r'_merged', '')
		except:
			pass


		midi_df['sample_id'] = pd.DataFrame("'" + midi_df['sample_id'] + ".wav'")

		midi_df.set_index('sample_id', inplace=True) 
		midi_df.index.names = ['name']
		
		midi_df.dropna(inplace=True)

		
		return midi_df


	def bow_executer(self, midi_df_mod, lld=False): 
		'''	
CALL IT TO START WHOLE PROCEDURE, INPUT A ALREADY FITTING DATAFRAME WHICH HAS COLUMN TITLES WHERE A FEATURE IS TIME FRAMED BY: FEATURE[t]ONGINGDESCRIPTION OR FEATURE_t WHERE 't' DESCRIBES CURRENT TIME POINT '''

		print("CALL IT TO START WHOLE PROCEDURE, INPUT A ALREADY FITTING DATAFRAME WHICH HAS COLUMN TITLES WHERE A FEATURE IS TIME FRAMED BY: FEATURE[t]ONGINGDESCRIPTION OR FEATURE_t WHERE 't' DESCRIBES CURRENT TIME POINT ")
		my_dir = os.getcwd()
		csv_name = my_dir + '/RESULTS/' + 'codebook' + '.csv'

		os.makedirs(my_dir + '/RESULTS/', exist_ok=True)

		codebook_df, time_interval_dict = self.codebook_generator(midi_df_mod, csv_name)

		first_sample_str = codebook_df['file_name'].iloc[0]
		o = codebook_df['file_name'] == first_sample_str
		if lld:
			idx_longest_vec = str(len(np.where(o==True)[0]))
		else: # functionals
			idx_longest_vec = str(int(codebook_df.file_name.where(o, first_sample_str).index[-1]))

		copy_sys = sys.path.copy()
		copy_sys.reverse()
		find_ind = copy_sys.index('../../stage-1/overall_used_tools') 
		start_ind = len(sys.path) - find_ind


		# use saved codebook for openXBOW:
		os.system('java -jar ' + sys.path[start_ind] + '/openXBOW/openXBOW.jar -i ' + csv_name + ' -B codebook -size ' + idx_longest_vec + ' -log -a 10 -standardizeInput')
		# -size 250, gives us the number of words to be generated
		# -i as input, -o as output, -l as label file (can be ignored)
		# -B as training request, -b as test request 

		# Now extract 'words':
		options   = ' -writeName -csvHeader'
		os.system('java -jar ' + sys.path[start_ind] + '/openXBOW/openXBOW.jar -i ' + csv_name + ' -o ' + my_dir + '/RESULTS/BoW_all.csv' + options + ' -b codebook')

		# Have a look at what we extracted and saved:
		bow_df = pd.read_csv(my_dir + '/RESULTS/BoW_all.csv', sep=';', index_col=0)
		print('We have extracted', len(bow_df.columns), 'BoAW-features from', len(bow_df), 'files')

		return bow_df, time_interval_dict


