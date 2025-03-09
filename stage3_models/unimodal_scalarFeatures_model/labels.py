import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as pyp


# Getting the used versions of imports:
import sys
sys.path.insert(0, '../../stage-1/overall_used_tools')
import requirements_check as rc

import matplotlib, collections
rc.check(sys, [pd, collections, np, matplotlib])


def categorizer(sublimity, vitality, unease, median_mode=True):

    c0,c1,c2,c3=0,0,0,0 # counters how often samples have 0/1/2/3 out of 3 possible labels marked as 1.

    label_sample_list = np.asarray([[x,y,z] for x,y,z in zip(sublimity,vitality,unease)])

    for ind,(a,b,c) in enumerate(label_sample_list):
	    

        if median_mode:
            biggest_place = np.argmax([a-np.percentile(sublimity,50), b-np.percentile(vitality,50), c-np.percentile(unease,50)])
            # have a look in which label value the sample overcomes the relating medians the most. This is taken as
            # the label of the sample.

            quartile_size = 65#75#68#65
            big_mask = np.asarray([a>=np.percentile(sublimity,quartile_size), b>=np.percentile(vitality,quartile_size), c>=np.percentile(unease,quartile_size)])
            # when a label value is bigger than the 3. quartile then the samples gets that label marked is given (1).
            # Here is the chance that a sample is positive in more than only one label.
        else:
            biggest_place = np.argmax([a-sublimity.mean(), b-vitality.mean(), c-unease.mean()])

            emo_threshold_sublim = sublimity.mean() + sublimity.std()
            emo_threshold_vital = vitality.mean() + vitality.std()
            emo_threshold_unease = unease.mean() + unease.std()

            big_mask = np.asarray([a>=emo_threshold_sublim, b>=emo_threshold_vital, c>=emo_threshold_unease])
        
        class_ar = np.zeros(3)
        class_ar[big_mask] = 1
        class_ar[biggest_place] = 1

        label_sample_list[ind] = class_ar
	    
        length = len(np.where(class_ar==1)[0])
        if length==3:
            c3+=1 # should not happen often that a music snippet transports all emotions
        elif length==2:
            c2+=1
        elif length==1:
            c1+=1
        else:
            c0+=1 # should not happen that a sample has no positive class; when music plays it should lead
            # to any emotion in human
    label_sample_list = label_sample_list.astype('int')
                               
    len_ar = len(label_sample_list)

    print('proportion of amount of samples (out of 370) with... ...zero ...one ...two ...three labels:')
    print(c0/len_ar, c1/len_ar, c2/len_ar, c3/len_ar)

    return label_sample_list, (c0,c1,c2,c3)

def visualizer(sublimity_ar, vitality_ar, unease_ar,median_mode=True):
    print('Put in regressive values!')

    label_sample_list, (c0,c1,c2,c3) = categorizer(sublimity_ar, vitality_ar, unease_ar,median_mode)

    # visualize big labels together:
    label_list = label_sample_list.tolist()
    count_list = []
    label_poss = [[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[0, 1, 1],[1, 0, 1],[1, 1, 1]]
    for i in label_poss:
        count_list.append(label_list.count(i))

    pyp.bar([i for i in range(len(count_list))],count_list, color="aqua");
    pyp.title('Label distribution concerning all sample labels at once')
    pyp.xlabel('possible label arrangements')
    pyp.ylabel('absolute frequency')
    pyp.xticks(np.arange(len(label_poss)), [str(i) for i in label_poss] )
    pyp.show()

    # visualize the frequency of the individual labels:
    pyp.bar([0,1,2,3],[c0,c1,c2,c3], color="aqua");
    pyp.title('Amount of present labels per samples')
    pyp.ylabel('absolute frequency')
    pyp.xlabel('number')
    pyp.xticks(np.arange(4), ['0', '1', '2', '3'] )
    pyp.show()

    # visualize the distribution WITHIN the individual labels:
    pyp.hist(sublimity_ar, color="aqua")
    pyp.title("Sublimity label distribution histogram")
    pyp.ylabel("absolute frequency")
    pyp.xlabel("value bins")
    pyp.show()

    pyp.boxplot(sublimity_ar)
    pyp.title("Sublimity label distribution boxplot")
    pyp.ylabel("value bins")
    pyp.show()

    pyp.hist(vitality_ar, color="aqua") 
    pyp.title("Vitality label distribution histogram")
    pyp.ylabel("absolute frequency")
    pyp.xlabel("value bins")
    pyp.show()

    pyp.boxplot(vitality_ar)
    pyp.title("Vitality label distribution boxplot")
    pyp.ylabel("value bins")
    pyp.show()

    pyp.hist(unease_ar, color="aqua")
    pyp.title("Unease label distribution histogram")
    pyp.ylabel("absolute frequency")
    pyp.xlabel("value bins")
    pyp.show()

    pyp.boxplot(unease_ar)
    pyp.title("Unease label distribution boxplot")
    pyp.ylabel("value bins")
    pyp.show()

    return

def get_categorical_labels(path='../../stage0_provided_information/gems-emotion-tags-main/data/GEMS_songs_overview.csv', median_mode=True):

    # Import the dataset containing all the labels:
    small = pd.read_csv(path, skiprows=1) # skiprows: keep the columns headers
    small = small.dropna(how='all') # remove all rows only containing nan
    label_df = small[['MP3-Code','sublimity','vitality','unease']]

    # Adapt the label_df to the datasets:
    label_df['MP3-Code']= label_df['MP3-Code'] + '_accompaniment' # add to each cell a string in a certain column
    label_df.rename(columns={'MP3-Code':'sample_id'},inplace=True) # rename column
    print("We import a csv file where we need the columns 'MP3-Code','sublimity','vitality','unease'. 'MP3-Code' describes the ID of each sample.")

    label_df.dropna(inplace=True)


    # convert labels into categorical:

    sublim = label_df.sublimity.to_numpy()
    vital = label_df.vitality.to_numpy()
    unease = label_df.unease.to_numpy()
    #print(sublim.mean(),vital.mean(),unease.mean())
    #print(sublim.std(),vital.std(),unease.std())

    label_sample_list, _ = categorizer(sublim, vital, unease, median_mode=median_mode)

    label_df['final_label'] = label_sample_list.tolist()


    del label_df['sublimity']
    del label_df['vitality']
    del label_df['unease']
    label_df['sample_id'] = "'" + label_df['sample_id'] + ".wav'"
    label_df.set_index('sample_id', inplace=True)
    label_df.index.names = ['name']

    return label_df


def get_regressive_labels(path='../../stage0_provided_information/gems-emotion-tags-main/data/GEMS_songs_overview.csv'):

    # Import the dataset containing all the labels:
    small = pd.read_csv(path, skiprows=1) # skiprows: keep the columns headers
    small = small.dropna(how='all') # remove all rows only containing nan
    label_df = small[['MP3-Code','sublimity','vitality','unease']]

    # Adapt the label_df to the datasets:
    label_df['MP3-Code']= label_df['MP3-Code'] + '_accompaniment' # add to each cell a string in a certain column
    label_df.rename(columns={'MP3-Code':'sample_id'},inplace=True) # rename column
    print("We import a csv file where we need the columns 'MP3-Code','sublimity','vitality','unease'. 'MP3-Code' describes the ID of each sample.")

    label_df.dropna(inplace=True)

    # convert labels into categorical:

    used_df = label_df  

    sublim = used_df.sublimity.to_numpy()
    vital = used_df.vitality.to_numpy()
    unease = used_df.unease.to_numpy()

    # to normalize the labels:
    sublim_mean = sublim.mean()
    vital_mean = vital.mean()
    unease_mean = unease.mean()

    sublim_std = sublim.std()
    vital_std = vital.std()
    unease_std = unease.std()

    label_sample_list = np.asarray([[(x-sublim_mean)/sublim_std,(y-vital_mean)/vital_std,(z-unease_mean)/unease_std] for x,y,z in zip(sublim,vital,unease)])


    label_df['final_label'] = label_sample_list.tolist()

    del label_df['sublimity']
    del label_df['vitality']
    del label_df['unease']

    label_df['sample_id'] = "'" + label_df['sample_id'] + ".wav'"
    label_df.set_index('sample_id', inplace=True)
    label_df.index.names = ['name']

    return label_df
