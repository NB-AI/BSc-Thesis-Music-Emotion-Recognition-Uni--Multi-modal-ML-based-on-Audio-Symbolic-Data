# for jSymbolic csv convertion:
import pandas as pd
import glob
import os
import numpy as np

# for Panda et al 2018 feature extraction:
import mido
from mido import Message, MidiFile, MidiTrack


# Getting the used versions of imports:
import sys
sys.path.insert(0, '../../stage-1/overall_used_tools')
import requirements_check as rc
rc.check(sys, [np,pd,mido])



def basic_tools_panda(directory_path_with_tracks, file_name, start_time=float('inf'),end_time=float('-inf')):
    """ Refering to Panda et al 2018. Get basic information out of the midi to later find out the 
    features, e.g. notes and their duration are here collected.
    start_time and end_time are used when we want to snip out only a part of the midi file to generate a smaller midi file."""
    
    # Loading the midi:
    mid = MidiFile(f'{directory_path_with_tracks}/{file_name}.mid', clip=True)


    # we create a midi when start and end time given:
    if start_time!=float('inf') and end_time!=float('-inf'): # when we want to cut out snippet
        # then we want to create a new midi file which collects only information within the time frame of start_time and end_time
        mid_create = MidiFile(f'{directory_path_with_tracks}/{file_name}.mid', clip=True) # we want to create the midi file mid_create which stores
        # the same information as the original midi file mid but only a snippet of it. Therefore, we simply make a copy of mid, namely mid_create,
        # whose tracks are overwritten by getting them empty. Then we can fill them up with the notes that are in the relevant time frame.
#MidiFile(clip=mid.clip,type=mid.type,ticks_per_beat=mid.ticks_per_beat,charset=mid.charset,debug=mid.debug)
        mid_create.tracks = [MidiTrack(name=i.name) for i in mid.tracks] # use the same names as in the original midi for the single tracks, furthermore,
        # with different names we dont create copies of MidiTrack which will be then the same variable

        create_mid = True
        creation_success = False

    else:
        create_mid = False
        creation_success = False

    # When these arrays used by one midi but the next doesn't need them, then reset them, such that no errors
    # occur:
    note_counter_ar = None
    ind_counter_ar = None
    ind_counter_ar_sal = None
    saved_note_ar = None
    note_ar = None
    time_passed = None
    old_note_on_ar = None

    time_passed_ar = np.zeros((len(mid.tracks),1))-1
    time_passed_ar_end = np.zeros((len(mid.tracks),1))

    s = np.array([[0]*len(mid.tracks)]).reshape((len(mid.tracks),1))-1
    note_collector = np.zeros((len(mid.tracks),1))-1
    nd_short = np.zeros((len(mid.tracks),1))-1
    pause_collector = np.zeros((len(mid.tracks),1))-1
    note_collector_all = np.zeros((len(mid.tracks),1))-1
    time_passed_ar_all = np.zeros((len(mid.tracks),1))-1
    track_names = np.asarray([])
    channel_obj = False # relevant for one track multi-channel midi files


    SAL = np.zeros((len(mid.tracks),1))-1
    CD = np.zeros((len(mid.tracks),1)).astype('str')
    s_CD = np.array([[0]*len(mid.tracks)]).reshape((len(mid.tracks),1)).astype('float').astype('str')
    nd = np.zeros((len(mid.tracks),1))-1

    length_mid_tracks = len(mid.tracks)

    if length_mid_tracks==1:
        one_tracker = True
    else:
        one_tracker = False
   
    for ind1, single_track in enumerate(mid.tracks): # look at each voice

        time_passed = 0 # for each track we start new time counting because tracks are independent of each other

        if length_mid_tracks>1:
            saved_note = -1
            ind_counter = -1
            ind_counter_sal = -1
            old_note_on = False       
            time_passed = 0 # can exist as an arr or as variable here
            note_counter = -1
            note = -2


        track_names = np.append(track_names, single_track.name)
        
        first_time_touching_time_interval = True

        sal_key_list = [] # store message info here to reuse for SAL
        sal_val_list = []

        for ind2, single_message in enumerate(single_track):


            time_passed += single_message.time

            ############## initializing the case that we have a one channeler, also possible when midi has several tracks but only one of them filled with notes ################
            if length_mid_tracks==1 and channel_obj==False: # called only once for one-trackers

                # then we have one track with several instruments/voices as whole

                time_passed_ar = np.zeros((16,1))-1
                time_passed_ar_end = np.zeros((16,1))

                s = np.array([[0]*16]).reshape((16,1))-1

                note_collector = np.zeros((16,1))-1
                nd_short = np.zeros((16,1))-1
                pause_collector = np.zeros((16,1))-1
                note_collector_all = np.zeros((16,1))-1
                time_passed_ar_all = np.zeros((16,1))-1

                SAL = np.zeros((16,1))-1
                CD = np.zeros((16,1)).astype('str')
                s_CD = np.array([[0]*16]).reshape((16,1)).astype('float').astype('str')
                
                nd = np.zeros((16,1))-1
                
                channel_obj = True

                note_counter_ar = np.zeros((16))-1
                ind_counter_ar = np.zeros((16))-1
                ind_counter_ar_sal = np.zeros((16))-1
                saved_note_ar = np.zeros((16))-1
                note_ar = np.zeros((16))-2
                old_note_on_ar = np.zeros((16)).astype(bool)
            ###########################################################

            
            if single_message.is_meta or single_message.type == 'sysex':

                if create_mid == True and one_tracker==False:
                    mid_create.tracks[ind1].append(single_message)

                elif create_mid == True and one_tracker==True:
                    mid_create.tracks[0].append(single_message)

            
            else: # no meta or sysex message

                try:
                    volumne = single_message.velocity   

                except:
                    volumne = -1

                type_ = single_message.type
                channel = single_message.channel

                if 'note_on' == type_ and volumne==0: 

                    type_ = 'note_off'                        



                ################## note_collector_all/time_passed_ar_all by looking at note_on #########################################
                try: # handling the case that we have note_on case (excluding note_on.velocity=0). collecting all note_ons, storing note when currently no note saved

                    try:
                        note_ar[channel] = single_message.note
                        note = note_ar[channel] 
                    except:
                        note = single_message.note

                    if type_ == 'note_on':


                        try:
                            note_counter_ar[channel] += 1
                            note_counter = int(note_counter_ar[channel])

                        except:
                            note_counter += 1

                        sal_key_list.append((channel,single_message.note))
                        sal_val_list.append(volumne)

                        if channel_obj == True:
                            #note_counter = channel
                            ind1 = channel

                        try:
                            note_collector_all[ind1, note_counter] = note

                            time_passed_ar_all[ind1,note_counter] += time_passed # before: was like before time_passed+=time_amount

                        except:
                            note_collector_all = np.append(note_collector_all, s, axis=1)
                            note_collector_all[ind1, note_counter] = note

                            time_passed_ar_all = np.append(time_passed_ar_all, s, axis=1) #expend array when ind beyond range
                            time_passed_ar_all[ind1,note_counter] += time_passed

                    try:
                        a = saved_note_ar[channel] 
                    except:
                        a = saved_note

                    if type_ == 'note_on' and a == -1: # when no note stored currently, then store the current note

                        try:
                            saved_note_ar[channel] = note
                            saved_note = int(saved_note_ar[channel])

                        except:
                            saved_note = note


                except:
                    pass

                ###########################################################


                ######################### nd_short having a look at note off when note is still active | note_collector/pause_collector/time_passed_ar looking at current note on when no older note is active ###############################
                try:
                    d = old_note_on_ar[channel]
                    e = saved_note_ar[channel]

                except:
                    d = old_note_on
                    e = saved_note

                if type_ == 'note_off' and d == True: #new
                        try:
                            nd_short[ind1,ind_counter] = single_message.time

                        except:
                            nd_short = np.append(nd_short, s, axis=1)
                            nd_short[ind1,ind_counter] = single_message.time


                if type_ == 'note_on' and d == False and e != -1: #  (excluding note_on.velocity=0), happens when note already stored and no older note active, i.e. 
                    # when note1 on; note2 on, note1 off --> note2 is ignored here.
                    # makes sense for: pause_collector because note1 on; note2 on time=40/0 --> influencing if notes following each other/are simultaneously. So or so,  
                    # there will be no pause inbetween notes generated. pause only generated for note2 when: note1 on; note1 off; note2 on time>0
                    # NO sense for note_collector: same example as above, with time_passed which counts higher for each message the notes follow each other meaning we 
                    # should use note_collector_all


                    if channel_obj == True:
                        ind1 = channel
                        ind_counter_ar[channel] += 1
                        ind_counter = int(ind_counter_ar[channel])
                        
                        note = saved_note_ar[channel]                            
                        saved_note = int(saved_note_ar[channel])

                        old_note_on_ar[channel] = True
                    else:
                        ind_counter += 1
                        note = saved_note

                        old_note_on = True

                    try:
                        time_passed_ar[ind1,ind_counter] += time_passed
                        note_collector[ind1,ind_counter] = note
                        pause_collector[ind1,ind_counter] = single_message.time 

                    except:
                        time_passed_ar = np.append(time_passed_ar, s, axis=1) # expand array when ind beyond range
                        time_passed_ar[ind1,ind_counter] += time_passed

                        note_collector = np.append(note_collector, s, axis=1)
                        note_collector[ind1,ind_counter] = note

                        pause_collector = np.append(pause_collector, s, axis=1)
                        pause_collector[ind1,ind_counter] = single_message.time 

                elif type_ == 'note_on':

                    try:
                        old_note_on_ar[channel] = True

                    except:
                        old_note_on = True
                ###########################################################


                ######################### SAL/CD/nd using note_off messages ##################################
                if type_ == 'note_off': 

                    if (channel, single_message.note) in sal_key_list:

                        position = sal_key_list.index((channel, single_message.note)) # gives the first position fulifilling condition/oldest stored note
                       
                        sal_stored_velocity = sal_val_list[position] 
                        sal_key_list.pop(position) #.remove((channel, single_message.note))
                        sal_val_list.pop(position) # remove per index

                        

                        if channel_obj == True: # we have one-tracker then
                            ind1 = channel
                            old_note_on_ar[channel] = False

                            saved_note_ar[channel] = -1
                            saved_note = int(saved_note_ar[channel])

                        else:
                             saved_note = -1

                             old_note_on = False

                        if single_message.velocity == 0:
                            sal_median = sal_stored_velocity

                        else:
                            sal_median = (sal_stored_velocity + single_message.velocity)/2


                        # considered for CD:
                        diff = single_message.velocity - sal_stored_velocity
                        if abs(diff) >= abs(sal_median/10*2) and single_message.velocity>0: # the threshold
                            if diff > 0:  
                                cd_to_add = 'c'
                            else:
                                cd_to_add = 'd'
                        else:
                            cd_to_add = 'n'

                        if channel_obj == True:
                            ind1 = channel
                            ind_counter_ar_sal[channel] += 1
                            ind_counter_sal = int(ind_counter_ar_sal[channel])

                        else:
                            ind_counter_sal += 1


                        try:
                            SAL[ind1,ind_counter_sal] = sal_median
                            CD[ind1,ind_counter_sal] = cd_to_add
                            nd[ind1,ind_counter_sal ] = single_message.time

                        except:

                            SAL = np.append(SAL, s, axis=1)
                            SAL[ind1,ind_counter_sal] = sal_median

                            CD = np.append(CD, s_CD, axis=1)
                            CD[ind1,ind_counter_sal ] = cd_to_add
        
                            nd = np.append(nd, s, axis=1)
                            nd[ind1,ind_counter_sal ] = single_message.time

                        try:
                            time_passed_ar_end[ind1,ind_counter_sal] += time_passed #(time_passed + time_amount)

                        except:
                            time_passed_ar_end = np.append(time_passed_ar_end, s, axis=1)
                            time_passed_ar_end[ind1,ind_counter_sal] += time_passed #(time_passed + time_amount)

                else:
                     pass
                ###########################################################

                ###################### creation midi snippet #####################################
                if create_mid == True:
                    if one_tracker == True:
                        indexer = 0
                    else: # we several tracks then
                        indexer = ind1

                    if (time_passed>=start_time and time_passed<=end_time) or single_message.type != 'note_on': 
                         mid_create.tracks[indexer].append(single_message) 
                         if single_message.type == 'note_on' and single_message.velocity > 0:
                             creation_success = True

                    elif type_ =='note_on': # note_on-s beyond the wanted time frame get their velocity set to 0 s.t. they are not more than a note_off/invisible

                         single_message.velocity=0 # also mid is influenced here
                         mid_create.tracks[indexer].append(single_message)
                ###########################################################

        if len(set(note_collector[ind1])) == 1: # e. g. when midi has two tracks but only one of them filled
            # with notes; so far no notes collected, we can skip that track
            length_mid_tracks -= 1 # When we reach length_mid_tracks = 1 it is treated like one channel one
            # line of music in the music sheet
            if length_mid_tracks == 1:
                one_tracker = True
            


    #################### outside of loops #######################################
    if nd.shape[1]<pause_collector.shape[1]: # the last 'note_off' for a 'note_on'
        # is missing, especially for the track 'P_AmyWinehouse_BackToB_accompaniment.mid'
        pause_collector = pause_collector[:,:-1]
        note_collector = note_collector[:,:-1]
        nd_short = nd_short[:,:-1]
        time_passed_ar = time_passed_ar[:,:-1]    
        print('here')


    if create_mid==True:

        store_direct = 'GeneratedMIDISnippets_webscraping'
        os.makedirs(store_direct, exist_ok=True)

        if creation_success == True:

            # check out the file such that no error message later when trying to store:
            if mid_create.type==0 and len(mid_create.tracks)!=1: # type 0 midi files only have one track filled with notes
                mid_create.type=1
                '''
                keep_ind = np.where(nd[:,0]!=-1)[0] # remove all tracks where no note on observed

                new_track_list=[]
                for d in keep_ind:
                    new_track_list.append(mid_create.tracks[d])

                mid_create.tracks = new_track_list
                # the length of tracks should be now not longer than 1        
                '''

            mid_create.save(f'{store_direct}/{file_name}.mid')
        else: # when no midi data collected store original midi

            mid2 = MidiFile(f'{directory_path_with_tracks}/{file_name}.mid', clip=True)
            # check out the file such that no error message later when trying to store:
            if mid2.type==0 and len(mid2.tracks)!=1:
                mid2.type = 1
                '''
                keep_ind = np.where(nd[:,0]!=-1)[0] # remove all tracks where no note on observed

                new_track_list=[]
                for d in keep_ind:
                    new_track_list.append(mid.tracks[d])

                mid.tracks = new_track_list
                # the length of tracks should be now not longer than 1        
                '''
            mid2.save(f'{store_direct}/{file_name}.mid')
    ###########################################################
    print('information extraction (and midi snippet generation) done')
    return pause_collector, nd, note_collector, note_collector_all,SAL,CD,time_passed_ar, nd_short, creation_success



def channel_mask(note_collector):
    ''' Only have finally a look at channels/voices where notes are played, i.e. not only Metadata is 
    stored '''
    # Have a look which tracks/voices simply have no notes/are empty:
    note_collector_track_watch = np.copy(note_collector)
    note_collector_track_watch[np.where(note_collector_track_watch==-1)] = 0
    summed_over_channels = np.sum(note_collector_track_watch,axis=1)
    mask_over_channels = np.where(summed_over_channels>0)
    return mask_over_channels # getting a mask to lay over the outcoming feature arrays. Only the arrays with notes in
    # it are concerned
