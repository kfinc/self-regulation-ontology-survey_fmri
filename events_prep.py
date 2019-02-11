
import pandas as pd
import numpy as np


def get_items_order():
    
    """Function which returns dictionary with ordering id (Q01-Q40) assigned to each question. 
    This dictionary can be further used to map all quesion to their unique (template) order, therefore, to obtain the same order of beta vales for each person
    Author: Karolina Finc
    """
    

    grit_items = [
        'New ideas and projects sometimes distract me from previous ones.',
        'Setbacks don\'t discourage me.',
        'I have been obsessed with a certain idea or project for a short time but later lost interest.',
        'I am a hard worker.',
        'I often set a goal but later choose to pursue a different one.',
        'I have difficulty maintaining my focus on projects that take more than a few months to complete.',
        'I finish whatever I begin.',
        'I am diligent.'
    ]

    brief_items = [
        'I am good at resisting temptation.',
        'I have a hard time breaking bad habits.',
        'I am lazy.',
        'I say inappropriate things.',
        'I do certain things that are bad for me, if they are fun.',
        'I refuse things that are bad for me.',
        'I wish I had more self-discipline.',
        'People would say that I have iron self-discipline.',
        'Pleasure and fun sometimes keep me from getting work done.',
        'I have trouble concentrating.',
        'I am able to work effectively toward long-term goals.',
        'Sometimes I can\'t stop myself from doing something, even if I know it is wrong.',
        'I often act without thinking through all the alternatives.'
     ]

    future_time_items = [
        'Many opportunities await me in the future.',
        'I expect that I will set many new goals in the future.',
        'My future is filled with possibilities.',
        'Most of my life lies ahead of me.',
        'My future seems infinite to me.',
        'I could do anything I want in the future.',
        'There is plenty of time left in my life to make new plans.',
        'I have the sense that time is running out.',
        'There are only limited possibilities in my future.',
        'As I get older, I begin to experience time as limited.'
     ]


    upps_items = [
        "Sometimes when I feel bad, I can't seem to stop what I am doing even though it is making me feel worse.",
        'Others would say I make bad choices when I am extremely happy about something.',
        'When I get really happy about something, I tend to do things that can have bad consequences.',
        'When overjoyed, I feel like I cant stop myself from going overboard.',
        'When I am really excited, I tend not to think of the consequences of my actions.',
        'I tend to act without thinking when I am really excited.'
    ]
    
    impulse_venture_items = [
        'Do you welcome new and exciting experiences and sensations even if they are a little frightening and unconventional?',
        'Do you sometimes like doing things that are a bit frightening?',
        'Would you enjoy the sensation of skiing very fast down a high mountain slope?'
    ]
    
    item_text = grit_items + brief_items + future_time_items + upps_items + impulse_venture_items
    item_id = ['Q{:0>2d}'.format(i+1) for i in range(len(item_text))]
    item_id_map = dict(zip(item_text, item_id))

    return item_id_map


def get_timing_correction(filey, TR=680, n_TRs=14):
    
    """Function to correct processing of a few problematic files need to change time_elapsed to reflect the fact that fmri triggers were sent outto quickly (at 8 times the rate), 
    thus starting the scan 14 TRs early. Those 14 TRs of data therefore need to be thrown out, which is accomplished by setting the "0" of the scan 14 TRs later
    Author: Ian Eisenberg
    
    """
    problematic_files = ['s568_MotorStop.csv', 's568_Stroop.csv',
                         's568_SurveyMedley.csv', 's568_DPX.csv',
                         's568_Discount.csv',
                         's556_MotorStop.csv', 's556_Stroop.csv',
                         's556_SurveyMedley.csv', 's556_DPX.csv',
                         's556_Discount.csv',
                         's561_WATT.csv', 's561_ANT.csv',
                         's561_TwoByTwo.csv', 's561_CCT.csv',
                         's561_StopSignal.csv',]
    tr_correction = TR * n_TRs
    if filey in problematic_files:
        return tr_correction
    else:
        return 0

def get_drop_columns(df, columns=None, use_default=True):
    """
    Function which helps to clean unncecessary columns
    Author: Ian Eisenberg
    """
    default_cols = ['block_duration', 'correct_response', 'exp_stage',
                    'feedback_duration', 'possible_responses',
                   'rt', 'stim_duration', 'text', 'time_elapsed',
                   'timing_post_trial', 'trial_num']
    drop_columns = []
    if columns is not None:
        drop_columns = columns
    if use_default == True:
        drop_columns = set(default_cols) | set(drop_columns)
    drop_columns = set(df.columns) & set(drop_columns)
    return drop_columns

def get_junk_trials(df):
    """
    Function which helps to identify junk trials
    Author: Ian Eisenberg
    """
    junk = pd.Series(False, df.index)
    if 'correct' in df.columns:
        junk = np.logical_or(junk,np.logical_not(df.correct))
    if 'rt' in df.columns:
        junk = np.logical_or(junk,df.rt < 50)
    return junk

def get_movement_times(df):
    """
    Time elapsed is evaluated at the end of a trial, so we have to subtract
    timing post trial and the entire block duration to get the time when
    the trial started. Then add the reaction time to get the time of movement
    Author: Ian Eisenberg
    """
    trial_time = df.time_elapsed - df.block_duration - df.timing_post_trial + \
                 df.rt
    return trial_time

def get_trial_times(df):
    """
    Time elapsed is evaluated at the end of a trial, so we have to subtract
    timing post trial and the entire block duration to get the time when
    the trial started.
    Author: Ian Eisenberg
    """
    trial_time = df.time_elapsed - df.block_duration - df.timing_post_trial
    return trial_time

def create_survey_event(df, duration=None):
    """
    Function preprocessing events and returning cleaned data frame.
    Author: Ian Eisenberg
    """
    
    columns_to_drop = get_drop_columns(df,
                                       use_default=False,
                                       columns = ['block_duration',
                                                  'trial_index',     #added
                                                  'internal_node_id', #added
                                                  'exp_id', #added
                                                  'key_press',
                                                  'options',
                                                  'response',
                                                  #'rt',
                                                  'stim_duration',
                                                  'stimulus', #added
                                                  'text',
                                                  'time_elapsed',
                                                  'timing_post_trial',
                                                  'trial_id',
                                                  'trial_type'])
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # add duration and response regressor
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)

    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    # time elapsed is at the end of the trial, so have to remove the block
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration',
                     'movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
