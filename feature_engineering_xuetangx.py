import cudf as cf
import cupy as cp
import pandas as pd
import numpy as np
import datetime, time

def feature_engineering_xuetangx(logdata, studentinfo_dropout_data):
    print("Feature Engineering ....\n")
    
    start_time = datetime.datetime.now()
    
    ## calculate duration of each login session 
    logdata['time'] = cf.to_datetime(logdata['time'], format='%Y-%m-%dT%H:%M:%S') # fixing datetime format of 'time' column 
    session_time_max = logdata.groupby(['enroll_id','session_id'])[['time']].max().reset_index().rename(columns={'time':'time_max'}) # time max in a login session
    session_time_min = logdata.groupby(['enroll_id','session_id'])[['time']].min().reset_index().rename(columns={'time':'time_min'}) # time min in a login session
    session_duration = session_time_max.merge(session_time_min, on=['enroll_id', 'session_id'])
    session_duration['duration'] = session_duration['time_max']-session_duration['time_min'] # substracting time max-time min to get session duration    
    session_duration['duration'] = session_duration.to_pandas()['duration'].dt.total_seconds() # duration in seconds

    ## counting Clickstream Activity in each Session
    logdata = cf.concat([logdata, cf.get_dummies(logdata['action'])], axis=1) #label encoding the "Action" column
    session_activity_counts_fulldim = logdata.drop(columns=['username','time']).groupby(['enroll_id','session_id']).aggregate('sum').reset_index()
    ## aggregating counts of "related" Clickstream Activities to reduce dimensionality
    session_activity_counts = session_activity_counts_fulldim.copy()
    columns = session_activity_counts.columns
    session_activity_counts['video_interaction'] = session_activity_counts[[ col for col in columns[columns.str.contains('video')] ]].sum(axis=1)
    session_activity_counts['quiz_interaction'] = session_activity_counts[[ col for col in columns[columns.str.contains('problem')] ]].sum(axis=1)
    session_activity_counts['forum_interaction'] = session_activity_counts[[ col for col in columns[columns.str.contains('thread|comment')] ]].sum(axis=1) 
    session_activity_counts['explore_courseinfo'] = session_activity_counts[[ col for col in columns[columns.str.contains('click|close')] ]].sum(axis=1)
    session_activity_counts = session_activity_counts[['enroll_id','session_id','video_interaction','quiz_interaction','forum_interaction','explore_courseinfo']]
    
    ########## SESSION
    ## merging Session Duration + Counts of Clickstream Activity (in Each Session)
    session_duration_activity = session_activity_counts.merge(session_duration[['enroll_id','session_id','duration','time_min']], how='left', on=['enroll_id','session_id']).sort_values('time_min')

    ############ WEEKLY
    ## clickstream activity per week
    session_weekly_activity = session_duration_activity.to_pandas().sort_values('time_min').groupby(['enroll_id', pd.Grouper(key='time_min', freq='1W')])[['video_interaction','quiz_interaction','forum_interaction','explore_courseinfo', 'duration']].sum().reset_index()
    session_weekly_activity.rename(columns={'time_min':'week', 
                                            'video_interaction':'video_interaction_weekly',
                                            'quiz_interaction':'quiz_interaction_weekly',
                                            'forum_interaction':'forum_interaction_weekly',
                                            'explore_courseinfo':'explore_courseinfo_weekly', 
                                            'duration':'duration_weekly'}, inplace=True)
    ## session counts per week
    session_weekly_counts = session_duration_activity.to_pandas().sort_values('time_min').groupby(['enroll_id', pd.Grouper(key='time_min', freq='1W')])[['session_id']].count().reset_index()
    session_weekly_counts.rename(columns={'time_min':'week', 'session_id':'session_counts_weekly'}, inplace=True)
    
    ## merge session activity/week & session counts/week into one table
    session_weekly_activity = cf.DataFrame(session_weekly_activity.merge(session_weekly_counts, on=['enroll_id','week'], how='left'))
    
    ## added _weeklydiff for each column above
    for featname, weeklyfeat in zip(['video','quiz','forum','explore_courseinfo','duration', 'session_counts'], 
                                   list(session_weekly_activity.drop(columns=['enroll_id','week']).columns)
                                    ):
        session_weekly_activity[featname+'_weeklydiff'] = (session_weekly_activity[['enroll_id','week',
                                                                                        weeklyfeat]]
                                                            .sort_values(['enroll_id','week'])
                                                            .drop(columns=['week'])
                                                            .groupby('enroll_id')
                                                            .diff(axis=0, periods=1)
                                                            .sum(axis=1)
                                                          )
    
    ############ weighted average function
    def weightedavg(data):
        enroll_id = np.array(data.enroll_id.unique())
        list_result_weightedavg = []

        for id in enroll_id:
            #getting the latest 'til the oldest records
            student_clickstream_records = data[data.enroll_id == id].sort_values('week',ascending=False).drop(columns=['enroll_id','week']) #getting the latest 'til the oldest records

            list_result_weightedavg.append([id])

            for column in student_clickstream_records:
                list_student_clickstream_records = list(student_clickstream_records[column])
                exponential_weight = []
                result_weightedavg = 0

                for i in range(len(list_student_clickstream_records)):
                    exponential_weight.append(np.exp(-1/5*i)) # getting the exponential weights
                    result_weightedavg += list_student_clickstream_records[i] * exponential_weight[-1] # multiplying data with corresponding exp. weight
                list_result_weightedavg[-1].append(result_weightedavg/sum(exponential_weight)) #this is the weighted average over multiple sessions based on time decaying method (with factor of 1/5)


        data_weightedavg = cf.DataFrame(dict(zip(data.drop(columns=['week']).columns,
                                                    [cp.array(list_result_weightedavg).T[i] for i in range(len(list_result_weightedavg[0]))]
                                                    ))
                                                    )

        return data_weightedavg

    ## weighted average of WEEKLY data based on its recency
    studentlog_weekly_weightedavg = weightedavg(session_weekly_activity.to_pandas())

    ## Course category & dropout (dropout/non-dropout) of each student
    studentlog_weekly_weightedavg = studentlog_weekly_weightedavg.merge(studentinfo_dropout_data[['enroll_id','course_category','dropout']], on='enroll_id', how='left')
    
    end_time = datetime.datetime.now()
    print('---DONE in %s---' % (end_time-start_time),'\n')
    
    # return logdata_perstudent_weightedavg
    return session_duration, session_activity_counts_fulldim, session_activity_counts, session_duration_activity, session_weekly_activity, studentlog_weekly_weightedavg