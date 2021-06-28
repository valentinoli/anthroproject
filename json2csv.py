import json
import csv
import pandas as pd
import numpy as np
import os
from glob import glob


def collect_one_feature(vid_data, aggregate="mean"):
    """
    Description: Collects the features over time of one processed video
    (stored in one json file)
    """
    def get_slope(row):
        """Given a row of a datafram representing a timeserie
        of the evolution of a facial feautre, compute the sum of positive
        slope over all pair of consecutive timestamp"""
        slope = 0
        for i in np.arange(0,len(row)-2,2):
            if row[i] < row[i+1]:
                slope = slope + row[i+1]- row[i]
        return slope


    expressions = vid_data["data"].get("expressions", None)

    if expressions is None: # camera couldn't detect a face
        result = pd.DataFrame(-np.ones(len(FEATURES))).transpose()
        result.columns = FEATURES
        result.drop(IGNORED_FEATURES,axis=1, inplace=True)
        result["nb_timestamps"] = 0
        # reorder the columns to have the nb_timestamps first
        columns_reorder = list(result.columns[-1:]) + list(result.columns[:-1])
        result = result[columns_reorder]
        return result

    df_features = pd.DataFrame(expressions)
    df_features["time"] = vid_data["timestamps"]
    df_features.set_index("time", inplace=True)
    df_features.drop(IGNORED_FEATURES,axis=1, inplace=True)

    if aggregate is None:
        return df_features

    # if we aggregate, also add a column giving the length of the time series before aggregation
    if aggregate == "mean":
        result = pd.DataFrame(df_features.mean(axis=0)).transpose()

    elif aggregate == "max":
        result = pd.DataFrame(df_features.max(axis=0)).transpose()

    elif aggregate == "slope":
        result = df_features.T
        result.columns = range(len(result.columns))
        result["slope"] = result.apply(lambda row:get_slope(row), axis=1)
        result = pd.DataFrame(result["slope"]).T

    else:
        raise ValueError('aggregate parameter can only be "mean" or "max" (passed: {0})'.format(aggregate))

    result["nb_timestamps"] = len(df_features)

    # reorder the columns to have the nb_timestamps first
    columns_reorder = list(result.columns[-1:]) + list(result.columns[:-1])
    result = result[columns_reorder]
    return result


def collect_features(video_data, aggregate="mean"):
    """
    Collects all features aggregated over time,
    from each processed videos (stored in one json file) and group them in a dataframe.
    """

    # one df per video
    df_videos = []

    # process videos one-by-one
    for i in range(NB_VIDEOS):
        df_videos.append(collect_one_feature(video_data[str(i+1)], aggregate=aggregate))

    # concatenated the three df

    result = pd.concat(df_videos)
    # add the video_id for each line of the df
    result["video_id"] = np.arange(NB_VIDEOS) + 1

    # reorder the columns to have the video_id first
    columns_reorder = list(result.columns[-1:]) + list(result.columns[:-1])
    result = result[columns_reorder]

    return result


def collect_answers(surveys_data, agg_survey):
    """Collect all answers and store them in a dataframe"""
    def get_val(survey_data, name):
        # given the question name and data of a particular survey data, return its answered value
        for pair in survey_data:
            if pair["name"] == name:
                return pair["value"]

    # collect answers in a NB_VIDEOS x NB_QUESTIONS + 1 array  (+ 1 to indicat video id)
    answers = np.zeros((NB_VIDEOS, NB_QUESTIONS + 1))

    # process videos one-by-one
    for i in range(NB_VIDEOS):
        answers[i][0] = i + 1 # set video id
        for idx,k in enumerate(ANSWERS_DICT.keys()):
            answers[i][idx+1] = get_val(surveys_data[str(i+1)],k)

    result = pd.DataFrame(data=answers, columns=["video_id"] + ANSWERS_COLUMNS)

    if agg_survey:
        for impression in IMPRESSIONS:
            result[impression] = result[IMPRESSIONS[impression]].mean(axis=1)
        result = result[["video_id"] + list(IMPRESSIONS.keys())]

    return result

def extract_data(data, agg_features="mean"):
    """Given raw data in a JSON format for one experiment,
    extract the video data, the answers data and the general data"""
    video_data = data["video"]

    df_features = collect_features(video_data, agg_features)
    return df_features




def store_all_time_series(filenames, agg="mean"):
    """Store for each pair (video_id, feature_name) a dataframe
    as returned by collect_time_series"""

    for video_id in map(str, range(1, NB_VIDEOS + 1)):
        for feature_name in CONSIDERED_FEATURES:
            temp = collect_time_series(filenames, feature_name, video_id, agg=agg)
            temp.to_csv(DATAFRAMES_TIMESERIES_PATH + "df_{0}_{1}.csv".format(feature_name, video_id))


def collect_time_series(filenames, feature_name, video_id, agg="mean"):
    """Given a feature_name, and a video_id, collect in a dataframe the
    time series of this facial features for all users for that particular video.
    The value for each timestamp is aggregated for each second"""

    nb_participants = len(filenames)

    time_series_list = []
    # process one file at a time
    for idx, file in enumerate(filenames):
        with open(file, encoding="utf-8") as json_file:
            video_data = json.load(json_file)["video"][video_id]

        time_series_list.append((collect_one_feature(video_data, aggregate=None)[feature_name]))

    # for each time_serie, change index and resample
    for idx, serie in enumerate(time_series_list):
        #set index to be a timestamp
        serie.index = pd.to_datetime(serie.index.values,unit="s")
        # resample to have one value for each second,
        if agg == "sum":
            serie = serie.resample(axis="index", rule="s").sum()
        elif agg == "mean":
            serie = serie.resample(axis="index", rule="s").mean()
        else:
            raise ValueError("Aggregator should be either sum or mean")
        serie.index = serie.index.time
        time_series_list[idx] = serie

    result = pd.concat(time_series_list, axis=1, sort=False).T
    result["user_id"] = np.arange(nb_participants)
    result["video_id"] = video_id
    result.set_index(["user_id","video_id"], inplace=True)
    return result





def store_all_df(filenames, agg_survey=False, agg_features="mean"):
    """Calls extract_data for each file in filenames,
    and collect everything in dataframes and store them"""

    list_df_features = []
    list_df_answers = []
    list_df_generals = []
    nb_participants = len(filenames)

    # process one file at a time
    for idx, file in enumerate(filenames):
        with open(file, encoding='utf8') as json_file:
            data = json.load(json_file)
            df_features = extract_data(data, agg_features)
            list_df_features.append(df_features)

    # concatenate data for each participant
    df_final_features = pd.concat(list_df_features)

    # set an id for each user
    user_id = np.arange(nb_participants).repeat(NB_VIDEOS)
    df_final_features["user_id"] = user_id


    df_final_features.set_index(["user_id","video_id"], inplace=True)

    df_final_features.to_csv(DATAFRAMES_PATH + "df_features_agg_" + agg_features + ".csv")

DATA_PATH = "data/csv/"
DATAFRAMES_PATH = DATA_PATH
DATAFRAMES_TIMESERIES_PATH = DATAFRAMES_PATH + "timeseries/"

os.makedirs(DATAFRAMES_TIMESERIES_PATH, exist_ok=True)

NB_VIDEOS = 2

ANSWERS_DICT = {"i1":"fake/natural", "ii2":"stagnant/lively", "ii5":"inert/interactive",
                   "i3":"unconscious/conscious", "iv1":"incompetent/competent",
                   "iv4":"unintelligent/intelligent", "ii3":"mechanical/organic",
                   "ii6":"unresponsive/responsive", "iv5":"foolish/sensible",
                   "iv2":"ignorant/knowledgeable", "i2":"machinelike/humanlike",
                   "i4":"artificial/lifelike", "ii1":"dead/alive", "iii4":"unpleasant/pleasant",
                   "i5":"rigid/smooth", "iii1":"dislike/like", "iv3":"irresponsible/responsible",
                   "iii5":"awful/nice", "iii2":"unfriendly/friendly", "iii3":"unkind/kind"}

ANSWERS_COLUMNS = ["fake/natural", "stagnant/lively", "inert/interactive",
                   "unconscious/conscious", "incompetent/competent",
                   "unintelligent/intelligent", "mechanical/organic",
                   "unresponsive/responsive", "foolish/sensible",
                   "ignorant/knowledgeable", "machinelike/humanlike",
                   "artificial/lifelike", "dead/alive", "unpleasant/pleasant",
                   "rigid/smooth", "dislike/like", "irresponsible/responsible",
                   "awful/nice", "unfriendly/friendly", "unkind/kind"]

NB_QUESTIONS = len(ANSWERS_COLUMNS)

ANTHROPOMORPHISM_COLUMNS = ["fake/natural", "machinelike/humanlike",
                            "unconscious/conscious", "artificial/lifelike",
                            "rigid/smooth"]
ANIMACY_COLUMNS = ["dead/alive", "stagnant/lively", "mechanical/organic",
                   "inert/interactive", "artificial/lifelike",
                   "unresponsive/responsive"]
LIKEABILITY_COLUMNS = ["dislike/like", "unfriendly/friendly",
                       "unkind/kind", "unpleasant/pleasant",
                       "awful/nice"]
INTELLIGENCE_COLUMNS = ["incompetent/competent", "ignorant/knowledgeable",
                        "irresponsible/responsible", "unintelligent/intelligent",
                        "foolish/sensible"]

IMPRESSIONS = {"Anthropomorphism": ANTHROPOMORPHISM_COLUMNS,
               "Animacy":ANIMACY_COLUMNS,
               "Likeability":LIKEABILITY_COLUMNS,
               "Intelligence":INTELLIGENCE_COLUMNS}

FEATURES = ['smile', 'innerBrowRaise', 'browRaise', 'browFurrow', 'noseWrinkle',
       'upperLipRaise', 'lipCornerDepressor', 'chinRaise', 'lipPucker',
       'lipPress', 'lipSuck', 'mouthOpen', 'smirk', 'eyeClosure', 'attention',
       'lidTighten', 'jawDrop', 'dimpler', 'eyeWiden', 'cheekRaise',
       'lipStretch']

IGNORED_FEATURES = ['attention', 'lidTighten', 'jawDrop', 'dimpler',
                    'eyeWiden', 'cheekRaise', 'lipStretch']

CONSIDERED_FEATURES = ['smile', 'innerBrowRaise', 'browRaise', 'browFurrow', 'noseWrinkle',
       'upperLipRaise', 'lipCornerDepressor', 'chinRaise', 'lipPucker',
       'lipPress', 'lipSuck', 'mouthOpen', 'smirk', 'eyeClosure']

aq10_fields = ["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10",
               "s1", "s2","s3","s4","s5","s6","s7","s8","s9", "s10", "totalScore", "considerDiagnosticAssessment"]
general_fields = ["age", "gender", "robotRealLife"]

sqpb_fields =  ["q1","q2","q3","q4","q5","q6","q7","q8"]

survey_fields = ["fake/natural", "stagnant/lively", "inert/interactive",
                   "unconscious/conscious", "incompetent/competent",
                   "unintelligent/intelligent", "mechanical/organic",
                   "unresponsive/responsive", "foolish/sensible",
                   "ignorant/knowledgeable", "machinelike/humanlike",
                   "artificial/lifelike", "dead/alive", "unpleasant/pleasant",
                   "rigid/smooth", "dislike/like", "irresponsible/responsible",
                   "awful/nice", "unfriendly/friendly", "unkind/kind"]

translation = {"i1":"fake/natural", "ii2":"stagnant/lively", "ii5":"inert/interactive",
                   "i3":"unconscious/conscious", "iv1":"incompetent/competent",
                   "iv4":"unintelligent/intelligent", "ii3":"mechanical/organic",
                   "ii6":"unresponsive/responsive", "iv5":"foolish/sensible",
                   "iv2":"ignorant/knowledgeable", "i2":"machinelike/humanlike",
                   "i4":"artificial/lifelike", "ii1":"dead/alive", "iii4":"unpleasant/pleasant",
                   "i5":"rigid/smooth", "iii1":"dislike/like", "iv3":"irresponsible/responsible",
                   "iii5":"awful/nice", "iii2":"unfriendly/friendly", "iii3":"unkind/kind"}

print(">>> Configuring AWS")
import shutil
from os.path import join
from pathlib import Path
home = str(Path.home())

shutil.copy("./credentials", join(home, ".aws", "credentials"))

print(">>> Downloading files from AWS S3")
print()
os.system('aws s3 cp s3://anthropobucket/devData/ ./data/json --recursive')
filenames = [y for x in os.walk("data") for y in glob(os.path.join(x[0], '*.json'))]
print()
print("Received {0} JSON files (one per participant)".format(len(filenames)))
print()

if len(filenames) == 0:
    raise ValueError('No files to process')

print(">>> Storing surveys answers as CSV")

prefix = DATA_PATH
with open(f'{prefix}aq10.csv', 'w', newline='') as aqfile:
    writer_aq = csv.DictWriter(aqfile, fieldnames=aq10_fields)
    writer_aq.writeheader()
    with open(f'{prefix}general.csv', 'w', newline='') as gefile:
        writer_ge = csv.DictWriter(gefile, fieldnames=general_fields)
        writer_ge.writeheader()
        with open(f'{prefix}survey1.csv', 'w', newline='') as sufile1:
            writer_su1 = csv.DictWriter(sufile1, fieldnames=survey_fields)
            writer_su1.writeheader()
            with open(f'{prefix}survey2.csv', 'w', newline='') as sufile2:
                writer_su2 = csv.DictWriter(sufile2, fieldnames=survey_fields)
                writer_su2.writeheader()
                with open(f'{prefix}sqpb.csv', 'w', newline='') as sqfile:
                    writer_sq = csv.DictWriter(sqfile, fieldnames=sqpb_fields)
                    writer_sq.writeheader()
                    for f in filenames:
                        data = json.load(open(f, 'r', encoding='utf-8'))
                        general = data["general"]
                        survey = data["survey"]
                        aq10 = data["aq10"]
                        sqpb = data["sqpb"]

                        #general writing
                        writer_ge.writerow(general)

                        #aq writing
                        dict_aq10 = {e["name"]:e["value"] for e in aq10["data"]}
                        dict_aq10.update({"s" + str(i+1):e["score"] for i,e in enumerate(aq10["data"])})
                        dict_aq10["totalScore"] = aq10["totalScore"]
                        dict_aq10["considerDiagnosticAssessment"] = aq10["considerDiagnosticAssessment"]
                        writer_aq.writerow(dict_aq10)

                        #sq writing
                        sqpb_dict = {e["name"]:e["value"] for e in sqpb["data"]}
                        writer_sq.writerow(sqpb_dict)

                        #survey writing
                        sur1 = survey["1"]
                        survey_dict1 = {translation[e["name"]]:e["value"] for e in sur1}
                        sur2 = survey["2"]
                        survey_dict2 = {translation[e["name"]]:e["value"] for e in sur2}
                        writer_su1.writerow(survey_dict1)
                        writer_su2.writerow(survey_dict2)


print(">>> Storing videos features as CSV (this might take a while)")
store_all_df(filenames, agg_features="max", agg_survey=False)
store_all_df(filenames, agg_features="mean", agg_survey=False)
store_all_df(filenames, agg_features="slope", agg_survey=False)
store_all_time_series(filenames)
