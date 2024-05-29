import json
import os
import pandas as pd

data_type = "train"
# temporal
with open("dataset/%s_allmetadata_json/%s_temporalspatial_information.json" % (data_type, data_type)) as f:
    train_ts = json.load(f)
    vuid_arr = []
    uid_arr = []
    time_arr = []
    lon_arr, lat_arr, geoacc_arr = [], [], []
    ts_dict = {}
    for k, v in enumerate(train_ts):
        vuid = v['Uid'] + '/' + v['Pid']
        vuid_arr.append(vuid)
        uid_arr.append(v['Uid'])
        time_arr.append(v['Postdate'])
        if v['Geoaccuracy'] == '0':
            lon_arr.append(0)
            lat_arr.append(0)
            geoacc_arr.append(0)
        else:
            lon_arr.append(v['Longitude'])
            lat_arr.append(v['Latitude'])
            geoacc_arr.append(v['Geoaccuracy'])
    id_arr = list(range(0, len(time_arr)))
    time_sort, id_sort = (list(t) for t in zip(*sorted(zip(time_arr, id_arr))))
    vuid_sort, uid_sort = [], []
    lon_sort, lat_sort, geoacc_sort = [], [], []
    for id in id_sort:
        uid_sort.append(uid_arr[id])
        vuid_sort.append(vuid_arr[id])
        lon_sort.append(lon_arr[id])
        lat_sort.append(lat_arr[id])
        geoacc_sort.append(geoacc_arr[id])
f.close()

# category
with open("dataset/%s_allmetadata_json/%s_category.json" % (data_type, data_type)) as f:
    train_cat = json.load(f)
    cat_arr, subcat_arr, concept_arr = [], [], []
    cat_sort, subcat_sort, concept_sort = [], [], []
    for v in train_cat:
        cat_arr.append(v['Category'])
        subcat_arr.append(v['Subcategory'])
        concept_arr.append(v['Concept'])
    for id in id_sort:
        cat_sort.append(cat_arr[id])
        subcat_sort.append(subcat_arr[id])
        concept_sort.append(concept_arr[id])
f.close()
# text
with open("dataset/%s_allmetadata_json/%s_text.json" % (data_type, data_type)) as f:
    train_text = json.load(f)
    text_arr, text_sort = [], []
    tags_arr, tags_sort = [], []
    for v in train_text:
        text_arr.append(v['Title'])
        tags_arr.append(v['Alltags'])
    for id in id_sort:
        text_sort.append(text_arr[id])
        tags_sort.append(tags_arr[id])
f.close()
with open("dataset/user_additional.txt") as p:
    addition_dict = {}
    while True:
        add = []
        line = p.readline().split()
        if not line:
            break
        pathalias = line[0]
        for i in line[1:]:
            add.append(int(i))
        addition_dict[pathalias] = add
f.close()
# additional
with open("dataset/%s_allmetadata_json/%s_additional_information.json" % (data_type, data_type)) as f:
    train_add = json.load(f)
    public_arr, public_sort, path_arr, path_sort = [], [], [], []
    add_arr, add_sort = [], []
    for v in train_add:
        vuid = v['Uid'] + '/' + v['Pid']
        public_arr.append(v['Ispublic'])
        path_arr.append(v['Pathalias'])
        add_arr.append(addition_dict[v['Pathalias']])
    for id in id_sort:
        public_sort.append(public_arr[id])
        path_sort.append(path_arr[id])
        add_sort.append(add_arr[id])
f.close()

with open("dataset/train_allmetadata_json/train_label.txt") as f:
    label_arr, label_sort = [], []
    for ls in f.readlines():
        l = ls.strip().split(' ')[0]
        label_arr.append(float(l))
    for id in id_sort:
        label_sort.append(label_arr[id])
f.close()
# user
with open("dataset/%s_allmetadata_json/%s_user_data.json" % (data_type, data_type)) as f:
    train_user = json.load(f)
    ispro_arr, ispro_sort, pcount_arr, pcount_sort, canpro_arr, canpro_sort = [], [], [], [], [], []
    loc_arr, loc_sort = [], []
    tzid_arr, tzid_sort, tzoffset_arr, tzoffset_sort = [], [], [], []
    pfirst_arr, pfirst_sort, pfirst_taken_arr, pfirst_taken_sort = [], [], [], []
    loc_str = ''
    for v in train_user:
        ispro_arr.append(v['ispro'])
        pcount_arr.append(v['photo_count'])
        canpro_arr.append(v['canbuypro'])
        tzid_arr.append(v['timezone_timezone_id'])
        tzoffset_arr.append(v['timezone_offset'])
        pfirst_arr.append(v['photo_firstdate'])
        pfirst_taken_arr.append(v['photo_firstdatetaken'])
    for id in id_sort:
        ispro_sort.append(ispro_arr[id])
        pcount_sort.append(pcount_arr[id])
        canpro_sort.append(canpro_arr[id])
        tzid_sort.append(tzid_arr[id])
        tzoffset_sort.append(tzoffset_arr[id])
        pfirst_sort.append(pfirst_arr[id])
        pfirst_taken_sort.append(pfirst_taken_arr[id])

f.close()
'''
additional data
'''
totalview_sort, totaltag_sort, totalgeotag_sort, totalfave_sort, totalingroup_sort, photocount_sort, followercount_sort, followingcount_sort = [],[],[],[],[],[],[],[]
for add in add_sort:
    totalview_sort.append(add[0])
    totaltag_sort.append(add[1])
    totalgeotag_sort.append(add[2])
    totalfave_sort.append(add[3])
    totalingroup_sort.append(add[4])
    photocount_sort.append(add[5])
    followercount_sort.append(add[6])
    followingcount_sort.append(add[7])

data_cols = ["vuid", "user_id", "ispro", "canbuy_pro", "ispublic", "tz_id", "tz_offset", "post_date", "pfirst_date", "pfirst_date_taken", 
"title", "category" , "subcategory" , "concept", "tags", "longitude", "latitude", "geoacc", 
"totalViews", "totalTags", "totalGeotagged", "totalFaves", "totalInGroup", "photoCount", "followerCount", "followingCount",
"label"]
all_data = [vuid_sort, uid_sort, ispro_sort, canpro_sort, public_sort, tzid_sort, tzoffset_sort, time_sort, pfirst_sort, pfirst_taken_sort, text_sort, cat_sort, subcat_sort, concept_sort, tags_sort, lon_sort, lat_sort, geoacc_sort, totalview_sort, totaltag_sort, totalgeotag_sort, totalfave_sort, totalingroup_sort, photocount_sort, followercount_sort, followingcount_sort, label_sort]

dataframe = pd.DataFrame(zip(*all_data), columns=data_cols)
# use new columns to transform user_id, category, subcategory, concept to one-hot encoding
dataframe['concept_id'] = dataframe['concept'].factorize()[0]
dataframe['category_id'] = dataframe['category'].factorize()[0]
dataframe['subcategory_id'] = dataframe['subcategory'].factorize()[0]




# dataframe = pd.get_dummies(dataframe, columns=["category", "subcategory", "concept"])
# transform pfirt_date, pfirst_date_taken to year
# check if the data contains 'None', if so transform it to unix time 1970-01-01
# dataframe['pfirst_date'] = dataframe['pfirst_date'].apply(lambda x: x if x != 'None' else 0)
# dataframe['pfirst_date_taken'] = dataframe['pfirst_date_taken'].apply(lambda x: x if x != 'None' else 0)
import time
def timestamp_to_year(timestamp):
    try:
        timestamp = int(timestamp)  # 尝试将值转换为整数
        timeArray = time.localtime(timestamp)
        return time.strftime("%Y", timeArray)
    except (ValueError, TypeError):
        timestamp = 0
        return timestamp_to_year(timestamp)
        # return None  # 如果转换失败，返回 None
# dataframe = dataframe[dataframe['pfirst_date'] != 'None']
# dataframe = dataframe[dataframe['pfirst_date_taken'] != 'None']
dataframe['pfirst_year'] = dataframe['pfirst_date'].apply(timestamp_to_year)
dataframe['pfirst_taken_year'] = dataframe['pfirst_date_taken'].apply(timestamp_to_year)
# dataframe['pfirst_year'] = pd.to_datetime(dataframe['pfirst_date'], unit='s', origin='unix').dt.year
# dataframe['pfirst_taken_year'] = pd.to_datetime(dataframe['pfirst_date_taken'], unit='s', origin='unix').dt.year

dataframe['post_date'] = pd.to_datetime(dataframe['post_date'], unit='s', origin='unix')
# transform post_date to hour, day in week, month
dataframe['post_hour'] = dataframe['post_date'].dt.hour
dataframe['post_day'] = dataframe['post_date'].dt.dayofweek
dataframe['post_month'] = dataframe['post_date'].dt.month

print("check if the image path exists......")
with open("dataset/train_allmetadata_json/train_img_path.txt", "r") as t:
    # iterate over the file
    for line in t:
        # split the line into a list
        line = line.split()
        # extract the image path
        img_path = line[0]
        # print(img_path)
        img_path = os.path.abspath(img_path)
        # print(img_path)
        # check if the path exists, if not, delete the corresponding line in the dataframe
        if not os.path.exists(img_path):
            print(img_path)
            uid = img_path.split('\\')[-2]
            vid = img_path.split('\\')[-1].split('.')[0]
            vuid = uid + '/' + vid
            dataframe = dataframe[dataframe.vuid != vuid]
t.close()

# to csv file
dataframe.to_csv('./dataset/train_data_preprocessed.csv', index=True)