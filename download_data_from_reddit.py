import numpy as np
import requests
import json
import csv
import time
import datetime
import os


def get_pushshift_data(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title=' + str(query) + '&size=1000&after=' + str(
        after) + '&before=' + str(before) + '&subreddit=' + str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def collect_sub_data(subm):
    sub_data = list()  # list to store data points
    title = subm['title']
    url = subm['url']
    try:
        # if flair is available then get it, else set 'NaN'
        flair = subm['link_flair_text']
    except KeyError:
        flair = 'NaN'
    author = subm['author']
    sub_id = subm['id']
    score = subm['score']
    try:
        # if selftext is available then get it, else set it empty
        selftext = subm['selftext']
        list_of_empty_markers = ['[removed]', '[deleted]']
        # many times selftext would be removed or deleted, if thats the case then set it empty
        if selftext in list_of_empty_markers:
            selftext = ''
    except:
        selftext = ''
    created = datetime.datetime.fromtimestamp(subm['created_utc'])  # 1520561700.0
    numComms = subm['num_comments']
    permalink = subm['permalink']

    sub_data.append((sub_id, title, selftext, url, author, score, created, numComms, permalink, flair))
    sub_stats[sub_id] = sub_data


def write_subs_to_file(filename):
    upload_count = 0
    if os.path.exists(filename):
        keep_header = False
    else:
        keep_header = True

    with open(filename, 'a', newline='') as file:
        a = csv.writer(file, delimiter=',')
        headers = ['post_id', 'title', 'selftext', 'url', 'author', 'score', 'publish_date', 'num_of_comments',
                   'permalink', 'flair']
        if keep_header:
            a.writerow(headers)
        for sub in sub_stats:
            a.writerow(sub_stats[sub][0])
            upload_count += 1
        # print(str(upload_count) + ' submissions have been uploaded')


if __name__ == '__main__':
    # Download reddit posts from sub_reddit with keywords given by key_word

    sub_reddit = 'bitcoin'
    key_word = 'bitcoin'

    output_filename = 'reddit_data.csv'
    # search all the posts from start_date to end_date overall
    start_date = datetime.datetime(2018, 1, 1, 0)
    end_date = datetime.datetime(2019, 11, 21, 0)

    # in each itration get reddit posts for one day, to avoid getting blocked by server
    one_day = datetime.timedelta(hours=24)
    after_date = start_date
    after = str(int(after_date.timestamp()))
    before_date = start_date + one_day
    before = str(int(before_date.timestamp()))

    while after_date < end_date:
        print('-' * 80)
        print(after_date, ' -> ', before_date)
        print('-' * 80)

        sub_count = 0
        sub_stats = {}

        data = get_pushshift_data(key_word, after, before, sub_reddit)

        max_count = 100
        count = 0
        while len(data) > 0 and count < max_count:
            print('count ', count)
            for submission in data:
                collect_sub_data(submission)
                sub_count += 1

            print(len(data))
            print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
            after = data[-1]['created_utc']
            data = get_pushshift_data(key_word, after, before, sub_reddit)
            # print(data)
            # print(data['data'][0]['author'])
            count = count + 1

        # keep saving data collected in each iteration
        write_subs_to_file(output_filename)

        # move to next day
        after_date += one_day
        after = str(int(after_date.timestamp()))
        before_date += one_day
        before = str(int(before_date.timestamp()))

        # randomly sleep before starting next iteration
        time.sleep(np.random.randint(1, 3))
