import tweepy
from tweepy import OAuthHandler
import csv


api = tweepy.API(auth)



screen_name_list = ["@sociauxling"]

for name in screen_name_list:
    user = api.get_user(name)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []  

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = name, count = 200,tweet_mode='extended', include_rts=False)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = name, count=200, max_id=oldest, tweet_mode='extended')

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1


    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text.encode('utf-8'), tweet.favorite_count,tweet.retweet_count] for tweet in alltweets]
    tweet_time = [index[1] for index in outtweets]
    tweet_list = [index[2] for index in outtweets]
    tweek_likes = [index[3] for index in outtweets]
    tweet_retweets = [index[4] for index in outtweets]

    for x in outtweets:
    	print(x)

    with open(f'new_mine_tweets.csv', 'w') as f:
    	writer = csv.writer(f)
    	writer.writerow(["id","created_at","text", "likes", "retweets"])

    	for tweetinfo in outtweets:
    		writer.writerows([tweetinfo])
   