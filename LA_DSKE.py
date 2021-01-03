from __future__ import division
import MySQLdb;
import numpy as np
import spacy
import re
import operator
import math
import csv
from collections import OrderedDict, defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import time
import statistics
import pandas as pd
from numpy import isnan
from scipy.stats import shapiro
import random

nlp = spacy.load('en_core_web_sm')
spacy_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)

# annotated_words={'islam', 'biggest', 'isis', 'hell', 'drumpf', 'attack', 'dumbass', 'nasty', 'twats', 'feminist', 'retard', 'moron', 'donald', 'sunni', 'goddamn', 'bad', 'fatuous', 'ungrateful', 'clingy', 'prick', 'annoying', 'chemical', 'decisions', 'fake', 'dirty', 'disgusting', 'worst', 'bitch', 'weapons', 'ugly', 'tease', 'kick', 'bryce', 'frontin', 'dick', 'hoe', 'rot', 'republican', 'witch', 'pathetic', 'mice', 'ass', 'fuck', 'beat', 'hate', 'wrong', 'group', 'mad', 'humans', 'sick', 'bloody', 'bombs', 'anger', 'bear', 'jihadis', 'trynna', 'fascist', 'party', 'dumb', 'kill', 'females', 'idiot', 'evil', 'nigga', 'montel', 'legit', 'village', 'lyin', 'stupid', 'people', 'damn', 'racist', 'dictator', 'sarandon', 'stfu', 'bastard'}
# annotated_words={'business', 'email', 'app', 'tickets', 'league', 'trends', 'sterling', 'first', 'grand', 'giftsforhim', 'vacation', 'santa', 'support', 'bait', 'club', 'code', 'hot', 'search', 'love', 'luxury', 'celebrity', 'theatre', 'online', 'shop', 'johnson', 'place', 'news', 'dance', 'availability', 'download', 'placement', 'easy', 'website', 'video', 'opportunities', 'event', 'ass', 'minutes', 'ultimate', 'lifetime', 'visit', 'movie', 'trip', 'original', 'campaign', 'latest', 'win', 'full', 'beads', 'fashion', 'adventures', 'stores', 'wallet', 'ipad', 'xxx', 'bets', 'returns', 'instagram', 'purse', 'best', 'product', 'click', 'wifi', 'podcast', 'holiday', 'adidas', 'suede', 'new', 'game', 'book', 'coupon', 'gossip', 'bag', 'vintage', 'fuck', 'modern', 'clothing', 'better', 'start-ups', 'loans', 'show', 'watch', 'free', 'hair', 'silver', 'gold', 'card', 'great', 'car', 'amazon', 'page', 'youtube', 'night', 'thank', 'season', 'girl', 'pic', 'today', 'winner', 'bet', 'top', 'gift', 'music','woman', 'link', 'set', 'available', 'job', 'blog', 'soundcloud', 'weekend', 'sale', 'radiodisney', 'daily', 'april', 'home', 'ebay', 'shoe', 'month', 'chance', 'size', 'money', 'good', 'travel', 'plan', 'launch', 'collection', 'opening', 'check', 'here'}
# annotated_words={'kick', 'attack', 'ass', 'stupid', 'moron', 'females', 'drumpf', 'evil', 'wrong', 'rot', 'village', 'mice', 'clingy', 'bad', 'feminist', 'lil', 'terrify', 'shit', 'chemical', 'terrible', 'motherfucker', 'insane', 'bullshit', 'prick', 'bitch', 'freak', 'syria', 'girl', 'biggest', 'hoe', 'nasty', 'shitty', 'asshole', 'anger', 'dictator', 'donald', 'wrestlemania', 'frontin', 'republican', 'bear', 'dumb', 'nigga', 'ungrateful', 'part-time', 'sunni', 'pathetic', 'legit', 'idiot','dog', 'goddamn', 'sick', 'hell', 'awful', 'people', 'annoying', 'trump', 'twats', 'beat', 'witch', 'ugly', 'sarandon', 'bryce', 'worst', 'bombs', 'disgusting', 'retard', 'lyin', 'dick', 'montel', 'dirty', 'fatuous', 'damn', 'racist', 'islam', 'group', 'weapons', 'fuck', 'humans', 'woman', 'bloody', 'fake', 'jihadis', 'fascist', 'isis', 'hate', 'party', 'trynna', 'words', 'kill', 'dumbass', 'tease', 'mad', 'stfu','asf', 'hurt', 'tf', 'crap', 'hypocrite', 'wtf', 'clown', 'pussy', 'adult', 'troll', 'crazy', 'scumbag'} 
# annotated_words={'khalistan2020', 'khalistan', 'sikh', 'kashmir', 'indian', 'kashmirwantsfreedom', 'referendum2020', 'freedom', 'free', 'august', 'freekashmir', 'right', 'people', 'kashmirkhalistan_ajointcall', 'kashmirunderthreat', 'citizen', 'freejagginow', 'jagtar', 'kashmiris', 'movement', 'referendum', 'voice', 'state', 'community', 'independence', 'kashmiri', 'terrorist', 'flag', 'support', 'kashmirbanaygapakistan', 'torture', 'jaggi', 'justice', 'hindutva', 'independent', 'trial', 'johal', 'detainedfreejagginow', 'abduct', 'hardkaur', 'freekhalistan', 'khalistanmovement', 'guru', 'fight', 'sjf', 'ban', 'brave', 'jaggis'}
annotated_words={'ghazwaehind', 'muslim', 'kashmir', 'time', 'jihad', 'india', 'pakistan', 'indian', 'army', 'allah', 'kashmirhamarahai', 'zaidzamanhamid', 'war', 'ready', 'islam', 'pak', 'savekashmirso', 'zionist', 'kashmiri', 'right', 'kashmirbleed', 'solution', 'peaceforchange', 'final', 'article370', 'pakistani', 'standwithkashmir', 'full', 'taliban', 'decision', 'force', 'savekashmirforhumanity', 'reason', 'quran', 'battle', 'kufr', 'kaffir', 'allahuackber', 'nonmuslim', 'soldier', 'call', 'mujahideen', 'naraetakbeer', 'conquer', 'sunnahjihadkhilafat', 'eyeopener', 'brother', 'menace', 'Hunduvta', 'terror', 'Bloodshed', 'warrior', 'WeStandWithKashmir', 'KashmirHamaraHai', 'KashmirParFinalFight', 'battle', 'ReleaseHafizSaeed', 'KashmirBanayGaPakistan', 'massacare', 'FreeKashmir', 'shahadat', 'martyred', 'Labayk', 'KashmirBleeds', 'Khilafat', 'Munafiq', 'KashmirUnderThreat', 'Zionists', 'warfare', 'MasoodAzhar', 'YasinMalick', 'Afzalguru', 'MullaUmar', 'soldiers', 'Freedom', 'fighters', 'HinduTerror', 'JihadIsFinalTreatmentForIndia', 'weapons', 'DesiringMartyrdom', 'blood', 'Allah', 'believers', 'followers', 'join', 'Kuffar', 'sacrifice', 'Ummat-e-Muslima', 'camps', 'suicide', 'fighter'}

# seed_words={'hate', 'nigga', 'idiot'}
# seed_words={'free', 'click', 'win'}
# seed_words={'moron', 'bitch', 'asshole'}
# seed_words={'khalistan2020', 'sikh', 'freejagginow'}
seed_words={'kashmir', 'jihad', 'ghazwaehind'}

#This file contains final version of code which is updated on Github.
def word_seedsim(word_list, bigram_count, embedding_model, term_freq_dict, nor_term_dict):
	word_seedword_sim={}
	noofseedword=len(seed_words)

	for word in word_list:
		embedd_based_sim=0
		cooccu_prox=[]
		word_sword_cocount=0
		num_count=0
		for sword in seed_words:
			bigram_fir=tuple([word]+[sword])
			bigram_sec=tuple([sword]+[word])
			
			if bigram_count[bigram_fir]>0 or bigram_count[bigram_sec]>0:
				word_sword_cocount=bigram_count[bigram_fir]+bigram_count[bigram_sec]
			
			cooccu_prox.append((word_sword_cocount/(term_freq_dict[word])))
			try:
				cos_sim=cosine_similarity(np.array(embedding_model[word]).reshape(1,100), np.array(embedding_model[sword]).reshape(1, 100))[0][0]
				embedd_based_sim=embedd_based_sim + cos_sim
				num_count=num_count+1
			except:
				continue

		if num_count!=0:
			embedd_based_sim=embedd_based_sim/num_count
		
		cooccu_prox=statistics.mean(cooccu_prox)
		
		num=(term_freq_dict[word]*1.0)/(sum(term_freq_dict.values()))
		
		if word in nor_term_dict.keys():
			den=(nor_term_dict[word]*1.0)/(sum(nor_term_dict.values()))
			domain_rel=(num/den)
		else:
			domain_rel=num
		
		semantic_sim=(embedd_based_sim + cooccu_prox + domain_rel)/3
		word_seedword_sim[word]=semantic_sim

	return word_seedword_sim

def indexing_fun(word_list):
	i=0
	word_index={}
	index_word={}
	for word in word_list:
		word_index[word]=i
		i=i+1

	for word in word_index.keys():
		index_word[word_index[word]]=word

	return word_index, index_word

def word_cooccur(word_list, bigram_count, word_index):
	nrow=ncol=len(word_list)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(word_list)):
		for j in range(i+1, len(word_list)):
			word_cooccur_count=0

			bigram_fir=tuple([word_list[i]]+[word_list[j]])
			bigram_sec=tuple([word_list[j]]+[word_list[i]])
			
			if bigram_count[bigram_fir]>0 or bigram_count[bigram_sec]>0:
				word_cooccur_count=bigram_count[bigram_fir]+bigram_count[bigram_sec]
	
			word_cooccur_mat[word_index[word_list[i]]][word_index[word_list[j]]]=word_cooccur_count
			word_cooccur_mat[word_index[word_list[j]]][word_index[word_list[i]]]=word_cooccur_count

	word_cooccur_mat_nor=word_cooccur_mat/np.max(word_cooccur_mat)
	return word_cooccur_mat_nor


#This function observes the effect of different number of datasets on G value of stop and relevant words
def termhood_relevance_deviation_cal():
	connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
	cursor=connection.cursor(MySQLdb.cursors.DictCursor)
	#table_list=['AbusiveTweets2000', 'Genuine_tweets_original', 'ISISTweets', 'GarethBale_Tweets', 'Our_Bot_Tweets', 'Socialbots1_tweets']
	dataset_list=['hateful', 'spam', 'abusive', 'Khalistan_Movement', 'Jihad_Tweets']
	term_list=['he', 'of', 'and', 'the', 'hate', 'idiot', 'bad', 'hell']

	l=1
	for i in range(2, len(dataset_list)+1):
		termset=set()
		global_term_dict={}
		global_term_dict1={}
		for j in range(i):
			cursor.execute("SELECT tweet FROM "+table_list[j]"")
			results=cursor.fetchall()
			term_freq_dict={}

			tweet_list=[]
			for row in results:
				tweet=re.sub(r'\n', '', row['tweet']).strip()
				tweet=re.sub(r'http\S+', '', row['tweet']).strip()
				tweet=re.sub(r'[.:!\'/(@#$)?-]', '', tweet)
				tweet_list.append()
			
			for eachtweet in tweet_list:
				tokens=eachtweet.lower().split(' ')
				for token in tokens:
					termset.add(token)
					if token in term_freq_dict.keys():
						term_freq_dict[token]=term_freq_dict[token]+1
					else:
						term_freq_dict[token]=1

			rank=voc=len(term_freq_dict.keys())
			sorted_list_term=sorted(term_freq_dict.items(), key=operator.itemgetter(1))
			
			term_freq_dict1={}
			for word, freq in list(reversed(sorted_list_term)):
				term_freq_dict1[word]=(rank*1.0)/voc
				rank=rank-1
			# print(term_freq_dict1['he'], "===" ,term_freq_dict1['of'], "====", term_freq_dict1['and'], "====", term_freq_dict1['the'])
			# print(term_freq_dict1['hate'], "===" ,term_freq_dict1['idiot'], "====", term_freq_dict1['bad'], "====", term_freq_dict1['hell'], "\n")
			
			global_term_dict[table_list[j]]=term_freq_dict1

		global_term_index_dic={}
		print("Number of datasets==", i)
		
		for term in termset:
			term_index=[]
			for key in global_term_dict.keys():
				if term in global_term_dict[key].keys():
					term_index.append(global_term_dict[key][term])
				else:
					term_index.append(0)

			if term in term_list:
				print(term, "===", term_index)
		
			#Code for incrimental approach
			term_index_avg=np.mean(term_index)

			term_index_dev=0
			term_index_incre=[]
			for k in range(len(term_index)):
				term_index_incre.append(term_index[k])		
				first_term=np.mean(term_index_incre)
				term_index_dev=term_index_dev+abs(first_term-max(term_index_incre))
				
				if term in term_list:
					print(term, "===", term_index_incre, "===", first_term, "===", max(term_index_incre),"===" ,term_index_dev)
			
			global_term_index=term_index_avg*(1.0/term_index_dev)

			if term in term_list:
				print("$$$$", term, "===", term_index_avg, "===", global_term_index)
	
			global_term_index_dic[term]=global_term_index

		if l<=1:
			prehe=global_term_index_dic['he']
			preof=global_term_index_dic['of']
			preand=global_term_index_dic['and']
			prethe=global_term_index_dic['the']
			#preis=global_term_index_dic['is']

			prehate=global_term_index_dic['hate']
			preidiot=global_term_index_dic['idiot']
			prebad=global_term_index_dic['bad']
			prehell=global_term_index_dic['hell']
			#preugly=global_term_index_dic['ugly']
			print(l, "***", prehe, "===" ,preof, "====", preand, "====", prethe)
			print(l, "***", prehate, "===" ,preidiot, "====", prebad, "====", prehell, "\n")
			
			l=l+1

		else:
			change_he=abs((global_term_index_dic['he']-prehe))*100/global_term_index_dic['he']
			change_of=abs((global_term_index_dic['of']-prehe))*100/global_term_index_dic['of']
			change_and=abs((global_term_index_dic['and']-prehe))*100/global_term_index_dic['and']
			change_the=abs((global_term_index_dic['the']-prehe))*100/global_term_index_dic['the']
			#change_is=(preis-global_term_index_dic['is'])*100/preis

			change_hate=abs((global_term_index_dic['hate']-prehe))*100/global_term_index_dic['hate']
			change_idiot=abs((global_term_index_dic['idiot']-prehe))*100/global_term_index_dic['idiot']
			change_bad=abs((global_term_index_dic['bad']-prehe))*100/global_term_index_dic['bad']
			change_hell=abs((global_term_index_dic['hell']-prehe))*100/global_term_index_dic['hell']
			#change_ugly=(preugly-global_term_index_dic['ugly'])*100/preugly

			prehe=global_term_index_dic['he']
			preof=global_term_index_dic['of']
			preand=global_term_index_dic['and']
			prethe=global_term_index_dic['the']
			#preis=global_term_index_dic['is']

			prehate=global_term_index_dic['hate']
			preidiot=global_term_index_dic['idiot']
			prebad=global_term_index_dic['bad']
			prehell=global_term_index_dic['hell']
			#preugly=global_term_index_dic['ugly']
			l=l+1

			print(l, "***", prehe, "===" ,preof, "====", preand, "====", prethe)
			print(l, "***", prehate, "===" ,preidiot, "====", prebad, "====", prehell, "\n")
			print("cha_he==", change_he, "==change_of==", change_of, "==change_and==", change_and, "==cha_the==", change_the)
			print("cha_hate==", change_hate, "==cha_idiot==", change_idiot, "==cha_bad==", change_bad, "==cha_hell==", change_hell, "\n")


#Final function to find the stop and irrelevant words
def stopwords_final():
	connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
	cursor=connection.cursor(MySQLdb.cursors.DictCursor)
	#table_list=['AbusiveTweets2000', 'Genuine_tweets_original', 'ISISTweets', 'GarethBale_Tweets', 'Our_Bot_Tweets', 'Socialbots1_tweets']
	dataset_list=['hateful', 'spam', 'abusive', 'Khalistan_Tweets', 'Jihad_Tweets']
	
	termset=set()
	global_term_dict={}
	for dataset in dataset_list:
		cursor.execute("SELECT tweet FROM benchmark_dataset_text where label='%s'" % (dataset))
		results=cursor.fetchall()
		term_freq_dict={}
		tweet_list=[]

		for row in results:
			tweet=re.sub(r'\n', '', row['tweet']).strip()
			tweet=re.sub(r'http\S+', '', tweet)
			tweet=re.sub(r'[.:!\'/(@#$)?-]', '', tweet).strip().lower()
			tweet_list.append(tweet)

		sampled_tweets=random.sample(tweet_list, 1000)

		for eachtweet in sampled_tweets:
			tokens=eachtweet.strip().split()
			for token in tokens:
				termset.add(token)
				if token in term_freq_dict.keys():
					term_freq_dict[token]=term_freq_dict[token]+1
				else:
					term_freq_dict[token]=1

		rank=voc=len(term_freq_dict.keys())
		sorted_list_term=sorted(term_freq_dict.items(), key=operator.itemgetter(1))

		term_freq_dict1={}
		for word, freq in list(reversed(sorted_list_term)):
			term_freq_dict1[word]=(rank*1.0)/voc
			rank=rank-1
		
		global_term_dict[dataset]=term_freq_dict1

	global_term_index_dic={}

	print("termset==", len(termset))
	for term in termset:
		term_index=[]
		for dataset in global_term_dict.keys():
			if term in global_term_dict[dataset].keys():
				term_index.append(global_term_dict[dataset][term])
			else:
				term_index.append(0)
		
		term_index_avg=np.mean(term_index)	

		term_index_dev=0
		term_index_incre=[]
		for k in range(len(term_index)):
			term_index_incre.append(term_index[k])		
			first_term=np.mean(term_index_incre)
			term_index_dev=term_index_dev+abs(first_term-max(term_index_incre))

		global_term_index=term_index_avg*(1.0/math.exp(term_index_dev))
		global_term_index_dic[term]=global_term_index

	glo_sorted_list_term=sorted(global_term_index_dic.items(), key=operator.itemgetter(1))	
	annotated_words=['the', 'to', 'RT', 'a', 'ISIS', 'of', 'in', 'and', 'is', 'I', 'on', 'for', 'you', 'that', 'it', 'with', 'have', 'are', 'this', 'at', 'be', 'my', '&', 'The', 'from', 'as', 'by', 'US', 'not', 'we', 'your', 'me', 'but', 'was', '"', 'just', 'all', 'like', 'about', 'Iraq', 'will', 'our', 'more', 'so', 'an', 'up', 'has', 'love', 'they', 'out', 'people', 's', 'i', 'us', 'who', 'can', 'do', 'one', 'its', 'Islamic', 'via', 'when', 'Im', 'Mosul', 'But', 'if', 'get', 'dont', 'those', 'State', 'now', 'or', 'what', 'no', 'he', 'how', 'time', 'their', 'his', 'war', 'know', 'You', 'near', 'A', 'Pokemon', 'dead', 'wants', 'damn', 'yall', 'them', 'think', 'This', 'hands', 'We', 'Muslim', 'there', 'race', 'today', 'hate', 'idiot', 'nigga', 'niggas', 'idiots', 'ass', 'hell', 'trump', 'bad', 'bitches', 'disgusting', 'sick', 'stupid', 'ugly', 'bitch', 'every', 'evil', 'retarded', 'country', 'tired', 'look', 'does', 'fuck', 'crazy', 'women', 'potus', 'keep', 'any', 'trying', 'doing', 'bloody', 'wanna', 'always', 'god', 'nazi', 'put', 'cause', 'friends', 'said', 'president', 'talking', 'made', 'making', 'nasty', 'woman', 'everyone', 'annoying', 'retard', 'gon', 'give', 'chemical', 'run', 'anyone', 'america', 'yet', 'stop', 'once', 'parissaxo', 'feminist', 'also']

	number_selected_words=len(global_term_index_dic.keys())*0.01
	candidate_stops={}
	for word, index in list(reversed(glo_sorted_list_term))[0:number_selected_words]:
		candidate_stops[word]=index
	
	top_stopwords=set(candidate_stops.keys())

	print("Number of stopwords==", top_stopwords,"====", len(top_stopwords))
	
	common_stopwords=spacy_stopwords.intersection(top_stopwords)
	print("common stopwords spacy==", len(common_stopwords), "\n")
	
	common_annotatewords=annotated_words.intersection(top_stopwords)
	print(common_annotatewords, "===", len(common_annotatewords))
	
	iteration(top_stopwords)

	print("End of the program!!!")


def iteration(stop_words):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		dataset=input("Please enter the dataset: ")
		
		cursor.execute("SELECT tweet FROM "+dataset+" ")
		results=cursor.fetchall()
		stop_words=list(stop_words)

		embedding_model=Word2Vec.load('embedding/Jihad_Tweets/Jihad_Tweets_model2.bin')

		term_freq_dict={}
		tweets_words=[]
		bigram_list=[]
		tweet_list=[]
		for row in results:
			tweet=re.sub(r'http\S+', '', row['tweet']).strip()
			tweet=re.sub(r'[.:!\'/(@#$?-]', '', tweet).strip().lower()

			if tweet.startswith("RT"):
				tweet=tweet[len("RT"):].strip()

			tweet_list.append(tweet)
			sampled_tweets=random.sample(list, 1000)
			
		for eachtweet in sampled_tweets:
			tweet_words=[]
			processed= list(filter(None, re.split('; |\$|\)|\?|!|-|:|/|;|; |;| |\n|\. |\.', eachtweet)))
			
			for word in processed:
				word=word.strip()

				if len(word)>2 and word not in stop_words:
					tweet_words.append(word)
					if word in term_freq_dict.keys():
						term_freq_dict[word]=term_freq_dict[word]+1
					else:
						term_freq_dict[word]=1
				
			blist=[]
			for i in range(len(tweet_words)):
				for j in range(i+1, len(tweet_words)):
					bigram=tuple([tweet_words[i]]+[tweet_words[j]])
					blist.append(bigram)

			bigram_list=bigram_list+blist

		tweets_words=list(term_freq_dict.keys())
		bigram_count=Counter(bigram_list)
		word_count=len(tweets_words)

		cursor.execute("SELECT tweet FROM benchmark_dataset_text where label='normal'")
		nor_results=cursor.fetchall()

		nor_term_dict={}

		generic_tweet_list=[]
		for row in nor_results:
			tweet=re.sub(r'http\S+', '', row['tweet']).strip()
			tweet=re.sub(r'[.:!\'/(@#$?-]', '', tweet).strip().lower()

			if tweet.startswith("RT"):
				tweet=tweet[len("RT"):].strip()

			generic_tweet_list.append(tweet)
			
		for eachtweet in generic_tweet_list:
			processed=nlp(eachtweet)

			for word in processed:
				if len(str(word).strip())>2:
					word=str(word).strip()
					if word in nor_term_dict.keys():
						nor_term_dict[word]=word_freq=nor_term_dict[word]+1
					else:
						nor_term_dict[word]=1
		
		word_seedword_sim=word_seedsim(tweets_words, bigram_count, embedding_model, term_freq_dict, nor_term_dict)
		word_index, index_word= indexing_fun(tweets_words)
		word_cooccur_mat_nor=word_cooccur(tweets_words, bigram_count, word_index)
		
		word_cooccur_mat_nor=word_cooccur_mat_nor/np.count_nonzero(word_cooccur_mat_nor, axis=0)
		NaNs=isnan(word_cooccur_mat_nor)
		word_cooccur_mat_nor[NaNs]=0
		
		word_rank=np.zeros((word_count, 1))
		for word in word_seedword_sim.keys():
			word_rank[word_index[word]][0]=word_seedword_sim[word]

		flag=True
		pre_word_rank=word_rank
		damp_val=(1-0.85)/word_count
		damp_vec=np.full((word_count, 1), damp_val)

		while flag==True:
			uprank=0.95*(np.dot(word_cooccur_mat_nor, word_rank))
			word_rank=damp_vec+uprank
			rankdiff=sum(abs(pre_word_rank-word_rank))
			pre_word_rank=word_rank

			if rankdiff <= 0.00001:
				flag=False

		final_word_rank={}
		for index, rank in enumerate(word_rank):
			final_word_rank[index_word[index]]=rank

		sorted_list=sorted(final_word_rank.items(), key=operator.itemgetter(1))

		sel_words_set=set()
		for key in list(reversed(sorted_list))[0:80]:
			sel_words_set.add(key[0])

		print("selected words==", sel_words_set, "==", len(sel_words_set))
		commonwords=annotated_words.intersection(sel_words_set)
		
		precision=(len(commonwords)*1.0)/len(sel_words_set)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)

		print("common words==", commonwords, "==", len(commonwords), "===len_annot==", len(annotated_words))
		print("Precision==", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")

	finally:
		connection.close()



# termhood_relevance_deviation_cal()
stopwords_final()