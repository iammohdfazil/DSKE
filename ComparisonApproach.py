from __future__ import division
import MySQLdb;
import numpy as np
import spacy
import re
import operator
import math
import collections
from nltk.stem import PorterStemmer
from collections import OrderedDict
from nltk.tag import StanfordPOSTagger as POS_tag
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
import statistics
from numpy import isnan, Inf
import nltk
from collections import OrderedDict, Counter
import time
import random
start_time=time.time()

nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
np.seterr(divide='ignore', invalid='ignore')

# annotated_words={'islam', 'biggest', 'isis', 'hell', 'drumpf', 'attack', 'dumbass', 'nasty', 'twats', 'feminist', 'retard', 'moron', 'donald', 'sunni', 'goddamn', 'bad', 'fatuous', 'ungrateful', 'clingy', 'prick', 'annoying', 'chemical', 'decisions', 'fake', 'dirty', 'disgusting', 'worst', 'bitch', 'weapons', 'ugly', 'tease', 'kick', 'bryce', 'frontin', 'dick', 'hoe', 'rot', 'republican', 'witch', 'pathetic', 'mice', 'ass', 'fuck', 'beat', 'hate', 'wrong', 'group', 'mad', 'humans', 'sick', 'bloody', 'bombs', 'anger', 'bear', 'jihadis', 'trynna', 'fascist', 'party', 'dumb', 'kill', 'females', 'idiot', 'evil', 'nigga', 'montel', 'legit', 'village', 'lyin', 'stupid', 'people', 'damn', 'racist', 'dictator', 'sarandon', 'stfu', 'bastard'}
# annotated_words={'business', 'email', 'app', 'tickets', 'league', 'trends', 'sterling', 'first', 'grand', 'giftsforhim', 'vacation', 'santa', 'support', 'bait', 'club', 'code', 'hot', 'search', 'love', 'luxury', 'celebrity', 'theatre', 'online', 'shop', 'johnson', 'place', 'news', 'dance', 'availability', 'download', 'placement', 'easy', 'website', 'video', 'opportunities', 'event', 'ass', 'minutes', 'ultimate', 'lifetime', 'visit', 'movie', 'trip', 'original', 'campaign', 'latest', 'win', 'full', 'beads', 'fashion', 'adventures', 'stores', 'wallet', 'ipad', 'xxx', 'bets', 'returns', 'instagram', 'purse', 'best', 'product', 'click', 'wifi', 'podcast', 'holiday', 'adidas', 'suede', 'new', 'game', 'book', 'coupon', 'gossip', 'bag', 'vintage', 'fuck', 'modern', 'clothing', 'better', 'start-ups', 'loans', 'show', 'watch', 'free', 'hair', 'silver', 'gold', 'card', 'great', 'car', 'amazon', 'page', 'youtube', 'night', 'thank', 'season', 'girl', 'pic', 'today', 'winner', 'bet', 'top', 'gift', 'music','woman', 'link', 'set', 'available', 'job', 'blog', 'soundcloud', 'weekend', 'sale', 'radiodisney', 'daily', 'april', 'home', 'ebay', 'shoe', 'month', 'chance', 'size', 'money', 'good', 'travel', 'plan', 'launch', 'collection', 'opening', 'check', 'here'}
# annotated_words={'kick', 'attack', 'ass', 'stupid', 'moron', 'females', 'drumpf', 'evil', 'wrong', 'rot', 'village', 'mice', 'clingy', 'bad', 'feminist', 'lil', 'terrify', 'shit', 'chemical', 'terrible', 'motherfucker', 'insane', 'bullshit', 'prick', 'bitch', 'freak', 'syria', 'girl', 'biggest', 'hoe', 'nasty', 'shitty', 'asshole', 'anger', 'dictator', 'donald', 'wrestlemania', 'frontin', 'republican', 'bear', 'dumb', 'nigga', 'ungrateful', 'part-time', 'sunni', 'pathetic', 'legit', 'idiot','dog', 'goddamn', 'sick', 'hell', 'awful', 'people', 'annoying', 'trump', 'twats', 'beat', 'witch', 'ugly', 'sarandon', 'bryce', 'worst', 'bombs', 'disgusting', 'retard', 'lyin', 'dick', 'montel', 'dirty', 'fatuous', 'damn', 'racist', 'islam', 'group', 'weapons', 'fuck', 'humans', 'woman', 'bloody', 'fake', 'jihadis', 'fascist', 'isis', 'hate', 'party', 'trynna', 'words', 'kill', 'dumbass', 'tease', 'mad', 'stfu','asf', 'hurt', 'tf', 'crap', 'hypocrite', 'wtf', 'clown', 'pussy', 'adult', 'troll', 'crazy', 'scumbag'} 
# annotated_words={'khalistan2020', 'khalistan', 'sikh', 'kashmir', 'indian', 'kashmirwantsfreedom', 'referendum2020', 'freedom', 'free', 'august', 'freekashmir', 'right', 'people', 'kashmirkhalistan_ajointcall', 'kashmirunderthreat', 'citizen', 'freejagginow', 'jagtar', 'kashmiris', 'movement', 'referendum', 'voice', 'state', 'community', 'independence', 'kashmiri', 'terrorist', 'flag', 'support', 'kashmirbanaygapakistan', 'torture', 'jaggi', 'justice', 'hindutva', 'independent', 'trial', 'johal', 'detainedfreejagginow', 'abduct', 'hardkaur', 'freekhalistan', 'khalistanmovement', 'guru', 'fight', 'sjf', 'ban', 'brave', 'jaggis'}
annotated_words={'ghazwaehind', 'muslim', 'kashmir', 'time', 'jihad', 'india', 'pakistan', 'indian', 'army', 'allah', 'kashmirhamarahai', 'zaidzamanhamid', 'war', 'ready', 'islam', 'pak', 'savekashmirso', 'zionist', 'kashmiri', 'right', 'kashmirbleed', 'solution', 'peaceforchange', 'final', 'article370', 'pakistani', 'standwithkashmir', 'full', 'taliban', 'decision', 'force', 'savekashmirforhumanity', 'reason', 'quran', 'battle', 'kufr', 'kaffir', 'allahuackber', 'nonmuslim', 'soldier', 'call', 'mujahideen', 'naraetakbeer', 'conquer', 'sunnahjihadkhilafat', 'eyeopener', 'brother', 'menace', 'Hunduvta', 'terror', 'Bloodshed', 'warrior', 'WeStandWithKashmir', 'KashmirHamaraHai', 'KashmirParFinalFight', 'battle', 'ReleaseHafizSaeed', 'KashmirBanayGaPakistan', 'massacare', 'FreeKashmir', 'shahadat', 'martyred', 'Labayk', 'KashmirBleeds', 'Khilafat', 'Munafiq', 'KashmirUnderThreat', 'Zionists', 'warfare', 'MasoodAzhar', 'YasinMalick', 'Afzalguru', 'MullaUmar', 'soldiers', 'Freedom', 'fighters', 'HinduTerror', 'JihadIsFinalTreatmentForIndia', 'weapons', 'DesiringMartyrdom', 'blood', 'Allah', 'believers', 'followers', 'join', 'Kuffar', 'sacrifice', 'Ummat-e-Muslima', 'camps', 'suicide', 'fighter'}

# seed_words={'system', 'control', 'linear'}
# seed_words={'hate', 'nigga', 'idiot'}
# seed_words={'free', 'click', 'show'}
# seed_words={'fuck', 'bitch', 'ass'}
# seed_words={'khalistan2020', 'sikh', 'freejagginow'}
seed_words={'kashmir', 'jihad', 'ghazwaehind'}


def word_seedsim(tweets_words, seed_words, embedding_model):
	word_seedword_sim={}
	noofseedword=len(seed_words)
	for word in tweets_words:
		avg_sim=0
		for sword in seed_words:
			try:
				cos_sim=cosine_similarity(np.array(embedding_model[word]).reshape(1,100), np.array(embedding_model[sword]).reshape(1, 100))[0][0]
				avg_sim=avg_sim+cos_sim
			except:
				continue
		avg_sim=avg_sim/noofseedword
		word_seedword_sim[word]=avg_sim
	return word_seedword_sim

def word_cooccur_TextRank(tweets_words, word_index, word_pair_freq):
	nrow=ncol=len(tweets_words)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(tweets_words)):
	 	for j in range(len(tweets_words)):
	 		
	 		word_pair=tweets_words[i]+":"+tweets_words[j]
 			if word_pair in word_pair_freq.keys():
	 			word_cooccur_mat[word_index[tweets_words[i]]][word_index[tweets_words[j]]]=word_pair_freq[word_pair]
	 		
	word_cooccur_mat_nor=word_cooccur_mat/np.sum(word_cooccur_mat, axis=0)
	NaNs=isnan(word_cooccur_mat_nor)
	word_cooccur_mat_nor[NaNs]=0

	return word_cooccur_mat_nor

def word_cooccur_RAKE(tweets_words, word_index, term_freq_dict, candidate_keywords):
	nrow=ncol=len(tweets_words)
	word_cooccur_mat=np.zeros((nrow, ncol))
	
	for i in range(len(tweets_words)):
		f_index=word_index[tweets_words[i]]
		word_cooccur_mat[f_index][f_index]=term_freq_dict[tweets_words[i]]
		for j in range(i+1, len(tweets_words)):
			s_index=word_index[tweets_words[j]]
			co_occur_count=0
			for candidate in candidate_keywords:
				if tweets_words[i] in candidate.split(' ') and tweets_words[j] in candidate.split(' '):
					co_occur_count=co_occur_count+1

			word_cooccur_mat[f_index][s_index]=co_occur_count
			word_cooccur_mat[s_index][f_index]=co_occur_count

	return word_cooccur_mat

def adj_matrix_PAGERANK(tweets_words, word_index, tweets_words_list):
	nrow=ncol=len(tweets_words)
	word_adj_mat=np.zeros((nrow, ncol))
	
	for i in range(len(tweets_words)):
	 	for j in range(i+1, len(tweets_words)):
	 		flag=False
	 		for tw_word_list in tweets_words_list:
	 			if tweets_words[i] in tw_word_list and tweets_words[j] in tw_word_list:
	 				flag=True
	 				break

	 		if flag==True:
		 		word_adj_mat[word_index[tweets_words[i]]][word_index[tweets_words[j]]]=1
		 		word_adj_mat[word_index[tweets_words[j]]][word_index[tweets_words[i]]]=1

	return word_adj_mat


def word_cooccur_CNW(tweets_words, word_index, tweets_words_list, term_freq_dict_up):
	nrow=ncol=len(tweets_words)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_adj=np.zeros((nrow, ncol))
	
	for i in range(len(tweets_words)):
		f_index=word_index[tweets_words[i]]
		for j in range(len(tweets_words)):
			s_index=word_index[tweets_words[j]]
			co_occur_count=0
			for tw_word_list in tweets_words_list:
				if tweets_words[i] in tw_word_list and tweets_words[j] in tw_word_list:
					co_occur_count=co_occur_count+1

			f_count=term_freq_dict_up[tweets_words[i]]
			s_count=term_freq_dict_up[tweets_words[j]]
			edge_weight=(co_occur_count*1.0)/(f_count+s_count-co_occur_count)
			word_cooccur_mat[f_index][s_index]=edge_weight
			word_cooccur_mat[s_index][f_index]=edge_weight
			word_cooccur_mat_adj[f_index][s_index]=edge_weight
			word_cooccur_mat_adj[s_index][f_index]=edge_weight

	nonzer_indices=np.nonzero(word_cooccur_mat_adj)
	word_cooccur_mat_adj[nonzer_indices]=1					#Code to replace nonzero values with 1
	
	return word_cooccur_mat, word_cooccur_mat_adj


def word_cooccur_CIW(tweets_words, word_index, bigram_count, term_count_dict, embedding_model):
	nrow=ncol=len(tweets_words)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(tweets_words)):
		for j in range(i+1, len(tweets_words)):
			
			bigram_fir=tuple([tweets_words[i]]+[tweets_words[j]])
			bigram_sec=tuple([tweets_words[j]]+[tweets_words[i]])
			
			if bigram_count[bigram_fir]>0 or bigram_count[bigram_sec]>0:
				freq_1=term_count_dict[tweets_words[i]]
				freq_2=term_count_dict[tweets_words[j]]
				
				try:
					euclidean_dis=cosine_similarity(np.array(embedding_model[tweets_words[i]]).reshape(1,100), np.array(embedding_model[tweets_words[j]]).reshape(1, 100))[0][0]
				except:
					euclidean_dis=0
				
				if euclidean_dis==0:
					informativeness=0
				else:
					informativeness=(freq_1*freq_2)/euclidean_dis

				word_cooccur_count=bigram_count[bigram_fir]+bigram_count[bigram_sec]
				phraseness=(2*word_cooccur_count)/(freq_1+freq_2)			#It is dice coefficient or score
				
				attraction_score=informativeness*phraseness
		
				word_cooccur_mat[word_index[tweets_words[i]]][word_index[tweets_words[j]]]=attraction_score
				word_cooccur_mat[word_index[tweets_words[j]]][word_index[tweets_words[i]]]=attraction_score

	word_cooccur_mat_nor=word_cooccur_mat/np.sum(word_cooccur_mat, axis=0)
	NaNs=isnan(word_cooccur_mat_nor)
	word_cooccur_mat_nor[NaNs]=0

	return word_cooccur_mat_nor


def indexing_fun(tweets_words):
	i=0
	word_index={}
	index_word={}
	for word in tweets_words:
		word_index[word]=i
		i=i+1

	for word in word_index.keys():
		index_word[word_index[word]]=word

	return word_index, index_word

def dataset_SARNA(table_name, label, rowcount):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cur=connection.cursor(MySQLdb.cursors.DictCursor)
		
		cur.execute("SELECT * FROM "+table_name+" where tweet_lang='en' limit "+str(rowcount)+"")
		result=cur.fetchall()
		
		path_to_model='StanPOSTagger/models/english-bidirectional-distsim.tagger'
		path_to_taggerjar='StanPOSTagger/stanford-postagger.jar'
		ps = PorterStemmer()

		tweets_words_list=[]
		term_count_dict={}
		for row in result:
			tweet_words=set()
			tweet=re.sub(r'http\S+', '', row['tweet']).strip()
			st = POS_tag(model_filename=path_to_model, path_to_jar=path_to_taggerjar, encoding='utf8')
			tags=st.tag(tweet.split())
			
			valid_tags=['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
			for each_tag in tags:
				if each_tag[1] in valid_tags:
					stem_word=ps.stem(each_tag[0])
					tweet_words.add(stem_word.strip())

					if stem_word.strip() in term_count_dict.keys():
						term_count_dict[stem_word.strip()]=term_count_dict[stem_word.strip()]+1
					else:
						term_count_dict[stem_word.strip()]=1
			
			tweets_words_list.append(list(tweet_words))

		tweets_words=list(set(term_count_dict.keys())-seed_words)

	finally:
		connection.close()

	return seed_words, tweets_words, tweets_words_list, term_count_dict


def cooccur_AND_network_SARNA(tweets_words, seed_words, tweets_words_list, term_count_dict):
	network_dict=OrderedDict()

	for sword in seed_words:
		for nword in tweets_words:
			key=sword+":"+nword
			word_cooccur_count=0
			for tw_word_list in tweets_words_list:
				if sword in tw_word_list and nword in tw_word_list:
					word_cooccur_count=word_cooccur_count+1
			
			prob_sword=(term_count_dict[sword]*1.0)/len(tweets_words_list)
			prob_nword=(term_count_dict[nword]*1.0)/len(tweets_words_list)

			prob_cooccur=(word_cooccur_count*1.0)/len(tweets_words_list)
			
			prob_new_when_seed=prob_seed_when_new=0.0
			if prob_sword!=0.0:
				prob_new_when_seed=prob_cooccur/prob_sword
			
			if prob_nword!=0.0:
				prob_seed_when_new=prob_cooccur/prob_nword

			if key in network_dict:
				network_dict[key].append(prob_sword)
				network_dict[key].append(prob_nword)
				network_dict[key].append(prob_cooccur)
				network_dict[key].append(prob_new_when_seed)
				network_dict[key].append(prob_seed_when_new)
			else:
				network_dict[key]=[]
				network_dict[key].append(prob_sword)
				network_dict[key].append(prob_nword)
				network_dict[key].append(prob_cooccur)
				network_dict[key].append(prob_new_when_seed)
				network_dict[key].append(prob_seed_when_new)

	return network_dict


def link_pruning(network_dict):
	significant_link=[]
	significant_words=[]
	for key, value in network_dict.items():
		new_when_seed_numerator=value[3]-value[2]
		seed_when_new_numerator=value[4]-value[2]
		chi_square_value=0
		if value[2]!=0:
			key1=key.split(":")
			new_when_seed_numerator_factor=(new_when_seed_numerator*new_when_seed_numerator)/value[2]
			seed_when_new_numerator_factor=(seed_when_new_numerator*seed_when_new_numerator)/value[2]
			chi_square_value=new_when_seed_numerator_factor+seed_when_new_numerator_factor
			
			if chi_square_value<3.841:
				significant_link.append(key)
				significant_words.append(key1[1])
	print ("significnt keys===", significant_link)

	print("selected words==", significant_words, "==", len(significant_words))
	commonwords=annotated_words.intersection(set(significant_words))
	
	precision=(len(commonwords)*1.0)/len(significant_words)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	if precision==0 and recall==0:
		fscore=0
	else:
		fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision==", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


def term_freq_in_tweets(tweets_words, tweets_words_list):
	term_tweetcount_dict={}
	
	for word in tweets_words:
		term_freq=0
		for tw_word_list in tweets_words_list:
			if word in tw_word_list:
				term_freq=term_freq+1
		term_tweetcount_dict[word]=term_freq
	return term_tweetcount_dict


def random_sampling(dataset, rowcount):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT tweet FROM benchmark_dataset_text WHERE label='%s'" % (dataset))
		results=cursor.fetchall()
		tweet_list=[]
		for row in results:
			tweet=re.sub(r'http\S+', '', row['tweet']).strip()
			tweet=re.sub(r'[.:!\'/(@#$?-]', '', tweet).lower()
			if tweet.startswith("RT"):
				tweet=tweet[len("RT"):].strip()

			tweet_list.append(tweet)
		
		sampled_tweet=random.sample(tweet_list, rowcount)
	finally:
		cursor.close()
		connection.close()

	return sampled_tweet


def dataset_tf(dataset, rowcount):
	sampled_tweet= random_sampling(dataset, rowcount)			# select a random sample of n tweets from the total tweets
	term_count_dict={}
	tweets_words_list=[]
	for tweet in sampled_tweet:
		tweet_words=[]											#list to store words in each tweet
		processed=nlp(tweet)									#pass the tweet from nlp pipeline
		for word in processed:
			stemword=word.lemma_
			tweet_words.append(stemword)

			if stemword in term_count_dict.keys():				#condition to find the occurrence frequency of each word
				term_count_dict[stemword]=term_count_dict[stemword]+1
			else:
				term_count_dict[stemword]=1
		tweets_words_list.append(tweet_words)

	tweets_words=list(term_count_dict.keys())
	
	return tweets_words, tweets_words_list, term_count_dict 


#	BaseLine#1
def tf_based_keywords():
	print("==tf_based_keywords==")
	dataset=input("Please enter the dataset: ")
	tweets_words, tweets_words_list, term_count_dict=dataset_tf(dataset, 1000)
	sorted_list_term=sorted(term_count_dict.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


# Baseline#2
def tf_idf_based_keywords():
	print("==tf_idf_based_keywords==")
	dataset=input("Please enter the dataset: ")
	tweets_words, tweets_words_list, term_count_dict=dataset_tf(dataset, 1000)
	term_tweetcount_dict=term_freq_in_tweets(tweets_words, tweets_words_list)	#function to find the occurrence of each word in number of tweets

	tweet_count=len(tweets_words_list)
	term_rel_dict={}
	for word in tweets_words:
		tf=math.log10(term_count_dict[word])+1
		idf=math.log10(1+(tweet_count/term_tweetcount_dict[word]))
		tf_idf=tf*idf
		term_rel_dict[word]=tf_idf

	sorted_list_term=sorted(term_rel_dict.items(), key=operator.itemgetter(1))	#sorting function to sort the terms based on their values

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


#	BaseLine#3
def embedding_based_keywords():
	print("==embedding_based_keywords==")
	dataset=input("Please enter the dataset: ")
	tweets_words, tweets_words_list, term_count_dict=dataset_tf(dataset, 1000)
	embedding_model=Word2Vec.load('embedding/Khalistan_Movement/Khalistan_Movement_model2.bin')
	word_seedword_sim=word_seedsim(tweets_words, seed_words, embedding_model)

	sorted_list_emb=sorted(word_seedword_sim.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_list_emb))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")



#	BaseLine#4
def pageRank():
	print("==pageRank==")
	dataset=input("Please enter the dataset: ")
	
	valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
	sampled_tweet=random_sampling(dataset, 1000)		

	tweets_words_list=[]
	tweets_words_conet=set()
	for tweet in sampled_tweet:
		tweet_words=[]
		processed=nlp(tweet)
		for word in processed:
			stemword=word.lemma_
			
			if word.tag_ in valid_keywords and len(word)>2:
				tweet_words.append(stemword)
				tweets_words_conet.add(stemword)
		
		tweets_words_list.append(tweet_words)

	tweets_words=list(tweets_words_conet)
	word_count=len(tweets_words)

	print("phase1 complete")
	word_index, index_word= indexing_fun(tweets_words)
	word_adj_mat=adj_matrix_PAGERANK(tweets_words, word_index, tweets_words_list)
	word_adj_mat_nor=word_adj_mat/np.sum(word_adj_mat, axis=0)
	NaNs=isnan(word_adj_mat_nor)						# Function to find the nan in the matrix generated due to 0/0 dividion
	word_adj_mat_nor[NaNs]=0							# Replaces nan with 0
	print("phase2 complete")
	
	word_rank=np.full((word_count, 1), 1)
	damp_val=(1-0.85)/word_count
	damp_vec=np.full((word_count, 1), damp_val)
	flag=True
	pre_word_rank=word_rank

	print("phase3 complete")
	while flag==True:
		uprank=0.85*(np.dot(word_adj_mat_nor, word_rank))
		word_rank=damp_vec+uprank
		rankdiff=sum(abs(pre_word_rank-word_rank))
		pre_word_rank=word_rank

		if rankdiff <= 0.0001:
			flag=False

	final_word_rank={}
	for index, rank in enumerate(word_rank):
		final_word_rank[index_word[index]]=rank

	sorted_rank=sorted(final_word_rank.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_rank))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")



#	Domain-based#1
#This is the implementation of th index proposed by Kit and Liu in Measuring mono-word termhood by rank difference via corpus comparison
def term_domain_specificity():
	print("==term_domain_specificity===")
	dataset=input("Please enter the dataset: ")
	tweets_words, tweets_words_list, term_count_dict=dataset_tf(dataset, 1000)
	dataset_con=input("Please enter the contrasting dataset: ")
	tweets_words_con, tweets_words_list_con, term_count_dict_con=dataset_tf(dataset_con, 1000)

	sorted_wordlist=sorted(term_count_dict.items(), key=operator.itemgetter(1))
	
	tds_dict={}
	term_count_domain=sum(term_count_dict.values())
	print("term_count_domain==", term_count_domain)
	
	for key in list(reversed(sorted_wordlist)):
		nume=(term_count_dict[key[0]]*1.0)/term_count_domain
		if key[0] in term_count_dict_con.keys():
			deno_n=(term_count_dict_con[key[0]]*1.0)/sum(term_count_dict_con.values())
			tds=nume/deno_n
			tds_dict[key[0]]=tds

	sorted_tds=sorted(tds_dict.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_tds))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")



#	Domain-based#2
#This is the implementation of tds index proposed by Park et al. in An empirical analysis of word error rate and keyword error rate
def termhood():
	print("==termhood==")
	dataset=input("Please enter the dataset: ")
	tweets_words, tweets_words_list, term_count_dict=dataset_tf(dataset, 1000)
	dataset_con=input("Please enter the contrasting dataset: ")
	tweets_words_con, tweets_words_list_con, term_count_dict_con=dataset_tf(dataset_con, 1000)

	sorted_wordlist=sorted(term_count_dict.items(), key=operator.itemgetter(1))
	sorted_wordlist_con=sorted(term_count_dict_con.items(), key=operator.itemgetter(1))

	sorted_wordlist_ss=collections.OrderedDict(list(reversed(sorted_wordlist_con)))

	sorted_list=list(sorted_wordlist_ss.keys())

	voc_size_h=len(tweets_words)
	voc_size_s=len(tweets_words_con)
	rank_h=voc_size_h

	thd_dict={}
	for key in list(reversed(sorted_wordlist))[:80]:
		if key[0] in term_count_dict_con.keys():
			firstterm=(rank_h/voc_size_h)
			rank_s=sorted_list.index(key[0])
			secondterm=(rank_s/voc_size_s)
			thd=firstterm-secondterm
			thd_dict[key[0]]=thd
			rank_h=rank_h-1

	sorted_thd=sorted(thd_dict.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_thd))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


#	Domain-based#3
def RAKE():
	print("RAKE")

	dataset=input("Please enter the dataset:")
	sampled_tweet=random_sampling(dataset, 1000)

	candidate_keywords=[]
	term_freq_dict={}
	for tweet in sampled_tweet:
		tweet=tweet.replace(',', ', ')
		tweet=tweet.replace('.', '. ')
		tweet=tweet.replace(';', '; ')
		tweet=tweet.replace(':', ': ')
		tweet=tweet.replace('  ', ' ')
		tweet=tweet.replace('?', ' ')
		tokens=list(filter(None, re.split('\$|\)|!|-|/| |\n', tweet)))

		keyword=[]
		for token in tokens:
			token=token.strip().lower()
			
			if len(token)>2 and token not in spacy_stopwords and token[-1]!=',' and token[-1]!='.' and token[-1]!=';' and token[-1]!=':':
				if token in term_freq_dict.keys():				#if token is not typed using str, it will be an nlp token and throw an error
					term_freq_dict[token]=term_freq_dict[token]+1
				else:
					term_freq_dict[token]=1

				keyword.append(token)				#token is added so to obtain consecutive words
			
			elif token[-1]==',' and token[-1]=='.' and token[-1]==';' and token[-1]==':':
				token=token[:-1]
				if token in term_freq_dict.keys():				#if token is not typed using str, it will be an nlp token and throw an error
					term_freq_dict[token]=term_freq_dict[token]+1
				else:
					term_freq_dict[token]=1
				
				keyword.append(token)
				candidate_keywords.append(' '.join(keyword))
				keyword=[]
			
			else:
				if len(keyword)==0:
					continue
				else:
					candidate_keywords.append(' '.join(keyword))
					keyword=[]

	tweets_words=list(term_freq_dict.keys())
	print("phase1")
	
	word_index, index_word= indexing_fun(tweets_words)
	word_cooccur_mat=word_cooccur_RAKE(tweets_words, word_index, term_freq_dict, candidate_keywords)
	word_deg_vec=np.sum(word_cooccur_mat, axis=1)	#This function sums the matrix row wise and generate a colmn vector
	print("phase2")

	word_deg_dict={}
	for index, deg in enumerate(word_deg_vec):
		word_deg_dict[index_word[index]]=deg

	word_rele={}
	for word in word_deg_dict.keys():
		word_rel=(word_deg_dict[word]*1.0)/term_freq_dict[word]
		word_rele[word]=word_rel

	sorted_list=sorted(word_rele.items(), key=operator.itemgetter(1))
	print("phase3")
	
	sel_words_list=[]
	for key in list(reversed(sorted_list))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")



#	Graph-based#1
def TextRank():
	print("==textrank==")

	dataset=input("Please enter the dataset:")
	valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
	sampled_tweet=random_sampling(dataset, 1000)

	tweets_words_conet=set()
	word_pair_freq={}
	
	for tweet in sampled_tweet:
		tokens=nlp(tweet)

		for i in range(len(tokens)-1):
			if len(tokens[i])>2 and len(tokens[i+1])>2:
				if tokens[i].tag_ in valid_keywords and tokens[i+1].tag_ in valid_keywords:
					word_pair=tokens[i].lemma_+":"+tokens[i+1].lemma_

					if word_pair in word_pair_freq.keys():
						word_pair_freq[word_pair]=word_pair_freq[word_pair]+1
					else:
						word_pair_freq[word_pair]=1
		
		for word in tokens:
			stemword=word.lemma_
			
			if word.tag_ in valid_keywords and len(word)>2:
				tweets_words_conet.add(stemword)
		
	tweets_words=list(tweets_words_conet)

	word_index, index_word= indexing_fun(tweets_words)
	word_adj_mat_nor=word_cooccur_TextRank(tweets_words, word_index, word_pair_freq)
	
	word_count=len(tweets_words)
	word_rank=np.full((word_count, 1), 1)
	damp_val=(1-0.85)
	damp_vec=np.full((word_count, 1), damp_val)
	
	flag=True
	pre_word_rank=word_rank

	while flag==True:
		uprank=0.85*(np.dot(word_adj_mat_nor, word_rank))
		word_rank=damp_vec+uprank
		rankdiff=sum(abs(pre_word_rank-word_rank))
		pre_word_rank=word_rank

		if rankdiff <= 0.0001:
			flag=False

	final_word_rank={}
	for index, rank in enumerate(word_rank):
		final_word_rank[index_word[index]]=rank

	sorted_rank=sorted(final_word_rank.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_rank))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


# Graph-based 2
#Second graph-based approach implementation of Corpus-independent Generic Keyphrase Extraction Using Word Embedding Vectors  by Rui Wang  
def CorpusIndependentWang():
	start_time = time.time()
	print("==CorpusIndependentWang==")

	dataset=input("Please enter the dataset:")
	sampled_tweet=random_sampling(dataset, 560)
	valid_tags=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
	# embedding_model=Word2Vec.load('embedding/hate_model2.bin')
	embedding_model=Word2Vec.load('embedding/Jihad_Tweets/Jihad_Tweets_model2.bin')
	print("embedding loaded")
		
	term_count_dict={}
	bigram_list=[]

	for tweet in sampled_tweet:
		split_text = list(filter(None, re.split('; |\$|\)|\?|!|-|:|/|;|; |;| |\n|\. |\.', tweet)))
		bigram = list(nltk.bigrams(split_text))
		bigram_list=bigram_list+bigram

		tokens=nlp(tweet)
		# tags=st.tag(split_text)				#Spacy is significantly faster than Stanford parser
		
		for each_token in tokens:
			if each_token.tag_ in valid_tags:
				word=each_token.text

				if word in term_count_dict.keys():
					term_count_dict[word]=term_count_dict[word]+1
				else:
					term_count_dict[word]=1

	tweets_words=list(set(term_count_dict.keys()))
	bigram_count=Counter(bigram_list)

	print("Bigram_count Completed")

	word_index, index_word= indexing_fun(tweets_words)
	word_cooccur_mat_nor=word_cooccur_CIW(tweets_words, word_index, bigram_count, term_count_dict, embedding_model)
	print("Cooccurrence Matrix Completed")
	
	word_count=len(tweets_words)
	word_rank=np.full((word_count, 1), 1)
	damp_val=(1-0.85)
	damp_vec=np.full((word_count, 1), damp_val)
	
	flag=True
	pre_word_rank=word_rank

	while flag==True:
		uprank=0.85*(np.dot(word_cooccur_mat_nor, word_rank))
		word_rank=damp_vec + uprank
		diff=abs(pre_word_rank-word_rank)
		NaNs=isnan(diff)
		diff[NaNs]=0
		rankdiff=sum(diff)
		pre_word_rank=word_rank

		if rankdiff <= 0.0001:
			flag=False

	final_word_rank={}
	for index, rank in enumerate(word_rank):
		final_word_rank[index_word[index]]=rank

	sorted_rank=sorted(final_word_rank.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_rank))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	den=precision+recall
	if den==0.0:
		fscore=0.0
	else:
		fscore=(2*precision*recall)/den

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")

	print("--- %s seconds ---" % (time.time() - start_time))



#Third graph-based approach implementation of A graph based keyword extraction model using collective node weight
def graph_CNW():		#CNW stands for collective node weight
	print("==CNW==")

	dataset=input("Please enter the dataset:")
	sampled_tweet=random_sampling(dataset, 1000)	

	tweets_words_list=[]
	term_freq_dict={}
	for tweet in sampled_tweet:
		tweet_words=[]
		tokens=nlp(tweet)						#Spacy NLP pipeline

		for token in tokens:
			if len(token)>2 and token.is_stop==False:
				if str(token) in term_freq_dict.keys():
					term_freq_dict[str(token)]=term_freq_dict[str(token)]+1
				else:
					term_freq_dict[str(token)]=1

				tweet_words.append(str(token))
		tweets_words_list.append(tweet_words)
	
	AOC=st.mean(term_freq_dict.values())			#Threshold calculation to select the relevant word
	print(AOC, "===phase1")

	term_freq_dict_up={}				#this dict has only those terms having frequency greater than the threshold AOC
	tweets_words=[]
	for term in term_freq_dict.keys():
		if term_freq_dict[term]>AOC:
			term_freq_dict_up[term]=term_freq_dict[term]
			tweets_words.append(term)

	word_index, index_word= indexing_fun(tweets_words)
	word_cooccur_mat, word_cooccur_mat_adj=word_cooccur_CNW(tweets_words, word_index, tweets_words_list, term_freq_dict_up)
	print("phase2")

	word_deg_vec=np.sum(word_cooccur_mat_adj, axis=1)			#sum the adjacency martix by row producing the degree of each node
	dist_from_cent=max(word_deg_vec)-np.array(word_deg_vec) 
	dist_from_cent=1/dist_from_cent
	dist_from_cent[dist_from_cent==Inf]=1						#It replaces each Inf in the list with 1

	selectivity_centrality=np.sum(word_cooccur_mat, axis=1)/np.sum(word_cooccur_mat_adj, axis=1)		#it divide the sum of weight of edges by number of connections(edges) 
	imp_neigh_nodes=np.dot(word_cooccur_mat_adj, selectivity_centrality)/np.sum(word_cooccur_mat_adj, axis=1)	#it find the relevance of a word based on the weight of their neighbors
	
	part_word_weight=dist_from_cent+selectivity_centrality+imp_neigh_nodes		#Sum of three relevance scores

	#loop to find the word corresponding to a index through enumeration
	part_word_weight_dict={}
	for index, weight in enumerate(part_word_weight):
		part_word_weight_dict[index_word[index]]=weight

	#Loop to add the word frequency to find the final weight
	final_weight_dict={}
	for word in tweets_words:
		final_weight_dict[word]=part_word_weight_dict[word]+term_freq_dict_up[word]

	print("phase3")

	sorted_rank=sorted(final_weight_dict.items(), key=operator.itemgetter(1))
	
	sel_words_list=[]
	for key in list(reversed(sorted_rank))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")



#Third graph-based approach implementation of PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents
def PositionRank():
	print("==PositionRank==")
	dataset=input("Please enter the dataset:")
	valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
	sampled_tweet=random_sampling(dataset, 1000)

	word_pair_freq={}
	pos_rank_dict={}
	for tweet in sampled_tweet:
		tokens=nlp(tweet)					#Spacy NLP pipeline

		for i in range(len(tokens)-1):
			if len(tokens[i])>2 and len(tokens[i+1])>2:
				if tokens[i].tag_ in valid_keywords and tokens[i+1].tag_ in valid_keywords:
					word_pair=tokens[i].lemma_+":"+tokens[i+1].lemma_

					if word_pair in word_pair_freq.keys():
						word_pair_freq[word_pair]=word_pair_freq[word_pair]+1
					else:
						word_pair_freq[word_pair]=1

		j=1
		for token in tokens:
			pos_rank=1/j
			j=j+1
			if len(token)>2 and token.is_stop==False:
				stemword=token.lemma_
				if stemword in pos_rank_dict.keys():
					pos_rank_dict[stemword]=pos_rank_dict[stemword]+pos_rank
				else:
					pos_rank_dict[stemword]=pos_rank
	
	tweets_words=list(pos_rank_dict.keys())

	word_index, index_word= indexing_fun(tweets_words)
	word_adj_mat_nor=word_cooccur_TextRank(tweets_words, word_index, word_pair_freq)
	
	word_count=len(tweets_words)
	word_rank=np.full((word_count, 1), 1/word_count)
	
	damp_vec_mat=np.zeros((word_count, 1))
	for word in tweets_words:
		damp_vec_mat[word_index[word]][0]=pos_rank_dict[word]

	damp_vec_mat=damp_vec_mat/np.sum(damp_vec_mat, axis=0)
	damp_vec_mat=0.15*damp_vec_mat
	
	flag=True
	pre_word_rank=word_rank

	while flag==True:
		uprank=0.85*(np.dot(word_adj_mat_nor, word_rank))
		word_rank=damp_vec_mat+uprank
		rankdiff=sum(abs(pre_word_rank-word_rank))
		pre_word_rank=word_rank

		if rankdiff <= 0.001:
			flag=False

	final_word_rank={}
	for index, rank in enumerate(word_rank):
		final_word_rank[index_word[index]]=rank

	sorted_rank=sorted(final_word_rank.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_rank))[0:80]:
		sel_words_list.append(key[0])

	print("selected words==", sel_words_list, "==", len(sel_words_list))
	commonwords=annotated_words.intersection(set(sel_words_list))
	
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)

	print("common words==", commonwords, "==", len(commonwords))
	print("len_annot==", len(annotated_words))
	print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")


def compared_approach_sarna():
	print("==sarna==")
	seed_words, tweets_words, tweets_words_list, term_count_dict=dataset_SARNA("Jihad_Tweets", "hateful", 1000)
	print("==phase1==")
	network_dict=cooccur_AND_network_SARNA(tweets_words, seed_words, tweets_words_list, term_count_dict)
	print("==phase2==")
	link_pruning(network_dict)



# def graph_construction():
# compared_approach_sarna()
# Baseline Approaches
# tf_based_keywords()
# tf_idf_based_keywords()
# embedding_based_keywords()
# pageRank()
# ReferenceVectorAlgorithm()

#Contrasting Corpora-Based Approaches
# term_domain_specificity()
# termhood()
# RAKE()

#Graph_based_approaches
# TextRank()
# CorpusIndependentWang()
# graph_CNW()
# PositionRank()
# compared_approach_sarna()
# testing()