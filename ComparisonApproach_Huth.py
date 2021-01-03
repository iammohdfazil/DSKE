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
from random import sample
import itertools
start_time=time.time()

nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
np.seterr(divide='ignore', invalid='ignore')

# seed_words={'system', 'control', 'linear'}
# seed_words={'hate', 'nigga', 'idiot'}
# seed_words={'free', 'click', 'show'}
# seed_words={'fuck', 'bitch', 'ass'}
# seed_words={'khalistan2020', 'sikh', 'freejagginow'}
# seed_words={'kashmir', 'jihad', 'ghazwaehind'}


def word_seedsim_pro(word_list, bigram_count, embedding_model, term_freq_dict, nor_term_dict):
	word_seedword_sim={}
	noofseedword=len(seed_words)

	for word in word_list:
		avg_sim=0
		tf_ratio=[]
		word_sword_cocount=0
		num_count=0
		for sword in seed_words:
			bigram_fir=tuple([word]+[sword])
			bigram_sec=tuple([sword]+[word])
			
			if bigram_count[bigram_fir]>0 or bigram_count[bigram_sec]>0:
				word_sword_cocount=bigram_count[bigram_fir]+bigram_count[bigram_sec]
			
			tf_ratio.append(word_sword_cocount/(term_freq_dict[word]+term_freq_dict[sword]-word_sword_cocount))

			try:
				cos_sim=cosine_similarity(np.array(embedding_model[word]).reshape(1,100), np.array(embedding_model[sword]).reshape(1, 100))[0][0]
				avg_sim=avg_sim+cos_sim
				num_count=num_count+1
			except:
				continue
			
		if num_count!=0:
			avg_sim=avg_sim/num_count

		tf_ratio=statistics.mean(tf_ratio)
		num=(term_freq_dict[word]*1.0)/(sum(term_freq_dict.values()))
		if word in nor_term_dict.keys():
			den=(nor_term_dict[word]*1.0)/(sum(nor_term_dict.values()))
			domain_spec=(num/den)
		else:
			domain_spec=num
		semantic_sim=avg_sim*tf_ratio
		word_seedword_sim[word]=semantic_sim
	return word_seedword_sim


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


def word_seedsim(words_set, embedding_model):
	word_seedword_sim={}
	noofseedword=len(seed_words)
	
	for word in words_set:
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

def word_cooccur_TextRank(words_list, word_index, bigram_count):
	nrow=ncol=len(words_list)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(words_list)):
	 	for j in range(len(words_list)):
	 		
	 		bigram=tuple([words_list[i]]+[words_list[j]])
 			if bigram in bigram_count.keys():
	 			word_cooccur_mat[word_index[words_list[i]]][word_index[words_list[j]]]=bigram_count[bigram]
	 		
	word_cooccur_mat_nor=word_cooccur_mat/np.sum(word_cooccur_mat, axis=0)
	NaNs=isnan(word_cooccur_mat_nor)
	word_cooccur_mat_nor[NaNs]=0

	return word_cooccur_mat_nor

def word_cooccur_RAKE(words_list, word_index, term_freq_dict, bigram_count):
	nrow=ncol=len(words_list)
	word_cooccur_mat=np.zeros((nrow, ncol))
	
	for i in range(len(words_list)):
		f_index=word_index[words_list[i]]
		word_cooccur_mat[f_index][f_index]=term_freq_dict[words_list[i]]
		for j in range(i+1, len(words_list)):
			s_index=word_index[words_list[j]]

			bigram=tuple([words_list[i]]+[words_list[j]])
			co_occur_count=bigram_count[bigram]

			word_cooccur_mat[f_index][s_index]=co_occur_count
			word_cooccur_mat[s_index][f_index]=co_occur_count

	return word_cooccur_mat

def adj_matrix_PAGERANK(words_list, word_index, paper_words_list):
	nrow=ncol=len(words_list)
	word_adj_mat=np.zeros((nrow, ncol))
	
	for i in range(len(words_list)):
	 	for j in range(i+1, len(words_list)):
	 		flag=False
	 		for pa_word_list in paper_words_list:
	 			if words_list[i] in pa_word_list and words_list[j] in pa_word_list:
	 				flag=True
	 				break

	 		if flag==True:
		 		word_adj_mat[word_index[words_list[i]]][word_index[words_list[j]]]=1
		 		word_adj_mat[word_index[words_list[j]]][word_index[words_list[i]]]=1

	return word_adj_mat


def word_cooccur_CNW(words_set, word_index, sentence_words_list, term_freq_dict_up):
	nrow=ncol=len(words_set)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_adj=np.zeros((nrow, ncol))
	
	for i in range(len(words_set)):
		f_index=word_index[words_set[i]]
		for j in range(len(words_set)):
			s_index=word_index[words_set[j]]
			co_occur_count=0
			for sen_word_list in sentence_words_list:
				if words_set[i] in sen_word_list and words_set[j] in sen_word_list:
					co_occur_count=co_occur_count+1

			f_count=term_freq_dict_up[words_set[i]]
			s_count=term_freq_dict_up[words_set[j]]
			edge_weight=(co_occur_count*1.0)/(f_count+s_count-co_occur_count)
			word_cooccur_mat[f_index][s_index]=edge_weight
			word_cooccur_mat[s_index][f_index]=edge_weight
			word_cooccur_mat_adj[f_index][s_index]=edge_weight
			word_cooccur_mat_adj[s_index][f_index]=edge_weight

	nonzer_indices=np.nonzero(word_cooccur_mat_adj)
	word_cooccur_mat_adj[nonzer_indices]=1					#Code to replace nonzero values with 1
	
	return word_cooccur_mat, word_cooccur_mat_adj

def word_cooccur_PositionRank(words_list, word_index, word_pair_freq):
	nrow=ncol=len(words_list)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(words_list)):
	 	for j in range(len(words_list)):
	 		
	 		word_pair=words_list[i]+":"+words_list[j]
 			if word_pair in word_pair_freq.keys():
	 			word_cooccur_mat[word_index[words_list[i]]][word_index[words_list[j]]]=word_pair_freq[word_pair]
	 		
	word_cooccur_mat_nor=word_cooccur_mat/np.sum(word_cooccur_mat, axis=0)
	NaNs=isnan(word_cooccur_mat_nor)
	word_cooccur_mat_nor[NaNs]=0

	return word_cooccur_mat_nor

def word_cooccur_CIW(words_list, word_index, bigram_count, term_count_dict, embedding_model):
	nrow=ncol=len(words_list)
	word_cooccur_mat=np.zeros((nrow, ncol))
	word_cooccur_mat_nor=np.zeros((nrow, ncol))
	
	for i in range(len(words_list)):
		for j in range(i+1, len(words_list)):
			
			bigram_fir=tuple([words_list[i]]+[words_list[j]])
			bigram_sec=tuple([words_list[j]]+[words_list[i]])
			
			if bigram_count[bigram_fir]>0 or bigram_count[bigram_sec]>0:
				freq_1=term_count_dict[words_list[i]]
				freq_2=term_count_dict[words_list[j]]
				
				try:
					sem_dis=cosine_similarity(np.array(embedding_model[words_list[i]]).reshape(1,100), np.array(embedding_model[words_list[j]]).reshape(1, 100))[0][0]
				except:
					sem_dis=0
				
				if sem_dis==0:
					attraction_force=0
				else:
					attraction_force=(freq_1*freq_2)/sem_dis

				word_cooccur_count=bigram_count[bigram_fir]+bigram_count[bigram_sec]
				dice_score=(2*word_cooccur_count)/(freq_1+freq_2)
				
				attraction_score=attraction_force*dice_score
		
				word_cooccur_mat[word_index[words_list[i]]][word_index[words_list[j]]]=attraction_score
				word_cooccur_mat[word_index[words_list[j]]][word_index[words_list[i]]]=attraction_score

	word_cooccur_mat_nor=word_cooccur_mat/np.sum(word_cooccur_mat, axis=0)
	NaNs=isnan(word_cooccur_mat_nor)
	word_cooccur_mat_nor[NaNs]=0

	return word_cooccur_mat_nor


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

def dataset_SARNA(table_name, rowcount):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cur=connection.cursor(MySQLdb.cursors.DictCursor)
		
		cur.execute("SELECT * FROM "+table_name+" limit "+str(rowcount)+"")
		result=cur.fetchall()
		
		# path_to_model='StanPOSTagger/models/english-bidirectional-distsim.tagger'
		# path_to_taggerjar='StanPOSTagger/stanford-postagger.jar'
		# ps = PorterStemmer()
		# st = POS_tag(model_filename=path_to_model, path_to_jar=path_to_taggerjar, encoding='utf8')
		valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

		sentence_words_list=[]
		term_count_dict={}
		annotated_words=[]
		for row in result:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[:!\'/(@),#$?-]', '', row['Full']).lower().split('.')

			for sentence in text:
				sentence_words=set()
				tokens=nlp(sentence)
				# tokens=st.tag(sentence)

				for token in tokens:
					stem_word=token.lemma_.lower().strip()
					if token.tag_ in valid_keywords:
						# stem_word=ps.stem(token[0])
						sentence_words.add(stem_word)

						if stem_word in term_count_dict.keys():
							term_count_dict[stem_word]=term_count_dict[stem_word]+1
						else:
							term_count_dict[stem_word]=1

				sentence_words_list.append(list(sentence_words))

		annotated_words=set(annotated_words)
		words_list=list(set(term_count_dict.keys())-seed_words)

	finally:
		connection.close()

	return seed_words, words_list, sentence_words_list, term_count_dict, annotated_words


def cooccur_AND_network_SARNA(seed_words, words_list, sentence_words_list, term_count_dict):
	network_dict=OrderedDict()

	for sword in seed_words:
		for nword in words_list:
			key=sword+":"+nword
			word_cooccur_count=0
			for tw_word_list in sentence_words_list:
				if sword in tw_word_list and nword in tw_word_list:
					word_cooccur_count=word_cooccur_count+1

			try:			
				prob_sword=(term_count_dict[sword]*1.0)/len(sentence_words_list)
			except KeyError as e:
				prob_sword=0.0

			try:
				prob_nword=(term_count_dict[nword]*1.0)/len(sentence_words_list)
			except KeyError as e:
				prob_nword=0.0

			prob_cooccur=(word_cooccur_count*1.0)/len(sentence_words_list)
			
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


def link_pruning(network_dict, annotated_words):
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

	print("selected words==", significant_words[0:80], "==", len(significant_words[0:80]))
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


def term_freq_in_papers(words_set, paper_words_list):
	term_papercount={}				#Dictionary to tore the number of papers in which this term is found 
	
	for word in words_set:
		term_freq=0
		for tw_word_list in paper_words_list:
			if word in tw_word_list:
				term_freq=term_freq+1
		term_papercount[word]=term_freq
	return term_papercount

def dataset_tf(table_name, rowcount):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute("SELECT * FROM "+table_name+" limit "+str(rowcount)+" ")
		results=cursor.fetchall()

		term_count_dict={}
		paper_words_list=[]
		annotated_words=[]

		for row in results:											#iterate through each tweet
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list
			
			paper_words=[]										#list to store words in each tweet
			text=re.sub(r'[.:!\'/(@#$?-]', '', row["Full"]).lower()		#regular expression to filter stopwords
			
			processed=nlp(text)									#pass the tweet from nlp pipeline
			for word in processed:
				stemword=word.text 							#for embedding based approaches
				# stemword=word.lemma_
				paper_words.append(stemword)

				if stemword in term_count_dict.keys():				#condition to find the occurrence frequency of each word
					term_count_dict[stemword]=term_count_dict[stemword]+1
				else:
					term_count_dict[stemword]=1
			paper_words_list.append(paper_words)

		words_set=list(term_count_dict.keys())
		annotated_words=set(annotated_words)
	finally:
		cursor.close()
		connection.close()
		print("MySQL connection is closed")
	
	return words_set, paper_words_list, term_count_dict, annotated_words


def dataset_tf1(dataset, rowcount):
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute("SELECT tweet FROM benchmark_dataset_text WHERE label='%s'" % (dataset))
		results=cursor.fetchall()

		term_count_dict={}
		for row in results:											#iterate through each tweet
			tweet=re.sub(r'http\S+', '', row["tweet"]).strip()		#regular expression to filter URL
			tweet=re.sub(r'[.:!\'/(@#$?-]', '', tweet).lower()		#regular expression to filter stopwords

			if tweet.startswith("RT"):								#regular expression to filter RT from retweets
				tweet=tweet[len("RT"):].strip()

			tweet_list.append(tweet)
		
		sampled_tweet=random.sample(tweet_list, rowcount)
		
		for eachtweet in sampled_tweet:
			processed=nlp(eachtweet)									#pass the tweet from nlp pipeline
			for word in processed:
				stemword=word.lemma_

				if stemword in term_count_dict.keys():				#condition to find the occurrence frequency of each word
					term_count_dict[stemword]=term_count_dict[stemword]+1
				else:
					term_count_dict[stemword]=1

		tweets_words=set(term_count_dict.keys())			
	finally:
		cursor.close()
		connection.close()
		print("MySQL connection is closed")
	
	return tweets_words, term_count_dict


def Proposed_Approach():
	print("==Proposed_Approach==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()
		
		valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
		embedding_model=Word2Vec.load('embedding/Huth/Huth_model1.bin')	#
		# embedding_model=load_embedding()							#To read embedding model from pretrained Glove embedding stored in .txt format
		
		term_freq_dict={}
		annotated_words=[]
		bigram_list=[]
		for row in results:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[:!\'/(@),#$?-]', '', row['Full']).split('.')
			for sentence in text:
				sentence_words=[]
				processed=nlp(sentence.strip())
				for word in processed:
					stemword=word.text.lower()

					if word.tag_ in valid_keywords and len(word)>2:
						sentence_words.append(stemword)
						
						if stemword in term_freq_dict.keys():
							term_freq_dict[stemword]=term_freq_dict[stemword]+1
						else:
							term_freq_dict[stemword]=1
				
				blist=[]
				for i in range(len(sentence_words)):
					for j in range(i+1, len(sentence_words)):
						bigram=tuple([sentence_words[i]]+[sentence_words[j]])
						blist.append(bigram)

				bigram_list=bigram_list+blist

		annotated_words=set(annotated_words)
		words_list=list(term_freq_dict.keys())
		word_count=len(words_list)
		bigram_count=Counter(bigram_list)

		cursor.execute("SELECT * FROM BenchmarkDataset where tweet_lang='en' and label='normal' limit 1000")
		nor_results=cursor.fetchall()

		nor_term_dict={}
		for row in nor_results:
			tweet=re.sub(r'http\S+', '', row['tweet']).strip()
			tweet=re.sub(r'[.:!\'/(@#$?-]', '', tweet).lower()
			if tweet.startswith("RT"):
				tweet=tweet[len("RT"):].strip()

			processed=nlp(tweet)
			for word in processed:
				if len(str(word).strip())>2:
					stemword=word.text
	
					if stemword in nor_term_dict.keys():
						nor_term_dict[stemword]=word_freq=nor_term_dict[stemword]+1
					else:
						nor_term_dict[stemword]=1
		
		word_index, index_word= indexing_fun(words_list)
		word_seedword_sim=word_seedsim_pro(words_list, bigram_count, embedding_model, term_freq_dict, nor_term_dict)
		print("after word_cooccur==")
		word_cooccur_mat_nor=word_cooccur(words_list, bigram_count, word_index)
		print("after word_cooccur==")

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

		sel_words_list=[]
		for key in list(reversed(sorted_list))[0:3400]:
			sel_words_list.append(key[0])

		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		commonwords=annotated_words.intersection(set(sel_words_list))
	
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)

		print("common words==", commonwords, "==", len(commonwords))
		print("len_annot==", len(annotated_words))
		print("Precision===", precision, "		Recall==", recall, "		Fscore==", fscore, "\n")
		print("--- %s seconds ---" % (time.time() - start_time))

	finally:
		connection.close()


#	BaseLine#1
def tf_based_keywords():
	print("==tf_based_keywords==")
	words_set, paper_words_list, term_count_dict, annotated_words=dataset_tf("Huth_Dataset_Validation", 500)
	sorted_list_term=sorted(term_count_dict.items(), key=operator.itemgetter(1))

	print("No of annotated keywords===", len(annotated_words))

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:3400]:
		sel_words_list.append(key[0])
	
	commonwords=annotated_words.intersection(set(sel_words_list))

	print("common words==", commonwords, "==", len(commonwords))
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)
	print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)


# Baseline#2
def tf_idf_based_keywords():
	print("==tf_idf_based_keywords")
	words_set, paper_words_list, term_count_dict, annotated_words=dataset_tf("Huth_Dataset_Validation", 500)
	term_papercount=term_freq_in_papers(words_set, paper_words_list)	#function to find the occurrence of each word in number of tweets

	paper_count=len(paper_words_list)
	term_rel_dict={}
	for word in words_set:
		tf=math.log10(term_count_dict[word])+1
		idf=math.log10(1+(paper_count/term_papercount[word]))
		tf_idf=tf*idf
		term_rel_dict[word]=tf_idf

	sorted_list_term=sorted(term_rel_dict.items(), key=operator.itemgetter(1))	#sorting function to sort the terms based on their values

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:3400]:
		sel_words_list.append(key[0])
	
	commonwords=annotated_words.intersection(set(sel_words_list))

	print("common words==", commonwords, "==", len(commonwords))
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)
	print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)


#	BaseLine#3
def embedding_based_keywords():
	print("==embedding_based_keywords==")
	embedding_model=Word2Vec.load('embedding/Huth/Huth_model1.bin')
	words_set, paper_words_list, term_count_dict, annotated_words=dataset_tf("Huth_Dataset_Validation", 500)
	word_seedword_sim=word_seedsim(words_set, embedding_model)

	sorted_list_term=sorted(word_seedword_sim.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:3400]:
		sel_words_list.append(key[0])
	
	commonwords=annotated_words.intersection(set(sel_words_list))

	print("common words==", commonwords, "==", len(commonwords))
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)
	print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)


#	BaseLine#4
def pageRank():
	print("==pageRank==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()
		
		valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']

		paper_words_list=[]
		words_set=set()
		annotated_words=[]
		for row in results:
			text=re.sub(r'[:!\'/(@),#$?-]', '', row['Full']).split('.')

			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list
	
			for sentence in text:
				sentence_words=[]
				processed=nlp(sentence)
				
				for word in processed:
					stemword=word.lemma_.lower()
					if word.tag_ in valid_keywords and len(word)>2:
						sentence_words.append(stemword)
						words_set.add(stemword)		#set to hold all the unique words 
				
				paper_words_list.append(sentence_words)
			# print(paper_words_list)

		annotated_words=set(annotated_words)
		words_list=list(words_set)
		word_count=len(words_list)

		print("phase1 complete")
		word_index, index_word= indexing_fun(words_list)
		word_adj_mat=adj_matrix_PAGERANK(words_list, word_index, paper_words_list)
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

		sorted_list_term=sorted(final_word_rank.items(), key=operator.itemgetter(1))

		sel_words_list=[]
		for key in list(reversed(sorted_list_term))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		print("common words==", commonwords, "==", len(commonwords))
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)
	finally:
		connection.close()


#	Domain-based#1
#This is the implementation of th index proposed by Kit and Liu in Measuring mono-word termhood by rank difference via corpus comparison
def term_domain_specificity():
	print("==term_domain_specificity===")
	words_set, paper_words_list, term_count_dict, annotated_words=dataset_tf("Huth_Dataset_Validation", 500)

	dataset=input("Please enter the dataset")
	tweets_words_con, term_count_dict_con=dataset_tf1(dataset, 1000)

	sorted_wordlist=sorted(term_count_dict.items(), key=operator.itemgetter(1))
	
	tds_dict={}
	term_count_domain=sum(term_count_dict.values())
	
	for key in list(reversed(sorted_wordlist)):
		nume=(term_count_dict[key[0]]*1.0)/term_count_domain
		if key[0] in term_count_dict_con.keys():
			deno_n=(term_count_dict_con[key[0]]*1.0)/sum(term_count_dict_con.values())
			tds=nume/deno_n
			tds_dict[key[0]]=tds

	sorted_list_term=sorted(tds_dict.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_list_term))[0:3400]:
		sel_words_list.append(key[0])
	
	commonwords=annotated_words.intersection(set(sel_words_list))

	print("selected words==", sel_words_list[0:80], "==", len(sel_words_list))
	print("common words==", len(commonwords))
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)
	print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)


#	Domain-based#2
#This is the implementation of tds index proposed by Park et al. in An empirical analysis of word error rate and keyword error rate
def termhood():
	print("==termhood==")
	words_set, paper_words_list, term_count_dict, annotated_words=dataset_tf("Huth_Dataset_Validation", 500)
	
	dataset=input("Please enter the dataset")
	tweets_words_con, term_count_dict_con=dataset_tf1(dataset, 1000)

	sorted_wordlist=sorted(term_count_dict.items(), key=operator.itemgetter(1))
	sorted_wordlist_s=sorted(term_count_dict_con.items(), key=operator.itemgetter(1))

	sorted_wordlist_ss=collections.OrderedDict(list(reversed(sorted_wordlist_s)))

	sorted_list=list(sorted_wordlist_ss.keys())

	voc_size=len(words_set)
	voc_size_con=len(tweets_words_con)
	rank=voc_size

	thd_dict={}
	for key in list(reversed(sorted_wordlist))[:80]:
		if key[0] in term_count_dict_con.keys():
			firstterm=(rank/voc_size)
			rank_con=sorted_list.index(key[0])
			secondterm=(rank_con/voc_size_con)
			thd=firstterm-secondterm
			thd_dict[key[0]]=thd
			rank=rank-1

	sorted_thd=sorted(thd_dict.items(), key=operator.itemgetter(1))

	sel_words_list=[]
	for key in list(reversed(sorted_thd))[0:3400]:
		sel_words_list.append(key[0])
	
	commonwords=annotated_words.intersection(set(sel_words_list))

	print("common words==", commonwords, "==", len(commonwords))
	precision=(len(commonwords)*1.0)/len(sel_words_list)
	recall=(len(commonwords)*1.0)/len(annotated_words)
	fscore=(2*precision*recall)/(precision+recall)
	print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)

#	Domain-based#3
def RAKE():
	print("RAKE")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()

		candidate_keywords=[]
		term_freq_dict={}
		annotated_words=[]
		bigram_list=[]
		for row in results:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[:.!\'/(@),#$?-]', '', row['Full']).lower()
			processed=nlp(text)												#pass the tweet from nlp pipeline
			keyword=[]
			for token in processed:
				if len(token)>2 and token.is_stop==False:
					if str(token) in term_freq_dict.keys():					#if token is not typed using str, it will be an nlp token and throw an error
						term_freq_dict[str(token)]=term_freq_dict[str(token)]+1
					else:
						term_freq_dict[str(token)]=1

					keyword.append(str(token))								#token is added so to obtain consecutive words
				else:
					if len(keyword)==0 or len(keyword)==1:
						keyword=[]
						continue
					else:
						blist=[]
						for i in range(len(keyword)):
							for j in range(i+1, len(keyword)):
								bigram=tuple([keyword[i]]+[keyword[j]])
								blist.append(bigram)
						bigram_list=bigram_list+blist
						keyword=[]

		bigram_count=Counter(bigram_list)
		annotated_words=set(annotated_words)
		words_list=list(term_freq_dict.keys())
		print("phase1")

		word_index, index_word= indexing_fun(words_list)
		print("phase1/2")
		word_cooccur_mat=word_cooccur_RAKE(words_list, word_index, term_freq_dict, bigram_count)
		print("phase1/2")
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
		for key in list(reversed(sorted_list))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		print("common words==", commonwords, "==", len(commonwords))
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)
	finally:
		connection.close()

#	Graph-based#2
def TextRank():
	print("==textrank==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()
		
		valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']

		term_count_dict={}
		words_set=set()
		annotated_words=[]
		bigram_list=[]
		for row in results:											#iterate through each tweet
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			split_text = list(filter(None, re.split('; |\$|\)|\?|!|-|:|/|;|; |;| |\n|\. |\.', row["Full"].lower())))
			bigram = list(nltk.bigrams(split_text))
			bigram_list=bigram_list+bigram
			paper_words=[]										#list to store words in each tweet
			text=re.sub(r'[.:!\'/(@#$?-]', '', row["Full"]).lower()		#regular expression to filter stopwords
			
			processed=nlp(text)									#pass the tweet from nlp pipeline
			for word in processed:
				stemword=word.lemma_
				
				if word.tag_ in valid_keywords and len(word)>2:
					words_set.add(stemword)

		words_list=list(words_set)
		annotated_words=set(annotated_words)
		bigram_count=Counter(bigram_list)
		
		word_pairs=set()
		tweets_words_conet=set()
			
		word_index, index_word= indexing_fun(words_list)
		word_adj_mat_nor=word_cooccur_TextRank(words_list, word_index, bigram_count)
		
		word_count=len(words_list)
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
		for key in list(reversed(sorted_rank))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		print("common words==", commonwords, "==", len(commonwords))
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)
	finally:
		connection.close()

#Second graph-based approach implementation of A graph based keyword extraction model using collective node weight
def graph_CNW():		#CNW stands for collective node weight
	print("==CNW==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()

		sentence_words_list=[]
		annotated_words=[]
		term_freq_dict={}
		for row in results:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[:!\'/(@),#$?-]', '', row['Full']).lower().split('.')

			for sentence in text:
				sentence_words=[]
				tokens=nlp(sentence)

				for token in tokens:
					if len(token)>2 and token.is_stop==False:
						if str(token) in term_freq_dict.keys():
							term_freq_dict[str(token)]=term_freq_dict[str(token)]+1
						else:
							term_freq_dict[str(token)]=1

						sentence_words.append(str(token))
				sentence_words_list.append(sentence_words)
		
		annotated_words=set(annotated_words)
		AOC=st.mean(term_freq_dict.values())			#Threshold calculation to select the relevant word
		print(AOC, "===phase1")

		term_freq_dict_up={}				#this dict has only those terms having frequency greater than the threshold AOC
		wordslist=[]
		for term in term_freq_dict.keys():
			if term_freq_dict[term]>AOC:
				term_freq_dict_up[term]=term_freq_dict[term]
				wordslist.append(term)

		wordslist=list(set(wordslist))
		word_index, index_word= indexing_fun(wordslist)
		word_cooccur_mat, word_cooccur_mat_adj=word_cooccur_CNW(wordslist, word_index, sentence_words_list, term_freq_dict_up)
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
		for word in wordslist:
			final_weight_dict[word]=part_word_weight_dict[word]+term_freq_dict_up[word]

		print("phase3")

		sorted_rank=sorted(final_weight_dict.items(), key=operator.itemgetter(1))
		
		sel_words_list=[]
		for key in list(reversed(sorted_rank))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		print("common words==", commonwords, "==", len(commonwords))
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)
	finally:
		connection.close()


#Third graph-based approach implementation of PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents
def PositionRank():
	print("==PositionRank==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()

		valid_keywords=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']

		annotated_words=[]
		word_pair_freq={}
		pos_rank_dict={}
		for row in results:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[:!.\'/(@),#$?-]', '', row['Full']).lower()

			tokens=nlp(text)					#Spacy NLP pipeline

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
				if len(token)>2 and token.is_stop==False and token.tag_ in valid_keywords:
					stemword=token.lemma_
					if stemword in pos_rank_dict.keys():
						pos_rank_dict[stemword]=pos_rank_dict[stemword]+pos_rank
					else:
						pos_rank_dict[stemword]=pos_rank
		
		words_list=list(pos_rank_dict.keys())
		annotated_words=set(annotated_words)

		word_index, index_word= indexing_fun(words_list)
		word_adj_mat_nor=word_cooccur_PositionRank(words_list, word_index, word_pair_freq)
		
		word_count=len(words_list)
		word_rank=np.full((word_count, 1), 1/word_count)
		
		damp_vec_mat=np.zeros((word_count, 1))
		for word in words_list:
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
		for key in list(reversed(sorted_rank))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		# print("common words==", commonwords, "==", len(commonwords))
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)
	finally:
		connection.close()


#Paper Corpus-independent Generic Keyphrase Extraction Using Word Embedding Vectors  by Rui Wang  
def CorpusIndependentWang():
	start_time = time.time()
	print("==CorpusIndependentWang==")
	try:
		connection=MySQLdb.connect("localhost", "root", "mfazil@ms", "Keyword_Expansion")
		cursor=connection.cursor(MySQLdb.cursors.DictCursor)

		cursor.execute("SELECT * FROM Huth_Dataset_Validation limit 500")
		results=cursor.fetchall()

		valid_tags=['NN', 'NNP', 'NNS', 'NNPS', 'JJ']
		embedding_model=Word2Vec.load('embedding/Hateful/Hateful_model1.bin')
		
		print("embedding loaded")

		term_count_dict={}
		bigram_list=[]
		annotated_words=[]
		for row in results:
			keywords=row["Keywords"].split(',')
			keywords_list=[]
			
			for eachkey in keywords:
				eachkey=re.sub(r'[\' \"]', '', eachkey).lower().strip()		#To remove the '"' sign from each string
				if len(eachkey) > 2 and eachkey.isdigit()==False:
					keywords_list.append(eachkey)
			
			annotated_words=annotated_words+keywords_list

			text=re.sub(r'[.:!\'/(@#$?-]', '', row['Full']).lower()

			split_text = list(filter(None, re.split('; |\$|\)|\?|!|-|:|/|;|; |;| |\n|\. |\.', text)))
			bigram = list(nltk.bigrams(split_text))
			bigram_list=bigram_list+bigram

			tokens=nlp(text)
			
			for token in tokens:
				if token.tag_ in valid_tags:
					word=token.text

					if word in term_count_dict.keys():
						term_count_dict[word]=term_count_dict[word]+1
					else:
						term_count_dict[word]=1

		words_list=list(term_count_dict.keys())
		annotated_words=set(annotated_words)
		bigram_count=Counter(bigram_list)

		print("Bigram_count Completed")

		word_index, index_word= indexing_fun(words_list)
		word_cooccur_mat_nor=word_cooccur_CIW(words_list, word_index, bigram_count, term_count_dict, embedding_model)
		print("Cooccurrence Matrix Completed")
		
		word_count=len(words_list)
		word_rank=np.full((word_count, 1), 1)
		damp_val=(1-0.85)
		damp_vec=np.full((word_count, 1), damp_val)
		
		flag=True
		pre_word_rank=word_rank

		while flag==True:
			uprank=0.85*(np.dot(word_cooccur_mat_nor, word_rank))
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
		for key in list(reversed(sorted_rank))[0:3400]:
			sel_words_list.append(key[0])
	
		commonwords=annotated_words.intersection(set(sel_words_list))

		print("selected words==", sel_words_list[0:80], "==", len(sel_words_list[0:80]))
		print("common words==", commonwords, "==", len(commonwords))
		precision=(len(commonwords)*1.0)/len(sel_words_list)
		recall=(len(commonwords)*1.0)/len(annotated_words)
		fscore=(2*precision*recall)/(precision+recall)
		print("Precision===", precision, "		Recall===", recall, "		Fscore===", fscore)

		print("--- %s seconds ---" % (time.time() - start_time))
	finally:
		connection.close()


def compared_approach_sarna():
	print("==sarna==")
	seed_words, words_list, sentence_words_list, term_count_dict, annotated_words=dataset_SARNA("Huth_Dataset_Validation", 500)
	print("==phase1==")
	network_dict=cooccur_AND_network_SARNA(seed_words, words_list, sentence_words_list, term_count_dict)
	print("==phase2==")
	link_pruning(network_dict, annotated_words)


# Proposed_Approach()

#def graph_construction():
# Baseline Approaches
# tf_based_keywords()
# tf_idf_based_keywords()
# embedding_based_keywords()
# pageRank()
# CorpusIndependentWang()
# ReferenceVectorAlgorithm()

#Contrasting Corpora-Based Approaches
# term_domain_specificity()
# termhood()
# RAKE()

#Graph_based_approaches
# TextRank()
# graph_CNW()
# PositionRank()
# compared_approach_sarna()