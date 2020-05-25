# coding: utf-8
# Echo Chamber Model
# agent.py
# Last Update: 20190410
# by Kazutoshi Sasahara

import numpy as np
from social_media import Message
import random


class Agent(object):
    def __init__(self, user_id, epsilon, truth, screen_diversity, num_agents):
        self.user_id = user_id
        #変更
        self.opinion = np.random.uniform(0.0, 1.0)
        self.reliability = np.random.uniform(0.0, 1.0, num_agents)
        self.unreliability = np.random.uniform(0.0, 1.0, num_agents)
        self.epsilon = epsilon
        self.truth = truth
        self.screen_diversity = screen_diversity
        self.orig_msg_ids_in_screen = []

        
    def set_orig_msg_ids_in_screen(self, screen):
        self.orig_msg_ids_in_screen = screen.orig_msg_id.values

        
    def evaluate_messages(self, screen):
        #笹原はここで（スクリーン（友人のみ）の中の）異なる意見を遮断している
        self.concordant_msgs = []
        self.discordant_msgs = []
        if len(screen) > 0:
            self.concordant_msgs = screen[abs(self.opinion - screen.content) < self.epsilon]
            self.discordant_msgs = screen[abs(self.opinion - screen.content) >= self.epsilon]

    '''
    def update_opinion(self, mu):
        if len(self.concordant_msgs) > 0:
            self.opinion = self.opinion + mu * np.mean(self.concordant_msgs.content - self.opinion)
    '''

    #比較用<1>：更新式のみ変更  
    #他の比較→リンクの切断・接続（とりあえず保留）  
    #自分たちの更新式は異なる意見は遮断せず，更新式を変えて更新している　→　どちらが良いか？       
    def update_opinion(self,screen):
        if len(screen)>0:
            df_len=len(screen)
            n=random.randrange(df_len)
            #保存していない？
            if self.truth == True:
                if float(screen.at[screen.index[n],'content'])>=0.8:  #tail(1)でいいのか？random？
                    self.opinion = self.reliability[int(screen.at[screen.index[n],'who_posted'])] * self.opinion / (self.unreliability[int(screen.at[screen.index[n],'who_posted'])] * (1 - self.opinion) + self.reliability[int(screen.at[screen.index[n],'who_posted'])] * self.opinion)
                elif float(screen.at[screen.index[n],'content'])<=0.2:
                    self.opinion = (1-self.reliability[int(screen.at[screen.index[n],'who_posted'])]) * self.opinion / ((1-self.unreliability[int(screen.at[screen.index[n],'who_posted'])]) * (1 - self.opinion) + (1-self.reliability[int(screen.at[screen.index[n],'who_posted'])]) * self.opinion)

            elif self.truth == False:
                if float(screen.at[screen.index[n],'content'])>= 0.8:
                    self.opinion = self.unreliability[int(screen.at[screen.index[n],'who_posted'])] * (1-self.opinion) / (self.unreliability[int(screen.at[screen.index[n],'who_posted'])] * (1 - self.opinion) + self.reliability[int(screen.at[screen.index[n],'who_posted'])] * self.opinion)
                elif float(screen.at[screen.index[n],'content'])<= 0.2:
                    self.opinion = ((1-self.unreliability[int(screen.at[screen.index[n],'who_posted'])]) * (1-self.opinion)) / ((1-self.unreliability[int(screen.at[screen.index[n],'who_posted'])]) * (1 - self.opinion) + (1-self.reliability[int(screen.at[screen.index[n],'who_posted'])])* self.opinion)


            
    def post_message(self, msg_id, p):
        if len(self.concordant_msgs) > 0 and np.random.random() < p:
            # repost a friend's message selected at random
            idx = (np.random.choice(self.concordant_msgs.index))
            selected_msg = self.concordant_msgs.ix[idx]
            return Message(msg_id=int(msg_id), orig_msg_id=int(selected_msg.orig_msg_id),
                           who_posted=int(self.user_id), who_originated=int(selected_msg.who_originated),
                           content=selected_msg.content)
        else:
            # post a new message
            new_content = self.opinion
            return Message(msg_id=int(msg_id), orig_msg_id=int(msg_id),
                           who_posted=int(self.user_id), who_originated=int(self.user_id), content=new_content)

        
    def decide_follow_id_at_random(self, friends, num_agents):
        prohibit_list = list(friends) + [self.user_id]
        return int(np.random.choice([i for i in range(num_agents) if i not in prohibit_list]))

    
    def decide_unfollow_id_at_random(self, discordant_messages):
        unfollow_candidates = discordant_messages.who_posted.values
        return int(np.random.choice(unfollow_candidates))

    
    def decide_to_rewire(self, social_media, following_methods):
        unfollow_id = None
        follow_id = None

        if len(self.discordant_msgs) > 0:
            # decide whom to unfollow
            unfollow_id = self.decide_unfollow_id_at_random(self.discordant_msgs)
            # decide whom to follow
            following_method = np.random.choice(following_methods)
            friends = social_media.G.neighbors(self.user_id)

            # Repost-based selection if possible; otherwise random selection
            if following_method == 'Repost':
                friends_of_friends = list(set(self.concordant_msgs.who_originated.values) - set(friends))
                if len(friends_of_friends) > 0:
                    follow_id = int(np.random.choice(friends_of_friends))
                else:
                    follow_id = self.decide_follow_id_at_random(friends, social_media.G.number_of_nodes())

            # Recommendation-basd selection if possible; otherwise random selection
            elif following_method == 'Recommendation':
                similar_agents = social_media.recommend_similar_users(self.user_id, self.epsilon, social_media.G.number_of_nodes())
                if len(similar_agents) > 0:
                    follow_id = int(np.random.choice(similar_agents))
                else:
                    follow_id = self.decide_follow_id_at_random(friends, social_media.G.number_of_nodes())

            # Random selection
            else:
                follow_id = self.decide_follow_id_at_random(friends, social_media.G.number_of_nodes())

        return unfollow_id, follow_id