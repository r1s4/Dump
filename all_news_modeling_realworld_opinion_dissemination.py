import networkx as nx
import pandas as pd
import numpy as np
import csv
import random

N = 100 #N回for文を回した中から最適解を探す

def import_file():
    #ファイルの読み込み
    df_matrix= pd.read_csv('./matrix.csv')
    #df_H_matrix= pd.read_csv('./H_matrix.csv')
    print(df_matrix)
    df_matrix = df_matrix.set_index('user_id')
    all_user_ids=df_matrix.index

    graph=nx.read_gml('./Graph.gml')
    #graph_reverse=nx.reverse(graph)
    adj_dic = nx.to_dict_of_lists(graph)    
    #辞書型：{target1:[source1,source2,...],...}
    #つまり，target(key)のフォロワーがvalue
    
    return df_matrix,graph,adj_dic,all_user_ids

def import_RT_users(news):
    #RTしたユーザのみのリストを作る
    df_tweet = pd.read_csv('~/EM/data/20200825/{}/coronavirus.csv'.format(news))    #tweets_dataがある場所を指定
    df_tweet = df_tweet.set_index('user_id')
    user_ids=df_tweet.index
    return user_ids

#初期値の設定
def initial_values(all_user_ids):
    #信念値，信用度，不信度の初期値を設定（辞書型：{userA:0.5,userB:0.7,...}
    #初期値は[0.0,1.0]の中から，0.01刻みの離散値をランダムに渡す
    belief={}
    t={}
    f={}
    for i in all_user_ids:
        #辞書型
        belief_list = np.arange(0.0,1.1,0.1)
        t_list = np.arange(0.0,1.1,0.1)
        f_list = np.arange(0.0,1.1,0.1)
        reliability_list = np.arange(0.0, 1.01, 0.01)
        belief[str(i)]= np.random.choice(belief_list)
        t[str(i)]=np.random.choice(t_list)
        f[str(i)]=np.random.choice(f_list)
    return belief,t,f

def neighbors_list(graph,user_id,adj_dic):
    #user_idのフォロワーのリストを返す

    friends_user_id_list=[]
    
    friends_user_id_list=adj_dic[str(user_id)] 
    friends_user_id_list=[int(s) for s in friends_user_id_list]

    return friends_user_id_list

def weighting(df_matrix):
    #重みの計算
    df_w_0=(df_matrix == 0)
    w_0= 1/ (df_w_0.values.sum())
    df_w_1=(df_matrix == 1)
    w_1= 1/ (df_w_1.values.sum())
    df_w_minus1=(df_matrix == -1)
    w_minus1= 1/ (df_w_minus1.values.sum())

    print("行列Gにおける0の数: ",df_w_0.values.sum())
    print("行列Gにおける1の数: ",df_w_1.values.sum())
    print("行列Gにおける-1の数: ",df_w_minus1.values.sum())

    print("w_0: ", w_0)
    print("w_1: ", w_1)
    print("w_-1: ", w_minus1)


    
    return w_0,w_1,w_minus1

def output_files(temp_df_H_matrix,temp_initial_belief,temp_initial_f,temp_initial_t):       
    #for文で回した中でもっとも結果がよかったときのデータを書き込む
    #行列H
    temp_df_H_matrix.fillna(0).to_csv("H_matrix.csv")

    #各初期値
    df_initial_belief=pd.DataFrame(temp_initial_belief,index=["initial_belief"])
    df_initial_t=pd.DataFrame(temp_initial_t,index=["initial_t"])
    df_initial_f=pd.DataFrame(temp_initial_f,index=["initial_f"])
    df_csv = pd.concat([df_initial_belief,df_initial_t,df_initial_f])
    df_csv = df_csv.T
    df_csv.to_csv('./initial_values.csv')


def main():
    df_matrix,graph,adj_dic,all_user_ids=import_file()
    w_0,w_1,w_minus1 = weighting(df_matrix)
    
    #news_list=[Fake1,Fake2,Fake3,Fake4,Fake5,True1,True2,True3,True4,True5,Fake-Yachin,True-Yachin,Fake-WHO,True-WHO]
    #truth_list = [False,False,False,False,False,True,True,True,True,True,False,True,False,True] 
    #news_list=['Fake3','Fake5','Fake-WHO','True1','True3','True-WHO']
    #truth_list = [False,False,False,True,True,True]
    news_list=['Fake3','Fake5','Fake-WHO','True1','True3','True-WHO']
    truth_list = [False,False,False,True,True,True]


    sum_true=0
    temp_initial_belief=[]
    temp_initial_t=[]
    temp_initial_f=[]
    for i in range(N):
        print(i,"回目")
        diff=0
        df_diff_0=pd.DataFrame(index=[], columns=[])
        df_diff_1=pd.DataFrame(index=[], columns=[])
        df_diff_minus1=pd.DataFrame(index=[], columns=[])
        df_H_matrix=df_matrix.copy()
        initial_belief,initial_t,initial_f=initial_values(all_user_ids) #初期信念値はどのニュースも同じ（変更の可能性あり）
        for news,truth in zip(news_list, truth_list):   #全てのニュースに対してfor
            print("news : ",news)
            belief,t,f=initial_belief,initial_t,initial_f

            #news1をRTしたユーザのリスト
            retweet_users_news1 = import_RT_users(news)
            belief[str(retweet_users_news1[-1])]=1.0    #ニュースを最初に拡散した人物（センサー，retweet_users_news1[-1]）の信念値は1.0とする
            
            #センサーのフォロワーリスト
            senser_agent_followers_list=[]
            senser_agent_followers_list=neighbors_list(graph,retweet_users_news1[-1],adj_dic)

            true_opinion_agents=[]  #RTした人物（＝信念値が0.8以上になった＝意見を発信した）を格納する → 最後に行列の値を1にする
            true_opinion_agents.append(retweet_users_news1[-1])
            rt_receivers_list=[]    #RTを受け取った人物（＝RTした人物のフォロワー）を格納する → rt_receivers_listに存在かつtrue_opinion_agentsに存在しない人物は，行列の値が-1になる

            count1=0
            #センサーのフォロワーの処理
            for rt_receiver in senser_agent_followers_list:
                count1=count1+1
                rt_receivers_list.append(rt_receiver)
                #print(rt_receiver , " initial belief: ", belief[str(rt_receiver)])
                if truth==True:
                    if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは更新を行わない
                        belief[str(rt_receiver)] = belief[str(rt_receiver)]
                    else:
                        belief[str(rt_receiver)] = (belief[str(rt_receiver)] * t[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                elif truth==False:
                    # P(z = False |o = True)のときの信念値の更新式
                    if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは，更新を行わない
                        belief[str(rt_receiver)]=belief[str(rt_receiver)]
                    else:
                        belief[str(rt_receiver)] = ((1 - belief[str(rt_receiver)]) * f[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                if (belief[str(rt_receiver)]>=0.8) and (rt_receiver not in true_opinion_agents):    #RTするのは1回のみなので
                    true_opinion_agents.append(rt_receiver)
                #print(rt_receiver , " updated belief: ", belief[str(rt_receiver)])

            #信念値の更新をrt_receivers_listが空になるまで（breakするまで）行う
            count2=0
            #true_opinion_agentsのエージェントの意見を発信
            for rt_agent in true_opinion_agents:    #rt_agent : 意見を発信した（RTした）人
                if rt_agent == retweet_users_news1[-1]:
                    continue
                #count=count+1
                #print(rt_agent, " is RT agent")
                followers_list=[]   #RTしたエージェントのフォロワーのリスト ＝ rt_agentからRTを受け取ったユーザのリスト
                followers_list=neighbors_list(graph,rt_agent,adj_dic)
                for rt_receiver in followers_list:
                    count2=count2+1
                    rt_receivers_list.append(rt_receiver)
                    #print(rt_receiver , " initial belief: ", belief[str(rt_receiver)])
                    if truth==True:
                        if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは更新を行わない
                            belief[str(rt_receiver)] = belief[str(rt_receiver)]
                        else:
                            belief[str(rt_receiver)] = (belief[str(rt_receiver)] * t[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                    elif truth==False:
                        # P(z = False |o = True)のときの信念値の更新式
                        if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは，更新を行わない
                            belief[str(rt_receiver)]=belief[str(rt_receiver)]
                        else:
                            belief[str(rt_receiver)] = ((1 - belief[str(rt_receiver)]) * f[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                    #rt_receiverの信念値が0.8を超えた場合　かつ　RTした（する）人のリスト(true_opinion_agents)にrt_receiverがない場合（RTするのは1回のみなので）
                    if (belief[str(rt_receiver)]>=0.8) and (rt_receiver not in true_opinion_agents): 
                        true_opinion_agents.append(rt_receiver)
                        #print(rt_receiver, " list append")
                    #print(rt_receiver , " updated belief: ", belief[str(rt_receiver)])
                
                if len(retweet_users_news1)<len(true_opinion_agents):
                    break
                #if count>100:
                #    break

            count3=0
            count4=0
            while len(retweet_users_news1)>len(true_opinion_agents):
                senser=random.choices(list(set(retweet_users_news1)- (set(retweet_users_news1)&set(true_opinion_agents))))

                belief[senser[0]]=1.0    #ニュースを最初に拡散した人物（センサー，retweet_users_news1[-1]）の信念値は1.0とする
            
                #センサーのフォロワーリスト
                senser_agent_followers_list=[]
                senser_agent_followers_list=neighbors_list(graph,senser[0],adj_dic)

                second_true_opinion_agents=[]  #RTした人物（＝信念値が0.8以上になった＝意見を発信した）を格納する → 最後に行列の値を1にする
                true_opinion_agents.append(senser[0])
                second_true_opinion_agents.append(senser[0])
                
                
                #センサーのフォロワーの処理
                for rt_receiver in senser_agent_followers_list:
                    count3=count3+1
                    rt_receivers_list.append(rt_receiver)
                    #print(rt_receiver , " initial belief: ", belief[str(rt_receiver)])
                    if truth==True:
                        if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは更新を行わない
                            belief[str(rt_receiver)] = belief[str(rt_receiver)]
                        else:
                            belief[str(rt_receiver)] = (belief[str(rt_receiver)] * t[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                    elif truth==False:
                        # P(z = False |o = True)のときの信念値の更新式
                        if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは，更新を行わない
                            belief[str(rt_receiver)]=belief[str(rt_receiver)]
                        else:
                            belief[str(rt_receiver)] = ((1 - belief[str(rt_receiver)]) * f[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                    if (belief[str(rt_receiver)]>=0.8) and (rt_receiver not in true_opinion_agents):    #RTするのは1回のみなので
                        true_opinion_agents.append(rt_receiver)
                        second_true_opinion_agents.append(rt_receiver)
                    #print(rt_receiver , " updated belief: ", belief[str(rt_receiver)])

                #信念値の更新をrt_receivers_listが空になるまで（breakするまで）行う
                #true_opinion_agentsのエージェントの意見を発信
                for rt_agent in second_true_opinion_agents:    #rt_agent : 意見を発信した（RTした）人
                    if rt_agent == senser[0]:
                        continue
                    #count=count+1
                    #print(rt_agent, " is RT agent")
                    followers_list=[]   #RTしたエージェントのフォロワーのリスト ＝ rt_agentからRTを受け取ったユーザのリスト
                    followers_list=neighbors_list(graph,rt_agent,adj_dic)
                    for rt_receiver in followers_list:
                        count4=count4+1
                        rt_receivers_list.append(rt_receiver)
                        #print(rt_receiver , " initial belief: ", belief[str(rt_receiver)])
                        if truth==True:
                            if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは更新を行わない
                                belief[str(rt_receiver)] = belief[str(rt_receiver)]
                            else:
                                belief[str(rt_receiver)] = (belief[str(rt_receiver)] * t[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                        elif truth==False:
                            # P(z = False |o = True)のときの信念値の更新式
                            if ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)])) ==0:   #0除算のときは，更新を行わない
                                belief[str(rt_receiver)]=belief[str(rt_receiver)]
                            else:
                                belief[str(rt_receiver)] = ((1 - belief[str(rt_receiver)]) * f[str(rt_receiver)]) / ((belief[str(rt_receiver)]) * t[str(rt_receiver)] + f[str(rt_receiver)] * (1 - belief[str(rt_receiver)]))
                        #rt_receiverの信念値が0.8を超えた場合　かつ　RTした（する）人のリスト(true_opinion_agents)にrt_receiverがない場合（RTするのは1回のみなので）
                        if (belief[str(rt_receiver)]>=0.8) and (rt_receiver not in true_opinion_agents): 
                            true_opinion_agents.append(rt_receiver)
                            second_true_opinion_agents.append(rt_receiver)
            

            print("実際RTしたユーザの数:　",len(retweet_users_news1))
            print("シミュレーションでRTしたユーザの数: ", len(true_opinion_agents))
            print("更新回数： ", count1+count2+count3+count4 )


            for user_id in all_user_ids:
                if user_id in true_opinion_agents:
                    df_H_matrix.at[user_id,news]=1
                elif ( user_id in rt_receivers_list ) & ( user_id not in true_opinion_agents ):
                    df_H_matrix.at[user_id,news]=-1
                else:
                    df_H_matrix.at[user_id,news]=0


            #行列Gと行列Hの値が不一致　かつ　値=0 (or 1 or -1)
            df_diff_0[news] = ( df_matrix[news] != df_H_matrix[news] ) & ( df_matrix[news] == 0 )
            df_diff_1[news] = ( df_matrix[news] != df_H_matrix[news] ) & ( df_matrix[news] == 1 ) 
            df_diff_minus1[news] = ( df_matrix[news] != df_H_matrix[news] ) & ( df_matrix[news] == -1 )

            print("不一致数(0): ",df_diff_0[news].values.sum())
            print("不一致数(1): ",df_diff_1[news].values.sum())
            print("不一致数(-1): ",df_diff_minus1[news].values.sum())



        print("不一致数(0): ",df_diff_0.values.sum())
        print("不一致数(1): ",df_diff_1.values.sum())
        print("不一致数(-1): ",df_diff_minus1.values.sum())

        #f(x)の計算 HにおいてGとの不一致数 × 重み
        diff = ( df_diff_0.values.sum() * w_0 ) + ( df_diff_1.values.sum() * w_1 ) + ( df_diff_minus1.values.sum() * w_minus1 )
        print("diff : ",diff)    
        if i==0:
            sum_true=diff
            temp_df_H_matrix=df_H_matrix
            temp_initial_belief=belief
            temp_initial_t=t
            temp_initial_f=f
        else:
            if diff<sum_true:
                sum_true=diff
                temp_df_H_matrix=df_H_matrix
                temp_initial_belief=belief
                temp_initial_t=t
                temp_initial_f=f
        print("H_matrix : ", temp_df_H_matrix)
        print("sum_True : ",sum_true)


    output_files(temp_df_H_matrix,temp_initial_belief,temp_initial_f,temp_initial_t)
    
    

    

if __name__ == "__main__":
	main()