#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import *
import random
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import io
import os
from sklearn import metrics

#Q 2 predict using decision tree classifier
Data = pd.read_csv("C:/Users/DELL/Downloads/Decision making/output_test.csv")
Data = pd.DataFrame(data=Data)
Data.columns = ['NO','0th pos', '1st pos', '2nd pos', '3rd pos', '4th pos', '5th pos', '6th pos', '7th pos', '8th pos', 'Player','Move_Pos']
train_set, test_set = train_test_split(Data, test_size = 0.2, random_state=42)

#train set for x
Data_train_x=train_set.iloc[:,1:10]
#the output column extracted from training
Data_train_y=train_set.iloc[:,-1]
Data_train_y=Data_train_y.astype('int')
    
#test 9 positions of the board 
Data_test_x=test_set.iloc[:,1:10]
    
#the output column extracted from testing
Data_test_y=test_set.iloc[:,-1]
Data_test_y=Data_test_y.astype('int')
    

#Train the classifier on the max accuracy in every iteration
dtclassifier = DecisionTreeClassifier(criterion='gini',random_state=42)
dtclassifier.fit(Data_train_x, Data_train_y)
Pred_test_y=dtclassifier.predict(Data_test_x)
print("Accuracy:",metrics.accuracy_score(Data_test_y,Pred_test_y))


# In[2]:


#Q4,5,6
class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is 2 whereas player 1 has the first move.
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2. This is the initial board state - all positions are empty.
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update the state board by replacing 0 with the player playing the move at the position/move of the board.
            Must update playerJustMoved.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state. That is return all the positional values of the zroes in the state board.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            #Winning possibilities of the board
            if self.board[x] == self.board[y] == self.board[z]: # check if values in all the 3 positions is of the same player
                # check if the player that just moved is same as the value in the winning positions, if yes return 1 else 0 stating that the other player wins.
                if self.board[x] == playerjm: 
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []:
            return 0.5 # draw
        return False 

    def __repr__(self): # This is how the return value is defined for this class.
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]] # . for 0, X for 1 and O for 2 positional values
            if i % 3 == 2: s += "\n"
        return s


class Node:
    """ A node in the game tree. Note : wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        #parentNode stores all the parents from the rootnode until the current node for backpropogation, during which it deletes until it is None.
        self.parentNode = parent # "None" for the root node.
        self.childNodes = []
        self.wins = 0
        self.visits = 0 #The number of itermax passed
        self.untriedMoves = state.GetMoves() # future child nodes. The available positions to be played at any point of the game.
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1] #pick the highest
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node.
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self): # Added other variables to be returned to check the flow of the variable during testing small iterations.
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + " PJM:" + str(self.playerJustMoved) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

def Random_Move(s,m):
#random value between 0 and 1. 
#if the value in range 0 to 0.9 the decision tree will choose move.
    
    random_no=random.uniform(0.0, 1.0)

    if random_no <= 0.9:
        rand_list=list(s.board)
        rand_list.append(s.playerJustMoved)
        next_move=m.predict([rand_list])
        next_move=next_move[0]
    
#If decision tree classifier select move that is not in the untried moves : select randomly for the next move.
        if next_move not in s.GetMoves():
            next_move=random.choice(s.GetMoves())
            st_bd=list(s.board) 
            st_bd.append(s.playerJustMoved)
            st_bd.append(next_move) 
            train_state.append(st_bd)            
    else:
        next_move=random.choice(s.GetMoves())
    return next_move
    
def UCT(first_t, class_dct, rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree. This updates the parent node as well.

        # Rollout
        while state.GetMoves() != []: # while state is non-terminal
            if first_t==1: # if no initial input states is given
                r=random.choice(state.GetMoves())
            else:
                r=Random_Move(state, class_dct)
            state.DoMove(r)
        
        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            gr=state.GetResult(node.playerJustMoved)
            node.Update(gr) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - Commented to save space for the output
    #if verbose: print(rootnode.TreeToString(0))
    #else: print(rootnode.ChildrenToString())
    
    mos=sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move

    return mos # return the move that was most visited as the best move.
                
def UCTPlayGame(first_t, class_dct): 
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState()
  
    temp_a=[[0,0,0,0,0,0,0,0,0]] # empty list for collect stages of one game
    best_move=[] 
    store_move=[] #List of player move
    while state.GetMoves() != []:
        #print(str(state)) 

        if state.playerJustMoved == 1: 
            m = UCT(first_t, class_dct, rootstate=state, itermax=10, verbose=False)  # play with values for itermax and verbose = True.
        else:
            m = UCT(first_t, class_dct, rootstate=state, itermax=10, verbose=False) 
        #print("Best Move: " + str(m) + "\n") 
        state.DoMove(m)
        
        store_move.append(state.playerJustMoved)
        best_move.append(m)
        temp_a.append(list(state.board))
        
        if state.GetResult(state.playerJustMoved) != False:
            #print(str(state)) 
            break
    
    if state.GetResult(state.playerJustMoved) == 1.0:
        #print("Player " + str(state.playerJustMoved) + " wins!") 
        winner=state.playerJustMoved
    elif state.GetResult(state.playerJustMoved) == 0.0:
        #print("Player " + str(3 - state.playerJustMoved) + " wins!") 
        winner=(3-state.playerJustMoved)
    else: 
      #print("Nobody wins!") #Commented to save space for the output
      winner=0

#Add Na to the last move to prevent blank space 
    best_move.append('NA')
    store_move.append('NA')

    for i in range(len(temp_a)): 
        temp_a[i].append(store_move[i]) 
        temp_a[i].append(best_move[i]) 

    return temp_a, winner 

def xo_dataset(num_games, first_t, class_dct):
#create data set using UCT for both players. 
    final_data=[] 
    final_data1=[] 
    
    player1win=0
    player2win=0
    Tie=0
    winner=0
    for i in range(num_games): # Run the UCTPlayGame 
        returnds, winner =UCTPlayGame(first_t, class_dct) 
        final_data.append(returnds) 
        if winner==1:
            player1win+=1
        elif winner==2:
            player2win+=1
        else:
            Tie+=1

    #Convert list of lists to one list
    for i in final_data:  
        for j in i:
            final_data1.append(j) 

    #give columns name
    datatbp=pd.DataFrame(final_data1, columns=['0th pos', '1st pos', '2nd pos', '3rd pos', '4th pos', '5th pos', '6th pos', '7th pos', '8th pos', 'Player', 'Move'])
     
    #Remove the rows that have 'NA' for the last stage of the game 
    datatbp.drop(datatbp[datatbp.Move == 'NA'].index, inplace=True)
    datatbp.reset_index(drop=True, inplace=True) 

    #print("Wins : Player1 : ", player1win, " Player2 : ", player2win, " Nobody wins : ", Tie) 
    
    return datatbp


# In[3]:


#Create Decision Tree classifier
def decision_tree(Data_in, get_depth, depth_max):
    #Split data into train and test set
    train_set, test_set = train_test_split(Data_in, test_size = 0.2, random_state=42)
    #x and y for training
    Data_train_x=train_set.iloc[:,0:10]
    Data_train_y=train_set.iloc[:,-1]
    Data_train_y=Data_train_y.astype('int')
    Data_test_x=test_set.iloc[:,0:10]
    
    #y for testing
    Data_test_y=test_set.iloc[:,-1]
    Data_test_y=Data_test_y.astype('int')
    
    if(get_depth==1):
        depth_range = list(range(1,40))
        accuracy_list = []
        for depth in depth_range:
            classifier_decision1 = DecisionTreeClassifier(criterion='gini',max_depth=depth,random_state=42)
            classifier_decision1.fit(Data_train_x, Data_train_y)
            score = classifier_decision1.score(Data_test_x, Data_test_y)
            accuracy_list.append(score)
        depth_max =(accuracy_list.index(max(accuracy_list)) + 1)

    
    #Train the classifier on the max accuracy in every iteration
    classifier_decision = DecisionTreeClassifier(criterion='gini',max_depth=depth_max,random_state=42)
    classifier_decision.fit(Data_train_x, Data_train_y)
    Pred_test_y=classifier_decision.predict(Data_test_x)

    return classifier_decision, depth_max

#Collect our classifiers in to dictionary
def class_dict(data_num, num_of_games, primer, Data_csv):
    get_depth=1
    depth_max=0
    dict_classifier=dict()
    classifier=None

    if (primer==1):
        classifier, depth_max =decision_tree(Data_csv, get_depth, depth_max)
        dict_classifier[int('0')]=classifier
        data_num=data_num-1
        first_t=0
        get_depth=0
    else:
        first_t=1
    
    for i in range(data_num):
        data_0=xo_dataset(num_of_games, first_t, classifier) 
        combind_train=pd.DataFrame(train_state,columns=['0th pos', '1st pos', '2nd pos', '3rd pos', '4th pos', '5th pos', '6th pos', '7th pos', '8th pos', 'Player','Move'])
        data_0=pd.concat([data_0,combind_train],ignore_index=True, axis=0)

        if(first_t==1):
            data_0.to_csv('newout.csv')
            csv.reader('newout.csv') 
        classifier, depth_max =decision_tree(data_0, get_depth, depth_max)

        if(primer==1):
            dict_classifier.update({i+1:classifier})
        else:
            dict_classifier.update({i:classifier})
        first_t=0
        get_depth=0
        train_state.clear()
    return dict_classifier


# In[4]:


#play game using classifiers to play against each other
def Random_Move1(decisionclassifier, s):
    rand_list=list(s.board)
    rand_list.append(s.playerJustMoved)

    r=decisionclassifier.predict([rand_list])
    r=r[0]
    if r not in s.GetMoves():
        r=random.choice(s.GetMoves())
    return r

def UCTclass(play1_class, play2_class,rootstate, itermax, verbose = False):

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)            
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout 
        while state.GetMoves() != []: # while state is non-terminal
            if(state.playerJustMoved==1):
                r=Random_Move1(play1_class,state)
            else:
                r=Random_Move1(play2_class,state)
            state.DoMove(r)

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    #if (verbose): print(rootnode.TreeToString(0))
    #else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

#Play a sample game between UCT  
def UCTPlayGame0(play1_class, play2_class):
    
    state = OXOState() 
    
    while (state.GetMoves() != []):
        #print(str(state))
        if state.playerJustMoved == 1:
            m = UCTclass(play1_class, play2_class,rootstate = state, itermax = 10, verbose = False) 
        else:
            m = UCTclass(play1_class, play2_class, rootstate = state, itermax =10, verbose = False)
        #print("Best Move: " + str(m) + "\n") 
        state.DoMove(m)
        if state.GetResult(state.playerJustMoved) != False:
            #print(str(state)) 
            break
    if state.GetResult(state.playerJustMoved) == 1.0:
        #print("Player " + str(state.playerJustMoved) + " wins!") 
        return state.playerJustMoved
    elif state.GetResult(state.playerJustMoved) == 0.0:
        #print("Player " + str(3 - state.playerJustMoved) + " wins!") 
        return (3-state.playerJustMoved)
    else:
        #print("Nobody wins!") 
        return 0


# In[7]:


#game play for classifier 
if __name__ == "__main__":
    # create a global list for unexpected states from the Decision tree. 
    #use csv file data to train model 
    global train_state
    train_state=[]
    gamesno=int(input("Enter number of games play: "))
    classno=100
    primer=int(input("use provided data(1),generate data set(0) : "))
    if(primer==1):
        #Data_csv = pd.read_csv("C:/Users/DELL/Downloads/Decision making/output_test.csv")
        Data_csv = pd.read_csv("C:/Users/DELL/Downloads/Decision making/newout2.csv")
    else:
        Data_csv=0
    dict_class=class_dict(classno, gamesno, primer, Data_csv)
    
    for i in range(0,len(dict_class),10):
        DecisionTree10=0
        Other=0
        Tie=0
        j=i
        for k in range(i, i+9): 
            winner = UCTPlayGame0(dict_class[9+10*(i//10)], dict_class[k]) 
            if winner==1:
                DecisionTree10+=1
            elif winner==2:
                Other+=1
            else:
                Tie+=1
        print("10th DecisionTree vs Other : ", "Win   ",DecisionTree10, " Loss   ", Other, " Tie  ", Tie,"Percentage of winning:",round(DecisionTree10/9,2))


# In[ ]:


Enter number of games play: 60
use provided data(1),generate data set(0) : 1
10th DecisionTree vs Other :  Win    4  Loss    1  Tie   4 Percentage of winning: 0.44
10th DecisionTree vs Other :  Win    3  Loss    1  Tie   5 Percentage of winning: 0.33
10th DecisionTree vs Other :  Win    3  Loss    3  Tie   3 Percentage of winning: 0.33
10th DecisionTree vs Other :  Win    4  Loss    4  Tie   1 Percentage of winning: 0.44
10th DecisionTree vs Other :  Win    6  Loss    1  Tie   2 Percentage of winning: 0.67
10th DecisionTree vs Other :  Win    5  Loss    2  Tie   2 Percentage of winning: 0.56
10th DecisionTree vs Other :  Win    5  Loss    3  Tie   1 Percentage of winning: 0.56
10th DecisionTree vs Other :  Win    3  Loss    4  Tie   2 Percentage of winning: 0.33
10th DecisionTree vs Other :  Win    7  Loss    2  Tie   0 Percentage of winning: 0.78
10th DecisionTree vs Other :  Win    4  Loss    4  Tie   1 Percentage of winning: 0.44

