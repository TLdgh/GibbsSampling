import numpy as np
import math
import plotly.express as px
import pandas as pd

class DecodeMessage():
    def __init__(self, message: str, M_big: np.ndarray, niter=5000):
        self.alphabets=self.strsplit("abcdefghijklmnopqrstuvwxyz ")
        self.string_split=self.strsplit(message)
        self.matCoord=[]
        self.M_big=M_big
        
        self.GibbsFun(niter)
            
            
    def MakeMatrix(self, message):
        M=np.ones((27,27))
        message_pairs=[(message[i], message[i+1]) for i in range(len(message)-1)]
        
        def find_index(p, ref):
            return (ref.index(p[0]), ref.index(p[1]))
        self.matCoord=[find_index(pair, self.alphabets) for pair in message_pairs]

        for ind in self.matCoord:
            M[ind[0], ind[1]]+=1
        
        return(M/M.sum())

    def Loglike(self, message):
        M=self.MakeMatrix(message)
        p=0
        for ind in self.matCoord:
            p+=M[ind[0], ind[1]]*math.log(self.M_big[ind[0], ind[1]])
        return p
    
    def Recover(self, message, subFunction):
        message_enum=[[i for i, alpha in enumerate(self.alphabets) if alpha == j] for j in message]
        message_enum = [item 
                        for sublist in message_enum 
                        for item in sublist]

        new_message=[self.alphabets[j] for j in [subFunction[i] for i in message_enum]]
        return new_message
    
    def GibbsFun(self, niter):
        totalresult=pd.DataFrame({"message":[], "LogL":[]})      
        
        for i in range(30):
            print(f"Attempt {i+1}:")
            SubF_old=np.random.choice(list(range(0,27,1)), 27, replace=False)
            LogL=self.Loglike(message=self.Recover(self.string_split, subFunction=SubF_old))
            
            for iter in range(niter):
                sigma=np.random.choice(list(range(0,27,1)), 2, replace=False)
                SubF_new=SubF_old.copy()
                SubF_new[sigma[0]], SubF_new[sigma[1]] = SubF_new[sigma[1]], SubF_new[sigma[0]]
                newLogL=self.Loglike(message=self.Recover(self.string_split, subFunction=SubF_new))
                
                if(1<=math.exp(newLogL-LogL) ):
                    SubF_old=SubF_new.copy()
                    LogL=newLogL
                
                res=pd.DataFrame({"message":[''.join(self.Recover(self.string_split, subFunction=SubF_old))], 
                                "LogL":[LogL]})
                
                totalresult=pd.concat([totalresult, res]).sort_values(by="LogL", ascending=False)
            
                if(iter%2000==0):
                    print(totalresult.iloc[0,0])
                    
        print(totalresult.head())

    def strsplit(self, message):
        return [char for char in message]
    
    def EncodeMessage(message):
        alph=[a for a in "abcdefghijklmnopqrstuvwxyz "]
        map=[a for a in "defghijklmnopqrstuvwxyz abc"]

        message_encoded=""

        for letter in message:
            original_index=alph.index(letter)
            message_encoded += map[original_index]

        return message_encoded
    
    
class ClassicalGibbs():
    def __init__(self, niter=2000):
        self.GibbsFun(niter)
    
    def f_cond(self,u):
        if(u>=0 and u<=1/4):
            v=np.random.uniform(0,u+1/4,1)
        elif(u>1/4 and u<=3/4):
            v=np.random.uniform(u-1/4,u+1/4,1)
        elif(u>3/4 and u<=1):
            v=np.random.uniform(u-1/4,1,1)
        return v.item()
    
    def GibbsFun(self,niter):
        x=0
        y=0
        x_new=[]
        y_new=[]
        
        for i in range(niter):
            x_new.append(self.f_cond(y))
            y_new.append(self.f_cond(x_new[i]))
            x=x_new[i]
            y=y_new[i]
        
        res={"x_new": x_new, "y_new": y_new} 
        fig=px.scatter(res, x="x_new", y="y_new") 
        fig.show()