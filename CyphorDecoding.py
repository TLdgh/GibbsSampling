import numpy as np
import math

class DecodeMessage():
    def __init__(self, message: str, M_big: np.ndarray):
        self.alphabets=self.strsplit("abcdefghijklmnopqrstuvwxyz ")
        self.string_split=self.strsplit(message)
        self.pair_indices=[]
        self.M_big=M_big
        
        SubF_old=np.random.choice(list(range(0,27,1)), 27, replace=False)
        LogL=self.Loglike(message=self.Recover(self.string_split, subFunction=SubF_old))
        
        for iter in range(10000):
            sigma=np.random.choice(list(range(0,27,1)), 2, replace=False)
            SubF_new=SubF_old.copy()
            SubF_new[sigma[0]], SubF_new[sigma[1]] = SubF_new[sigma[1]], SubF_new[sigma[0]]
            newLogL=self.Loglike(message=self.Recover(self.string_split, subFunction=SubF_new))
            if(1<=math.exp(newLogL-LogL)):
                SubF_old=SubF_new.copy()
                LogL=newLogL
            
            if(iter%1000==0):
                print(''.join(self.Recover(self.string_split, subFunction=SubF_old)))
            
            
    def MakeMatrix(self, message):
        M=np.ones((27,27))
        message_pairs=[(message[i], message[i+1]) for i in range(len(message)-1)]
        
        def find_index(p, ref):
            return (ref.index(p[0]), ref.index(p[1]))
        
        self.pair_indices=[find_index(pair, self.alphabets) for pair in message_pairs]

        for ind in self.pair_indices:
            M[ind[0], ind[1]]+=1
        
        return(M/M.sum())

    def Loglike(self, message):
        M=self.MakeMatrix(message)
        p=0
        for ind in self.pair_indices:
            p+=M[ind[0], ind[1]]*math.log(self.M_big[ind[0], ind[1]])
        return p
    
    def Recover(self, message, subFunction):
        message_enum=[[i for i, alpha in enumerate(self.alphabets) if alpha == j] for j in message]
        message_enum = [item 
                     for sublist in message_enum 
                        for item in sublist]

        new_message=[self.alphabets[j] for j in [subFunction[i] for i in message_enum]]
        
        return new_message
    
    #def GibbsFun():
        
    def strsplit(self, message):
        return [char for char in message]
    
    
    
    
    
    
