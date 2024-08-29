import streamlit as st

import time

from BlackScholes import Plots
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from tqdm import tqdm
import sklearn
from scipy.stats import norm 

N = norm.cdf
epsilon = 1e-10






class BlackScholes:

    def __init__(self, St, r):
        self.St = St
        self.r = r
        

    def BS_Call(self, T,sigma,K):  

        epsilon = 1e-10  # Small value to avoid division by zero

        d1 = (np.log(self.St/(K + epsilon)) + (self.r + 0.5*sigma**2)*(T + epsilon))/(sigma*np.sqrt(T + epsilon))

        d2 = d1 - sigma*np.sqrt(T)
        C = N(d1)*self.St -N(d2)*K*np.exp(-self.r*(T))
        return C


    def BS_Put(self, T,sigma,K):  

        epsilon = 1e-10

        d1 = (np.log(self.St/(K+epsilon)) + (self.r + 0.5*sigma**2)*(T+epsilon))/(sigma*np.sqrt(T+epsilon))
        d2 = d1 - sigma*np.sqrt(T)
        P = N(-d2)*K*np.exp(-self.r*(T)) - N(-d1)*self.St
        return P


    def CallPrice(self, T,sigma,K):
        return self.BS_Call(T,sigma,K)


    def PutPrice(self, T,sigma,K):
        return self.BS_Put(T,sigma,K)
    






class Plots(BlackScholes):

    def __init__(self, St, r):
        self.St = St
        self.r = r
    
        self.BS = BlackScholes(self.St,self.r)



    def BS_Time_Strike_Call(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        time = np.linspace(0, T, 100)
        strikeprices = np.linspace(i, f, 100)


        '''Call Pricing Heatmap'''
        ZCall = np.zeros((len(time), len(strikeprices)))
        for i, t in tqdm(enumerate(time)):
            for j, k in enumerate(strikeprices):
                ZCall[i, j] = self.BS.CallPrice(t, sigma,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.flipud(np.transpose(np.fliplr(ZCall))), cmap='viridis', aspect='auto', origin='lower', extent=[time[0], time[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (time[-1] - time[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (time[-1] - time[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val


        xloc = [ time[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ time[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = time[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZCall[i * len(time) // num_boxes_x_loc, j * len(strikeprices) // num_boxes_y_loc]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(time[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Time to Maturity (T) in Years')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Black-Scholes Call Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0)
        fig.colorbar(imk, label='Call Price')
        
        plt.savefig('streamlitapp/plots/BS/BSTSC.png', transparent=True)
        return fig



    def BS_Time_Strike_Put(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        time = np.linspace(0, T, 100)
        strikeprices = np.linspace(i, f, 100)

        '''Put Pricing Heatmap'''
        ZPut = np.zeros((len(time), len(strikeprices)))
        for i, t in enumerate(time):
            for j, k in enumerate(strikeprices):
                ZPut[i, j] = self.BS.PutPrice(t, sigma,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.transpose(ZPut), cmap='viridis', aspect='auto', origin='lower', extent=[time[0], time[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (time[-1] - time[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (time[-1] - time[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val



        xloc = [ time[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ time[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = time[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZPut[i * (len(time)-1) // num_boxes_x_val, j * (len(strikeprices)-1) // num_boxes_y_val]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(time[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Time to Maturity (T) in Years')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Black-Scholes Put Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Put Price')

        plt.savefig('streamlitapp/plots/BS/BSTSP.png', transparent=True)
        return fig




    def BS_Volatility_Strike_Call(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        volatily = np.linspace(1/2*sigma, 3/2*sigma, 100)
        strikeprices = np.linspace(i, f, 100)


        '''Call Pricing Heatmap'''
        ZCallVS = np.zeros((len(volatily), len(strikeprices)))
        for i, v in enumerate(volatily):
            for j, k in enumerate(strikeprices):
                ZCallVS[i, j] = self.BS.CallPrice(T, v,k)

        fig, ax = plt.subplots(figsize=(12,8.4))  
        imk = ax.imshow((np.transpose(ZCallVS)), cmap='viridis', aspect='auto', origin='lower', extent=[volatily[0], volatily[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (volatily[-1] - volatily[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (volatily[-1] - volatily[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val


        xloc = [ volatily[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ volatily[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = volatily[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZCallVS[i * len(volatily) // num_boxes_x_loc, j * len(strikeprices) // num_boxes_y_loc]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(volatily[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Black-Scholes Call Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Call Price')
        
        plt.savefig('streamlitapp/plots/BS/BSVSC.png', transparent=True)
        return fig




    def BS_Volatility_Strike_Put(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        volatily = np.linspace(1/2*sigma, 3/2*sigma, 100)
        strikeprices = np.linspace(i, f, 100)

        '''Put Pricing Heatmap'''
        ZPutVS = np.zeros((len(volatily), len(strikeprices)))
        for i, v in enumerate(volatily):
            for j, k in enumerate(strikeprices):
                ZPutVS[i, j] = self.BS.PutPrice(T, v,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.transpose(ZPutVS), cmap='viridis', aspect='auto', origin='lower', extent=[volatily[0], volatily[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (volatily[-1] - volatily[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (volatily[-1] - volatily[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val



        xloc = [ volatily[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ volatily[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = volatily[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZPutVS[i * (len(volatily)-1) // num_boxes_x_val, j * (len(strikeprices)-1) // num_boxes_y_val]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(volatily[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Black-Scholes Put Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Put Price')
        
        
        plt.savefig('streamlitapp/plots/BS/BSVSP.png', transparent=True)
        return fig

    
    






class BinomialPriceTree:

    def __init__(self, S0, r, N):
        self.S0 = S0
        self.r = r
        self.N = N
        self.p = np.zeros(self.N)



    def u(self,sigma,T):
        return  np.exp(sigma * np.sqrt(T / self.N))
    def d(self,sigma,T):
        return  1 / (np.exp(sigma * np.sqrt(T / self.N)))
    


    def PutPrice(self, T,sigma,K):
        p0=(self.u(sigma,T)**2  - self.u(sigma,T)*np.exp(-self.r*T/self.N))/(self.u(sigma,T)**2-1 + epsilon)

        p1 = np.exp(-self.r * T/self.N) - p0


        for i in range(self.N):
            val = K - self.S0 * self.u(sigma,T)**(2*i-self.N)
            
            if val>=0:
                self.p[i] = val
            else:
                self.p[i] = 0
        
        for j in range(self.N-1,0,-1):
            for i in range(j):
                self.p[i] = p0 * self.p[i+1] + p1 * self.p[i]

                exercise = K - self.S0 * self.u(sigma,T)**(2*i - j)

                if self.p[i]<= exercise:
                    self.p[i] = exercise


        return self.p[0]
    

    def CallPrice(self, T, sigma, K):
        p0 = (self.u(sigma, T)**2  - self.u(sigma, T) * np.exp(-self.r * T / self.N)) / (self.u(sigma, T)**2 - 1 + epsilon)
        p1 = np.exp(-self.r * T / self.N) - p0

        for i in range(self.N):
            val = self.S0 * self.u(sigma, T)**(2 * i - self.N) - K
            self.p[i] = max(val, 0)
        
        for j in range(self.N - 1, 0, -1):
            for i in range(j):
                self.p[i] = p0 * self.p[i + 1] + p1 * self.p[i]

                exercise = self.S0 * self.u(sigma, T)**(2 * i - j) - K
                if self.p[i] <= exercise:
                    self.p[i] = exercise

        return self.p[0]
    






class BPTPlots(BinomialPriceTree):

    def __init__(self, S0, r,N):
        self.S0 = S0
        self.r = r
        self.N = N
    
        self.BPT = BinomialPriceTree(self.S0,self.r,self.N)



    def BPT_Time_Strike_Call(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        time = np.linspace(0, T, 100)
        strikeprices = np.linspace(i, f, 100)


        '''Call Pricing Heatmap'''
        ZCall = np.zeros((len(time), len(strikeprices)))
        for i, t in enumerate(time):
            for j, k in enumerate(strikeprices):
                ZCall[i, j] = self.BPT.CallPrice(t, sigma,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.transpose(ZCall), cmap='viridis', aspect='auto', origin='lower', extent=[time[0], time[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (time[-1] - time[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (time[-1] - time[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val


        xloc = [ time[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ time[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = time[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZCall[i * len(time) // num_boxes_x_loc, j * len(strikeprices) // num_boxes_y_loc]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(time[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Time to Maturity (T) in Years')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Binomial Price Tree Call Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0)
        fig.colorbar(imk, label='Call Price')
        
        plt.savefig('streamlitapp/plots/BPT/BPTTSC.png', transparent=True)
        return fig



    def BPT_Time_Strike_Put(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        time = np.linspace(0, T, 100)
        strikeprices = np.linspace(i, f, 100)

        '''Put Pricing Heatmap'''
        ZPut = np.zeros((len(time), len(strikeprices)))
        for i, t in enumerate(time):
            for j, k in enumerate(strikeprices):
                ZPut[i, j] = self.BPT.PutPrice(t, sigma,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.transpose(ZPut), cmap='viridis', aspect='auto', origin='lower', extent=[time[0], time[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (time[-1] - time[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (time[-1] - time[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val



        xloc = [ time[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ time[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = time[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZPut[i * (len(time)-1) // num_boxes_x_val, j * (len(strikeprices)-1) // num_boxes_y_val]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(time[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Time to Maturity (T) in Years')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Binomial Price Tree Put Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Put Price')

        plt.savefig('streamlitapp/plots/BPT/BPTTSP.png', transparent=True)
        return fig




    def BPT_Volatility_Strike_Call(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        volatily = np.linspace(1/2*sigma, 3/2*sigma, 100)
        strikeprices = np.linspace(i, f, 100)


        '''Call Pricing Heatmap'''
        ZCallVS = np.zeros((len(volatily), len(strikeprices)))
        for i, v in enumerate(volatily):
            for j, k in enumerate(strikeprices):
                ZCallVS[i, j] = self.BPT.CallPrice(T, v,k)

        fig, ax = plt.subplots(figsize=(12,8.4))  
        imk = ax.imshow(np.transpose((ZCallVS)), cmap='viridis', aspect='auto', origin='lower', extent=[volatily[0], volatily[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (volatily[-1] - volatily[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (volatily[-1] - volatily[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val


        xloc = [ volatily[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ volatily[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = volatily[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZCallVS[i * len(volatily) // num_boxes_x_loc, j * len(strikeprices) // num_boxes_y_loc]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(volatily[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Binomial Price Tree Call Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Call Price')
        
        plt.savefig('streamlitapp/plots/BPT/BPTVSC.png', transparent=True)
        return fig




    def BPT_Volatility_Strike_Put(self,T,sigma, K,Range):


        i = (1-Range)*K
        f = (1+Range)*K

        volatily = np.linspace(1/2*sigma, 3/2*sigma, 100)
        strikeprices = np.linspace(i, f, 100)

        '''Put Pricing Heatmap'''
        ZPutVS = np.zeros((len(volatily), len(strikeprices)))
        for i, v in tqdm(enumerate(volatily)):
            for j, k in enumerate(strikeprices):
                ZPutVS[i, j] = self.BPT.PutPrice(T, v,k)

        fig, ax = plt.subplots(figsize=(12, 8.4))  
        imk = ax.imshow(np.transpose(ZPutVS), cmap='viridis', aspect='auto', origin='lower', extent=[volatily[0], volatily[-1], strikeprices[0], strikeprices[-1]])

        
        num_boxes_x_loc = 10
        num_boxes_y_loc = 10  
        box_width_loc = (volatily[-1] - volatily[0]) / num_boxes_x_loc
        box_height_loc = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_loc

        num_boxes_x_val = 9
        num_boxes_y_val = 9 
        box_width_val = (volatily[-1] - volatily[0]) / num_boxes_x_val
        box_height_val = (strikeprices[-1] - strikeprices[0]) / num_boxes_y_val



        xloc = [ volatily[0]+i*box_width_loc for i in range(num_boxes_x_loc)]
        xvals = [ volatily[0]+i*box_width_val for i in range(0,num_boxes_x_val+1)]
        ax.set_xticks(xloc + box_width_loc/2) 
        ax.set_xticklabels([f"{x:.2f}" for x in xvals]) 

        yloc = [ strikeprices[0]+i*box_height_loc for i in range(num_boxes_y_loc)]
        yvals = [ strikeprices[0]+i*box_height_val for i in range(0,num_boxes_y_val+1)]
        ax.set_yticks(yloc + box_height_loc/2) 
        ax.set_yticklabels([f"{x:.2f}" for x in yvals]) 



        for i in range(num_boxes_x_loc):
            for j in range(num_boxes_y_loc):

                box_x = volatily[0] + i * box_width_loc
                box_y = strikeprices[0] + j * box_height_loc
                box_value =ZPutVS[i * (len(volatily)-1) // num_boxes_x_val, j * (len(strikeprices)-1) // num_boxes_y_val]
                ax.text(box_x + box_width_loc / 2, box_y + box_height_loc / 2, f"{box_value:.2f}", ha='center', va='center', color='white')

        for i in range(num_boxes_x_loc):
            ax.axvline(volatily[0] + i * box_width_loc, color='black', linestyle='-', linewidth=0.5)
        for j in range(num_boxes_y_loc ):
            ax.axhline(strikeprices[0] + j * box_height_loc, color='black', linestyle='-', linewidth=0.5)


        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Strike Price (K)')
        ax.set_title('Binomial Price Tree Put Price Heatmap')
        ax.tick_params(axis='both', which='both', length=0) 
        fig.colorbar(imk, label='Put Price')
        
        
        plt.savefig('streamlitapp/plots/BPT/BPTVSP.png', transparent=True)
        return fig

    
    







if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True



def Action(St,K,r,T,sigma,R,model,optiontype,N=0):
    if optiontype == "European":

        st.markdown("<h2 style='text-align: center; '>{}</h2>".format(model), unsafe_allow_html=True)

        
        figs = tqdm([Plots(St, r).BS_Time_Strike_Call(T,sigma, K,R),Plots(St, r).BS_Time_Strike_Put(T,sigma, K,R),Plots(St, r).BS_Volatility_Strike_Call(T,sigma, K,R),Plots(St, r).BS_Volatility_Strike_Put(T,sigma, K,R)])

        path = "streamlitapp/plots/BS/"

        figpaths = [path+"BSTSC.png",path+"BSTSP.png",path+"BSVSC.png",path+"BSVSP.png"]


        row1 = st.columns([6,6])
        row2 = st.columns([4,4])

        



        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center; '>Call Price</h3>", unsafe_allow_html=True)
            st.divider()
            val = round(BlackScholes(St, r).CallPrice(T,sigma,K),2)
            st.markdown("<h5 style='text-align: center; '>{}</h5>".format(val), unsafe_allow_html=True)
            


            
            
            st.image(figpaths[0])
            st.image(figpaths[2])

        with col2:
            st.markdown("<h3 style='text-align: center; '>Put Price</h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown("<h5 style='text-align: center; '>{}</h5>".format(round(BlackScholes(St, r).PutPrice(T,sigma,K),2)), unsafe_allow_html=True)
            st.image(figpaths[1])
            st.image(figpaths[3])


        




    if optiontype == "American":

        st.markdown("<h2 style='text-align: center; '>{}</h2>".format(model), unsafe_allow_html=True)

        
        figs = tqdm([BPTPlots(St, r,N).BPT_Time_Strike_Call(T,sigma, K,R),BPTPlots(St, r,N).BPT_Time_Strike_Put(T,sigma, K,R),BPTPlots(St, r,N).BPT_Volatility_Strike_Call(T,sigma,K,R),BPTPlots(St, r,N).BPT_Volatility_Strike_Put(T,sigma, K,R)])

        path = "streamlitapp/plots/BPT/"

        figpaths = [path+"BPTTSC.png",path+"BPTTSP.png",path+"BPTVSC.png",path+"BPTVSP.png"]


        row1 = st.columns([6,6])
        row2 = st.columns([4,4])

        



        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center; '>Call Price</h3>", unsafe_allow_html=True)
            st.divider()
            val = round(BinomialPriceTree(St, r,N).CallPrice(T,sigma,K),2)
            st.markdown("<h5 style='text-align: center; '>{}</h5>".format(val), unsafe_allow_html=True)
            


            
            
            st.image(figpaths[0])
            st.image(figpaths[2])

        with col2:
            st.markdown("<h3 style='text-align: center; '>Put Price</h3>", unsafe_allow_html=True)
            st.divider()
            val = round(BinomialPriceTree(St, r,N).PutPrice(T,sigma,K),2)
            st.markdown("<h5 style='text-align: center; '>{}</h5>".format(val), unsafe_allow_html=True)
            st.image(figpaths[1])
            st.image(figpaths[3])











st.markdown("<h1 style='text-align: center; '>Option Pricing Models</h1>", unsafe_allow_html=True)

st.divider()



# Using "with" notation
with st.sidebar:
    optiontype = st.radio(
        "Option Type",
        ("European","American")
    )

    if optiontype == "European":
        model = st.radio(
        "Model",
        ("Black Scholes")
    )
        
    if optiontype == "American":
        model = st.radio(
        "Model",
        ("Binomial Price Tree")
        )
        


    St = st.number_input("Asset Price",value=100.00)
    K = st.number_input("Strike Price",value=100.00)
    r = st.number_input("Risk-free Rate",value=0.05)
    T = st.number_input("Time to Maturity (Years)",value=1.00)
    sigma = st.number_input("Volatility",value=0.2)
    R = st.number_input("Strike Price Range", value=0.2)
    

    if model=='Binomial Price Tree':
        N = st.number_input("Number of Nodes",value=10)
    else:
        N=0



    st.button("Generate Heatmap", key=None, help=None, on_click=click_button, args=None, kwargs=None, type="primary", disabled=False, use_container_width=False)

if st.session_state.clicked:
    plt.close('all')
    Action(St,K,r,T,sigma,R,model,optiontype,N)
    st.session_state.clicked = False
