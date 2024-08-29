import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from tqdm import tqdm
import sklearn
from scipy.stats import norm 

N = norm.cdf




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

    
    





