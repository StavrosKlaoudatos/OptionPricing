import streamlit as st

import time
from BlackScholes import BlackScholes
from BlackScholes import Plots
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from BinomialPricing import BinomialPriceTree, BPTPlots



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
