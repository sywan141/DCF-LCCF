# Obi-Wan 2024/6/12; DCF & LCCF v1.1.2
# Take measurement errors into considerations
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Reset the time series
def reset(arr_a,arr_b):
    mint= np.min((np.min(arr_a[:,0]),np.min(arr_b[:,0])))
    arr_a[:,0]=arr_a[:,0]-mint
    arr_b[:,0]=arr_b[:,0]-mint
    return arr_a,arr_b

# detrend
def dtrd(arr,ply):
    if ply==1:
        fc_fit=lambda x,a,b:a*x+b
        popt,pcov = curve_fit(fc_fit,arr[:,0],arr[:,1])
        arr[:,1]=arr[:,1]-fc_fit(arr[:,0],popt[0],popt[1])
    elif ply==2:
        fc_fit=lambda x,a,b,c:a*x**2+b*x+c
        popt,pcov = curve_fit(fc_fit,arr[:,0],arr[:,1])
        arr[:,1]=arr[:,1]-fc_fit(arr[:,0],popt[0],popt[1],popt[2])
    return arr

'''Edelson-Krolik DCF function
    Inputs: 
    two arrays,should contain 3 columns each(observed time, observed flux, measurement errors,dtype = ndarrays)
    taos(Edges of each time lag detection bin, dtype = ndarrays)
    bin(The length of each time lag detection bin, dtype = float)
    Returns:
    central time lag values t_mid(dtype = ndarray)
    DCF values(dtype = ndarray)
    DCF errors(dtype = ndarray)
'''
def DCF_cal(arr_a,arr_b,taos,bin):
    if (arr_a.shape!=arr_b.shape):
        raise ValueError('The input arrays should have the same size')
    else:
        dt = np.zeros((len(arr_a[:,0]),len(arr_b[:,0])))
        UDCF = np.zeros((len(arr_a[:,0]),len(arr_b[:,0])))
        DCF = np.zeros(len(taos)-1)
        DCF_err = np.zeros(len(taos)-1)
        M = np.zeros(len(taos)-1)

        ma = np.mean(arr_a[:,1])
        mb = np.mean(arr_b[:,1])
        sa=np.std(arr_a[:,1],ddof=1)
        sb=np.std(arr_b[:,1],ddof=1)
        t_mid = 0.5*(taos[1:]+taos[:-1]) # give taus

        for i in range(0,len(arr_a)):
            for j in range(0,len(arr_b)):
                dt[i,j]=arr_a[i,0]-arr_b[j,0]
                UDCF[i,j] = [(arr_a[i,1]-ma)*(arr_b[j,1]-mb)]/np.sqrt((sa**2-arr_a[i,2]**2)*(sb**2-arr_b[j,2]**2)) # Take measurement errors for data points into consideration 
        for k in range(0,len(taos)-1):
            tao_min = taos[k]   # give the edges of bins (length-1)
            tao_max = taos[k]+bin
            flag = (dt >= tao_min) & (dt < tao_max)
            M[k]=np.sum(flag) # only true values(1) are added to M
            DCF[k]=np.sum(UDCF[flag])/M[k]
            DCF_err[k] = np.sqrt(np.sum((UDCF[flag]-DCF[k])**2))/(M[k]-1)
    return t_mid, DCF,DCF_err


''' 'Local' correlation function
    similar with DCF function, but the means and variances are all 'local'
    Inputs: 
    two arrays,should contain 3 columns each(observed time, observed flux, measurement errors,dtype = ndarrays)
    taos(Edges of each time lag detection bin, dtype = ndarrays)
    bin(The length of each time lag detection bin, dtype = float)
    Returns:
    central time lag values t_mid(dtype = ndarray)
    LCCF values(dtype = ndarray)
    LCCF errors(dtype = ndarray)
'''
def LCCF_cal(arr_a,arr_b,taos,bin):
    if(arr_a.shape!=arr_b.shape):
        raise ValueError('The input arrays should have the same size')
    else:
        dt = np.zeros((len(arr_a[:,0]),len(arr_b[:,0])))
        LCCF = np.zeros(len(taos)-1)
        LCCF_err = np.zeros(len(taos)-1)
        M = np.zeros(len(taos)-1)
        t_mid = 0.5*(taos[1:]+taos[:-1])
   
        for i in range(0,len(arr_a)):
            for j in range(0,len(arr_b)):
                dt[i,j]=arr_a[i,0]-arr_b[j,0]
        for k in range(0,len(taos)-1):
            tao_min = taos[k]
            tao_max = taos[k]+bin
            idx_a,idx_b = np.where((dt >= tao_min) & (dt < tao_max)) # select index pairs which satisfy given time lag
            ma = np.mean(arr_a[idx_a,1])
            mb = np.mean(arr_b[idx_b,1])
            sa=np.std(arr_a[idx_a,1],ddof=1)
            sb=np.std(arr_b[idx_b,1],ddof=1)
            M[k]= len(idx_a)
            ULCCF = (arr_a[idx_a,1]-ma)*(arr_b[idx_b,1]-mb)/np.sqrt((sa**2-arr_a[idx_a,2]**2)*(sb**2-arr_b[idx_b,2]**2)) #Take measurement errors for data points into consideration 
            LCCF[k] = np.sum(ULCCF)/M[k]
            LCCF_err[k] = np.sqrt(np.sum((ULCCF-LCCF[k])**2))/(M[k]-1)
    return t_mid, LCCF,LCCF_err