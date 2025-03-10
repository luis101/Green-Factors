

### Implementiation of Pastor et al (2022) green factor


## Load packages

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import linearmodels as lm
from scipy import stats
from regpyhdfe import Regpyhdfe
from fixedeffect.fe import fixedeffect
from stargazer.stargazer import Stargazer
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import gc
import glob
import time
import subprocess
import warnings
import dli

pd.options.mode.chained_assignment = None  # default='warn'

## Set directories

input_dir = "C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Data"
zip_dir = "C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Data\\Financials"

output_dir = "C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Projects\\QRSS"

res_out = "C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Projects\\Results"


#Daterange to be used:
    
#mindate = '09/31/2013'
#maxdate = '05/01/2022'
#maxdate = '01/01/2022'

#mindate = '09-30-2013'
mindate = '08-31-2013'
#maxdate = '04-30-2022'
#maxdate = '05-01-2022'
#maxdate = '12-31-2021'
maxdate = '09-30-2022'

#Set script directory to current working directory:
    
script = os.getcwd()
#os.chdir(script+'\\Data')
os.chdir('C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Data')


#%% Stock market data:    
 
ret = pd.read_csv(input_dir+'\\Financials\\firm_returns_monthly.csv', low_memory=False)
#ret = pd.read_csv(input_dir+'\\Financials\\returns_monthly.csv', low_memory=False)

zipfile = ZipFile(input_dir+'\\Financials\\industries.zip')
ind = pd.read_csv(zipfile.open(zipfile.namelist()[0]))  
#ind = pd.read_csv(input+'\\Financials\\industry_affiliation_monthly.csv', low_memory=False)
#ret = pd.read_csv(input+'\\Financials\\returns_monthly.csv', low_memory=False)
#ind = pd.read_csv(input+'\\Financials\\industries_monthly.csv', low_memory=False)

#ret = pd.merge(ret, ind, on=['issue_id', 'month_id', 'Date', 'company_name',
#                            'capital_iq', 'snlinstitutionid'])
#ret = pd.merge(ret, ind, on=['issue_id', 'month_id', 'company_name',
#                             'capital_iq', 'snlinstitutionid'], suffixes=['', '_ind'], indicator=True)
ret = pd.merge(ret, ind, on=['capital_iq', 'month_id'], suffixes=['', '_ind'], 
               indicator=True, how='inner')

print(ret['_merge'].value_counts())
ret.drop(['_merge'], axis=1, inplace=True)

try:
    ret = ret.drop(['Unnamed: 0'], axis = 1)
except:
    pass

ret = ret.rename(columns={'capital_iq':'CIQ_ID'})
ret['month_id'] = pd.to_datetime(ret['month_id']).dt.to_period("M")

ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)
for m in ['market_cap', 'market_cap_usd']:
    ret[m] = ret.groupby('CIQ_ID')[m].shift(1)
    ret[m] = ret[m]/1000
    
# Drop if CIQ ID not available:
ret = ret[~ret['CIQ_ID'].isnull()]

del ind
gc.collect()

# Prepare data filtering:

countf = pd.read_csv(input_dir+'\\Financials\\zero_return_filter.csv', low_memory=False)


# %%Apply filters:

# Winsorize returns by month at 1% and 99% percentiles
# Set max return to 300% if cumulative 2 month return less than 50%
# Set min excess return to -100%

print(len(ret))

def month_return_adjust(rd, ret_vars):
    for r in ret_vars:
        rd[r+'2'] = ((1+rd[r])*(1+rd[r].shift())-1)*100
        rd[r] = rd[r]*100
        #rd[r] = rd.groupby('month_id')[r].transform(lambda x: stats.mstats.winsorize(x, 
        #                                                                             limits=[0.01, 0.01]))
        #rd[r] = rd.groupby('month_id')[r].transform(lambda x: np.maximum(x.quantile(.01), 
        #                                                                 np.minimum(x, x.quantile(.99))))
        #rd.loc[(rd[r]>300)&(rd[r+'2']<50),r] = 300 
        rd.loc[(rd[r]>300)&(rd[r+'2']<50),r] = np.nan
        rd.loc[rd[r]<-100,r] = -100 
    
    return rd

ret.sort_values(['CIQ_ID', 'month_id'], inplace=True)

ret = month_return_adjust(ret, ['ret', 'ret_usd'])

# Drop small and illiquid firms:

#ret = ret[ret['cshtrd'] != 0]
ret = ret[ret['cshtrm'] != 0]
ret = ret[~((ret['prccd_usd']<1) & (ret['ret_usd']>300))]
ret = ret[~((ret['prccd_usd']<1) & (ret['ret_usd']<=-80))]
# data = data[data['prccd_usd']>1]
# data = data[data['market_cap_usd']>1]
# data = data[~((data['prccd_usd']<0.5) & (data['market_cap_usd']<1))]
print(len(ret))

# Drop if more than 30% zero return days per year

ret['year_id'] = pd.to_datetime(ret['Date']).dt.strftime('%Y').astype(int)
#ret = pd.merge(ret, countf, on=['issue_id', 'year_id'])
countf = countf.rename(columns={'capital_iq':'CIQ_ID'})
ret = pd.merge(ret, countf, on=['CIQ_ID', 'year_id'], how='left', indicator = True)
ret.loc[ret['perf'].isnull(), 'perf'] = 0.5
ret = ret[ret['perf'] < 0.3]
print(len(ret))

# Number of firms per country and month, require at least 10:

firms = ret.groupby(['country', 'month_id'])['ret'].count()
firms = firms.reset_index()
firms.rename(columns={'ret': 'firmno'}, inplace=True)

ret = pd.merge(ret, firms, on=['country', 'month_id'], how='outer')
ret = ret[ret['firmno'] >= 10]
print(len(ret))

# Drop time period where CSA data not available:

ret = ret[pd.to_datetime(ret['Date']) > '01-01-2013']
ret.drop('_merge', axis=1, inplace=True)
print(len(ret))

gc.collect()


#%% Import ESG data

import csa_data_exec

csa = csa_data_exec.csa_import()

csa = csa[csa['_merge']=='both']

#csa.drop(['CAM_ID', 'CAM_NAME', 'CAM_TYPE', 'EVR_PUBLISH_TIME', 'EVR_ID',
#          'aspect_type', 'Potential Score Contribution Combined', 'Data Availability Public',
#          'Data Availability Private', 'Data Availability Combined', 'SCORE_IMP_TEMP', '_merge'], 
#          axis=1, inplace=True)
csa.drop(['Potential Score Contribution Combined', 'Data Availability Public',
          'Data Availability Private', 'Data Availability Combined', 'SCORE_IMP_TEMP', '_merge'], 
          axis=1, inplace=True)
gc.collect()

# Only keep the respective level of aggregated scores

#csa['CAM_TYPE'].value_counts()
#csa = csa[csa['CAM_TYPE']=='CA']

#question = csa[csa['QUESTION'].notnull()]
#criterion = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].notnull())]
dimension = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].isnull())&(csa['DIMENSION'].notnull())]
esg = csa[(csa['QUESTION'].isnull())&(csa['CRITERION'].isnull())&(csa['DIMENSION'].isnull())]


#%% Prepare wide tables

#In wide tables the different score levels are next to each other instead of below each other, 
#this enables sorting for multiple variables 

#ESG

esg.loc[esg['DIMENSION'].isnull(), 'DIMENSION'] = "ESG"

esg.sort_values(['CSF_LCID', 'month_id'], inplace=True)
firm_inf = esg.groupby(['CSF_LCID', 'month_id'])['INDUSTRY', 'INDUSTRYGROUP', 
                                                 'CSF_LONGNAME', 'ISIN', 'GVKEY', 
                                                 'CIQ_ID', 'COUNTRYNAME', 'DJREGION',
                                                 'Date', 'CAM_YEAR', 'ASOF_DATE'].last()

#esg = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
#                      columns='DIMENSION', values=score)

esgs = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE')
esgi = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE_IMP')
esgn = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='SCORE_IMP_NP')
esgw = esg.pivot_table(index=['CSF_LCID', 'month_id'], 
                      columns='DIMENSION', values='WEIGHT')
esg = pd.merge(esgs, esgi, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
esg = pd.merge(esg, esgn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))
esg = pd.merge(esg, esgw, left_index=True, right_index=True, how='outer', suffixes=('',' W'))

esg = pd.merge(esg, firm_inf, left_index=True, right_index=True)
esg = esg.reset_index()

del esgs
del esgi
del esgn
del esgw

#Dimension:
 
dimension.replace({'DIMENSION':{'Economic Dimension':'Governance & Economic Dimension'}},
                 inplace=True)

#dimension = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
#                                  columns='DIMENSION', values=score)
   
dimensions = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE')
dimensioni = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE_IMP')
dimensionn = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='SCORE_IMP_NP')
dimensionw = dimension.pivot_table(index=['CSF_LCID', 'month_id'], 
                                  columns='DIMENSION', values='WEIGHT')
dimension = pd.merge(dimensions, dimensioni, left_index=True, right_index=True, how='outer', suffixes=('',' Imp'))
dimension = pd.merge(dimension, dimensionn, left_index=True, right_index=True, how='outer', suffixes=('',' NP'))
dimension = pd.merge(dimension, dimensionw, left_index=True, right_index=True, how='outer', suffixes=('',' W'))

dimension = pd.merge(dimension, firm_inf, left_index=True, right_index=True)
dimension = dimension.reset_index()

del dimensions
del dimensioni
del dimensionn
del dimensionw


#%% Industry-demeaned and standardized levels

def variable_adjust(data):
    
    var_name = list(data.loc[:,'month_id':'INDUSTRY'].columns)
    var_name = var_name[1:-1]
        
    data.sort_values(['CSF_LCID', 'month_id'], inplace=True)
    
    weights = var_name[int(3*len(var_name)/4):len(var_name)]
    variables = var_name[0:int(len(var_name)/4)]

    i=0
    
    for v in variables:
    
        for x in ['', ' Imp', ' NP']:
    
            data[v+x+'_us'] = -(100-data[v+x])*data[weights[i]]/100
            
            data[v+x+'_mean'] = data.groupby(['CAM_YEAR'])[v+x].transform('mean')
            data[v+x+'_dmf'] = data[v+x]-data[v+x+'_mean']

            data[v+x+'_ind'] = data.groupby(['INDUSTRY', 'CAM_YEAR'])[v+x].transform('mean')
            data[v+x+'_dm'] = data[v+x]-data[v+x+'_ind']

            data.drop([v+x+'_ind', v+x+'_mean'], axis=1, inplace=True)
        
        i+=1
        
    return data

dimension = variable_adjust(dimension)
esg = variable_adjust(esg)

gc.collect()


#%% Use both esg and dimension level data:
    
data = pd.merge(esg, dimension, on=['CSF_LCID', 'month_id', 'INDUSTRY',
'INDUSTRYGROUP', 'CSF_LONGNAME', 'ISIN', 'GVKEY', 'CIQ_ID',
'COUNTRYNAME', 'DJREGION', 'Date', 'CAM_YEAR', 'ASOF_DATE']) 

# Score types

cla = [x for x in list(data.columns) if x.endswith('Imp')]
cla = [x[:-4] for x in cla]
imp = [x for x in list(data.columns) if x.endswith('Imp')]
nnp = [x for x in list(data.columns) if x.endswith('NP')]

wt = list(data.loc[:, data.columns.str.contains(' W')].columns)
imp_all = list(data.loc[:, data.columns.str.contains('Imp')].columns)
np_all = list(data.loc[:, data.columns.str.contains('NP')].columns)
cla_all = [x.replace(' Imp', '') for x in imp_all]
us = list(data.loc[:, data.columns.str.contains('_us')].columns)


#%% Merge financial data with csa data

data = pd.merge(ret, data, on=['CIQ_ID', 'month_id'], 
                how='outer', suffixes=['', '_csa'], indicator=True)

# Keep specific dates

data = data[pd.to_datetime(data['Date']) > mindate]
data = data[pd.to_datetime(data['Date']) < maxdate]

data.sort_values(['CIQ_ID', 'month_id'], inplace=True)

print(data['_merge'].value_counts())
gc.collect()

# Forward fill, csa data to the next 24 months, industry data generally

info = [x for x in list(data.loc[:,'INDUSTRY':'DJREGION'].columns) if x != 'CIQ_ID']

data[info] = data.groupby(['CIQ_ID'])[info].ffill()
data[['CSF_LCID', 'Date', 'CAM_YEAR', 'ASOF_DATE']] = data.groupby(
    ['CIQ_ID'])[['CSF_LCID', 'Date', 'CAM_YEAR', 'ASOF_DATE']].ffill(limit=24)

data[us] = data.groupby(['CIQ_ID'])[us].ffill(limit=24)
data[cla_all] = data.groupby(['CIQ_ID'])[cla_all].ffill(limit=24)
data[imp_all] = data.groupby(['CIQ_ID'])[imp_all].ffill(limit=24)
data[np_all] = data.groupby(['CIQ_ID'])[np_all].ffill(limit=24)
data[wt] = data.groupby(['CIQ_ID'])[wt].ffill(limit=24)

summary = data.describe()
len(data)

# Keep data with CSA values

data = data[data['CSF_LCID'].notnull()]

print(len(data))

gc.collect()

# Only specific region

#data = data[data['DJREGION']=='NAM']


#%% Value-weighted cross-sectional average scores 

data['mcap_sum'] = data.groupby(['month_id'])['market_cap_usd'].transform('sum')
data['mcap_w'] = data['market_cap_usd']/data['mcap_sum']

variables = list(data.loc[:, data.columns.str.endswith('_us')].columns)

for v in variables:
    data[v+'_mw'] = data[v]*data['mcap_w']
    data[v+'_w'] = data.groupby(['month_id'])[v+'_mw'].transform('sum')
    
    data[v+'_dm'] = data[v] - data[v+'_w']
    data.drop([v+'_mw', v+'_w'], axis=1, inplace=True)
    #data[v+'_t'] = data[v+'_dm']*data['mcap_w']
    #data[v+'_t'] = data.groupby(['month_id'])[v+'_t'].transform('sum')
        
# Industry-average

variables = list(data.loc[:, data.columns.str.endswith('_us_dm')].columns)

ind_av = {}

for v in variables:
    average = data.groupby(['industry'])[v].mean()
    average.sort_values(ascending=False, inplace=True)
    ind_av[v] = average  
    
    
#%% GMB factor construction

# Set colors

colors = [(0, 43/235, 95/235),(0, 126/235, 174/235),(161/235, 195/235, 218/235)]
colors_pt = [(0, 43/235, 95/235),(6/235, 66/235, 61/235),
             (0, 126/235, 174/235),(6/235, 146/235, 125/235),
             (161/235, 195/235, 218/235),(153/235, 228/235, 215/235)]
colors_ls = [(122/235, 104/235, 85/235), (6/235, 146/235, 124/235)]

# Portfolio sorts

data.sort_values(['CIQ_ID', 'month_id'], inplace=True)

data_all = data.copy(deep=True)
#data = data[data['DJREGION']=='NAM'] 

var_name_us = list(data.loc[:, data.columns.str.endswith('_us_dm')].columns)
var_name_dm = [x.replace('_us', '') for x in var_name_us]
var_name = [x.replace('_us_dm', '') for x in var_name_us]

os.chdir('C:\\Users\\LUKAS_ZIMMERMANN\\OneDrive - S&P Global\\Projects')
    
import finfunc

#for v in var_name:
#    data = data[data[v]!=0]

data_c, num_c = finfunc.portfolio_sort(data, ['CSF_LCID', 'month_id'], 'month_id', 
                                     var_name, 10, zeros=False)
data_d, num_d = finfunc.portfolio_sort(data, ['CSF_LCID', 'month_id'], 'month_id', 
                                       var_name_dm, 10, zeros=False)
data_u, num_u = finfunc.portfolio_sort(data, ['CSF_LCID', 'month_id'], 'month_id', 
                                       var_name_us, 10, zeros=False)

def set_portfolios(data):
    var_name_pt = list(data.loc[:, data.columns.str.startswith('pt_')].columns)
    for v in var_name_pt:
        data.loc[data[v] <= 3, v] = 1
        data.loc[(data[v] <= 7)&(data[v] > 3), v] = 2
        data.loc[data[v] > 7, v] = 3
        
set_portfolios(data_c)
set_portfolios(data_d)
set_portfolios(data_u)            
        
# Return computation:

rdata_c = finfunc.portfolio_return(data_c, 'month_id', var_name, 3, 
                                   'ret_usd', weight='market_cap_usd')
rdata_d = finfunc.portfolio_return(data_d, 'month_id', var_name_dm, 3, 
                                   'ret_usd', weight='market_cap_usd')
rdata_u = finfunc.portfolio_return(data_u, 'month_id', var_name_us, 3, 
                                   'ret_usd', weight='market_cap_usd')

# Time-series regressions:

regt_c,reg_c = finfunc.ts_reg_det(rdata_c, var_name, top=3, controls=None, nw=6)
regt_d,reg_d = finfunc.ts_reg_det(rdata_d, var_name_dm, top=3, controls=None, nw=6)
regt_u,reg_u = finfunc.ts_reg_det(rdata_u, var_name_us, top=3, controls=None, nw=6)

#Plot return time-series:

#fct_df = rdata_u.copy(deep=True)
fct_df = pd.merge(rdata_c, rdata_u, left_index=True, right_index=True)
fct_df.loc[fct_df.index[0] + pd.offsets.MonthEnd(-1)] = [0]*len(fct_df.loc['2013-09'])
fct_df.sort_index(inplace=True)

factors = list(fct_df.loc[:,fct_df.columns.str.contains('ls')].columns)
#factors = ['Environmental Dimension_ls_3', 'Environmental Dimension Imp_ls_3',
#           'Environmental Dimension NP_ls_3']
#factors = ['Environmental Dimension_us_dm_ls_3', 'Environmental Dimension Imp_us_dm_ls_3',
#           'Environmental Dimension NP_us_dm_ls_3']

#All types of scores:

for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+s+'_ls_3', d+' Imp'+s+'_ls_3', d+' NP'+s+'_ls_3']
        #fac =[d+s+'_ls_3', d+' Imp'+s+'_ls_3']
        i=0
        for f in fac:
            #((1+(fct_df.set_index(['month_id']).f/100)).cumprod()).plot(title='Cumulative Returns of ' + f)
            #((1-(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + f).legend()
            ((1+(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[i]).legend()
            i+=1
        plt.savefig(output_dir+'\\'+d+s+'.png')
        plt.clf()
  
#Only imputed scores:        
  
for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+' Imp'+s+'_ls_3']
        for f in fac:
            ((1+(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[1])
        plt.savefig(output_dir+'\\'+d+s+'_IO.png')
        plt.clf()
 
#factors = ['Environmental Dimension_ret_1','Environmental Dimension_ret_3',
#           'Environmental Dimension Imp_ret_1','Environmental Dimension Imp_ret_3',
#           'Environmental Dimension NP_ret_1','Environmental Dimension NP_ret_3']   
#factors = ['Environmental Dimension_us_dm_ret_1', 'Environmental Dimension_us_dm_ret_3',
#           'Environmental Dimension Imp_us_dm_ret_1', 'Environmental Dimension Imp_us_dm_ret_3',
#           'Environmental Dimension NP_us_dm_ret_1', 'Environmental Dimension NP_us_dm_ret_3']

#All types of scores:

for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+s+'_ret_1', d+s+'_ret_3', d+' Imp'+s+'_ret_1', 
              d+' Imp'+s+'_ret_3', d+' NP'+s+'_ret_1', d+' NP'+s+'_ret_3']
        #fac =[d+s+'_ret_1', d+' Imp'+s+'_ret_1', d+' NP'+s+'_ret_1', 
        #      d+s+'_ret_3', d+' Imp'+s+'_ret_3', d+' NP'+s+'_ret_3']
        #fac =[d+s+'_ret_1', d+' Imp'+s+'_ret_1', 
        #      d+s+'_ret_3', d+' Imp'+s+'_ret_3']
        i=0
        for f in fac:
            #((1+(fct_df.set_index(['month_id']).f/100)).cumprod()).plot(title='Cumulative Returns of ' + f)
            #((1-(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + f).legend()
            ((1+(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors_pt[i]).legend()
            i+=1 
        plt.savefig(output_dir+'\\'+d+s+'_pt.png')
        plt.clf()

#Only imputed scores: 

for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+' Imp'+s+'_ret_1', d+' Imp'+s+'_ret_3']
        i=0
        for f in fac:
            ((1+(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors_ls[i]).legend(
                                                     ['Brown Portfolio', 'Green Portfolio'])
            i+=1 
        plt.savefig(output_dir+'\\'+d+s+'_pt_IO.png')
        plt.clf()


#%% Green factor construction

#zipfile = ZipFile(input_dir+'\\Financials\\betas_market.zip')
#beta = pd.read_csv(zipfile.open(zipfile.namelist()[1])) 
beta = pd.read_csv(input_dir+'\\Financials\\betas_monthly_data.csv')  
beta = beta[~beta['beta'].isnull()] 
beta['month_id'] = pd.to_datetime(beta['month_id']).dt.to_period('M') 
beta.rename(columns={'capital_iq':'CIQ_ID'}, inplace=True)

market = pd.read_csv(input_dir+'\\Financials\\returns_market_monthly.csv')  
market['month_id'] = pd.to_datetime(market['month_id']).dt.to_period('M') 

#data_all['month_id'] = pd.to_datetime(data_all['Date']).dt.to_period('M') 
data = pd.merge(data_all, beta, on=['CIQ_ID', 'month_id'])
data = pd.merge(data, market, on=['country', 'month_id'])

data['ret_e'] = data['ret_usd']-data['beta']*data['mkt_usd']*100

# Cross-sectional regressions

def format_results(df, name):
    n = df.iloc[0,2]
    dfn = df.reset_index()
    dfn = dfn.iloc[0:,1:3]
    #dfn = df.iloc[0:,0:2].unstack()
    dfn = dfn.unstack()
    dfn = dfn.reset_index()
    dfn.sort_values(['level_1'],inplace=True)
    dfn.drop('level_1', axis=1, inplace=True)
    dfn.rename(columns={'level_0':'coefficient', 0:'value'}, inplace=True)

    dfn = dfn.set_index('coefficient')
    dfn = pd.DataFrame(pd.Series(dfn['value']).append(pd.Series(n)))
    dfn.rename(columns={0:name}, inplace=True)

    index_names = []
    for i in range(1,len(df['coef'])+1):
        index_names.append('coef_'+str(i))
        index_names.append('t_stats_'+str(i))
    index_names.append('n')
    dfn = dfn.set_index(pd.Series(index_names))

    return dfn

id_vars = ['month_id', 'ret_usd', 'ret_e']
var_name_us.extend(id_vars)
var_name_dm.extend(id_vars)
var_name.extend(id_vars)

data_c = data[var_name]
data_d = data[var_name_dm]
data_u = data[var_name_us]

var_name_us = var_name_us[:-3]
var_name_dm = var_name_dm[:-3]
var_name = var_name[:-3]

results = {}
results_e = {}
results_dm = {}
results_us = {}

beta = {}
beta_e = {}
beta_dm = {}
beta_us = {}

i = 1
for v in var_name:
    print(v)
    
    data_c[v] = data_c[v]/10

    fmb, tb, reg, b = finfunc.fmb_reg_var(data_c,'month_id','ret_usd',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results[i]=r1
    beta[v]=pd.DataFrame(b)    
    i+=1
    
i = 1
for v in var_name:
    print(v)
    
    data_c[v] = data_c[v]/10

    fmb, tb, reg, b = finfunc.fmb_reg_var(data_c,'month_id','ret_e',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_e[i]=r1
    beta_e[v]=pd.DataFrame(b)    
    i+=1

i = 1    
for v in var_name_dm:
    print(v)
    
    data_d[v] = data_d[v]/100

    fmb, tb, reg, b = finfunc.fmb_reg_var(data_d,'month_id','ret_usd',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_dm[i]=r1
    beta_dm[v]=pd.DataFrame(b) 
    i+=1

i = 1    
for v in var_name_us:
    print(v)
    
    data_u[v] = data_u[v]/10

    fmb, tb, reg, b = finfunc.fmb_reg_var(data_u,'month_id','ret_e',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_us[i]=r1
    beta_us[v]=pd.DataFrame(b) 
    i+=1
    
#gr = data_u[data_u['month_id']=='2020-01']
#gr.dropna(subset=['ret_e'], inplace=True)
#g = np.array(gr['Environmental Dimension_us_dm'])
#re = np.array(gr['ret_e'])
#f = g.T.dot(re)/g.dot(g)

beta_us = pd.concat(beta_us, axis=1).sum(axis=1, level=0)
#fct_df = beta_us.copy(deep=True)
fct_df = pd.concat([beta_us, pd.DataFrame(data_u['month_id'].astype(str).unique())], axis=1)   
fct_df = fct_df.set_index(0)
#factors = list(fct_df.loc[:,fct_df.columns].columns)

sd = np.array(rdata_u[list(rdata_u.loc[:,rdata_u.columns.str.contains('ls')].columns)].describe().loc['std'])
sdr = np.array(fct_df.describe().loc['std'])
sc = np.divide(sd,sdr)

fct_sc = fct_df*sc 

#beta = beta_e.copy()
beta = pd.concat(beta, axis=1).sum(axis=1, level=0)
beta_df = pd.concat([beta, pd.DataFrame(data_c['month_id'].astype(str).unique())], axis=1)   
beta_df = beta_df.set_index(0)

sd = np.array(rdata_c[list(rdata_c.loc[:,rdata_c.columns.str.contains('ls')].columns)].describe().loc['std'])
sdr = np.array(beta_df.describe().loc['std'])
sc = np.divide(sd,sdr)

beta_sc = beta_df*sc 

fct_sc = pd.merge(fct_sc, beta_sc, left_index=True, right_index=True)

fct_sc.loc['2013-08'] = [0]*len(fct_sc.loc['2013-09'])
fct_sc.sort_index(inplace=True)
fct_sc.index = pd.to_datetime(fct_sc.index).to_period('M')

#factors = list(fct_sc.loc[:,fct_sc.columns.str.contains('Environmental Dimension')].columns)
#for f in factors:
#    ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + f).legend()

#All types of scores:

for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+s, d+' Imp'+s, d+' NP'+s]
        i=0
        for f in fac:
            #((1+(fct_df.set_index(['month_id']).f/100)).cumprod()).plot(title='Cumulative Returns of ' + f)
            #((1-(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + f).legend()
            ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[i]).legend()
            i+=1
        plt.savefig(output_dir+'\\'+d+s+'_green_factor.png')
        #plt.savefig(output_dir+'\\'+d+s+'_green_factor_e.png')
        plt.clf()
        
#Only imputed scores:        
    
for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['', '_us_dm']:
        fac =[d+' Imp'+s]
        for f in fac:
            ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[1])
        plt.savefig(output_dir+'\\'+d+s+'_green_factor_IO.png')
        #plt.savefig(output_dir+'\\'+d+s+'_green_factor_e_IO.png')
        plt.clf()


## Standardized comparison of both approaches

def standardize(data, variables):

    for v in variables:
        data[v+'_mean'] = data.groupby(['month_id'])[v].transform('mean')
        data[v+'_sd'] = data.groupby(['month_id'])[v].transform('std')
        data[v+'_stf'] = (data[v]-data[v+'_mean'])/data[v+'_sd']
        
    return data

data_c = standardize(data_c, var_name)
data_d = standardize(data_d, var_name_dm)
data_u = standardize(data_u, var_name_us)

var_name = list(data_c.loc[:, data_c.columns.str.contains('_stf')].columns)
var_name_dm = list(data_d.loc[:, data_d.columns.str.contains('_stf')].columns)
var_name_us = list(data_u.loc[:, data_u.columns.str.contains('_stf')].columns)

results = {}
results_e = {}
results_dm = {}
results_us = {}

beta = {}
beta_e = {}
beta_dm = {}
beta_us = {}

i = 1
for v in var_name:
    print(v)
    data_c[v] = data_c[v]/10
    
    fmb, tb, reg, b = finfunc.fmb_reg_var(data_c,'month_id','ret_usd',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results[i]=r1
    beta[v]=pd.DataFrame(b)    
    i+=1

i = 1
for v in var_name:
    print(v)
    
    fmb, tb, reg, b = finfunc.fmb_reg_var(data_c,'month_id','ret_e',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_e[i]=r1
    beta_e[v]=pd.DataFrame(b)    
    i+=1

i = 1    
for v in var_name_dm:
    print(v)
    data_d[v] = data_d[v]/10
    
    fmb, tb, reg, b = finfunc.fmb_reg_var(data_d,'month_id','ret_usd',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_dm[i]=r1
    beta_dm[v]=pd.DataFrame(b) 
    i+=1

i = 1    
for v in var_name_us:
    print(v)
    data_u[v] = data_u[v]/10
    
    fmb, tb, reg, b = finfunc.fmb_reg_var(data_u,'month_id','ret_e',v,intercept=False)
    print(fmb)
    
    r1 = format_results(fmb,v)
    results_us[i]=r1
    beta_us[v]=pd.DataFrame(b) 
    i+=1
    
beta_us = pd.concat(beta_us, axis=1).sum(axis=1, level=0)
fct_df = pd.concat([beta_us, pd.DataFrame(data_u['month_id'].astype(str).unique())], axis=1)   
fct_df = fct_df.set_index(0)

sd = np.array(rdata_u[list(rdata_u.loc[:,rdata_u.columns.str.contains('ls')].columns)].describe().loc['std'])
sdr = np.array(fct_df.describe().loc['std'])
sc = np.divide(sd,sdr)

fct_sc = fct_df*sc 

#beta = beta_e.copy()
beta = pd.concat(beta, axis=1).sum(axis=1, level=0)
beta_df = pd.concat([beta, pd.DataFrame(data_c['month_id'].astype(str).unique())], axis=1)   
beta_df = beta_df.set_index(0)

sd = np.array(rdata_c[list(rdata_c.loc[:,rdata_c.columns.str.contains('ls')].columns)].describe().loc['std'])
sdr = np.array(beta_df.describe().loc['std'])
sc = np.divide(sd,sdr)

beta_sc = beta_df*sc 

fct_sc = pd.merge(fct_sc, beta_sc, left_index=True, right_index=True)

fct_sc.loc['2013-08'] = [0]*len(fct_sc.loc['2013-09'])
fct_sc.sort_index(inplace=True)
fct_sc.index = pd.to_datetime(fct_sc.index).to_period('M')

#All types of scores:

for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['_stf', '_us_dm_stf']:
        fac =[d+s, d+' Imp'+s, d+' NP'+s]
        i=0
        for f in fac:
            #((1+(fct_df.set_index(['month_id']).f/100)).cumprod()).plot(title='Cumulative Returns of ' + f)
            #((1-(fct_df[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + f).legend()
            ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[i]).legend()
            i+=1
        plt.savefig(output_dir+'\\'+d+s+'_green_factor_std.png')
        #plt.savefig(output_dir+'\\'+d+s+'_green_factor_std_e.png')
        plt.clf()
 
#Only imputed scores:        
    
for d in ['ESG', 'Environmental Dimension', 'Governance & Economic Dimension', 'Social Dimension']:
    for s in ['_stf', '_us_dm_stf']:
        fac =[d+' Imp'+s]
        for f in fac:
            ((1+(fct_sc[f]/100)).cumprod()).plot(title='Cumulative Returns of ' + d,
                                                 ylabel='Value of 1$ Invested',
                                                 xlabel='Year',color=colors[1])
        plt.savefig(output_dir+'\\'+d+s+'_green_factor_std_IO.png')
        #plt.savefig(output_dir+'\\'+d+s+'_green_factor_std_e_IO.png')
        plt.clf()   
 
    