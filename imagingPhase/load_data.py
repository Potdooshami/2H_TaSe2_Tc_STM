import os
dir = './phaseAnalysis'
files = os.listdir(dir)
files_no_ext = [f.replace('.gwy', '') for f in files]
print('Directory of data:',dir)
# print(files)
print('list of files',files_no_ext)


import pandas as pd
fnTbl = 'dataTbl.xlsx'
df = pd.read_excel(fnTbl,engine='openpyxl')
df['fileName_no_ext'] = df['fileName_corrected_ntn'].str.replace('.sxm', '', regex=False)
df = df.set_index('fileName_no_ext')
dfCrrent =df.loc[files_no_ext]
print(dfCrrent)
dfCrrent = dfCrrent.iloc[[0,1,2,3,4,6,7,8]] #gwyddion files only


def get_pxlsz(fn):
    sz_nano = dfCrrent.loc[fn][['scan_range_1_sxm','scan_range_2_sxm']].values *1000000000
    pxl = dfCrrent.loc[fn][['scan_pixels_1_sxm','scan_pixels_2_sxm']].values
    return pxl,sz_nano

def loadgwy(fn,pxl):
    fnlong = dir +'/'+ fn +'.gwy'
    # print(":-)")
    import gwyfile
    import numpy as np
    container = gwyfile.load(fnlong)
    arr = container['/0/data']['data']
    arr = arr.reshape(pxl)
    print('I can read the variables in outside')
    return arr
