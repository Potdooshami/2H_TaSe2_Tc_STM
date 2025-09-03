import os
dir = './phaseAnalysis'
files = os.listdir(dir)
files_no_ext = [f.replace('.gwy', '') for f in files]


import pandas as pd
fnTbl = 'dataTbl.xlsx'
df = pd.read_excel(fnTbl,engine='openpyxl')
df['fileName_no_ext'] = df['fileName_corrected_ntn'].str.replace('.sxm', '', regex=False)
df = df.set_index('fileName_no_ext')
dfCrrent =df.loc[files_no_ext]
dfCrrent = dfCrrent.iloc[[0,1,2,3,4,6,7,8]] #gwyddion files only


# fnlong = dir +'/'+ fn +'.gwy'
# sz_nano = dfCrrent.loc[fn][['scan_range_1_sxm','scan_range_2_sxm']].values *1000000000
# pxl = dfCrrent.loc[fn][['scan_pixels_1_sxm','scan_pixels_2_sxm']].values