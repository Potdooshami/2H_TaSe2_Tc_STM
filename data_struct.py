import pickle
import pandas as pd
with open('dataCache/pa.pkl','rb') as f:
    pa = pickle.load(f)
with open('dataCache/hddf.pkl','rb') as f:
    df = pickle.load(f)
with open('dataCache/C02_arrcln.pkl', 'rb') as f:
    arr_cln, pk_choose = pickle.load(f)
with open('dataCache/C02_angle.pkl', 'rb') as f:
    angle_restores = pickle.load(f)    

new_dt = {'fns':'110K_highres',
    'arr_clns':arr_cln,
    'colors':'#1f77b4',
    'nms':'110K(2)',
    'Ts':110,
    'nano':40,
    'sz':1024,
    'pxl20nm':512,
    'k123':pk_choose}
df = pd.concat([df, pd.DataFrame([new_dt])], ignore_index=True)
pa[0].append(angle_restores)