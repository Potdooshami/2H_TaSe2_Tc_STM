# load data fraction and intergrate it---------------------------------------------
import pickle
import pandas as pd
with open('dataCache/pa.pkl','rb') as f:
    pa = pickle.load(f)
    # 1st index: phase or amplitude
    # 2nd index: temperature
    # 3rd index: k123
with open('dataCache/hddf.pkl','rb') as f:
    df = pickle.load(f)
with open('C02_arrcln.pkl', 'rb') as f:
    arr_cln, pk_choose = pickle.load(f)
with open('C02_angle.pkl', 'rb') as f:
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

# phase correcting---------------------------------------------------------------
phi1 = -2.8
phi2 = 1.5
phi3 = 3
phis = [phi1,phi2,phi3]
for ik in range(3): #add a constant guage
    pa[0][5][ik] = pa[0][5][ik] - phis[ik]
for idt in range(len(pa[0])):
    for ik in range(3): # change a guage
        pa[0][idt][ik] = -pa[0][idt][ik]    

def load_C05(pa=pa,df=df):
    idx_ordered= [0,4,2,5,3,1]
    idx_final = [idx_ordered[i] for i in [0, 3, 4, 5]]
    p = pa[0]
    for idt in range(len(p)):
        p[idt][0], p[idt][1] = p[idt][1], p[idt][0]
        # p[idt] = p[idt] + phis[idt]    
    df['phase'] = p
    df = df.iloc[idx_final]
    return df