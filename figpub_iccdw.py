from figsets import *
REDUCE_FACTOR = 0.01
rf = REDUCE_FACTOR
#----------------------------------------------------------------------------------------
w_latt = 1/3
u = 1/(2 + w_latt)
xs = [0,w_latt*u,1-u]
fig1 =PubFig('2col',u)
f = fig1
f.add_child([0,0,u*w_latt,u],
            label='a',
            comment='<schematic of lattice>\n' \
            'vertical,top view lattice CDW'
            ).reduce(w_reduce=rf,h_reduce=rf*10,anchor='center')
f.add_child([u*w_latt,0,u,u],
            label='b',
            comment='<Topography with island-layer>\n' \
            'topography + almost commensurate island'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
ax = f.add_child([1-u,0,u,u],
            label='c',
            comment='<phase and phase shift>\n' \
            'phase of domain, phase shift of domain'
            )
ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
xy = ax.get_point('bottom')
f.add_child(xy=xy,wh=[.1,.1],anchor='bottom',
            label='c_1',
            comment='<orderparameter space diagram>\n'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
#------------------------------------------------------------
w_fft = .7
h_1dfft = 1-w_fft 
u = 1/(3 + w_fft)


fig2 = PubFig('2col',u)
f = fig2
f.add_child([0,u*h_1dfft,w_fft*u,w_fft*u],
            label='a',
            comment='<2dFFT>\n' \
            '...').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
f.add_child([0,0,w_fft*u,h_1dfft*u],
            label='b',
            comment='<1dFFT>\n' \
            '...'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
lbls = 'cdefgh'
c= 0
xs = np.arange(3)*u +w_fft*u
ys = [0,0]
for ik,x,comm_k in zip(range(3),xs,['k1','k2','k3']):

    for iinfo,y,comm_info in  zip(range(1),ys,['map','histogram']):
        lbl = lbls[c]
        c = c+1
        xywh = [x,y,u,u]
        f.add_child(xywh,
                    label=lbl,
                    comment=comm_k+'\n'+comm_info
                    ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
#----------------------------------------------------------------------        
num_T = 4
num_plot = 2
h_bel = 1
l_fft = .35
l_k23 = .3
T_list = ['78K','110K','115K','118K']
xs_T = np.arange(num_T)/num_T
xs_plot = np.arange(num_plot)/num_plot
ys = np.array([0,h_bel,h_bel+1])/num_T 
fig3 = PubFig('2col',(2+h_bel)/num_T,width_rescale = .8)
f = fig3
for iT,T,x,lbl_topo,lbl_phase in zip(
    range(num_T),T_list,xs_T,['a','b','c','d'],['e','f','g','h']):
    xywh = [x,ys[2],1/num_T,1/num_T]
    ax = f.add_child(xywh,
                label=lbl_topo,
                comment=T)
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    xy = ax.get_point('top_right')
    f.add_child(anchor='top_right',xy=xy,wh=[l_fft/num_T,l_fft/num_T],label=lbl_topo+'_1',
                comment='2dFFT'
                )
    
    xywh = [x,ys[1],1/num_T,1/num_T]
    ax = f.add_child(xywh,
                label=lbl_phase,
                comment='k1_phasemap')
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center') 
    xy = ax.get_point('top_right')
    ax = f.add_child(anchor='top_right',xy=xy,wh=[l_k23/num_T,l_k23/num_T],label=lbl_phase+'_1',
                comment='k3'
                )
    xy = ax.get_point('top_left')
    f.add_child(anchor='top_right',xy=xy,wh=[l_k23/num_T,l_k23/num_T],label=lbl_phase+'_2',
                comment='k2'
                )
for iinfo,x,lbl,cmts in zip(range(num_plot),xs_plot,'ijk',
    ['<FFT_related>\n'
    'pk_dlt,pk_intensity',
     '<phase_map related>\n'
     'pinningness, return_length']):
    xywh = [x,ys[0],1/num_plot,h_bel/num_T]
    ax = f.add_child(xywh,
                label=lbl,
                comment='cmt'
                )
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center'
                         )
    ax.reduce(w_reduce=rf*3,h_reduce=rf*3,anchor='top_right')
#----------------------------------------------------------------------------------------
num_case = 3
rh_1d = .5
ys = [rh_1d/num_case,0]
h_tot = (1+rh_1d)/num_case
fig4 = PubFig('2col',h_tot)
f = fig4
cmt_T = ['sl','iccdw','melted']
cmt_dim = ['2d vis','1d vis']
for icase,x in zip(range(num_case),np.arange(num_case)/num_case):
    f.add_child([x,ys[0],1/num_case,1/num_case],label=chr(ord('a')+icase),
                comment=cmt_T[icase]+cmt_dim[0]
                ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    f.add_child([x,ys[1],1/num_case,rh_1d/num_case],label=chr(ord('d')+icase),
                comment=cmt_T[icase]+cmt_dim[1]
                ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    





#-----
print('run iccdw project')
f = fig4
f.plot_layout()
f.fig.show()
breakpoint()

