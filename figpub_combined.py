from figsets import *
fig1 = PubFig('2col',.8,width_rescale=.8)
REDUCE_FACTOR = 0.01
rf = REDUCE_FACTOR
bl = fig1.add_child([0,.5,.9,.3],label ='a',
                    comment='<schematic of DW>\n' \
                    'crypy: lattice+triangleCDW+DW\n' \
                    'annotation: lattice unitcell+CDW unitcell + visualize relative shift +\n' \
                    '3Q direction+phase&phaseshift value')
bl.reduce(w_reduce=rf,h_reduce=rf,anchor='left')
xy = bl.get_point('bottom_right')


#fig1.get_child(-1)
fig1.add_child(label ='a_0',wh=[.2,.2],xy=xy,anchor='bottom_right',
                comment='<orderparameter space diagram>\n' \
                '')
fig1.add_child([.9,.7,.1,.1],label ='a_1',comment='<cartoon of r-type dw>')
fig1.add_child([.9,.6,.1,.1],label ='a_2')
fig1.add_child([.9,.5,.1,.1],label ='a_3')

ax = fig1.add_child([0,0,.5,.5],'b',comment='<topo of DW>\n' \
'scalebar + nesting box of inset')
ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_left')
xy = ax.get_point('top_right')
fig1.add_child(label = 'b_0',xy=xy,wh=[.2,.2],anchor='top_right',comment='<topo zoom>')

ax = fig1.add_child([.5,0,.5,.5],'c',comment='<DWN phase map>\n' \
'nesting box of b, scalebar')
ax.reduce(w_reduce=rf,h_reduce=rf,anchor='bottom_right')     
xy = ax.get_point('bottom_left')
fig1.add_child(label = 'c_0',xy=xy,wh=[.2,.2],anchor='bottom_left',comment='<diagram of DWN>')
#----------------------------------------------------------------------------------------------------------------
fig2 = PubFig('2col',.5)
lu = .25
ls = [lu, lu/2,.2, .8-lu*(3/2)]
xs = []
ys = [0,.25]

fig2.add_child([0,0,ls[-1],.5],label='a',comment='<schematic of vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1],lu,lu,lu],label='b',comment='<topo of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1]+lu,lu*(3/2),lu/2,lu/2],label='c',comment='<shiftmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1]+lu,lu*(1),lu/2,lu/2],label='d',comment='<tripletmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1],0,lu,lu],label='e',comment ='<... of L>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1]+lu,lu*(1/2),lu/2,lu/2],label='f').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1]+lu,lu*(0),lu/2,lu/2],label='g').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig2.add_child([ls[-1]+lu*(3/2),0,.2,.5],label='h',
                comment='<schematic of vortex joining>\n' \
                'R,L vortex\n' \
                '3 R-L bonding\n' \
                'honeycomb tile').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
#----------------------------------------------------------------------------------------------------------------    
h1 = .1 
h2 =  .3
h3 = .3
hs = [h1,h2,h3]
h123 = h1+h2+h3
fig3 = PubFig('1col',1+h123)    
fig3.add_child([0,h123,1,1],label='a',
                comment='<schematic of DWN>\n' \
                'background white with phase value\n' \
                'directional DW + vortex symbol\n' \
                'bounding Loop set1: unique boundary decomposition,\n' \
                'bounding Loop set2: Vortex calculation').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
lbls = 'bcd'
ys = [h2+h3,h3,0]
sublbls =['','_1','_2']
w = 1/3
for iloop,lbl_main in zip(range(3),lbls):
    x= iloop/3
    for iinfo,lbl_sub,y,h in zip(range(3),sublbls,ys,hs):
        xywh =[x,y,w,h]
        label = lbl_main+lbl_sub
        fig3.add_child(xywh,label=label).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.get_child('b').comment = '<pathwords>:boundcary decomposition'
fig3.get_child('b_1').comment = '<geometric>'
fig3.get_child('b_2').comment = '<algebraic>'
fig3.get_child('c').comment = 'charge calculation 1'
fig3.get_child('d').comment = 'charge calculation 2'


#----------------------------------------------------------------------------------------------------------------
h1 = .7
hk23 = .15
fig4 = PubFig('1col',1+ h1)
alikes ='ac'
blikes ='bd'
xs =[0,.5]
for ind,alike,blike,x in zip(range(2),alikes,blikes,xs):
    ax = fig4.add_child([x,.5+h1,.5,.5],label=alike)
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    xy =ax.get_point('top_right')
    fig4.add_child(xy=xy,wh=[.2,.2],anchor='top_right',label= alike +'_1')    
    ax = fig4.add_child([x,+h1,.5,.5],label=blike)
    ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
    xy =ax.get_point('top_right')
    ax = fig4.add_child(xy=xy,wh=[hk23,hk23],anchor='top_right',label=blike+ '_2')
    xy =ax.get_point('top_left')
    fig4.add_child(xy=xy,wh=[hk23,hk23],anchor='top_right',label=blike+ '_1')
fig4.add_child([0,0,1,h1],label='e',
            comment = '<cartoon of phase pinning>\n' \
            '2 layer of projected schematic. upper(lower) is IC(C)\n' \
            'left-side: arrow with text phase-pinning\n' \
            'right-side: 1d version of pinnint').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig4.get_child('a').comment = '<topo-78K>'
fig4.get_child('a_1').comment = '<FFT-78K>\nhighlight Q peaks'
fig4.get_child('b').comment = '<phase-78K>\nk1'
fig4.get_child('b_1').comment = 'k2'
fig4.get_child('b_2').comment = 'k3'
fig4.get_child('c').comment = '...-110K'


#----------------------------------------------------------------------------------------------------------------
my_paper = PubProject(fig1, fig2, fig3, fig4)
print("\n--- 모든 Figure 레이아웃 플롯팅 ---")