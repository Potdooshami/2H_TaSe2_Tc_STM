from figpub.figsets import *
from figpub.solLatt.layout import fig3 as fig2
from figpub.solLatt.layout import fig4 as fig3
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
if __name__ == '__main__':
    my_paper.plot_layouts()
    my_paper.show()
    print("\n--- 모든 Figure 레이아웃 플롯팅 ---")