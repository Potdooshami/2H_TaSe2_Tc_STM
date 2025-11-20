from figpub.figsets import *
REDUCE_FACTOR = 0.01
rf = REDUCE_FACTOR
#----------------------------------------------------------------------------------------
l_sch =.6
l_top =.4
h= l_top*2
fig1 = PubFig('1col',h)
f = fig1


f.add_child([0,h-l_sch,l_sch,l_sch],label='a',
            comment='<schematic of CDW>\n' \
            'crypy: lattice, triangle CDW,\n' \
            'unitcells'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
f.add_child([0,0,l_sch,h-l_sch],label='a_1',
            comment='<vertical view>\n' \
            'vertical view of lattice'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
f.add_child([l_sch,l_top,l_top,l_top],label='b',
            comment='<topography>'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
f.add_child([l_sch,0,l_top,l_top],label='c',
            comment='<FFT>'
            ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')

# #----------------------------------------------------------------------------------------
u = 1/5
fig2 = PubFig('2col',3*u)
f = fig2
lbls = 'abcdef'
c =0
for idw,y in zip(range(3),[2*u,u,0]):
    for iinfo,x in zip(range(2),[0,u]):
        c = c+1
        f.add_child([x,y,u,u],label=lbls[c-1]
                    ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
ax = f.add_child([2*u,0,3*u,3*u],label='g',
            comment ='<DWN shiftmap>\n' \
            '3dw cropbox\n' \
            'chiral vortex map\n' \
            'three jump retrurn')
ax.reduce(w_reduce=rf,h_reduce=rf,anchor='center')
xy = ax.get_point('bottom_left')
# f.add_child(anchor='bottom_left',xy=xy,wh=[.25,.25],label='h',
#             comment='<DWN schematic>'
#             ).reduce(w_reduce=rf*2,h_reduce=rf*2,anchor='center')
f.add_child([4*u,0,u,u],label='h',comment='<orderparameter space diagram>').translate(-rf,rf)        
f.get_child('a').comment ='<cartoon of a1>'
f.get_child('b').comment ='<topo of a1>'
f.get_child('c').comment ='<... of a2>'
f.get_child('e').comment ='<... of a3>'
    
# #----------------------------------------------------------------------------------------
fig3 = PubFig('2col',.5)
lu = .25
ls = [lu, lu/2,.2, .8-lu*(3/2)]
xs = []
ys = [0,.25]

fig3.add_child([0,0,ls[-1],.5],label='a',
               comment='<schematic of vortex>'
               ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1],lu,lu,lu],label='b',
               comment='<topo of R vortex>'
               ).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1]+lu,lu*(3/2),lu/2,lu/2],label='c',comment='<shiftmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1]+lu,lu*(1),lu/2,lu/2],label='d',comment='<tripletmap of R vortex>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1],0,lu,lu],label='e',comment ='<... of L>').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1]+lu,lu*(1/2),lu/2,lu/2],label='f').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1]+lu,lu*(0),lu/2,lu/2],label='g').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig3.add_child([ls[-1]+lu*(3/2),0,.2,.5],label='h',
                comment='<schematic of vortex joining>\n' \
                'R,L vortex\n' \
                '3 R-L bonding\n' \
                'honeycomb tile').reduce(w_reduce=rf,h_reduce=rf,anchor='center')
# #----------------------------------------------------------------------------------------

h1 = .1 
h2 =  .3
h3 = .3
hs = [h1,h2,h3]
h123 = h1+h2+h3
fig4 = PubFig('1col',1+h123)    
fig4.add_child([0,h123,1,1],label='a',
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
        fig4.add_child(xywh,label=label).reduce(w_reduce=rf,h_reduce=rf,anchor='center')
fig4.get_child('b_1').comment = '<geometric>'
fig4.get_child('b_2').comment = '<algebraic>'
fig4.get_child('c').comment = 'charge calculation 1'
fig4.get_child('b').comment = '<pathwords>:boundcary decomposition'
fig4.get_child('d').comment = 'charge calculation 2'
#----------------------------------------------------------------------------------------
fig1.figtitle = '3x3 Charge Density Wave (CDW) in 2H-TaSe2'
fig2.figtitle = '3 types of topological domain walls (DW)'
fig3.figtitle = 'two types of chiral vortices (R and L)'
fig4.figtitle = 'Phase Pinning at the Domain Walls'
fig1.keyword_info = ['Lattice structure', 
                     'CDW in real space', 
                     'FFT of Topo']
fig1.keyword_argument = ['atom(Se) and CDW both corrugation are observed',
                         'CDW center is hollow', 
                         '3x3 CDW from FFT']
fig2.keyword_info = ['three DWs topo', 
                     'three DWs cartoon',
                     'phaseshiftmap of DWN']
fig2.keyword_argument = ['atom corruation is fixed but CDW currgation localy shift', 
                         'three shift rgb has a b^3=g^3=r^3=rgb=0',
                         'shiftmap shows honeycomb-like DWN']    
fig3.keyword_info = ['schematic of DV', 
                     'phase infos of DV', 
                     'tinker toy model of DV']
fig3.keyword_argument = ['tri-junction of 3 differnet phase', 
                     'Identified by not only shift but also centerness',
                     'honeycomb structure is naturally formed by the existence of R,L vortex']  
fig4.keyword_info = ['DWN topological encoded','geometric diagram','algebraic diagram']
fig4.keyword_argument = ['Bulk-boundary correspendence(Uniqueness of boundary decomposition)',
                         'charge calculation from bounding loops',
                         'Topological invariant is rigorously defined by group']
                         


my_paper = PubProject(fig1, fig2, fig3, fig4,
                      title="Chiral vortex interlocking in topological soliton lattice",
                      synopsis=[
            "We investigated the 3x3 Charge Density Wave (CDW) in 2H-TaSe2.",
            "3 types of topological domain walls (DW) were observed.",
            "two types of chiral vortices (R and L) were identified",
            "Spectroscopy confirms the insulating nature of the C-phase."
        ])
#----------------------------------------------------------------------------------------
# child = my_paper.figs[0].get_child('a')
def imsert_im(ax,img_path):
    ax.imshow(plt.imread(img_path))
    ax.set_xticks([])
    ax.set_yticks([])
def set_draw(ind_fig,lbl_panel,img_path):
         my_paper.figs[ind_fig].get_child(lbl_panel).draw = lambda ax: imsert_im(
    ax, img_path)
set_draw(0,'a','assets/lattice.png')
set_draw(1,'a','assets/dw3_0_False.png')
set_draw(1,'c','assets/dw3_1_False.png')
set_draw(1,'e','assets/dw3_2_False.png')
set_draw(1,'b','assets/dw_r_topo.png')
set_draw(1,'d','assets/dw_g_topo.png')
set_draw(1,'f','assets/dw_b_topo.png')


set_draw(1,'g','assets/solLatt_shiftmap.png')
set_draw(2,'a','assets/vortices.png')

set_draw(2,'b','assets/vor1_topo.png')
set_draw(2,'c','assets/vor1_dw.png')
set_draw(2,'d','assets/vor1_vertex.png')
set_draw(2,'e','assets/vor2_topo.png')
set_draw(2,'f','assets/vor2_dw.png')
set_draw(2,'g','assets/vor2_vertex.png')


if __name__ == '__main__':
    # my_paper.plot_layouts()
    # my_paper.plot_draws()
    # my_paper.show()
    my_paper.create_report(filename="Report_solLatt.pptx")
