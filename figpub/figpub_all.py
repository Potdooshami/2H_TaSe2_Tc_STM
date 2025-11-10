print('run figpub_iccdw')
from  figpub.iccdw.layout import my_paper as p1
print('run figpub_combined')
from figpub.figpub_combined import my_paper as p2
print('run figpub_solitionLattice')
from figpub.solLatt.layout import my_paper as p3

p1.plot_layouts()
p2.plot_layouts()
p3.plot_layouts()
p1.show()