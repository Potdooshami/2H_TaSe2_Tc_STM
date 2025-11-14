from crypy_examples.pts_on_poly import lattice_points_in_polygon



def gen_hex(R):
    tht = 2*np.pi*(1/12+np.arange(0,6)/6)
    v_xy =R*(2/np.sqrt(3))*np.array([np.cos(tht),np.sin(tht)])
    polygon = v_xy.transpose()
    return polygon

def lattice_points_in_hex(a1, a2,R ,contain_boundary=True):
    polygon = gen_hex(R)
    if contain_boundary:
        R = R +0.01
    else:
        R = R -0.01
    foo,indices = lattice_points_in_polygon(a1, a2, polygon)
    return indices