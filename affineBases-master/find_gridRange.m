function [z1Lim,z2Lim] = find_gridRange(I_oB__pC,rect_pC)
%{
phrphr@postech.ac.kr
-----------------------------------
<< Date >> 
2025_02_10__17_14: Birthday

<< Purpose >>
 find the range of lattice near the window
<< Input >>

<< Output >> 

%}


arguments
    I_oB__pC (3,3) double   
    rect_pC (2,4) double %rectangular window: [[x1 x2 x3 x4];[y1 y2 y3 y4]]
end
n_coords = affine_augCal(I_oB__pC,rect_pC);
z1Lim = [floor(min(n_coords(1, :))) ceil(max(n_coords(1, :)))];
z2Lim = [floor(min(n_coords(2, :))) ceil(max(n_coords(2, :)))];
end
