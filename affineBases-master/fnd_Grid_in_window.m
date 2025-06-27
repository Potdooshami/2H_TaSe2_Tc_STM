function tbl_line = fnd_Grid_in_window(I_oB__pC,rect_pC)
%{
phrphr@postech.ac.kr
-----------------------------------
<< Date >> 
2025_02_11__12_39: Birthday (I think optimization is suck...)

<< Purpose >>
 find the set of grid line which is in window

<< Input >>


<< Output >> 
they have 4 attribute
gridSeg_x (num_line,2): 1st:line id 2nd:start,end 
gridSeg_y (num_line,2): 1st:line id 2nd:start,end 
ord_bss (num_line,1): 1st:line id 2nd:start,end
scalar (num_line,1):  1st:line id 2nd:start,end
%}
arguments
    I_oB__pC (3,3) double % bases change Matrix from pC to oB   
    rect_pC (2,4) double %rect_pC_pCangular window: [[x1 x2 x3 x4];[y1 y2 y3 y4]]
end
I_pC__oB = inv(I_oB__pC);
[z1Lim,z2Lim] = find_gridRange(I_oB__pC,rect_pC);
n1_min = z1Lim(1);n1_max = z1Lim(2);n2_min = z2Lim(1);n2_max = z2Lim(2);
% uncroped line의 atom좌표계 정의
cpk1 = (((n1_min:n1_max).*[1;0])+[0;1])';
cpk2 = (((n2_min:n2_max).*[1;0])+[0;2])';
composite_primary_key = array2table([cpk1;cpk2],"VariableNames",{'scalar','ind_bss'});
num_n1 = n1_max - n1_min + 1;
num_n2 = n2_max - n2_min + 1;
nn1nn2_1 = [cpk1(:,[1 1]) [n2_min n2_max].*ones(num_n1,1)];
nn1nn2_2 = [[n1_min n1_max].*ones(num_n2,1) cpk2(:,[1 1])];
nn1nn2 = [nn1nn2_1;nn1nn2_2];
lseg = array2table(nn1nn2,"VariableNames",{'z1_start' 'z1_end' 'z2_start' 'z2_end'});
lseg = [composite_primary_key lseg];

% uncroped line의 pixel좌표계 정의
cols =  ["z1_start","z2_start";"z1_end","z2_end"];
se = ["start" "end"];
%% AffineTransform

for ind = 1:2
    lattice_oB = table2array(lseg(:,cols(ind,:)));
    lattice_pC = affine_augCal(I_pC__oB,lattice_oB');
    lseg = [lseg array2table(lattice_pC','VariableNames',["x", "y"]+"_"+se(ind))];
end

% croped line 정의 pixel 좌표계 리턴


poly1 = polyshape(rect_pC(1,:),rect_pC(2,:));
for ind = 1:size(lseg,1)
    lsegNow = lseg(ind,:);
    lineseg = [lsegNow.x_start lsegNow.y_start;
        lsegNow.x_end lsegNow.y_end];
    in = intersect(poly1,lineseg);    
    isCrossing(ind) = ~isempty(in);
    ins{ind}=in;
end
visible_gridLine = ins(isCrossing);
% lseg(isCrossing,1:2)
% visible_gridLine
num_ln = size(visible_gridLine,2);
grdX = zeros(num_ln,2);
grdY = grdX;
for ind = 1:num_ln;
    grdX(ind,:) = (visible_gridLine{ind}(:,1))';
    grdY(ind,:) = (visible_gridLine{ind}(:,2))';
end
tbl_line = lseg(isCrossing,1:2);
tbl_line.grdX = grdX;
tbl_line.grdY = grdY;

end