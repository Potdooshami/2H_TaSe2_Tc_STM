function latinWin_oB = fnd_Lattice_in_window(I_oB__pC,rect_pC)
%{
phrphr@postech.ac.kr
-----------------------------------
<< Date >> 
2025_02_10__20_38: Birthday

<< Purpose >>
find lattice points in rectengular

<< Input >>


<< Output >> 
latinWin_oB (2,:) lattice points in rectengular in pixel Coord
%}
arguments
    I_oB__pC (3,3) double % bases change Matrix from pC to B   
    rect_pC (2,4) double %rectangular window: [[x1 x2 x3 x4];[y1 y2 y3 y4]]
end
[z1Lim,z2Lim] = find_gridRange(I_oB__pC,rect_pC);
n1_min = z1Lim(1);n1_max = z1Lim(2);n2_min = z2Lim(1);n2_max = z2Lim(2);

% n1, n2 격자 생성
[n1, n2] = meshgrid(n1_min:n1_max, n2_min:n2_max) ;

% 격자점의 실제 좌표 계산
I_pC__oB = inv(I_oB__pC); % bases change Matrix from oB(Lattice Coord) to pC(Pixel Coord)
liwCan_oB = [n1(:) n2(:)]'; %lattice in window candidate oB coordinate
liwCan_pC = affine_augCal(I_pC__oB,liwCan_oB);


% 직사각형 내부 격자점 확인
in_rect = inpolygon(liwCan_pC(1,:)', liwCan_pC(2,:)', rect_pC(1, :), rect_pC(2, :));

% 내부 점의 (n1, n2) 정수쌍 반환
latinWin_oB = [n1(in_rect), n2(in_rect)];
end