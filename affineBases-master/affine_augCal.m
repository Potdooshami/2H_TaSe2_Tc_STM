function v_pC = affine_augCal(I_pC__oB,v_oB)
%{
phrphr@postech.ac.kr
-----------------------------------
<< Date >> 
2025_02_08__15_28: Birthday

<< Purpose >>
증강행렬을 이용할 때, 단점은 매번 벡터를 augment하고 deaugment해 줘야 한다.
이게 귀찮고 은근히 많이 틀려서 코드륵 작성하기로 함.

<< Input >>


<< Output >> 

%}
arguments
    I_pC__oB (3,3) double % 증강행렬
    v_oB (2,:) double % 벡터들 oB좌표계
end
v_oB(3,:) = 1;
v_pC = I_pC__oB*v_oB;
v_pC = v_pC(1:2,:);
end