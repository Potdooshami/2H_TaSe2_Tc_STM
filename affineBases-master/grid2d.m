classdef grid2d    
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    properties
        offset_stdpts (1,2) double = [nan nan] %[o-p]_pC
        a1_stdpts (1,2) double = [nan nan] % [a_1]_pC
        a2_stdpts (1,2) double = [nan nan] % [a_2]_pC
        sz_stdpts (1,2) double = [nan nan] % [a_3]
        clr_a1 = [1 0 0]
        clr_a2 = [0 1 0]
        clr_faceWindow = [0 1 1];
        clr_edgeWindow = [0 0 1];
    end
    properties (Dependent)
        augMat (3,3) table % [I]_pC^oB
        baryBss (2,3) table
        inLatt table 
        inGrid table 
    end
    properties (Dependent,Hidden)
        rect (2,4) double
        is_fully_defined (1,1) boolean
    end
    
    methods % about dependent properties
        function v = get.augMat(obj)
            v = [obj.a1_stdpts' obj.a2_stdpts' obj.offset_stdpts';0 0 1];
            v = array2table(v,"RowNames",{'x','y','aug'} ,"VariableNames",{'a1','a2','o'});
        end
        function v = get.baryBss(obj)%barycentric basis            
            v = [[0;0] obj.a1_stdpts' obj.a2_stdpts']-obj.offset_stdpts';
            v = array2table(v,"RowNames",{'x','y'} ,"VariableNames",{'p0(=o)','p1(=o+v1)','p2(=o+v2)'});
        end
        function v = get.rect(obj)
            sz = obj.sz_stdpts;
            v = [0 sz(1) sz(1) 0;
                0 0 sz(2) sz(2)];
        end
        function v = get.inLatt(obj)            
            if obj.is_fully_defined
                I_pC__oB = table2array(obj.augMat);
                I_oB__pC = inv(I_pC__oB) ;
                v_oB =  transpose(fnd_Lattice_in_window(I_oB__pC,obj.rect));
                v_pC = affine_augCal(I_pC__oB,  v_oB);
                v = table();
                v.Lattice =v_oB';
                v.pixel = v_pC';
                % v = [ ];
                'reCalculated'
            end            
        end
        function v = get.inGrid(obj)
            if obj.is_fully_defined
                I_oB__pC = inv(table2array(obj.augMat)) ;
                v = fnd_Grid_in_window(I_oB__pC,obj.rect);
            end            
        end
        function v = get.is_fully_defined(obj)
            is_affine_defined =  ~any(isnan(table2array(obj.augMat)),"all");
            is_window_defined = ~any(isnan(obj.sz_stdpts),"all");
            v = is_affine_defined & is_window_defined;
        end
    end   
    methods % calculate something
        function pC = oB2pC(obj,oB)
            pC = affine_augCal(table2array(obj.augMat),oB);
        end
        function oB = pC2oB(obj,pC)
            oB = affine_augCal(inv(table2array(obj.augMat)),pC)
        end
    end
    methods % plot graphic object
        function pWindow(obj)
            fill(obj.rect(1, :), obj.rect(2, :), obj.clr_faceWindow, 'FaceAlpha', 0.3, 'EdgeColor', obj.clr_edgeWindow, 'LineWidth', 1.5);
        end
        function pBases(obj)
            O = obj.offset_stdpts;
            b1 = obj.a1_stdpts;
            b2 = obj.a2_stdpts;
            quiver(O(1), O(2), b1(1), b1(2), 0,'Color' ,obj.clr_a1, 'LineWidth', 2, 'MaxHeadSize', 0.5); % a1 벡터
            hold on
            quiver(O(1), O(2), b2(1), b2(2), 0,'Color' ,obj.clr_a2, 'LineWidth', 2, 'MaxHeadSize', 0.5); % a2 벡터
            hold off
        end
        function p = pLatticeWin(obj)
            xy = obj.inLatt.pixel
            z1 = obj.inLatt.Lattice(:,1);
            z2 =  obj.inLatt.Lattice(:,2);
            p = plot(xy(:, 1), xy(:, 2), 'k.', 'MarkerSize', 8, 'LineWidth', 1.5);
            rowZ1 = dataTipTextRow("z1",z1);
            rowZ2 = dataTipTextRow("z2",z2);
            p.DataTipTemplate.DataTipRows(end+1) = rowZ1;
            p.DataTipTemplate.DataTipRows(end+1) = rowZ2;
        end
        function pGridlineWin(obj)
            for ibss = 1:2
                % iSegNow = tbl_line.ind_bss == ibss;
                iSegNow = obj.inGrid.ind_bss == ibss;
                tbl_lineNow  = obj.inGrid(iSegNow,:);
                grdX1d =  tbl_lineNow.grdX;
                grdX1d(:,3) =  nan;
                grdX1d = grdX1d';
                grdX1d = grdX1d(:);
                grdY1d =  tbl_lineNow.grdY;
                grdY1d(:,3) =  nan;
                grdY1d = grdY1d';
                grdY1d = grdY1d(:);
                grd1d(ibss).X = grdX1d;
                grd1d(ibss).Y = grdY1d;
            end
            clrs = [obj.clr_a1; obj.clr_a2];
            for ibss = 1:2
                plot(grd1d(ibss).X,grd1d(ibss).Y,'Color',clrs(ibss,:))
            end

        end
        
    end
    methods % return other instance
        function newInstance = latticeFineTune(obj,indLatt,smallShift)            
            arguments
                obj
                indLatt (2,3) double 
                smallShift (2,3) double 
            end            
            I_old = table2array(obj.augMat); % I_pC__oB
            dlt = [smallShift;0 0 0]; % [[dlta dltb dltc]_pC;0 0 0]
            P = [indLatt;1 1 1]; % [Pa, Pb, Pc]_oB
            I_new = (I_old*P + dlt)*inv(P);
            newInstance = grid2d().oneLine(I_new,obj.sz_stdpts);
        end
    end
    methods % constructor
        function obj = oneLine(obj,I_pC__oB,sz) % define class in one line
            arguments
                obj
                I_pC__oB (3,3) double
                sz (1,2) double
            end
            obj.a1_stdpts = I_pC__oB(1:2,1)';
            obj.a2_stdpts = I_pC__oB(1:2,2)';
            obj.offset_stdpts = I_pC__oB(1:2,3)';
            obj.sz_stdpts = sz;            
        end
    end
    
end
