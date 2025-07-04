%% Optimization
function [xa]=optimize(elementParameters)
xSpan = elementParameters.xSpan;
ySpan = elementParameters.ySpan;
zSpan = elementParameters.zSpan;
xLength = elementParameters.xLength;
yLength = elementParameters.yLength;
storyHeight = elementParameters.storyHeight;
weight = elementParameters.weight;
numExecution = elementParameters.numExecution;
perExecution = elementParameters.perExecution;
iteration = elementParameters.iteration;
spanType = elementParameters.spanType;
frameType = elementParameters.frameType;
height = elementParameters.height;
rng('shuffle');
iter = 1;
optFval = 1e+9;
global nelx nely nelz lx ly lz pw span frame
span = spanType;
frame = frameType;
nelx = xSpan;
nely = ySpan;
nelz = zSpan;
lx = xLength;
ly = yLength;
lz = storyHeight;
pw = weight;
global  e pr nma F nlc Z Co

e = 205000.;
pr = 0.3;
nma = 2;
F = [235,325];
if frame(1) == 's'; nlc = 3; else; nlc = 2; end
Z = 1.;
Co = 0.2;
global nj nc ng nm ndf njf njsf njef nsj ns6 fht nly aj
[nj,nc,ng,nsj] = modelParameter();
nm = nc+ng;
[ndf,njf,njsf,njef] = rf_node_info();
ns6 = 6*nsj;

fht=height;
nly = nelz;
aj = ones(nm,1);
global x y z xe ye ze xr yr xi yi zi
[x,y,z,xe,ye,ze,xi,yi,zi] = nodeCodinate();
xr = x-x(end)/2;
yr = y-y(end)/2;
global js je jel c_g lm cyl nb
[js,je,jel,c_g] = memberInfo();

lm = memberLength();
cyl = directionCosine();
nb = bandwidth();
global Hn Bn twn tfn Dn tn
[Hn,Bn,twn,tfn,Dn,tn] = sectiongroup();

global ns isup pd
[ns,isup,pd] = supportInfo();
xr(ns) = 0;
yr(ns) = 0;
global f memberEnd memberM0 lyr
[f,memberEnd,memberM0,lyr] = loadsf();
global Hp Bp twp tfp Dp tp
[Hp,Bp,twp,tfp,Dp,tp] = arrangevalue();
global He Be twe tfe De te
[He,Be,twe,tfe,De,te] = sectionStandard();
global nvg nvc nvars repg repc
nvg = length(tfp);
nvc = length(Dp);

nvars = nvg+nvc;
[repg,repc] = varMember();
% disp(repg)
A = [];
bb = [];
[lb,ub] = setbound();

x0 = lb;

options = optimoptions('fmincon','Algorithm','interior-point',...
                       'UseParallel',true,...
                       'Display','off',...
                       'TolCon',1e-5,...
                       'MaxIter',Inf,...
                       'maxFunEvals',Inf,...
                       'TolFun',1e-5);

while iter <= perExecution
    [xo,fval] = fmincon(@continuousObjective,x0,A,bb,[],[],lb,ub,@continuousConstraints,options);
    if fval < optFval
        optSolution = xo;
        optFval = fval;
    end      
    Ho = xo(Hp);
    Bo = xo(Bp);
    tfo = xo(tfp);
    two = xo(tfp);
    for kk =1:length(xo(tfp))
        if tfo(kk)/2<9
            two(kk)=9;
        else
            two(kk)=tfo(kk)/2;
        end
    end
    Do = xo(Dp);
    to = xo(tp);
    xa=[Ho Bo two tfo Do to];
    iter = iter+1;
    [bri,brj,brc,cr,bsi,bsj,cs,deflect,rps] = analysis(xo)  ;
    if frame(1)=='x'
        a=nelx;
    else
        a=nely;
    end

    % disp(reshape(bri+1,a,[]).')
    % disp(reshape(brj+1,a,[]).')
    % disp([Ho.' Bo.' two.' tfo.'])
    
 
end



% xout=[xH xB xo(tfp).' xo(twp).'; xo(Dp).' xo(Dp).' xo(tp).' xo(tp).'];
% outputVariable(optSolution,exitflag,numExecution,iteration);
% fclose(fparameter);
% fclose(fsolution);
% fclose('all');
%% Definition of optimal function
    function fun = continuousObjective(xo)

        Ho = xo(Hp);
        Bo = xo(Bp);
        
        tfo = xo(tfp);
        two = xo(tfp);
        for k =1:length(xo(tfp))
            if tfo(k)/2<9
                two(k)=9;
            else
                two(k)=tfo(k)/2;
            end
        end
        Do = xo(Dp);
        to = xo(tp);
        a = zeros(1,nm);
        for i = 1:nm
            if c_g(i) == 1
                D = Do(Dn(i-ng));
                t = to(tn(i-ng));
                a(i) = D^2-(D-2*t)^2;
            else
                H = Ho(Hn(i));
                B = Bo(Bn(i));
                tw = two(twn(i));
                tf = tfo(tfn(i));
                a(i) = H*B-(B-tw)*(H-2*tf);
            end
        end
        fun = a*lm.'*1e-9;
    end
%
    function [c,ceq,constraint] = continuousConstraints(xo)
        Ho = xo(Hp);
        Bo = xo(Bp);
        tfo = xo(tfp);
        two = xo(tfp);
        for k =1:length(xo(tfp))
            if tfo(k)/2<9
                two(k)=9;
            else
                two(k)=tfo(k)/2;
            end
        end
        Do = xo(Dp);
        to = xo(tp);
        [a, asx, asy, aiy, aiz, zy, zz, zyf, zpy,~] = datasf(Ho,Bo,two,tfo,Do,to);
        s = stifsf(a,aiy,aiz);
        [s,f,r,sk] = suptsf(s,f);
        d = eqsoln_matlab(s,f);
        [RS,Mc] = rsltsf(d,r,sk,a,aiy,aiz);
        [st,stc] = stress(RS,Mc,a,asx,asy,zy,zz,zyf);
     
        [bri,brj,brc,cr,bsi,bsj,cs] = stress_ratio(Ho,Bo,two,tfo,st,stc,a,aiy,aiz);
        deflect = inter_story(d);
        rps = proof_stress(zpy);
%         column = thickness(Do);%上階との大きさの差:柱
%         [gb,gw] = gwidth(Ho,Bo);%上階との大きさの差:梁
%         [dbl,form] = deformation(Ho,RS,aiy);%中央たわみ変形
%         [wid_thick,wid_c] = wt_ratio(Ho,Bo,two,tfo,Do,to);% 鋼材種は手動変更（要検討）？？？？
%         [wid_gl] = wt_ratio_L(Bo,tfo); % 梁フランジの幅厚比の下限制約        
%         c = [bri(1,:),brj(1,:),brc(1,:),cr(1,:),...
%             bsi(1,:),bsj(1,:),cs(1,:),...
%             deflect(1,:),...
%             rps(1,:),...   
%             wid_thick(1,:),wid_c(1,:),wid_gl(1,:),...  
%             dbl(1,:),form(1,:)];%
        c = [bri(1,:),brj(1,:),brc(1,:),cr(1,:),...
            deflect(1,:),...
            rps(1,:)];%            



        ceq = [];
        constraint = struct;

%%%%%%%%VERSIN1
        constraint.bri = bri;
        constraint.brj = brj;
        constraint.brc = brc;
        constraint.cr = cr;
        constraint.deflect = deflect;     
        constraint.cs = cs;        
%         constraint.column = column;
        constraint.rps = rps;  

    end


    function [bri,brj,brc,cr,bsi,bsj,cs,deflect,rps] = analysis(xo)
        Ho = xo(Hp);
        Bo = xo(Bp);
        tfo = xo(tfp);
        two = xo(tfp);
        for k =1:length(xo(tfp))
            if tfo(k)/2<9
                two(k)=9;
            else
                two(k)=tfo(k)/2;
            end
        end
        Do = xo(Dp);
        to = xo(tp);
        [a, asx, asy, aiy, aiz, zy, zz, zyf, zpy,~] = datasf(Ho,Bo,two,tfo,Do,to);
        s = stifsf(a,aiy,aiz);
        [s,f,r,sk] = suptsf(s,f);
        d = eqsoln_matlab(s,f);
        [RS,Mc] = rsltsf(d,r,sk,a,aiy,aiz);
        [st,stc] = stress(RS,Mc,a,asx,asy,zy,zz,zyf);
     
        [bri,brj,brc,cr,bsi,bsj,cs] = stress_ratio(Ho,Bo,two,tfo,st,stc,a,aiy,aiz);
        deflect = inter_story(d);
        rps = proof_stress(zpy);

    end

%% Matrix method
    function [a,asx,asy,aiy,aiz,zy,zz,zyf,zpy,zpz] = datasf(Ho,Bo,two,tfo,Do,to)
        a = zeros(1,nm); % cross-sectional area
        asx = zeros(1,ng); % area for shear (the strong axis)
        asy = zeros(1,ng); % area for shear (the weak axis)
        aiy= zeros(1,nm); % second moment of area (the strong axis)
        aiz = zeros(1,nm); % second moment of area (the weak axis)
        zy = zeros(1,nm); % section modulus (the strong axis)
        zz = zeros(1,nm); % section modulus (the weak axis)
        zyf = zeros(1,ng); % section modulus for calculating allowable bending stress
        zpy = zeros(1,nm); % plastic section modulus (the strong axis)
        zpz = zeros(1,nm); % plastic section modulus (the weak axis)
        if frame(1) == 's'
            ngx = nelx*(nely+1)*nelz;
            fngx = nelx*(nely+1);
            fngy = (nelx+1)*nely;
            for i = 1:nm
            if c_g(i) == 1
                D = Do(Dn(i-ng));
                t = to(tn(i-ng));
                a(i) = D^2-(D-2*t)^2;

                aiy(i) = (D^4-(D-2*t)^4)/12;
                aiz(i) = aiy(i);
                zy(i) = aiy(i)/(D/2);
                zz(i) = zy(i);
                zpy(i) = D*t*(D-t)+(D-2*t)^2*t/2;
                zpz(i) = zpy(i);
            else
                H = Ho(Hn(i));
                B = Bo(Bn(i));
                tw = two(twn(i));
                tf = tfo(tfn(i));
                a(i) = H*B-(B-tw)*(H-2*tf);
                asx(i) = (H-2*tf)*tw;
                asy(i) = B*tf*2;
                aiz(i) = (tf*2*B^3+(H-2*tf)*tw^3)/12;
                zy(i) = aiy(i)/(H/2);
                zz(i) = aiz(i)/(B/2);
                zyf(i) = B*(H^3-(H-2*tf)^3)/(6*H);
%                 zyf(i) = aiy(i)/(H/2);
                zpy(i) = B*tf*(H-tf)+(H-2*tf)^2*tw/4;
                zpz(i) = tf*B^2/2+(H-2*tf)*tw^2/4;
                aiy(i) = (B*H^3-(B-tw)*(H-2*tf)^3)/12;
            end
            end
        elseif frame(2) == 'o'
            for i = 1:nm
                if c_g(i) == 1
                    D = Do(Dn(i-ng));
                    t = to(tn(i-ng));
                    a(i) = D^2-(D-2*t)^2;
                    
                    aiy(i) = (D^4-(D-2*t)^4)/12;
                    aiz(i) = aiy(i);
                    zy(i) = aiy(i)/(D/2);
                    zz(i) = zy(i);
                    zpy(i) = D*t*(D-t)+(D-2*t)^2*t/2;
                    zpz(i) = zpy(i);
                else
                    H = Ho(Hn(i));
                    B = Bo(Bn(i));
                    tw = two(twn(i));
                    tf = tfo(tfn(i));
                    a(i) = H*B-(B-tw)*(H-2*tf);
                    asx(i) = (H-2*tf)*tw;
                    asy(i) = B*tf*2;
                    aiy(i) = (B*H^3-(B-tw)*(H-2*tf)^3)/12;
                    aiz(i) = (tf*2*B^3+(H-2*tf)*tw^3)/12;
                    zy(i) = aiy(i)/(H/2);
                    zz(i) = aiz(i)/(B/2);
                    zyf(i) = B*(H^3-(H-2*tf)^3)/(6*H);
%                     zyf(i) = aiy(i)/(H/2);
                    zpy(i) = B*tf*(H-tf)+(H-2*tf)^2*tw/4;
                    zpz(i) = tf*B^2/2+(H-2*tf)*tw^2/4;
                end
            end
        else
            for i = 1:nm
                if c_g(i) == 1
                    D = Do(Dn(i-ng));
                    t = to(tn(i-ng));
                    a(i) = D^2-(D-2*t)^2;
                    aiy(i) = (D^4-(D-2*t)^4)/12;
                    aiz(i) = aiy(i);
                    zy(i) = aiy(i)/(D/2);
                    zz(i) = zy(i);
                    zpy(i) = D*t*(D-t)+(D-2*t)^2*t/2;
                    zpz(i) = zpy(i);
                else
                    H = Ho(Hn(i));
                    B = Bo(Bn(i));
                    tw = two(twn(i));
                    tf = tfo(tfn(i));
                    a(i) = H*B-(B-tw)*(H-2*tf);
                    asx(i) = (H-2*tf)*tw;
                    asy(i) = B*tf*2;
                    aiy(i) = (B*H^3-(B-tw)*(H-2*tf)^3)/12;
                    aiz(i) = (tf*2*B^3+(H-2*tf)*tw^3)/12;
                    zy(i) = aiy(i)/(H/2);
                    zz(i) = aiz(i)/(B/2);
                    zyf(i) = B*(H^3-(H-2*tf)^3)/(6*H);
%                     zyf(i) = aiy(i)/(H/2);
                    zpy(i) = B*tf*(H-tf)+(H-2*tf)^2*tw/4;
                    zpz(i) = tf*B^2/2+(H-2*tf)*tw^2/4;
                end
            end
        end
    end
%
    function s = stifsf(a,aiy,aiz)
        % Stiffness matrix for space frame.See Sec. 23-9,Ghali and Neville
        s = zeros(ndf,nb);
        t = zeros(3,3);
        for jm = 1:nm
          %{ 
           Generate a 3x3 transformation matrix [t],
           Eq. 23-18 of Ghali and Neville
          %}
          alx = x(je(jm))-x(js(jm));
          aly = y(je(jm))-y(js(jm));
          alz = z(je(jm))-z(js(jm));   
          alj = sqrt(alx*alx+aly*aly+alz*alz);
          t(1,1) = alx/alj;
          t(1,2) = aly/alj;
          t(1,3) = alz/alj;
          t(2,1) = cyl(jm,1);
          t(2,2) = cyl(jm,2);
          t(2,3) = cyl(jm,3);
          t(3,1) = t(1,2)*t(2,3)-t(1,3)*t(2,2);
          t(3,2) = t(1,3)*t(2,1)-t(1,1)*t(2,3);
          t(3,3) = t(1,1)*t(2,2)-t(1,2)*t(2,1);
%           tt = t'; % Generate the transpose of [t].

          %{
           Member stiffness matrix in local coordinates,
           Eq.6-6 of Ghali and Neville
          %}
          aaj = a(jm); aiyj = aiy(jm); aizj = aiz(jm); ajj = aj(jm);
          es = estfsf(alj,aaj,aiyj,aizj,ajj,e,pr);
          %%
          %{
          Transform element stiffness, [ES] from local to global directions;
          see Eq. 23-19 of Ghali and Neville. Store the result back in [ES].
          Partition [ES] into 3x3 submatrices and postmultipy each by [t]
          and store the product in [P]. Postmultiply [P] by the transpose
          of [t]. Store the resulting submatrices back in [ES].
          %}
          tm = blkdiag(t,t,t,t);
          es = tm'*es*tm;
          es0 = es;
%           h = zeros(3,3);
%           for n=1:4
%             for l=1:4
%               for i=1:3
%                 for j=1:3
%                   ies = 3*(n-1)+i;
%                   jes = 3*(l-1)+j;
%                   h(i,j) = es(ies,jes);
%                 end
%               end
%               h = tt*h*t;
%               for i=1:3
%                 for j=1:3
%                   ies = 3*(n-1)+i;
%                   jes = 3*(l-1)+j;
%                   es(ies,jes) = h(i,j);
%                 end
%               end
%             end
%           end
          % Input a constraint of rigid floor
%           es(1:6,6) = es(1:6,6)-es(1:6,1)*yr(js(jm))+es(1:6,2)*xr(js(jm));
%           es(:,12) = es(:,12)-es(:,7)*yr(je(jm))+es(:,8)*xr(je(jm));
%           es(6,6) = es(6,6)-es(1,6)*yr(js(jm))+es(2,6)*xr(js(jm));
%           es(6,12) = es(6,12)-es(1,12)*yr(js(jm))+es(2,12)*xr(js(jm));
%           es(12,12) = es(12,12)-es(7,12)*yr(je(jm))+es(8,12)*xr(je(jm));
%           es(6,7) = es(6,7)-es(1,7)*yr(js(jm))+es(2,7)*xr(js(jm));
%           es(6,8) = es(6,8)-es(1,8)*yr(js(jm))+es(2,8)*xr(js(jm));
%           es(6,9) = es(6,9)-es(1,9)*yr(js(jm))+es(2,9)*xr(js(jm));
%           es(6,10) = es(6,10)-es(1,10)*yr(js(jm))+es(2,10)*xr(js(jm));
%           es(6,11) = es(6,11)-es(1,11)*yr(js(jm))+es(2,11)*xr(js(jm));
%           es(6,1:5) = es(1:5,6);
%           es(12,1:11) = es(1:11,12);
%           es(7:8,6) = es(6,7:8);
          tg = eye(12);
          tg(1,6) = -yr(js(jm)); %tg(6,1) = -yr(1,jm);
          tg(2,6) = xr(js(jm)); %tg(6,2) = xr(1,jm);
          tg(7,12) = -yr(je(jm)); %tg(12,7) = -yr(2,jm);
          tg(8,12) = xr(je(jm)); %tg(12,8) = xr(2,jm);
          es2 = tg'*es0*tg;
          %---
          es = es2;
%           es(1,6) = es(1,6)-es(1,1)*y(js(jm))+es(1,2)*x(js(jm));
%           es(2,6) = es(2,6)-es(2,1)*y(js(jm))+es(2,2)*x(js(jm));
%           es(3,6) = es(3,6)-es(3,1)*y(js(jm))+es(3,2)*x(js(jm));
%           es(4,6) = es(4,6)-es(4,1)*y(js(jm))+es(4,2)*x(js(jm));
%           es(5,6) = es(5,6)-es(5,1)*y(js(jm))+es(5,2)*x(js(jm));
%           es(6,6) = es(6,6)-es(6,1)*y(js(jm))+es(6,2)*x(js(jm))-es(1,6)*y(js(jm))+es(2,6)*x(js(jm));
%           es(1,12) = es(1,12)-es(1,7)*y(je(jm))+es(1,8)*x(je(jm));
%           es(2,12) = es(2,12)-es(2,7)*y(je(jm))+es(2,8)*x(je(jm));
%           es(3,12) = es(3,12)-es(3,7)*y(je(jm))+es(3,8)*x(je(jm));
%           es(4,12) = es(4,12)-es(4,7)*y(je(jm))+es(4,8)*x(je(jm));
%           es(5,12) = es(5,12)-es(5,7)*y(je(jm))+es(5,8)*x(je(jm));
%           es(6,12) = es(6,12)-es(6,7)*y(je(jm))+es(6,8)*x(je(jm))-es(1,12)*y(js(jm))+es(2,12)*x(js(jm));
%           es(7,12) = es(7,12)-es(7,7)*y(je(jm))+es(7,8)*x(je(jm));
%           es(8,12) = es(8,12)-es(8,7)*y(je(jm))+es(8,8)*x(je(jm));
%           es(9,12) = es(9,12)-es(9,7)*y(je(jm))+es(9,8)*x(je(jm));
%           es(10,12) = es(10,12)-es(10,7)*y(je(jm))+es(10,8)*x(je(jm));
%           es(11,12) = es(11,12)-es(11,7)*y(je(jm))+es(11,8)*x(je(jm));
%           es(12,12) = es(12,12)-es(12,7)*y(je(jm))+es(12,8)*x(je(jm))-es(7,12)*y(je(jm))+es(8,12)*x(je(jm));
          % Identify the 12 coordinates at the start and end of member.
          im = [njf(6*(js(jm)-1)+1:6*js(jm)) njf(6*(je(jm)-1)+1:6*je(jm))];
          for i=1:12
            for j=1:12
              k = im(j)-im(i);
              if k>=0
                k = k+1;
                s(im(i),k) = s(im(i),k)+es(i,j);
              end
            end
          end
        end
    return
    end
%
    function es = estfsf(alj,aaj,aiyj,aizj,ajj,e,pr)
        %{
        Element stiffness matrix in local coordinates;member of space
        frame,(Eq. 6-6, Ghali and Neville).
        ALJ=member length;AJ=a;AIJ=I;ARDJ=memberEnd;PR=Poisson's ratio.
        %}
        es = zeros(12,12);
        g = e/(2.*(1.+pr));
        es(1,1) = e*aaj/alj;
        es(7,1) = -es(1,1);
        es(7,7) = es(1,1);
        es(2,2) = 12.0*e*aizj/(alj*alj*alj);
        es(8,2) = -es(2,2);
        es(8,8) = es(2,2);
        es(3,3) = 12.0*e*aiyj/(alj*alj*alj);
        es(9,3) = -es(3,3);
        es(9,9) = es(3,3);
        es(4,4) = g*ajj/alj;
        es(10,4) = -es(4,4);
        es(10,10) = es(4,4);
        es(6,2) = 6.0*e*aizj/(alj*alj);
        es(12,2) = es(6,2);
        es(12,8) = -es(6,2);
        es(5,3) = -6.0*e*aiyj/(alj*alj);
        es(11,3) = es(5,3);
        es(11,9) = -es(11,3);
        es(9,5) = 6.0*e*aiyj/(alj*alj);
        es(11,5) = 2.0*e*aiyj/alj;
        es(8,6) = -6.0*e*aizj/(alj*alj);
        es(12,6) = 2.0*e*aizj/alj;
        es(5,5) = 4.0*e*aiyj/alj;
        es(11,11) = es(5,5);
        es(6,6) = 4.0*e*aizj/alj;
        es(12,12) = es(6,6);
        es = es+tril(es,-1)';
%         for i=1:12
%           for j=(i+1):12
%             es(i,j) = es(j,i);
%           end
%         end
        return
    end
%
    function [s,f,r,sk] = suptsf(s,f)
        %{
         Adjust [S] and [F] in accordance with the support conditions.See
         Eq. 23-39 of Ghali and Neville. Note that before the adjustment,
         [S] generated by the subroutine STIFPF is the stiffness matrix of
         a free unsupported structure.
         Use {SK} and [R] to store appropriate elements on the diagonal of
         [S] and of [F] for use in calculation of the reactions by Eq. 23-38
         of Ghali and Neville.
        %}
        sk = zeros(1,ns6);
        r = zeros(ns6,nlc);
        for i=1:nsj
          m1 = 6*(i-1);
          n1 = 6*(ns(i)-1);
          for k = 1:6
            if(isup(i,k)==0)
              m = m1+k;
              n = n1+k;
              sk(m) = s(n,1);
              s(n,1) = s(n,1)*1.0e4;
              for j = 1:nlc
                r(m,j) = f(n,j);
                f(n,j) = -sk(m)*1.0e4*pd(i,k);
              end
            end
          end
        end
        return
    end
%
    function [RS,Mc] = rsltsf(d,r,sk,a,aiy,aiz)
        % Printing of nodal displacement reactions and member end forces.
        RS = zeros(12*nlc,nm);
        Mc = zeros(nlc,nm); % 中央曲げモーメント
        ngx = nelx*(nely+1)*nelz;
        perngx = nelx*(nely+1);
        perngy = (nelx+1)*nely;
        for k=1:nlc
          for l=1:nsj
            m1 = 6*(l-1);
            n1 = 6*(ns(l)-1);
            for j=1:6
              if(isup(l,j) == 0)
                m = m1+j;
                n = n1+j;
                r(m,k) = r(m,k)+sk(m)*1.0e4*(pd(l,j)-d(n,k))+sk(m)*d(n,k);
              end
            end
          end
          for jm=1:nm
            alx = x(je(jm))-x(js(jm));
            aly = y(je(jm))-y(js(jm));
            alz = z(je(jm))-z(js(jm));
            alj = sqrt(alx*alx+aly*aly+alz*alz);
            t(1,1) = alx/alj;
            t(1,2) = aly/alj;
            t(1,3) = alz/alj;
            t(2,1) = cyl(jm,1);
            t(2,2) = cyl(jm,2);
            t(2,3) = cyl(jm,3);
            t(3,1) = t(1,2)*t(2,3)-t(1,3)*t(2,2);
            t(3,2) = t(1,3)*t(2,1)-t(1,1)*t(2,3);
            t(3,3) = t(1,1)*t(2,2)-t(1,2)*t(2,1);
            aaj = a(jm); aiyj = aiy(jm); aizj = aiz(jm); ajj = aj(jm);
            es = estfsf(alj,aaj,aiyj,aizj,ajj,e,pr);
            
            % transform displacements at member ends into local coordinates.
            nd = [njf(6*(js(jm)-1)+1:6*js(jm)) njf(6*(je(jm)-1)+1:6*je(jm))];
            dt = zeros(12,1);
            h = d(nd,k);
            h(1) = h(1)-yr(js(jm))*h(6);
            h(2) = h(2)+xr(js(jm))*h(6);
            h(7) = h(7)-yr(je(jm))*h(12);
            h(8) = h(8)+xr(je(jm))*h(12);
            for m = 1:4
              ii = 3*(m-1)+(1:3);
              hh = h(ii);
              dt(ii) = t*hh;
            end

            %{
             For each member,calculate the product of its stiffness matrix by
             its displacement vector.Store result in {ARM}.
            %}

            arm = es*dt;
            
            % Add the fixed-end forces included in the input data to {ARM}.
            if jm <= ng
                if frame(1) == 's'
                    if ngx >= jm
                        if mod(jm,perngx) <= nelx || mod(jm,perngx) >= perngx-nelx
                            lcx = 1;
                            arm = arm+memberEnd(:,lcx);
                            Mc(k,jm) = memberM0(lcx);
                        else 
                            lcx = 2;
                            arm = arm+memberEnd(:,lcx);
                            Mc(k,jm) = memberM0(lcx);
                        end
                    else
                        if mod(jm-ngx,perngy) <= nely || mod(jm-ngx,perngy) > perngy-nely
                            lcy = 3;
                            arm = arm+memberEnd(:,lcy);
                            Mc(k,jm) = memberM0(lcy);
                        else
                            lcy = 4;
                            arm = arm+memberEnd(:,lcy);
                            Mc(k,jm) = memberM0(lcy);
                        end
                    end
                elseif frame(1) == 'x'
                    if frame(2) == 'o'
                        mef = 1;
                        orthogout = 3;
                        orthogin = 4;
                    else
                        mef = 2;
                        orthogout = 4;
                        orthogin = 5;
                    end
                   
%                   arm = arm+memberEnd(:,mef)+[memberEnd(1:3,orthogout);0;0;0;0;0;0;0;0;0];
                    arm = arm+memberEnd(:,mef);
                    Mc(k,jm) = memberM0(mef);
                else
                    if frame(2) == 'o'
                        mef = 3;
                        orthogout = 1;
                        orthogin = 2;
                    else
                        mef = 4;
                        orthogout = 2;
                        orthogin = 6;
                    end
                    % arm = arm+memberEnd(:,mef)+[memberEnd(1:3,orthogout);0;0;0;0;0;0;0;0;0];
                    arm = arm+memberEnd(:,mef);
                    Mc(k,jm) = memberM0(mef);
                end
            end
            RS(12*(k-1)+1:12*k,jm) = arm;
            Mc(k,jm) = Mc(k,jm)+(arm(5,1)-arm(11,1))/2;
          end
        end
    Mc = Mc(:,1:ng);
    return
    end
%
    function [d, ss] = eqsoln_matlab(s,f)
        % 		  Solution of Banded Symmetrical Equations
        % ----------------------------------------------------------------------
        % 		     Notation and dimensions of arrays
        % 							     rows  cols.
        %     NB     band width ; see Sec. 24-3 of Ghali and Nevile.
        %     NLC    number of load cases
        %     [D]    nodal displacements 			     NDF    NLC
        %     [F]    nodal restraining forces			 NDF    NLC
        %     [S]    stiffness matrix				     NDF    NB
        % ----------------------------------------------------------------------
        %      Solution of equilibrium equations [S][D]=-[F], using Cholesky's
        %      method; see Sec. 22-11 of Ghali and Neville.
        for i=2:nb
          s(i:ndf,i) = s(1:ndf-i+1,i);
        end
        ss = spdiags(s,0:nb-1,ndf,ndf);
        ds = decomposition(ss,'chol','upper');
        d = -ds\f;
        return
    end
%% Constraints
    function [st,stc] = stress(RS,Mc,a,asx,asy,zy,zz,zyf)
        st = zeros(12*nlc,nm);
        stc = zeros(nlc,ng);
        for i = 1:nm
            for j = 1:nlc
                st(12*(j-1)+1,i) = RS(12*(j-1)+1,i)/a(1,i);
                st(12*(j-1)+6,i) = RS(12*(j-1)+6,i)/zz(1,i);
                st(12*(j-1)+7,i) = RS(12*(j-1)+7,i)/a(1,i);
                st(12*j,i) = RS(12*j,i)/zz(1,i);
                if i <=ng
                    st(12*(j-1)+2,i) = RS(12*(j-1)+2,i)/asy(1,i);
                    st(12*(j-1)+3,i) = RS(12*(j-1)+3,i)/asx(1,i);
                    st(12*(j-1)+5,i) = RS(12*(j-1)+5,i)/zyf(1,i);
                    st(12*(j-1)+8,i) = RS(12*(j-1)+8,i)/asy(1,i);
                    st(12*(j-1)+9,i) = RS(12*(j-1)+9,i)/asx(1,i);
                    st(12*(j-1)+11,i) = RS(12*(j-1)+11,i)/zyf(1,i);
                    stc(j,i) = Mc(j,i)/zyf(1,i);
                else
                    st(12*(j-1)+2,i) = RS(12*(j-1)+2,i)/a(1,i);
                    st(12*(j-1)+3,i) = RS(12*(j-1)+3,i)/a(1,i);
                    st(12*(j-1)+5,i) = RS(12*(j-1)+5,i)/zy(1,i);
                    st(12*(j-1)+8,i) = RS(12*(j-1)+8,i)/a(1,i);
                    st(12*(j-1)+9,i) = RS(12*(j-1)+9,i)/a(1,i);
                    st(12*(j-1)+11,i) = RS(12*(j-1)+11,i)/zy(1,i);
                end
            end
        end
    return
    end
%
    function [bri,brj,brc,cr,bsi,bsj,cs] = stress_ratio(Ho,Bo,two,tfo,st,stc,a,aiy,aiz)
        if frame(1) == 's'
            ngx = nelx*(nely+1)*nelz;
        elseif frame(1) == 'x'
            ngx = nelx*nelz;
        else
            ngx = 0;
        end
        ratio = zeros(12*nlc,nm);
        ratioc = zeros(nlc,ng);
        ft = zeros(2,nma);
        fc = zeros(2,nm);
        fb = zeros(2,nm);
        fs = zeros(2,nma);
        RAM = zeros(1,nma); % critical slenderness ratio
        for g = 1:nma
            RAM(g) = pi*(e/(0.6*F(g)))^(1/2);
        end
        %--------------------------------------------------------------------------
        % Calculation of allowable tensile (ft) and shear (fs) stress
        %--------------------------------------------------------------------------
        for i = 1:nm
            if jel(i) ~= 1
                ft(1,i) = F(2)/1.5; ft(2,i) = F(2);
                fs(1,i) = F(2)/(1.5*sqrt(3)); fs(2,i) = fs(1,i) * 1.5;
            else
                ft(1,i) = F(1)/1.5; ft(2,i) = F(1);
                fs(1,i) = F(1)/(1.5*sqrt(3)); fs(2,i) = fs(1,i) * 1.5;
            end
        end
        %--------------------------------------------------------------------------
        % Calculation of allowable compressive (fc) stress
        %--------------------------------------------------------------------------
        iy = zeros(1,nm); iz = zeros(1,nm);
        ramday = zeros(1,nm); ramdaz = zeros(1,nm);
        xn = ones(1,nc); yn = ones(1,nc);
        fx = ones(1,nc); fy = ones(1,nc);
        an = zeros(1,nc); bn = zeros(1,nc);
        anx = zeros(1,nc); bnx = zeros(1,nc);
        any = zeros(1,nc); bny = zeros(1,nc);
        if frame(1) == 'y' || frame(1) == 'x'
            for k = 1:nc
                n = k+ng;
                men_a = find(js==js(n) | je==js(n));
                men_b = find(js==je(n) | je==je(n));
                men_a(men_a==n) = []; men_b(men_b==n) = [];
                gn = aiy(n)/lm(n);
                if isempty(men_a)
                    Ga = 1.0;
                else
                    gca = aiy(max(men_a))/lm(max(men_a));
                    if numel(men_a) == 3
                        gba1 = aiy(men_a(1))/lm(men_a(1));
                        gba2 = aiy(men_a(2))/lm(men_a(2));
                    else
                        gba1 = aiy(men_a(1))/lm(men_a(1));
                        gba2 = 0;
                    end
                    Ga = (gn+gca)/(gba1+gba2);
                end
                if max(men_b) <= ng
                    if numel(men_b) == 1
                        gbb1 = aiy(men_b(1))/lm(men_b(1));
                        gbb2 = 0;
                        gcb = 0;
                    else
                        gbb1 = aiy(men_b(1))/lm(men_b(1));
                        gbb2 = aiy(men_b(2))/lm(men_b(2));
                        gcb = 0;
                    end
                elseif numel(men_b) == 3
                    gbb1 = aiy(men_b(1))/lm(men_b(1));
                    gbb2 = aiy(men_b(2))/lm(men_b(2));
                    gcb = aiy(men_b(3))/lm(men_b(3));
                else
                    gbb1 = aiy(men_b(1))/lm(men_b(1));
                    gbb2 = 0;
                    gcb = aiy(men_b(2))/lm(men_b(2));
                end
                Gb = (gn+gcb)/(gbb1+gbb2);
                an(k) = -1*Ga*Gb/(6*(Ga+Gb));
                bn(k) = -1*6/(Ga+Gb);
            end
            while ~all(abs(fx) < 1e-3)
                fx = xn./tan(xn)+an.*(xn.^2)-bn;
                dfx = 1./tan(xn)-xn./((sin(xn)).^2)+2*an.*xn;
                xn = xn - fx./dfx;
            end
            kc = pi./xn;
            lk = lm; lk(ng+1:ng+nc) = kc.*lm(ng+1:ng+nc);
            for i = 1:nm
                iy(i) = (aiy(i)/a(i))^(1/2);
                iz(i) = (aiz(i)/a(i))^(1/2);
                ramday(i) = lk(i)/iy(i);
                ramdaz(i) = lk(i)/iz(i);
                ramda = max(ramday(i),ramdaz(i));
                if ramda<=RAM(jel(i))
                    nu = 3/2 + 2/3*(ramda/RAM(jel(i)))^2;
                    fc(1,i) = F(jel(i))/nu*(1.0-0.4*(ramdaz(i)/RAM(jel(i)))^2);
                else
                    fc(1,i) = 0.277*F(jel(i))/(ramda/RAM(jel(i)))^2;
                end
            end
        else
            for k = 1:nc
                n = k+ng;
                men_ax = find(js==js(n) | je==js(n));
                men_ay = find(js==js(n) | je==js(n));
                men_bx = find(js==je(n) | je==je(n));
                men_by = find(js==je(n) | je==je(n));
                men_ax(men_ax==n | (ngx<men_ax & men_ax<=ng)) = [];
                men_ay(men_ay==n | men_ay<=ngx) = [];
                men_bx(men_bx==n | (ngx<men_bx & men_bx<=ng)) = [];
                men_by(men_by==n | men_by<=ngx) = [];
                gn = aiy(n)/lm(n);
                if isempty(men_ax)
                    Gax = 1.0; Gay = 1.0;
                else
                    gca = aiy(max(men_ax))/lm(max(men_ax));
                    if numel(men_ax) == 3
                        gbax1 = aiy(men_ax(1))/lm(men_ax(1));
                        gbax2 = aiy(men_ax(2))/lm(men_ax(2));
                    else
                        gbax1 = aiy(men_ax(1))/lm(men_ax(1));
                        gbax2 = 0;
                    end
                    if numel(men_ay) == 3
                        gbay1 = aiy(men_ay(1))/lm(men_ay(1));
                        gbay2 = aiy(men_ay(2))/lm(men_ay(2));
                    else
                        gbay1 = aiy(men_ay(1))/lm(men_ay(1));
                        gbay2 = 0;
                    end
                    Gax = (gn+gca)/(gbax1+gbax2);
                    Gay = (gn+gca)/(gbay1+gbay2);
                end
                if max(men_bx) <= ng
                    gcb = 0;
                    if numel(men_bx) == 1
                        gbbx1 = aiy(men_bx(1))/lm(men_bx(1));
                        gbbx2 = 0;
                    else
                        gbbx1 = aiy(men_bx(1))/lm(men_bx(1));
                        gbbx2 = aiy(men_bx(2))/lm(men_bx(2));
                    end
                    if numel(men_by) == 1
                        gbby1 = aiy(men_by(1))/lm(men_by(1));
                        gbby2 = 0;
                    else
                        gbby1 = aiy(men_by(1))/lm(men_by(1));
                        gbby2 = aiy(men_by(2))/lm(men_by(2));
                    end
                else
                    gcb = aiy(max(men_bx))/lm(max(men_bx));
                    if numel(men_bx) == 3
                        gbbx1 = aiy(men_bx(1))/lm(men_bx(1));
                        gbbx2 = aiy(men_bx(2))/lm(men_bx(2));
                    else
                        gbbx1 = aiy(men_bx(1))/lm(men_bx(1));
                        gbbx2 = 0;
                    end
                    if numel(men_by) == 3
                        gbby1 = aiy(men_by(1))/lm(men_by(1));
                        gbby2 = aiy(men_by(2))/lm(men_by(2));
                    else
                        gbby1 = aiy(men_by(1))/lm(men_by(1));
                        gbby2 = 0;
                    end
                end
                Gbx = (gn+gcb)/(gbbx1+gbbx2);
                Gby = (gn+gcb)/(gbby1+gbby2);
                anx(k) = -1*Gax*Gbx/(6*(Gax+Gbx));
                any(k) = -1*Gay*Gby/(6*(Gay+Gby));
                bnx(k) = -1*6/(Gax+Gbx);
                bny(k) = -1*6/(Gay+Gby);
            end
            while ~all(abs(fx) < 1e-3) && ~all(abs(fy) < 1e-3)
                fx = xn./tan(xn)+anx.*(xn.^2)-bnx;
                fy = yn./tan(yn)+any.*(yn.^2)-bny;
                dfx = 1./tan(xn)-xn./((sin(xn)).^2)+2*anx.*xn;
                dfy = 1./tan(yn)-yn./((sin(yn)).^2)+2*any.*yn;
                xn = xn - fx./dfx; yn = yn - fy./dfy;
            end
            kcx = pi./xn; kcy = pi./yn;
            lkx = lm; lky = lm;
            lkx(ng+1:ng+nc) = kcx.*lm(ng+1:ng+nc);
            lky(ng+1:ng+nc) = kcy.*lm(ng+1:ng+nc);
            for i = 1:nm
                iy(i) = (aiy(i)/a(i))^(1/2);
                iz(i) = (aiz(i)/a(i))^(1/2);
                ramday(i) = lkx(i)/iy(i);
                ramdaz(i) = lky(i)/iz(i);
                ramda = max(ramday(i),ramdaz(i));
                if ramda<=RAM(jel(i))
                    nu = 3/2 + 2/3*(ramda/RAM(jel(i)))^2;
                    fc(1,i) = F(jel(i))/nu*(1.0-0.4*(ramda/RAM(jel(i)))^2);
                else
                    fc(1,i) = 0.277*F(jel(i))/(ramda/RAM(jel(i)))^2;
                end
            end
        end
        %--------------------------------------------------------------------------
        % Calculation of allowable bending stress (fb)
        %--------------------------------------------------------------------------
        Lb = zeros(1,nm); % stifness interval
        nstif = zeros(1,nm); % the number of stifness
        for i = 1:nm
            if c_g(i) == 1
                nstif(i) = 0;
                Lb(i) = lm(i);
            else
                if F(jel(i)) == 235
                    nstif(i) = ceil(max((ramdaz(i)-170)/20,lm(i)/3000-1));
                    Lb(i) = lm(i)/(nstif(i)+1);
                elseif F(jel(i)) == 325
                    nstif(i) = ceil(max((ramdaz(i)-130)/20,lm(i)/3000-1));
                    Lb(i) = lm(i)/(nstif(i)+1);
                else
                    nstif(i) = 0;
                end
            end
        end
        for k = 1:nm
            if c_g(k) == 2
                H = (Ho(Hn(k)));
                B = (Bo(Bn(k)));
                tw = (two(twn(k)));
                tf = (tfo(tfn(k)));
                lbi = Lb(k);
%                 If = tf*B^3/12;
%                 At = tf*B+tw*(H/6-tf);
%                 siy = sqrt(If/At);
                siy = sqrt((tf*B^3/12)/(tf*B+(H/6-tf)*tw));
                c = 1.0; %検討
                fb1 = (1-0.4*(lbi/siy)^2/(c*RAM(jel(k))^2))*ft(1,k);
                fb2 = 89000/(lbi*H/(2*tf*B));
                fb(1,k) = max(fb1,fb2);
                if fb(1,k) > ft(1,k)
                    fb(1,k) = ft(1,k);
                end
            else
                fb(1,k) = ft(1,k);
            end
        end
        for ij = 1:nm
            fb(2,ij) = fb(1,ij) * 1.5;
            fc(2,ij) = fc(1,ij) * 1.5;
        end
        %--------------------------------------------------------------------------
        % Calculation of stress ratio
        %--------------------------------------------------------------------------
        for l = 1:nm
            if st(1,l) >= 0
                ratio(1,l) = st(1,l)/ft(1,l);
                ratio(7,l) = st(7,l)/ft(1,l);
            else
                ratio(1,l) = st(1,l)/fc(1,l);
                ratio(7,l) = st(7,l)/fc(1,l);
            end
            ratio(2,l) = st(2,l)/fs(1,l);
            ratio(3,l) = st(3,l)/fs(1,l);
            ratio(5,l) = st(5,l)/fb(1,l);
            ratio(6,l) = st(6,l)/ft(1,l);
            ratio(8,l) = st(8,l)/fs(1,l);
            ratio(9,l) = st(9,l)/fs(1,l);
            ratio(11,l) = st(11,l)/fb(1,l);
            ratio(12,l) = st(12,l)/ft(1,l);
            if l <= ng
                ratioc(1,l) = stc(1,l)/fb(1,l);
            end
            if nlc > 1
                for m = 2:nlc
                    if st(12*(m-1)+1,l) >= 0
                        ratio(12*(m-1)+1,l) = st(12*(m-1)+1,l)/ft(2,l);
                        ratio(12*(m-1)+7,l) = st(12*(m-1)+7,l)/ft(2,l);
                    else
                        ratio(12*(m-1)+1,l) = st(12*(m-1)+1,l)/fc(2,l);
                        ratio(12*(m-1)+7,l) = st(12*(m-1)+7,l)/fc(2,l);
                    end
                    ratio(12*(m-1)+2,l) = st(12*(m-1)+2,l)/fs(2,l);
                    ratio(12*(m-1)+3,l) = st(12*(m-1)+3,l)/fs(2,l);
                    ratio(12*(m-1)+5,l) = st(12*(m-1)+5,l)/fb(2,l);
                    ratio(12*(m-1)+6,l) = st(12*(m-1)+6,l)/ft(2,l);
                    ratio(12*(m-1)+8,l) = st(12*(m-1)+8,l)/fs(2,l);
                    ratio(12*(m-1)+9,l) = st(12*(m-1)+9,l)/fs(2,l);
                    ratio(12*(m-1)+11,l) = st(12*(m-1)+11,l)/fb(2,l);
                    ratio(12*m,l) = st(12*m,l)/ft(2,l);
                    if l <= ng
                        ratioc(m,l) = stc(m,l)/fb(2,l);
                    end
                end
            end
        end
        %--------------------------------------------------------------------------
        % Stress inspection
        %--------------------------------------------------------------------------
        bri = zeros(1,ng*nlc); brj = zeros(1,ng*nlc); brc = zeros(1,ng*nlc);
        cr = zeros(1,nc*nlc);
        bsi = zeros(1,ng*nlc); bsj = zeros(1,ng*nlc);
        cs = zeros(1,nc*nlc);
        % bri = []; brj = []; cr = [];
        for cd = 1:nlc
            for ab = 1:nm
                if c_g(ab) == 2
                    i = ng*(cd-1)+ab;
                    bi1 = abs(ratio(12*(cd-1)+5,ab));
                    bi2 = abs(ratio(12*(cd-1)+6,ab));
                    bj1 = abs(ratio(12*(cd-1)+11,ab));
                    bj2 = abs(ratio(12*cd,ab));
                    bri(i) = max([bi1,bi2])-1.0; % i端曲げ応力度の検定 
                    brj(i) = max([bj1,bj2])-1.0; % j端曲げ応力度の検定 
                    brc(i) = abs(ratioc(cd,ab))-1.0; % 中央曲げ応力度の検定 
                    bsi1 = abs(ratio(12*(cd-1)+2,ab));
                    bsi2 = abs(ratio(12*(cd-1)+3,ab));
                    bsj1 = abs(ratio(12*(cd-1)+8,ab));
                    bsj2 = abs(ratio(12*(cd-1)+9,ab));
                    bsi(i) = max([bsi1,bsi2])-1.0; % i端せん断応力度の検定 
                    bsj(i) = max([bsj1,bsj2])-1.0; % j端せん断応力度の検定 
                else
                    j = nc*(cd-1)+ab-ng;
                    cc = abs(ratio(12*(cd-1)+1,ab));
                    cbi1 = abs(ratio(12*(cd-1)+5,ab));
                    cbi2 = abs(ratio(12*(cd-1)+6,ab));
                    cbj1 = abs(ratio(12*(cd-1)+11,ab));
                    cbj2 = abs(ratio(12*cd,ab));
                    cri = cc+cbi1+cbi2;
                    crj = cc+cbj1+cbj2;
                    cr(j) = max([cri,crj])-1.0; 
                    csi1 = abs(ratio(12*(cd-1)+2,ab));
                    csi2 = abs(ratio(12*(cd-1)+3,ab));
                    csj1 = abs(ratio(12*(cd-1)+8,ab));
                    csj2 = abs(ratio(12*(cd-1)+9,ab));
                    cs(j) = max([csi1,csi2,csj1,csj2])-1.0; 
                end
            end
        end
        
            if isempty(bri)
                bri = -0.8; brj = -1.0; % -1.0
            elseif isempty(cr)
                cr = -0.8; % -1.0
            end
        return
        
    end
%
    function [dbl,form] = deformation(Ho,RS,aiy)
        dbl = zeros(1,nvg);
        delta = zeros(1,nvg);
        form = zeros(1,nvg);
        perngx = nelx*(nely+1);
        perngy = (nelx+1)*nely;
        if frame(1) == 's'
            ngx = nelx*(nely+1)*nelz;
        elseif frame(1) == 'x'
            ngx = ng;
        else
            ngx = 0;
        end
        for i = 1:nvg
            dbl(i) = 1/15*(lm(repg(i))/Ho(Hn(repg(i))))-1;
            
            if frame(1) == 's'
                if ngx >= repg(i)
                    if mod(repg(i),perngx) <= nelx || mod(repg(i),perngx) >= perngx-nelx
                        mef = 1;
                    else 
                        mef = 2;
                    end
                else
                    if mod(repg(i)-ngx,perngy) <= nely || mod(repg(i)-ngx,perngy) > perngy-nely
                        mef = 3;
                    else
                        mef = 4;
                    end
                end
            elseif frame(1) == 'x'
                if frame(2) == 'o'
                    mef = 1;
                else
                    mef = 2;
                end
            else
                if frame(2) == 'o'
                    mef = 3;
                else
                    mef = 4;
                end
            end
            M0 = memberM0(1,mef);
            delta(i) = 5*M0*(lm(repg(i)))^2/(48*e*aiy(repg(i)))...
                -(-RS(5,repg(i))+RS(11,repg(i)))/(16*e*aiy(repg(i)))*(lm(repg(i)))^2;
            form(i) = (delta(i)/lm(repg(i)))*300-1; 
        end
        return
    end
%
    function deflect = inter_story(d)
        if frame(1) == 's'
            deflect = zeros(1,2*nly);
            delbyhx = d(njf(6*(lyr(1,1)-1)+1),2)/fht(1);
            delbyhy = d(njf(6*(lyr(1,1)-1)+2),3)/fht(1);
            deflect(1) = abs(delbyhx)*200-1; 
            deflect(11) = abs(delbyhy)*200-1; 
            for i = 2:nly
                du = d(njf(6*(lyr(i,1)-1)+1),2) - d(njf(6*(lyr(i-1,1)-1)+1),2);
                delbyhx = du/fht(i);
                deflect(i) = abs(delbyhx)*200-1; 
            end
            for i = 2:nly
                dv = d(njf(6*(lyr(i,1)-1)+2),3) - d(njf(6*(lyr(i-1,1)-1)+2),3);
                delbyhy = dv/fht(i);
                deflect(i+10) = abs(delbyhy)*200-1; 
            end
        else
         
            deflect = zeros(1,nly);
            if frame(1) == 'x'
                delbyhx = d(njf(6*(lyr(1,1)-1)+1),2)/fht(1);
                deflect(1) = abs(delbyhx)*200-1; 
                for i = 2:nly
                    du = d(njf(6*(lyr(i,1)-1)+1),2) - d(njf(6*(lyr(i-1,1)-1)+1),2);
                    delbyhx = du/fht(i);
                    deflect(i) = abs(delbyhx)*200-1; 
                end
            else
                delbyhy = d(njf(6*(lyr(1,1)-1)+2),2)/fht(1);
                deflect(1) = abs(delbyhy)*200-1; 
                for i = 2:nly
                    dv = d(njf(6*(lyr(i,1)-1)+2),2) - d(njf(6*(lyr(i-1,1)-1)+2),2);
                    delbyhy = dv/fht(i);
                    deflect(i) = abs(delbyhy)*200-1; 
                end
            end
        end
    end
%
    function [wid_thick,wid_c] = wt_ratio(Ho,Bo,two,tfo,Do,to)
        wid_thick = zeros(1,2*nvg);
        wid_c = zeros(1,nvc);
        for i = 1:nvg
            H = Ho(Hn(repg(i)));
            B = Bo(Bn(repg(i)));
            tw = two(twn(repg(i)));
            tf = tfo(tfn(repg(i)));
            btf = B/2/tf/9; %325->7.65
            dtw = (H-2*tf)/tw/60; %325->51.02
            wid_thick(2*i-1) = btf-1.0; % -1.00
            wid_thick(2*i) = dtw-1.0; %-1.00
        end
        for j = 1:nvc
            D = Do(Dn(repc(j)));
            t = to(tn(repc(j)));
            bt = D/t/28.06; 
            wid_c(j) = bt-1.0; % -1.00
        end
        return
    end
%
    function [wid_gl] = wt_ratio_L(Bo,tfo)
        wid_gl = zeros(1,nvg);
        for i = 1:nvg
            B = Bo(Bn(repg(i)));
            tf = tfo(tfn(repg(i)));
            btf = B/2/tf;
            wid_gl(i) = 4.0/btf-1.0; % -1.00
        end
    return
    end
%
    function column = thickness(Do)
        column = [];
        if frame(1) == 'y'
            if rem(nely,2)
                for i = 1:nc/2-1
                    if ~mod(i,nly)
                        continue
                    end
                    if Dn(i) == Dn(i+1)
                        continue
                    end
                    b1 = Do(Dn(i));
                    b2 = Do(Dn(i+1));
                    column = horzcat(column,(b2/b1-1.0)); % -1.00
                end        
            else
                for j = 1:max(Dn)-1
                    if ~mod(j,nly)
                        continue
                    end
                    if Dn(j) == Dn(j+1)
                        continue
                    end
                    b1 = Do(Dn(j));
                    b2 = Do(Dn(j+1));
                    column = horzcat(column,(b2/b1-1.0)); % -1.00
                end
            end
        elseif frame(1) == 'x'
            if rem(nelx,2)
                for i = 1:nc/2-1
                    if ~mod(i,nly)
                        continue
                    end
                    if Dn(i) == Dn(i+1)
                        continue
                    end
                    b1 = Do(Dn(i));
                    b2 = Do(Dn(i+1));
                    column = horzcat(column,(b2/b1-1.0)); % -1.00
                end
            else
                for j = 1:max(Dn)-1
                    if ~mod(j,nly)
                        continue
                    end
                    if Dn(j) == Dn(j+1)
                        continue
                    end
                    b1 = Do(Dn(j));
                    b2 = Do(Dn(j+1));
                    column = horzcat(column,(b2/b1-1.0)); % -1.00
                end
            end
        else
            if rem(nelx,2)
                if rem(nely,2) %奇数×奇数
                    for i = 1:(nely+1)/2
                        for j = 1:(nelx+1)/2*nelz
                            if ~mod(j,nly)
                                continue
                            end
                            k = j+(nelx+1)*nelz*(i-1);
                            if Dn(k) == Dn(k+1)
                                continue
                            end
                            c1 = Do(Dn(k));
                            c2 = Do(Dn(k+1));
                            column = horzcat(column,(c2/c1-1.0)); % -1.00
                        end
                    end            
                else %奇数×偶数
                    for i = 1:nely/2+1
                        for j = 1:(nelx+1)/2*nelz
                            if ~mod(j,nly)
                                continue
                            end
                            k = j+(nelx+1)*nelz*(i-1);
                            if Dn(k) == Dn(k+1)
                                continue
                            end
                            c1 = Do(Dn(k));
                            c2 = Do(Dn(k+1));
                            column = horzcat(column,(c2/c1-1.0)); % -1.00
                        end
                    end       
                end
            else
                if rem(nely,2) %偶数×奇数
                    for i = 1:(nely+1)/2
                        for j = 1:(nelx/2+1)*nelz
                            if ~mod(j,nly)
                                continue
                            end
                            k = j+(nelx+1)*nelz*(i-1);
                            if Dn(k) == Dn(k+1)
                                continue
                            end
                            c1 = Do(Dn(k));
                            c2 = Do(Dn(k+1));
                            column = horzcat(column,(c2/c1-1.0)); % -1.00
                        end
                    end
                else %偶数×偶数
                    for i = 1:nely/2+1
                        for j = 1:(nelx/2+1)*nelz
                            if ~mod(j,nly)
                                continue
                            end
                            k = j+(nelx+1)*nelz*(i-1);
                            if Dn(k) == Dn(k+1)
                                continue
                            end
                            c1 = Do(Dn(k));
                            c2 = Do(Dn(k+1));
                            column = horzcat(column,(c2/c1-1.0)); % -1.00
                        end
                    end
                end
            end
        end
        if isempty(column)
            column = -1.0; % -1.00
        end
        return
    end
%
    function [gb,gw] = gwidth(Ho,Bo)
        a=ceil(nelx/2)*(nelz-1);
        gb = zeros(1,a);
        gw = zeros(1,a);
        for i = 1:a
            gb(i) = Ho(i+nelx/2)/Ho(i)-1.0; % -1.00
            gw(i) = Bo(i+nelx/2)/Bo(i)-1.0; % -1.00
        end
  
        
        return
    end
%
    function rps = proof_stress(zpy)
        if frame(1) == 's'
            nfj = (nelx+1)*(nely+1);
            ngx = nelx*(nely+1)*nelz;
        elseif frame(1) == 'x'
            nfj = nelx+1;
            ngx = ng;
        else
            nfj = nely+1;
            ngx = 0;
        end
    nrps = nj - nfj*2;
    rps = zeros(1,nrps*2);
    pci = []; pbix = []; pbiy = [];
    for i = nfj+1:nfj+nrps
        men = [find(js==i) find(je==i)];
        for j = 1:numel(men)
            if c_g(men(j)) == 1
                pci = horzcat(pci,zpy(men(j))*F(jel(men(j))));
            elseif c_g(men(j)) == 2 && men(j) <= ngx
                pbix = horzcat(pbix,zpy(men(j))*F(jel(men(j))));
            else
                pbiy = horzcat(pbiy,zpy(men(j))*F(jel(men(j))));
            end
        end
        rpc = sum(pci);
        rpbx = sum(pbix);
        rpby = sum(pbiy);
        rps(2*(i-nfj)-1:2*(i-nfj)) = [1.5*rpbx/rpc-1,1.5*rpby/rpc-1]; 
        pci = []; pbix = []; pbiy = [];
    end
    return
    end
end
%
function [nj,nc,ng,nsj] = modelParameter()
global nelx nely nelz frame
if frame(1) == 'x'
    nj = (nelx+1)*(nelz+1);
    nc = (nelx+1)*nelz;
    ng = nelx*nelz;
    nsj = nelx+1;
elseif frame(1) == 'y'
    nj = (nely+1)*(nelz+1);
    nc = (nely+1)*nelz;
    ng = nely*nelz;
    nsj = nely+1;
else
    nj = (nelx+1)*(nely+1)*(nelz+1);
    nc = (nelx+1)*(nely+1)*nelz;
    ng = (nelx*(nely+1)+(nelx+1)*nely)*nelz;
    nsj = (nelx+1)*(nely+1);
end
end
%
function [ndf,njf,njsf,njef] = rf_node_info()
global nelx nely nelz nj frame
jd = 1:nj;
if frame(1) == 's'
    jd(1:(nelx+1)*(nely+1)) = [];
    jd(1:(nelx+1)*(nely+1):length(jd)) = [];
    njr = (nelx+1)*(nely+1)+1:(nelx+1)*(nely+1):(nelx+1)*(nely+1)*(nelz+1);
    fnjd = (nelx+1)*(nely+1);
elseif frame(1) == 'x'
    jd(1:(nelx+1)) = [];
    jd(1:(nelx+1):length(jd)) = [];
    njr = (nelx+1)+1:(nelx+1):(nelx+1)*(nelz+1);
    fnjd = nelx+1;
else
    jd(1:(nely+1)) = [];
    jd(1:(nely+1):length(jd)) = [];
    njr = (nely+1)+1:(nely+1):(nely+1)*(nelz+1);
    fnjd = nely+1;
end

jf = zeros(1,nj);
for i = 1:nj
    if all(jd ~= i)
        jf(i) = 6;
    else
        jf(i) = 3;
    end
end
njdp = 1:nj;
for i = 1:length(njr)
    njdp(njr(i):njr(i)+fnjd-1) = njr(i);
end
njef = cumsum(jf);
njsf = [1 njef(1:end-1)+1];
ndf = njef(end);
njf = zeros(1,nj*6);
for i = 1:nj
    if jf(i) == 6
        njf(6*(i-1)+1:6*i) = njsf(i):njef(i);
    else
        njf(6*(i-1)+1) = njf(6*(njdp(i)-1)+1);
        njf(6*(i-1)+2) = njf(6*(njdp(i)-1)+2);
        njf(6*(i-1)+3:6*i-1) = njsf(i):njef(i);
        njf(6*i) = njf(6*(njdp(i)));
    end
end
return
end
% 
function [x,y,z,xe,ye,ze,xi,yi,zi] = nodeCodinate()
global nj nelx nely nelz lx ly lz span frame
x = zeros(1,nj);
y = zeros(1,nj);
z = zeros(1,nj);
switch span
    case 0
        xe(2:nelx+1) = lx;
        ye(2:nely+1) = ly;
    case 1
        xe(2:nelx+1) = lx;
        ye(2:nely+1) = ly;
    otherwise
end
ze(2) = lz+0; ze(3:nelz+1) = lz;

xi = cumsum(xe);
yi = cumsum(ye);
zi = cumsum(ze);
if nelz == 0
    if frame(1) == 'y'
        y = yi;
    elseif frame(1) == 'x'
        x = xi;
    else
        for n = 1:nely+1
            i = (nelx+1)*(n-1)+1;
            j = (nelx+1)*n;
            x(i:j) = xi;
            y(i:j) = yi(n);
        end
    end
else
    for n = 1:nelz+1
        if frame(1) == 'y'
            i = (nely+1)*(n-1)+1;
            j = (nely+1)*n;
            y(i:j) = yi;
            z(i:j) = zi(n);
        elseif frame(1) == 'x'
            i = (nelx+1)*(n-1)+1;
            j = (nelx+1)*n;
            x(i:j) = xi;
            z(i:j) = zi(n);
        else
            for m = 1:nely+1
                i = (nelx+1)*(m-1)+1;
                j = (nelx+1)*m;
                s = i+(nelx+1)*(nely+1)*(n-1);
                t = j+(nelx+1)*(nely+1)*(n-1);
                x(s:t) = xi;
                y(s:t) = yi(m);
                z(s:t) = zi(n);
            end
        end
    end
end
end
%
function [js,je,jel,c_g] = memberInfo()
global nm nelx nely nelz frame
js = zeros(1,nm);
je = zeros(1,nm);
jel = zeros(1,nm);
c_g = zeros(1,nm);
if nelz == 0
    if frame(1) == 'y'
        js = 1:nely;
        je = 2:nely+1;
        jel(:) = 1; %許容応力度　柱　→　２，梁　→　１
        c_g(:) = 2; %柱か梁か　　柱　→　１，梁　→　２
    elseif frame(1) == 'x'
        js = 1:nelx;
        je = 2:nely+1;
        jel(:) = 1;
        c_g(:) = 2;
    else
        for n = 1:nely+1
            i = nelx*(n-1)+1;
            j = nelx*n;
            s = (nelx+1)*(n-1)+1;
            t = (nelx+1)*n;
            js(i:j) = s:t-1;
            je(i:j) = s+1:t;
        end
        for m = 1:nelx+1
            i = nelx*(nely+1)+nely*(m-1)+1;
            j = nelx*(nely+1)+nely*m;
            a = m;
            b = (nelx+1)*(nely-1)+m;
            c = nelx+1+m;
            d = (nelx+1)*nely+m;
            js(i:j) = a:nelx+1:b;
            je(i:j) = c:nelx+1:d;
        end
        jel(:) = 1;
        c_g(:) = 2;
    end
else
    if frame(1) == 'y'
        for n = 2:nelz+1
            i = nely*(n-2)+1;
            j = nely*(n-1);
            s = (nely+1)*(n-1)+1;
            t = (nely+1)*n;
            js(i:j) = s:t-1;
            je(i:j) = s+1:t;
            jel(i:j) = 1;
            c_g(i:j) = 2;
        end
        for m = 1:nely+1
            i = nely*nelz+nelz*(m-1)+1;
            j = nely*nelz+nelz*m;
            a = m;
            b = (nely+1)*(nelz-1)+m;
            c = nely+1+m;
            d = (nely+1)*nelz+m;
            js(i:j) = a:nely+1:b;
            je(i:j) = c:nely+1:d;
            jel(i:j) = 2;
            c_g(i:j) = 1;
        end
    elseif frame(1) == 'x'
        for n = 2:nelz+1
            i = nelx*(n-2)+1;
            j = nelx*(n-1);
            s = (nelx+1)*(n-1)+1;
            t = (nelx+1)*n;
            js(i:j) = s:t-1;
            je(i:j) = s+1:t;
            jel(i:j) = 1;
            c_g(i:j) = 2;
        end
        for m = 1:nelx+1
            i = nelx*nelz+nelz*(m-1)+1;
            j = nelx*nelz+nelz*m;
            a = m;
            b = (nelx+1)*(nelz-1)+m;
            c = nelx+1+m;
            d = (nelx+1)*nelz+m;
            js(i:j) = a:nelx+1:b;
            je(i:j) = c:nelx+1:d;
            jel(i:j) = 2;
            c_g(i:j) = 1;
        end
    else
        for l = 2:nelz+1
            for m = 1:nely+1
                i = nelx*(nely+1)*(l-2)+nelx*(m-1)+1;
                j = nelx*(nely+1)*(l-2)+nelx*m;
                s = (nelx+1)*(nely+1)*(l-1)+(nelx+1)*(m-1)+1;
                t = (nelx+1)*(nely+1)*(l-1)+(nelx+1)*m;
                js(i:j) = s:t-1;
                je(i:j) = s+1:t;
                jel(i:j) = 1;
                c_g(i:j) = 2;
            end
        end
        for l = 2:nelz+1
            for m = 1:nelx+1
                i = nelx*(nely+1)*nelz+(nelx+1)*nely*(l-2)+nely*(m-1)+1;
                j = nelx*(nely+1)*nelz+(nelx+1)*nely*(l-2)+nely*m;
                a = (nelx+1)*(nely+1)*(l-1)+m;
                b = (nelx+1)*(nely+1)*(l-1)+(nelx+1)*(nely-1)+m;
                c = (nelx+1)*(nely+1)*(l-1)+nelx+1+m;
                d = (nelx+1)*(nely+1)*(l-1)+(nelx+1)*nely+m;
                js(i:j) = a:nelx+1:b;
                je(i:j) = c:nelx+1:d;
                jel(i:j) = 1;
                c_g(i:j) = 2;
            end
        end
        for n = 1:nely+1
            for o = 1:nelx+1
                i = (nelx*(nely+1)+nely*(nelx+1))*nelz+...
                    nelz*(nelx+1)*(n-1)+nelz*(o-1)+1;
                j = (nelx*(nely+1)+nely*(nelx+1))*nelz+...
                    nelz*(nelx+1)*(n-1)+nelz*o;
                a = (nelx+1)*(n-1)+o;
                b = (nelz-1)*(nelx+1)*(nely+1)+(nelx+1)*(n-1)+o;
                c = (nelx+1)*(nely+1)+(nelx+1)*(n-1)+o;
                d = nelz*(nelx+1)*(nely+1)+(nelx+1)*(n-1)+o;
                js(i:j) = a:(nelx+1)*(nely+1):b;
                je(i:j) = c:(nelx+1)*(nely+1):d;
                jel(i:j) = 2;
                c_g(i:j) = 1;
            end
        end
    end
end
end
%
function lm = memberLength()
global nm x y z js je
lm = zeros(1,nm);
for i = 1:nm
    alx = x(je(i))-x(js(i));
    aly = y(je(i))-y(js(i));
    alz = z(je(i))-z(js(i));
    alj = sqrt(alx*alx+aly*aly+alz*alz);
    lm(1,i) = alj;
end
end
%
function [Hn,Bn,twn,tfn,Dn,tn] = sectiongroup()
global nelx nely nelz frame
if nelz == 0
    
else
    if frame(1) == 'y'
        if rem(nely,2)
            for n = 2:nelz+1
                i = nely*(n-2)+1;
                j = nely*(n-1);
                s = (nely+1)/2*(n-2)+1;
                t = (nely+1)/2*(n-1);
                Hn(i:j) = n-1;
                Bn(i:j) = n-1;
                twn(i:i-1+(nely+1)/2) = s:t;
                twn(i+(nely+1)/2:j) = t-1:-1:s;
                tfn = twn;
            end
            for m = 1:nely/2+1
                i = nelz*(m-1)+1;
                j = nelz*m;
                s = nelz*(nely+1-m)+1;
                t = nelz*(nely+1-(m-1));
                Dn(i:j) = i:j;
                Dn(s:t) = i:j;
                tn(i:j) = i:j;
                tn(s:t) = i:j;
            end
        else
            for n = 2:nelz+1
                i = nely*(n-2)+1;
                j = nely*(n-1);
                s = nely/2*(n-2)+1;
                t = nely/2*(n-1);
                Hn(i:j) = n-1;
                Bn(i:j) = n-1;
                twn(i:i-1+nely/2) = s:t;
                twn(j+1-nely/2:j) = t:-1:s;
                tfn(i:i-1+nely/2) = s:t;
                tfn(j+1-nely/2:j) = t:-1:s;
            end
            for m = 1:nely/2
                i = nelz*(m-1)+1;
                j = nelz*m;
                s = nelz*(nely+1-m)+1;
                t = nelz*(nely+1-(m-1));
                Dn(i:j) = i:j;
                Dn(s:t) = i:j;
                tn(i:j) = i:j;
                tn(s:t) = i:j;
            end
            Dn(nelz*nely/2:nelz*(nely/2+1)) =...
                nelz*nely/2:nelz*(nely/2+1);
            tn(nelz*nely/2:nelz*(nely/2+1)) =...
                nelz*nely/2:nelz*(nely/2+1);
        end
    elseif frame(1) == 'x'
        if rem(nelx,2)
            for n = 2:nelz+1
                i = nelx*(n-2)+1;
                j = nelx*(n-1);
                s = (nelx+1)/2*(n-2)+1;
                t = (nelx+1)/2*(n-1);
                Hn(i:j) = n-1;
                Bn(i:j) = n-1;
                twn(i:i-1+(nelx+1)/2) = s:t;
                twn(i+(nelx+1)/2:j) = t-1:-1:s;
                tfn = twn;
            end
            

            for m = 1:nelx/2
                i = nelz*(m-1)+1;
                j = nelz*m;

                s = nelz*(nelx+1-m)+1;
                t = nelz*(nelx+1-(m-1));
                Dn(i:j) = i:j;
                Dn(s:t) = i:j;
                tn(i:j) = i:j;
                tn(s:t) = i:j;
            end
   
        else
            for n = 2:nelz+1
                i = nelx*(n-2)+1;
                j = nelx*(n-1);
                s = nelx/2*(n-2)+1;
                t = nelx/2*(n-1);
                Hn(i:j) = n-1;
                Bn(i:j) = n-1;
                twn(i:i-1+nelx/2) = s:t;
                twn(j+1-nelx/2:j) = t:-1:s;
                tfn(i:i-1+nelx/2) = s:t;
                tfn(j+1-nelx/2:j) = t:-1:s;
            end
            for m = 1:nelx/2
                i = nelz*(m-1)+1;
                j = nelz*m;
                s = nelz*(nelx+1-m)+1;
                t = nelz*(nelx+1-(m-1));
                Dn(i:j) = i:j;
                Dn(s:t) = i:j;
                tn(i:j) = i:j;
                tn(s:t) = i:j;
            end

            Dn(nelz*nelx/2:nelz*(nelx/2+1)) =...
                nelz*nelx/2:nelz*(nelx/2+1);
            tn(nelz*nelx/2:nelz*(nelx/2+1)) =...
                nelz*nelx/2:nelz*(nelx/2+1);
        end
    else
        if rem(nelx,2)
            if rem(nely,2) %奇数×奇数
                for n = 2:nelz+1
                    for m = 1:(nely+1)/2
                        i = nelx*(nely+1)*(n-2)+nelx*(m-1)+1;
                        j = nelx*(nely+1)*(n-2)+nelx*m;
                        a = nelx*(nely+1)*(n-1)-nelx*m+1;
                        b = nelx*(nely+1)*(n-1)-nelx*(m-1);
                        s = (nelx+1)/2*(nely+1)/2*(n-2)+(nelx+1)/2*(m-1)+1;
                        t = (nelx+1)/2*(nely+1)/2*(n-2)+(nelx+1)/2*m;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+(nelx+1)/2-1) = s:t;
                        twn(i+(nelx+1)/2:j) = t-1:-1:s;
                        twn(a:a+(nelx+1)/2-1) = s:t;
                        twn(a+(nelx+1)/2:b) = t-1:-1:s;
                        tfn = twn;
                    end
                    for l = 1:(nelx+1)/2
                        i = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*(l-1)+1;
                        j = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*l;
                        a = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*l+1;
                        b = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*(l-1);
                        s = ((nely+1)/2*(nelx+1))/2*(n-2)...
                            +(nely+1)/2*(l-1)+1+(nelx+1)/2*(nely+1)/2*nelz;
                        t = ((nely+1)/2*(nelx+1))/2*(n-2)+(nely+1)/2*l...
                            +(nelx+1)/2*(nely+1)/2*nelz;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+(nely+1)/2-1) = s:t;
                        twn(i+(nely+1)/2:j) = t-1:-1:s;
                        twn(a:a+(nely+1)/2-1) = s:t;
                        twn(a+(nely+1)/2:b) = t-1:-1:s;
                        tfn = twn;
                    end
                end
                for k = 1:(nely+1)/2
                    for l = 1:(nelx+1)/2
                        a = nelz*(l-1)+nelz*(nelx+1)*(k-1)+1;
                        b = a+nelz-1;
                        c = -nelz*(l-1)+nelz*nelx+1+nelz*(nelx+1)*(k-1);
                        d = c+nelz-1;
                        e = nelz*(l-1)+1-nelz*(nelx+1)*(k-1)+(nelx+1)*nelz*nely;
                        f = e+nelz-1;
                        g = (nelx+1)*(nely+1)*nelz-nelz*l-nelz*(nelx+1)*(k-1)+1;
                        h = g+nelz-1;
                        s = nelz*(l-1)+1+nelz*(nelx+1)/2*(k-1);
                        t = s+nelz-1;
                        Dn(a:b) = s:t;
                        Dn(c:d) = s:t;
                        Dn(e:f) = s:t;
                        Dn(g:h) = s:t;
                        tn(a:b) = s:t;
                        tn(c:d) = s:t;
                        tn(e:f) = s:t;
                        tn(g:h) = s:t;                        
                    end
                end
            else %奇数×偶数
                for n = 2:nelz+1
                    for m = 1:nely/2+1
                        i = nelx*(nely+1)*(n-2)+nelx*(m-1)+1;
                        j = nelx*(nely+1)*(n-2)+nelx*m;
                        a = nelx*(nely+1)*(n-1)-nelx*m+1;
                        b = nelx*(nely+1)*(n-1)-nelx*(m-1);
                        s = (nelx+1)/2*(nely/2+1)*(n-2)+(nelx+1)/2*(m-1)+1;
                        t = (nelx+1)/2*(nely/2+1)*(n-2)+(nelx+1)/2*m;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+(nelx+1)/2-1) = s:t;
                        twn(i+(nelx+1)/2:j) = t-1:-1:s;
                        twn(a:a+(nelx+1)/2-1) = s:t;
                        twn(a+(nelx+1)/2:b) = t-1:-1:s;
                        tfn = twn;
                    end
                    for l = 1:(nelx+1)/2
                        i = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*(l-1)+1;
                        j = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*l;
                        a = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*l+1;
                        b = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*(l-1);
                        s = (nely/2)*(nelx+1)/2*(n-2)...
                            +nely/2*(l-1)+1+(nelx+1)/2*(nely/2+1)*nelz;
                        t = (nely/2)*(nelx+1)/2*(n-2)+nely/2*l...
                            +(nelx+1)/2*(nely/2+1)*nelz;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+nely/2-1) = s:t;
                        twn(i+nely/2:j) = t:-1:s;
                        twn(a:a+nely/2-1) = s:t;
                        twn(a+nely/2:b) = t:-1:s;
                        tfn = twn;
                    end
                end
                for k = 1:nely/2+1
                    for l = 1:(nelx+1)/2
                        a = nelz*(l-1)+nelz*(nelx+1)*(k-1)+1;
                        b = a+nelz-1;
                        c = -nelz*(l-1)+nelz*nelx+1+nelz*(nelx+1)*(k-1);
                        d = c+nelz-1;
                        e = nelz*(l-1)+1-nelz*(nelx+1)*(k-1)+(nelx+1)*nelz*nely;
                        f = e+nelz-1;
                        g = (nelx+1)*(nely+1)*nelz-nelz*l-nelz*(nelx+1)*(k-1)+1;
                        h = g+nelz-1;
                        s = nelz*(l-1)+1+nelz*(nelx+1)/2*(k-1);
                        t = s+nelz-1;
                        Dn(a:b) = s:t;
                        Dn(c:d) = s:t;
                        Dn(e:f) = s:t;
                        Dn(g:h) = s:t;
                        tn(a:b) = s:t;
                        tn(c:d) = s:t;
                        tn(e:f) = s:t;
                        tn(g:h) = s:t;                        
                    end
                end
            end
        else % 偶数×奇数
            if rem(nely,2)
                for n = 2:nelz+1
                    for m = 1:(nely+1)/2
                        i = nelx*(nely+1)*(n-2)+nelx*(m-1)+1;
                        j = nelx*(nely+1)*(n-2)+nelx*m;
                        a = nelx*(nely+1)*(n-1)-nelx*m+1;
                        b = nelx*(nely+1)*(n-1)-nelx*(m-1);
                        s = (nelx/2*(nely+1))/2*(n-2)+nelx/2*(m-1)+1;
                        t = (nelx/2*(nely+1))/2*(n-2)+nelx/2*m;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+nelx/2-1) = s:t;
                        twn(i+nelx/2:j) = t:-1:s;
                        twn(a:a+nelx/2-1) = s:t;
                        twn(a+nelx/2:b) = t:-1:s;
                        tfn = twn;
                    end
                    for l = 1:nelx/2+1
                        i = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*(l-1)+1;
                        j = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*l;
                        a = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*l+1;
                        b = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*(l-1);
                        s = (nely+1)/2*(nelx/2+1)*(n-2)...
                            +(nely+1)/2*(l-1)+1+nelx/2*(nely+1)/2*nelz;
                        t = (nely+1)/2*(nelx/2+1)*(n-2)+(nely+1)/2*l...
                            +nelx/2*(nely+1)/2*nelz;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+(nely+1)/2-1) = s:t;
                        twn(i+(nely+1)/2:j) = t-1:-1:s;
                        twn(a:a+(nely+1)/2-1) = s:t;
                        twn(a+(nely+1)/2:b) = t-1:-1:s;
                        tfn = twn;
                    end
                end
                for k = 1:(nely+1)/2
                    for l = 1:nelx/2+1
                        a = nelz*(l-1)+nelz*(nelx+1)*(k-1)+1;
                        b = a+nelz-1;
                        c = -nelz*(l-1)+nelz*nelx+1+nelz*(nelx+1)*(k-1);
                        d = c+nelz-1;
                        e = nelz*(l-1)+1-nelz*(nelx+1)*(k-1)+(nelx+1)*nelz*nely;
                        f = e+nelz-1;
                        g = (nelx+1)*(nely+1)*nelz-nelz*l-nelz*(nelx+1)*(k-1)+1;
                        h = g+nelz-1;
                        s = nelz*(l-1)+1+nelz*(nelx/2+1)*(k-1);
                        t = s+nelz-1;
                        Dn(a:b) = s:t;
                        Dn(c:d) = s:t;
                        Dn(e:f) = s:t;
                        Dn(g:h) = s:t;
                        tn(a:b) = s:t;
                        tn(c:d) = s:t;
                        tn(e:f) = s:t;
                        tn(g:h) = s:t;                        
                    end
                end
            else %偶数×偶数
                for n = 2:nelz+1
                    for m = 1:nely/2+1
                        i = nelx*(nely+1)*(n-2)+nelx*(m-1)+1;
                        j = nelx*(nely+1)*(n-2)+nelx*m;
                        a = nelx*(nely+1)*(n-1)-nelx*m+1;
                        b = nelx*(nely+1)*(n-1)-nelx*(m-1);
                        s = nelx/2*(nely/2+1)*(n-2)+nelx/2*(m-1)+1;
                        t = nelx/2*(nely/2+1)*(n-2)+nelx/2*m;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+nelx/2-1) = s:t;
                        twn(i+nelx/2:j) = t:-1:s;
                        twn(a:a+nelx/2-1) = s:t;
                        twn(a+nelx/2:b) = t:-1:s;
                        tfn = twn;
                    end
                    for l = 1:nelx/2+1
                        i = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*(l-1)+1;
                        j = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-2)+nely*l;
                        a = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*l+1;
                        b = nelx*(nely+1)*nelz+...
                            (nelx+1)*nely*(n-1)-nely*(l-1);
                        s = nely/2*(nelx/2+1)*(n-2)+nely/2*(l-1)+1+...
                            nelx/2*(nely/2+1)*nelz;
                        t = nely/2*(nelx/2+1)*(n-2)+nely/2*l+...
                            nelx/2*(nely/2+1)*nelz;
                        Hn(i:j) = n-1; Hn(a:b) = n-1;
                        Bn(i:j) = n-1; Bn(a:b) = n-1;
                        twn(i:i+nely/2-1) = s:t;
                        twn(i+nely/2:j) = t:-1:s;
                        twn(a:a+nely/2-1) = s:t;
                        twn(a+nely/2:b) = t:-1:s;
                        tfn = twn;
                    end
                end
                for k = 1:nely/2+1
                    for l = 1:nelx/2+1
                        a = nelz*(l-1)+nelz*(nelx+1)*(k-1)+1;
                        b = a+nelz-1;
                        c = -nelz*(l-1)+nelz*nelx+1+nelz*(nelx+1)*(k-1);
                        d = c+nelz-1;
                        e = nelz*(l-1)+1-nelz*(nelx+1)*(k-1)+(nelx+1)*nelz*nely;
                        f = e+nelz-1;
                        g = (nelx+1)*(nely+1)*nelz-nelz*l-nelz*(nelx+1)*(k-1)+1;
                        h = g+nelz-1;
                        s = nelz*(l-1)+1+nelz*(nelx/2+1)*(k-1);
                        t = s+nelz-1;
                        Dn(a:b) = s:t;
                        Dn(c:d) = s:t;
                        Dn(e:f) = s:t;
                        Dn(g:h) = s:t;
                        tn(a:b) = s:t;
                        tn(c:d) = s:t;
                        tn(e:f) = s:t;
                        tn(g:h) = s:t;                        
                    end
                end
            end
        end
    end
end
Hn=twn;
Bn=twn;
end
%
function [Hp,Bp,twp,tfp,Dp,tp] = arrangevalue()
global Hn Bn twn tfn Dn tn
Hp = 1:max(Hn);
Bp = Hp(end)+1:Hp(end)+max(Bn);
% twp = Bp(end)+1:Bp(end)+max(twn);
tfp = Bp(end)+1:Bp(end)+max(tfn);
Dp = tfp(end)+1:tfp(end)+max(Dn);
tp = Dp(end)+1:Dp(end)+max(tn);
twp=[1 1 1 1];

end
%
function cyl = directionCosine()
global nm js je x y z
cyl = zeros(nm,3);
ang = zeros(nm,1);
for i = 1:nm
    nns = js(i);
    nne = je(i);
    nan = pi*ang(i)/180;
    cyl(i,:) = ystar(x(nns),y(nns),z(nns),x(nne),y(nne),z(nne),nan);
end
end
%
function nb = bandwidth()
global nm js je njsf njef
nbi = 0;
for i=1:nm
  n = njsf(js(i))-njef(je(i));
%   n = js(i)-je(i);
  if(n<0)
    n = -n;
  end
  if(n>nbi)
    nbi = n;
  end
end
nb = nbi+1;
% nb = 6*(nbi+1);
end
%
function [ns,isup,pd] = supportInfo()
global nsj
ns = 1:nsj;
isup = zeros(nsj,6);
pd  = zeros(nsj,6);

end
%
function seismicForce = earthquake()
global nelz pw fht Z Co xi yi
seismicForce = zeros(nelz,1);
Qi = zeros(nelz,1);
Ci = zeros(nelz,1);
Ai = zeros(nelz,1);
alpha = zeros(nelz,1);
Sigmawi = zeros(nelz,1);
T = 0.03*sum(fht)*1e-3;
wi = xi(end)*yi(end)*(pw-1e-3);
if T < 0.6
    Rt = 1.0;
elseif 0.6 < T && T < 2*0.6
    Rt = 1-0.2*(T/0.6-1)^2;
else
    Rt = 1.6*0.6/T;
end
for i = nelz:-1:1
    if i == nelz
        Sigmawi(i) = wi;
    else
        Sigmawi(i) = Sigmawi(i+1)+wi;
    end
end
for i = 1:nelz
    alpha(i) = Sigmawi(i)/Sigmawi(1);
    Ai(i) = 1+(1/sqrt(alpha(i))-alpha(i))*(2*T/(1+3*T));
    Ci(i) = Z*Rt*Ai(i)*Co;
    Qi(i) = Ci(i)*Sigmawi(i);
end
for i = nelz:-1:1
    if i == nelz
        seismicForce(i) = Qi(i);
    else
        seismicForce(i) = Qi(i)-Qi(i+1);
    end
end
end
%
function [f,memberEnd,M0,lyr] = loadsf()
seismicForce = earthquake();
[memberEnd,M0] = memberForce();
[f,lyr] = arrangeLoad(seismicForce,memberEnd);
end
%
function [memberEnd,M0] = memberForce()
global lx ly pw
% [xout,xin(orthogonalyout),yout,yin(orthogonalxout),orthogonalxin,orthogonalyin]
memberEnd = zeros(12,6);
M0 = zeros(1,6);
if lx < ly
    wxy = pw*lx/2;
    Qx = wxy*lx/4;
    Qy = wxy*(ly-lx/2)/2;
    Cx = -5*wxy*lx^2/96;
    Cy = -wxy*(ly^3-2*((lx/2)^2)*ly+(lx/2)^3)/(12*ly);
    M0x = wxy*lx^2/12;
    M0y = wxy*(3*ly^2-4*(lx/2)^2)/24;
elseif lx > ly
    wxy = pw*ly/2;
    Qx = wxy*(lx-ly/2)/2;
    Qy = wxy*ly/4;
    Cx = -wxy*(lx^3-2*((ly/2)^2)*lx+(ly/2)^3)/(12*lx);
    Cy = -5*wxy*ly^2/96;
    M0x = wxy*(3*lx^2-4*(ly/2)^2)/24;
    M0y = wxy*ly^2/12;
else
    wxy = pw*lx/2;
    Qx = wxy*lx/4;
    Qy = wxy*ly/4;
    Cx = -5*wxy*lx^2/96;
    Cy = -5*wxy*ly^2/96;
    M0x = wxy*lx^2/12;
    M0y = wxy*ly^2/12;
end
memberEnd(3,1) = Qx;
memberEnd(9,1) = Qx;
memberEnd(5,1) = Cx;
memberEnd(11,1) = -Cx;
memberEnd(:,2) = memberEnd(:,1)*2; % 内構面
memberEnd(3,3) = Qy;
memberEnd(9,3) = Qy;
memberEnd(5,3) = Cy;
memberEnd(11,3) = -Cy;
memberEnd(:,4) = memberEnd(:,3)*2; % 内構面
memberEnd(:,5) = memberEnd(:,3)*4; 
memberEnd(:,6) = memberEnd(:,1)*4;
M0(1,1) = M0x;
M0(1,2) = M0x*2;
M0(1,3) = M0y;
M0(1,4) = M0y*2;
M0(1,5) = M0y*4;
M0(1,6) = M0x*4;

end
%
function [f,lyr] = arrangeLoad(seismicForce,memberEnd)
global x y z nelx nely nelz frame js je ng ndf nlc c_g cyl njf
f = zeros(ndf,nlc);

beam = find(c_g == 2);
fxj = zeros(nelz,nely+1);
fyj = zeros(nelz,nelx+1);
tt = zeros(3,3);
if frame(2) == 'o'
    v=1;
else
    v=2;
end
if frame(1) == 's'
    perNode = (nelx+1)*(nely+1);
    ngx = nelx*(nely+1)*nelz;
    ngy = (nelx+1)*nely*nelz;
    perngx = nelx*(nely+1);
    perngy = (nelx+1)*nely;
    for i = 1:nelz
        for j = 1:nely+1
            fxj(i,j) = perNode*i+(nelx+1)*(j-1)+1;
            f(njf(6*(fxj(i,j)-1)+1),2) = seismicForce(i)/(nely+1);
        end
        for j = 1:nelx+1
            fyj(i,j) = perNode*i+j;
            f(njf(6*(fyj(i,j)-1)+2),3) = seismicForce(i)/(nelx+1);
        end
    end
    for i = 1:nlc
        for j = 1:ngx
            lcase = i;
            if mod(j,perngx) <= nelx || mod(j,perngx) > perngx-nelx
                lcx = 1;
            else
                lcx = 2;
            end
            member = beam(j);
            alx = x(je(member))-x(js(member));
            aly = y(je(member))-y(js(member));
            alz = z(je(member))-z(js(member));
            al = sqrt(alx*alx+aly*aly+alz*alz);
            tt(1,1) = alx/al;
            tt(2,1) = aly/al;
            tt(3,1) = alz/al;
            tt(1,2) = cyl(member,1);
            tt(2,2) = cyl(member,2);
            tt(3,2) = cyl(member,3);
            tt(1,3) = tt(2,1)*tt(3,2)-tt(3,1)*tt(2,2);
            tt(2,3) = tt(3,1)*tt(1,2)-tt(1,1)*tt(3,2);
            tt(3,3) = tt(1,1)*tt(2,2)-tt(2,1)*tt(1,2);
            %------
            nn = njf(6*(js(member)-1)+1:6*js(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,lcx); tt*memberEnd(4:6,lcx)];
            % Repeat for forces at end node of the member.
            nn = njf(6*(je(member)-1)+1:6*je(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,lcx); tt*memberEnd(10:12,lcx)];
        end
        for k = ngx+1:ngx+ngy
            lcase = i;
            if mod(k-ngx,perngy) <= nely || mod(k-ngx,perngy) > perngy-nely
                lcy = 3;
            else
                lcy = 4;
            end
            member = beam(k);
            alx = x(je(member))-x(js(member));
            aly = y(je(member))-y(js(member));
            alz = z(je(member))-z(js(member));
            al = sqrt(alx*alx+aly*aly+alz*alz);
            tt(1,1) = alx/al;
            tt(2,1) = aly/al;
            tt(3,1) = alz/al;
            tt(1,2) = cyl(member,1);
            tt(2,2) = cyl(member,2);
            tt(3,2) = cyl(member,3);
            tt(1,3) = tt(2,1)*tt(3,2)-tt(3,1)*tt(2,2);
            tt(2,3) = tt(3,1)*tt(1,2)-tt(1,1)*tt(3,2);
            tt(3,3) = tt(1,1)*tt(2,2)-tt(2,1)*tt(1,2);
            %------
            nn = njf(6*(js(member)-1)+1:6*js(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,lcy); tt*memberEnd(4:6,lcy)];
            % Repeat for forces at end node of the member.
            nn = njf(6*(je(member)-1)+1:6*je(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,lcy); tt*memberEnd(10:12,lcy)];
        end
    end
    lyr = fxj(:,1);
elseif frame(1) == 'x'
    for i = 1:nelz
        fxj(i,1) = (nelx+1)*i+1;
        f(njf(6*(fxj(i,1)-1)+1),2) = seismicForce(i)/(nely+1);
    end
    for i = 1:nlc
        for j = 1:ng
            lcase = i;
            member = beam(j);
            if frame(2) == 'o'
                mef = 1;
                orthogout = 3;
                orthogin = 4;
            else
                mef = 2;
                orthogout = 4;
                orthogin = 5;
            end
            alx = x(je(member))-x(js(member));
            aly = y(je(member))-y(js(member));
            alz = z(je(member))-z(js(member));
            al = sqrt(alx*alx+aly*aly+alz*alz);
            tt(1,1) = alx/al;
            tt(2,1) = aly/al;
            tt(3,1) = alz/al;
            tt(1,2) = cyl(member,1);
            tt(2,2) = cyl(member,2);
            tt(3,2) = cyl(member,3);
            tt(1,3) = tt(2,1)*tt(3,2)-tt(3,1)*tt(2,2);
            tt(2,3) = tt(3,1)*tt(1,2)-tt(1,1)*tt(3,2);
            tt(3,3) = tt(1,1)*tt(2,2)-tt(2,1)*tt(1,2);
            %------
            nn = njf(6*(js(member)-1)+1:6*js(member));
            
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,mef); tt*memberEnd(4:6,mef)];
            % Repeat for forces at end node of the member.
            nn = njf(6*(je(member)-1)+1:6*je(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,mef); tt*memberEnd(10:12,mef)];
            

            % nn = njf(6*(js(member)-1)+1:6*js(member));
            % f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogout);0;0;0];          
            % nn = njf(6*(je(member)-1)+1:6*je(member));
            % f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,orthogout);0;0;0];
            if mod(j,nelx) == 1 || nelx == 1%%左端部に追加
                nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogout);0;0;0];
            elseif mod(j,nelx) == 0%%右端部に追加
                nn = njf(6*(je(member)-1)+1:6*je(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,orthogout);0;0;0];
                 nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogin);0;0;0];               
            else
                nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogin);0;0;0];
                % nn = njf(6*(je(member)-1)+1:6*je(member));
                % f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,orthogin);0;0;0];             
            end

        end
    end
    lyr = fxj(:,1);
else
    for i = 1:nelz
        fyj(i,1) = (nely+1)*i+1;
        f(njf(6*(fyj(i,1)-1)+2),2) = seismicForce(i)/(nelx+1);
    end
    for i = 1:nlc
        for j = 1:ng
            lcase = i;
            member = beam(j);
            if frame(2) == 'o'
                mef = 3;
                orthogout = 1;
                orthogin = 2;
            else
                mef = 4;
                orthogout = 2;
                orthogin = 6;
            end
            alx = x(je(member))-x(js(member));
            aly = y(je(member))-y(js(member));
            alz = z(je(member))-z(js(member));
            al = sqrt(alx*alx+aly*aly+alz*alz);
            tt(1,1) = alx/al;
            tt(2,1) = aly/al;
            tt(3,1) = alz/al;
            tt(1,2) = cyl(member,1);
            tt(2,2) = cyl(member,2);
            tt(3,2) = cyl(member,3);
            tt(1,3) = tt(2,1)*tt(3,2)-tt(3,1)*tt(2,2);
            tt(2,3) = tt(3,1)*tt(1,2)-tt(1,1)*tt(3,2);
            tt(3,3) = tt(1,1)*tt(2,2)-tt(2,1)*tt(1,2);
            %------
            nn = njf(6*(js(member)-1)+1:6*js(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,mef); tt*memberEnd(4:6,mef)];
            % Repeat for forces at end node of the member.
            nn = njf(6*(je(member)-1)+1:6*je(member));
            f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,mef); tt*memberEnd(10:12,mef)];
            
            if mod(j,nely) == 1 || nely == 1%%左端部に追加
                nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogout);0;0;0];
            elseif mod(j,nely) == 0%%右端部に追加
                nn = njf(6*(je(member)-1)+1:6*je(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(7:9,orthogout);0;0;0];
                 nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogin);0;0;0];               
            else
                nn = njf(6*(js(member)-1)+1:6*js(member));
                f(nn,lcase) = f(nn,lcase)+[tt*memberEnd(1:3,orthogin);0;0;0];
            end
        end
        
    end
    lyr = fyj(:,1);
end
end
%
function [He,Be,twe,tfe,De,te] = sectionStandard()
global rankStandard
filename = './csv/JPN_standard.xlsx';
%
sheet = 1;
girderStandard = xlsread(filename,sheet);
He = girderStandard(:,2);
Be = girderStandard(:,3);
twe = girderStandard(:,4);
tfe = girderStandard(:,5);
%
sheet = 2;
columnStandard = xlsread(filename,sheet);
De = columnStandard(:,2);
te = columnStandard(:,3);
%
sheet = 3;
rankStandard = xlsread(filename,sheet);
%
He = rmmissing(He);
Be = rmmissing(Be);
twe = rmmissing(twe);
tfe = rmmissing(tfe);
De = rmmissing(De);
te = rmmissing(te);
end
%
function c = ystar(xs,ys,zs,xe,ye,ze,angle)
%  Direction cosines of local y* axis.  The subroutine is used only
%  when the direction cosines are not given. In which case, the y*
%  axis is assumed horizontal and directed such that Lmdaz*z is positive
%  or zero.
al = sqrt((xe-xs)^2+(ye-ys)^2+(ze-zs)^2);
a1 = (xe-xs)/al;
a2 = (ye-ys)/al;
a3 = (ze-zs)/al;

albar = sqrt(a1^2+a2^2);
if (albar~=0)
  c1 = -a2/albar*cos(angle)-a3*a1/albar*sin(angle);
  c2 = a1/albar*cos(angle)-a2*a3/albar*sin(angle);
  c3 = albar*sin(angle);
  c = [c1 c2 c3];
end
if (abs(a3)==1)
  c = [-sin(angle) cos(angle) 0];
end
return
end
%
function [lb,ub] = setbound()
global Hp Bp twp tfp Dp tp
nval = tp(end);
lb = zeros(1,nval);
ub = zeros(1,nval);
lb(Hp) = 400;
ub(Hp) = 1000;
lb(Bp) = 200;
ub(Bp) = 400;
% lb(twp) = 9;
% ub(twp) = 22;
lb(tfp) = 12;
ub(tfp) = 40;
lb(Dp) = 350;
ub(Dp) = 1000;
lb(tp) = 12;
ub(tp) = 40;
end
% %
% function outputVariable(xo,exitflag,numExecution,iteration)
% global nelx nely nelz lx ly lz pw xe ye span
% global fparameter fsolution fvariable
% global nvg nvc repg repc Hn Bn twn tfn Dn tn Hp Bp twp tfp Dp tp
% nval = tp(end);
% Ho = xo(Hp);
% Bo = xo(Bp);
% two = xo(twp);
% tfo = xo(tfp);
% Do = xo(Dp);
% to = xo(tp);
% switch span
%     case 0
%         fprintf(fparameter,'%d,%d,%d,%d,%d,%d,%f,%d',nelx,nely,nelz,lx,ly,lz,pw,exitflag);
%     case 1
%         fprintf(fparameter,'%d,%d,%d,',nelx,nely,nelz);
%         for i = 2:nelx-1
%             fprintf(fparameter,'%d,',xe(i));
%         end
%         for i = 2:nely-1
%             fprintf(fparameter,'%d,',ye(i));
%         end
%         fprintf(fparameter,'%f',pw);
%     otherwise
% end
% for i = 1:nvg
%     H = Ho(Hn(repg(i)));
%     B = Bo(Bn(repg(i)));
%     tw = two(twn(repg(i)));
%     tf = tfo(tfn(repg(i)));
%     fprintf(fsolution,'%d,%f,%f,%f,%f,',i,H,B,tw,tf);
% end
% for i = 1:nvc
%     D = Do(Dn(repc(i)));
%     t = to(tn(repc(i)));
%     if i == nvc
%         fprintf(fsolution,'%d,%f,%f',i,D,t);
%     else
%         fprintf(fsolution,'%d,%f,%f,',i,D,t);
%     end
% end
% for i = 1:nval
%     if i == nval
%         fprintf(fvariable,'%f',xo(i));
%     else
%         fprintf(fvariable,'%f,',xo(i));
%     end
% end
% if iteration < numExecution
%     fprintf(fparameter,'\n');
%     fprintf(fsolution,'\n');
%     fprintf(fvariable,'\n');
% end
% end
function [repg,repc] = varMember()
global twn Dn nvg nvc
repg = zeros(1,nvg);
repc = zeros(1,nvc);

for i = 1:nvg
    repg(i) = find(twn == i,1);
end
for i = 1:nvc
    repc(i) = find(Dn == i,1);
end

end
%