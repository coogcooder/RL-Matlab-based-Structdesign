
function [xG,xC]=output(xSpan,ySpan,xLength,yLength,zSpan,frame,vecFrame)
%  if max(size(gcp)) == 0 % parallel pool needed
%     parpool % create the parallel pool
%  end
 storyHeight = 4000; 

 modelInput = './csv/param_3d.xlsx';
 model = readmatrix(modelInput);

 numExecution = 1;
 perExecution = 1;
 iteration = 1;
 spanType = 0; % 0: ‹Ï“™, 1: •s‹Ï“™
 frameType = [vecFrame,string(frame)]; % x or y or s, i or o or ''
 weight = model(iteration,7);
 height=model(2,1:zSpan);
 % Struct element Parameters variables %
 elementParameters = struct;
 elementParameters.xSpan = xSpan;
 elementParameters.ySpan = ySpan;
 elementParameters.zSpan = zSpan;
 elementParameters.xLength = xLength;
 elementParameters.yLength = yLength;
 elementParameters.storyHeight = storyHeight;
 elementParameters.weight = weight;
 elementParameters.height=height;
 elementParameters.numExecution = numExecution;
 elementParameters.perExecution = perExecution;
 elementParameters.iteration = iteration;
 elementParameters.spanType = spanType;
 elementParameters.frameType = frameType;
 
 if vecFrame=='x'
  ng=zSpan*(xSpan/2);
 else
  ng=zSpan*(ySpan/2); 
 end

 while iteration <= numExecution

     [xo]=optimize(elementParameters);
     xG=xo(1:4*ng);
     xC=reshape(reshape(xo(4*ng+1:end),[],2).',[],1);
%      disp("xG")
%      disp(xG)
%      disp("xC")
%      disp(xC)     
     iteration = iteration+1;
     elementParameters.iteration = iteration;

 end

