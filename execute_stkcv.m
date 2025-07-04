function [stress_bx,stress_by,stress_c,cof,deflect]=execute_stkcv(choice_section)

% element of parameters
% Input file %
modelInput = './csv/param_3d.xlsx';
variable = choice_section ;
% variable = readmatrix(variableInput);
model = readmatrix(modelInput);
numExecution = 1;
iteration = 1;
frameType = ['s','']; % x or y or s, i or o or ''
spanType = 0; % 0: ‹Ï“™, 1: •s‹Ï“™

while iteration <= numExecution
% Struct element Parameters variables %
xSpan = model(iteration,1);
ySpan = model(iteration,2);
zSpan = model(iteration,3);
xLength = model(iteration,4);
yLength = model(iteration,5);
storyHeight = model(iteration,6);
weight = model(iteration,7);
xo = rmmissing(variable(iteration,:));
height=model(2,1:zSpan);
elementParameters = struct;
elementParameters.xSpan = xSpan;
elementParameters.ySpan = ySpan;
elementParameters.zSpan = zSpan;
elementParameters.xLength = xLength;
elementParameters.yLength = yLength;
elementParameters.storyHeight = storyHeight;
elementParameters.weight = weight;
elementParameters.numExecution = numExecution;
elementParameters.iteration = iteration;
elementParameters.frameType = frameType;
elementParameters.xo = xo;
elementParameters.spanType = spanType;
elementParameters.height=height;

    [b_ratiox,b_ratioy,c_ratio,cof,deflect_]=optimize_stkcv(elementParameters);
    iteration = iteration+1;
    stress_bx=b_ratiox;
    stress_by=b_ratioy;
    stress_c=c_ratio;
    deflect=max(deflect_(1:zSpan),deflect_(1+zSpan:zSpan*2));

%     elementParameters.iteration = iteration;
end
end

