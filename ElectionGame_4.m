function [Result, LCan, tRun] = ElectionGame_4()

% 1. Initializations
% Baseline Platform Values
%   alpha   Big business
%   beta    Establishment ties
%   gamma   Foreign policy experience
%   delta   Libertarian	issues
%   epsilon Social issues

clear  all

format shortg

Run_Time = 20; % Specify maximum run time for re-opt process

% Baseline Platform Values (BPV) - Determine initial bounds
%     [ alpha |  beta | gamma | delta | epsilon]
BPV = [   0.1     0.8     0.5     0.9     0.6;   % A
          0.5     0.9    -0.1    -0.8    -0.4;   % B
          0.4    -1.0    -0.6     0.3     0.5;   % C
         -0.5    -0.9     0.1     0.8     0.4];  % D

% Voter Group Coefficients (VGC)
%     [ alpha |  beta | gamma | delta | epsilon]
VGC = [  -0.4     1.0     0.6    -0.3    -0.5;   % a
         -1.0     0.4     0.3    -0.6     0.9;   % b
          0.9     0.5    -0.8    -0.1     1.0;   % c
         -0.3     0.6    -1.0     0.4    -0.8;   % d
         -0.6     0.3    -0.4     1.0     0.1;   % e
          0.6    -0.3     0.4    -1.0    -0.1];  % f

disp('Initialized... Starting Proxy Opt')


% 2. Proxy NE as starting point
[Proxy,X] = ProxyNE(BPV,VGC)

% 3. Re-optimize for losing candidates to search for equilibrium
[Result, LCan, tRun]   = ReOptGA(VGC,Proxy,X,Run_Time)

end

%% Compute Proxy Nash Equilibrium
function [Proxy,X] = ProxyNE(BPV,VGC)

options1 =  optimoptions('linprog', ...
                         'Algorithm','dual-simplex', ...
                         'display', 'off');

% Create decision variable coefficients
VGCsum = sum(VGC);
f      = reshape(-repmat(VGCsum,4,1),1,20);

% Bounds
RLB  = max(BPV - 0.5,-1);  % Relative LB of -0.5 from Baseline not to exceed -1
RLBr = reshape(RLB,1,20);  % resize RLB
RUB  = min(BPV + 0.5, 1);  % Relative UB of +0.5 from Baseline not to exceed 1
RUBr = reshape(RUB,1,20);  % resize RUB

% Proxy Nash Equilibrium Optimization
[Policy, FVAL] = linprog(f,[],[],[],[],RLBr,RUBr,[],options1);
FVAL = -FVAL;

X = reshape(Policy,4,5);    % Adjusted Policy Matrix
%X(abs(X)<0.00001) = 0.0;    % Change small values to zero

Infl     = (VGC*X')';       % Voter Influence Matrix
[~, VGs] = max(Infl);       % Tally voter groups won by each candidate

% Results of proxy NE (Number VGs and Influence scores)
PxyVGs  = [sum(VGs==1); sum(VGs==2); sum(VGs==3); sum(VGs==4)];
PxyInfl = sum(Infl,2);
Proxy   = [PxyInfl PxyVGs];

disp('Proxy Opt Compete... Re-opt in progress')
end


%% Use GA to re-optimize for losing candidate
function [ResultRO, LCan, tRun] = ReOptGA(VGC, Proxy, X, MaxRun)

% Initializations
i           = 1;
Policies    = reshape(X',1,20)
Flag        = [0 0 0 0];
tMax        = inf;
options2    = optimoptions('ga', 'display', 'off', 'MaxTime',tMax);
Y           = X;
CurrentInfl = Proxy(:,1);
CurrentVGs  = Proxy(:,2);

LCan(i) = ID_LCan(CurrentVGs,CurrentInfl); % ID losing candidate

tStart      = tic;
while sum(Flag) < 3
    switch LCan
        % Select Candidate
        % Set appropriate Relative LB (RLB) of +0.5 from Baseline not to exceed -1
        % Set appropriate Relative UB (RUB) of +0.5 from Baseline not to exceed +1
        % Resize RLB/RUB into vector RLBr/RUBr
        
        case 1
            RLBr      = Y;
            RLBr(1,:) = max(X(1,:) - 0.5,-1);
            RLBr      = reshape(RLBr,20,1);
            RUBr      = Y;
            RUBr(1,:) = min(X(1,:) + 0.5, 1);
            RUBr      = reshape(RUBr,20,1);
            f = @ReOptVG1;
            
        case 2
            RLBr      = Y;
            RLBr(2,:) = max(X(2,:) - 0.5,-1);
            RLBr      = reshape(RLBr,20,1);
            RUBr      = Y;
            RUBr(2,:) = min(X(2,:) + 0.5, 1);
            RUBr      = reshape(RUBr,20,1);
            f = @ReOptVG2;
            
        case 3
            RLBr      = Y;
            RLBr(3,:) = max(X(3,:) - 0.5,-1);
            RLBr      = reshape(RLBr,20,1);
            RUBr      = Y;
            RUBr(3,:) = min(X(3,:) + 0.5, 1);
            RUBr      = reshape(RUBr,20,1);
            f = @ReOptVG3;
            
        case 4
            RLBr      = Y;
            RLBr(4,:) = max(X(4,:) - 0.5,-1);
            RLBr      = reshape(RLBr,20,1);
            RUBr      = Y;
            RUBr(4,:) = min(X(4,:) + 0.5, 1);
            RUBr      = reshape(RUBr,20,1);
            f = @ReOptVG4;
    end
    
    % GA function
    % X = ga(FITNESSFCN,NVARS,A,b,Aeq,beq,lb,ub,NONLCON,options)
    
    [Z,fval] = ga(f,20,[],[], [], [], RLBr, RUBr,[],options2);
    
    if ~isempty(Z)
        Z = reshape(Z,4,5);
    end
     
    % Voter Influence Matrix
    InflRO = Z*VGC';
    
    % Tally voter groups won by each candidate
    [~, VGsRO] = max(InflRO);
    
    % Results of proxy NE (Number VGs and Influence scores)
    VGsRO     = [sum(VGsRO==1); sum(VGsRO==2); sum(VGsRO==3); sum(VGsRO==4)];
    InflROsum = sum(InflRO,2);
    ResultRO  = [Z InflROsum VGsRO];
    Flag(LCan) = 1;
    
    LCan = ID_LCan(VGsRO,InflROsum); % ID losing candidate
    tRun = toc(tStart);
    
    % Terminate if runtime exceeded
    if tRun > MaxRun
        Flag = [1 1 1 1]
        break
    end
    
    i = i + 1;
end
end

%% Determine Losing Candidate
function LCan = ID_LCan(CurrentVGs,CurrentInfl)
    MinVG = min(CurrentVGs);                % ID least # VGs won by any candidate
    LCans = (CurrentVGs==MinVG);            % ID candidates with least VGs

    % ID candidate with lowest influence score to break ties
        LVGs          = LCans.*CurrentInfl; % Set vector with only scores of losing candidates
        LVGs(LVGs==0) = 9;                  % Set zeros to a sufficiently large number
        [~, LCan] = min(LVGs);              % Find losing candidate with lowest score
end

%
%
%% END