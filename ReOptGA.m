%% Use GA to re-optimize for losing candidate
function [Policies,Influence, VGWins, VGScores, LCan, tRun] ...
    = ReOptGA(VGC,Proxy, Infl, X, MaxRun)

% Initializations
i           = 1;
Policies    = reshape(X',1,20);      % A~1:5, B~6:10, C~11:15, D~16:20
Flag        = [0 0 0 0];
tMax        = inf;
options2    = optimoptions('ga', 'display', 'off', 'MaxTime',tMax);
Y           = X;
CurrentInfl = Proxy(:,1);
CurrentVGs  = Proxy(:,2);

Influence   = reshape(Infl',1,24);
VGWins      = CurrentVGs';
VGScores    = CurrentInfl';

times       = 0;

LCan(i) = ID_LCan(CurrentVGs,CurrentInfl); % ID losing candidate

tStart0      = tic;
while sum(Flag) < 4
    tic;
    switch LCan(i)
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
    i = i + 1;
    
    % GA function
    [Z,fval]      = ga(f,20,[],[], [], [], RLBr, RUBr,[],options2);
    
    % Error check for empty vector
    if isempty(Z)
        disp('Terminated: GA returned empty policy')
        break
    end
    
    Policies(i,:) = Z;
    Z = reshape(Z,4,5);
    
    % Voter Influence Matrix
    InflRO         = Z*VGC';
    Influence(i,:) = reshape(InflRO',1,24);
    
    % Tally voter groups won by each candidate
    [~, VGsRO]   = max(InflRO);
    
    % Results of proxy NE (Number VGs and Influence scores)
    VGsRO       = [sum(VGsRO==1);sum(VGsRO==2);sum(VGsRO==3);sum(VGsRO==4)];
    InflROsum   = sum(InflRO,2);
    %ResultRO    = [Z InflROsum VGsRO];
    Flag(LCan)  = 1;
    
    VGScores(i,:) = InflROsum';
    VGWins(i,:)   = VGsRO';
    LCan(i)       = ID_LCan(VGsRO,InflROsum); % ID losing candidate
    tRun(i)       = toc;
    
    % Terminate if runtime exceeded
    if toc(tStart0) > MaxRun
        Flag = [1 1 1 1];
        disp('Terminated: Max run time exceeded')
        break
    end
end
end
