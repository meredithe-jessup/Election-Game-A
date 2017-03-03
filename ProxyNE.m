%% Compute Proxy Nash Equilibrium
function [Proxy, Infl, X] = ProxyNE(BPV,VGC)

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
Infl     = (VGC*X')';       % Voter Influence Matrix
[~, VGs] = max(Infl);       % Tally voter groups won by each candidate

% Results of proxy NE (Number VGs and Influence scores)
PxyVGs  = [sum(VGs==1); sum(VGs==2); sum(VGs==3); sum(VGs==4)];
PxyInfl = sum(Infl,2);
Proxy   = [PxyInfl PxyVGs];

disp('Proxy Opt Complete... Re-opt in progress')
end

