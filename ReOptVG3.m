function NumVGs = ReOptVG3(XPolicy)

VGC = [ -0.4  1.0  0.6 -0.3 -0.5;   % a
        -1.0  0.4  0.3 -0.6  0.9;   % b
         0.9  0.5 -0.8 -0.1  1.0;   % c
        -0.3  0.6 -1.0  0.4 -0.8;   % d
        -0.6  0.3 -0.4  1.0  0.1;   % e
         0.6 -0.3  0.4 -1.0 -0.1];  % f
% Adjusted Policy Matrix
X = reshape(XPolicy,4,5);

% Voter Influence Matrix
Infl = X*VGC';
[~, VGs] = max(Infl);
NumVGs = -sum(VGs==3);
end