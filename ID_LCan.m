%% Determine Losing Candidate
function LCan = ID_LCan(CurrentVGs,CurrentInfl)
    MinVG = min(CurrentVGs);                % ID least # VGs won by any candidate
    LCans = (CurrentVGs==MinVG);            % ID candidates with least VGs

    % ID candidate with lowest influence score to break ties
        LVGs          = LCans.*CurrentInfl; % Set vector with only scores of losing candidates
        LVGs(LVGs==0) = 9;                  % Set zeros to a sufficiently large number
        [~, LCan] = min(LVGs);              % Find losing candidate with lowest score
end
