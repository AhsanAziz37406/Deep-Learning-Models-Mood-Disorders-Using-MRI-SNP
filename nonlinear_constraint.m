function [c, ceq] = nonlinear_constraint(x, numFeatures)
    % Example nonlinear constraint function
    % Define constraints here, such as ensuring a certain number of features are selected
    c = [];
    ceq = numFeatures - sum(x);  % Constraint: Select exactly numFeatures
end
