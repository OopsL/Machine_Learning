function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_new = theta;
n = length(theta);

for iter = 1:num_iters
for j = 1 : n
theta_new(j) = theta(j) - (alpha / m) * sum((X * theta - y) .* X(:, j));
end
theta = theta_new;

% Save the cost J in every iteration
J_history(iter) = computeCost(X, y, theta);

end

end
