function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

tempTheta=theta;
tempTheta(1)=0;
h = X * theta;
J = (((h-y)'*(h-y))+lambda*(tempTheta'*tempTheta))/(2*m);


grad(1)=1/m * X(:,1)'*(h-y);
%grad(2)=1/m * X(:,1)'*(h-y) + lambda/m *theta(2);
%for i=2:size(theta,1)
%for i=1:m 
%grad(2) = grad(2) + (h(i)-y(i))*X(i,2) + lambda/m * theta(2);
%end
%grad(2)=grad(2)/m;
%end
for j=2:size(X,2)
for i=1:m
grad(j) = grad(j) + ((h(i)-y(i))*X(i,j))/m;
end
grad(j) = grad(j) + (lambda/m)*theta(j); 
end











% =========================================================================

grad = grad(:);

end
