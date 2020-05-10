function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
pos_vals = [0.01;0;03;0.1;0.3;1;3;10;30];
c_sigma_error = [];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


for counter_c = 1:length(pos_vals)
   for counter_sigma = 1:length(pos_vals)
        temp_c = pos_vals(counter_c);
        temp_sigma = pos_vals(counter_sigma);
        model = svmTrain(X, y, temp_c, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval)); 
        c_sigma_error = [c_sigma_error;temp_c, temp_sigma, error];
   end
end


[M,I] = min(c_sigma_error);
min_location = I(3);
C = c_sigma_error(min_location,1);
sigma = c_sigma_error(min_location,2);


% =========================================================================

end
