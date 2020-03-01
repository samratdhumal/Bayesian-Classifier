%The goal of this project is to construct a classifier 
%such that for any given values of ?1 and ?2, 
%it can predict the performed task (?1, ?2, C3, C4, ?5). 

%Authors:-
%Anchal Sojatiya – 50321238 – (anchalso@buffalo.edu)
%Samratsinh Dhumal – 50321053 – (samratsi@buffalo.edu)
%School of Engineering and Applied Sciences, SUNY Buffalo
%Buffalo, NY


clear all
clc

% Loading the given dataset
load('data.mat')

% splitting f1 to train and test, finding mean and variance
data_train_f1 = F1(1:100,:);
data_test_f1 = F1(101:1000,:);
data_mean(1,:) = round(mean(data_train_f1,1),3);
data_variance(1,:) = round(var(data_train_f1,1),3);

% splitting f2 to train and test, finding mean and variance
data_train_f2 = F2(1:100,:);
data_test_f2 = F2(101:1000,:);
data_mean(2,:) = round(mean(data_train_f2,1),3);
data_variance(2,:) = round(var(data_train_f2,1),3);

% Calculating Z1
for i = 1:1000
Z1(i,:) = zscore(F1(i,:));
Z2(i,:) = zscore(F2(i,:));
end

% splitting z1 to train and test, finding mean and variance
data_train_z1 = Z1(1:100,:);
data_test_z1 = Z1(101:1000,:);
data_mean_z1(1,:) = round(mean(data_train_z1,1),3);
data_variance_z1(1,:) = round(var(data_train_z1,1),3);

% initializing comparision matrix to make comparision with predicition
comparision_matrix = zeros(900, 5);
for val=1:5
    comparision_matrix(:,val) = val;
end

% X = F1
[class_accuracy_case1,error_rate_case1] = classifier(data_test_f1,data_mean(1,:),data_variance(1,:),comparision_matrix);
% X = Z1
[class_accuracy_case2,error_rate_case2] = classifier(data_test_z1,data_mean_z1(1,:),data_variance_z1(1,:),comparision_matrix);
% X= F2
[class_accuracy_case3,error_rate_case3] = classifier(data_test_f2,data_mean(2,:),data_variance(2,:),comparision_matrix);

% X = [Z1 F2]
mean_z1=data_mean_z1(1,:);
var_z1=data_variance_z1(1,:);
mean_z2=data_mean(2,:);
var_z2=data_variance(2,:);
for i = 1:length(data_test_z1(:,1))
  for j = 1:length(mean_z1(1,:))
      for k = 1:length(mean_z1(1,:))
        z_1(j,k) = (data_test_z1(i,j) - mean_z1(1,k))/(sqrt(var_z1(1,k)));
        z_2(j,k) = (data_test_f2(i,j) - mean_z2(1,k))/(sqrt(var_z2(1,k)));
      end
      p_1 = normpdf(z_1(j,:));
      p_2 = normpdf(z_2(j,:));
      [~,pred(i,j)] = max(p_1.*p_2);
  end
end
[class_accuracy_case4,error_rate_case4] = accuracy_calculator(comparision_matrix,pred);

% Plots
generate_plot(Z1,F2,"Z1 vs F2")
generate_plot(F1,F2,"F1 vs F2")
generate_plot(Z1,Z2,"Z1 vs Z2")

figure;
bar_label = categorical({'X = F1','X = Z1','X = F2','X = [Z1 F2]'});
bar_label = reordercats(bar_label,{'X = F1','X = Z1','X = F2','X = [Z1 F2]'});
data_bar=[class_accuracy_case1,error_rate_case1; class_accuracy_case2,error_rate_case2; class_accuracy_case3,error_rate_case3; class_accuracy_case4,error_rate_case4];
bar(bar_label,data_bar,0.4,'stacked');
hold on;
title("Accuracy and Error Rate");
xlabel("Cases");
ylabel("Accuracy and Error Rate");
legend('Accuracy','Error');
hold off;

%%% Functions

% Classifies values and returns error and accuracy 
function [class_accuracy,error_rate] = classifier(data_input,data_mean,data_variance,comparision_matrix)
for i = 1:length(data_input(:,1))
  for j = 1:length(data_mean(1,:))
      for k = 1:length(data_mean(1,:))
        z(j,k) = (data_input(i,j) - data_mean(1,k))/(sqrt(data_variance(1,k)));
      end
      [~,predicted_mat(i,j)] = max(normpdf(z(j,:))/5); 
  end
end
[class_accuracy,error_rate] = accuracy_calculator(comparision_matrix,predicted_mat);
end

% Calculates accuracy and error rate of classifier
function [class_acc,err_rate] = accuracy_calculator(comparision_matrix,prediction)
error_matrix = comparision_matrix - round(prediction,3);
index=error_matrix==0;
error=sum(index(:));
class_acc = round((error/4500)*100,3);
err_rate = 100-class_acc;
end

% Plotting scatter plots
function generate_plot(X,Y,title_name)
figure;
hold on
for i = 1:5
scatter(X(:,i),Y(:,i),'.')
hold on
end
title(sprintf('Scatterplot %s',title_name))
xlabel(X)
ylabel(Y)
legend('C1','C2','C3','C4','C5')
hold off
end