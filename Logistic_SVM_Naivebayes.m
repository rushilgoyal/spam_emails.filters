
    %% Calculating mean and Standard Deviation
    train_mean = mean(train_data');
    train_std = std(train_data');

    %% Normalizing Data
    train_mean_mat = repmat(train_mean,size(train_data,2),1);
    train_std_mat = repmat(train_std,size(train_data,2),1);

    test_mean_mat = repmat(train_mean,size(test_data,2),1);
    test_std_mat = repmat(train_std,size(test_data,2),1);

    norm_train_data = rdivide(train_data - train_mean_mat',train_std_mat');
    norm_test_data = rdivide(test_data - test_mean_mat',test_std_mat');

    %Calculating std and mean of the normalized training set to cross-check the normalization process
    a = std(norm_train_data);
    b = mean(norm_train_data);

    %Using a different function ìbsxfunî to subtract the mean: just an additional command
    C= bsxfun (@minus, train_data_transpose, mean(train_data_transpose));

    %Question 2.3
    %%Applying logistic regression model to the training dataset
    logistic_regression = mnrfit(norm_train_data',ytrain');
    prob = mnrval(logistic_regression,norm_train_data');

    %creating a third column based on the condition: If prob(spam)> prob(not spam) , then 1 else 2

    %%populating and tabulating results
    for k=1:size(prob,1)
        if(prob(k,1)>prob(k,2))
            prob(k,3)=1
        else
            prob(k,3)=2
        end
    end
    results_regression=prob(:,3);
    tabulate(results_regression);

      Value    Count   Percent
          1     1911     62.35%
          2     1154     37.65%
    %Here 2 represents non-spam emails, so the percentage of emails that are
    %not spam according to our model is 37.65%

    %%Creating a Confusion Matrix
    confusion_mat=confusionmat(ytrain,results_regression);
    % 1069	149
    % 85	1762

    %% Calculating training accuracy

    training_accuracy = trace(confusionmat) ./sum(confusionmat(:))
    training_accuracy =

        0.9237

    %% Evaluating test results and tabulating them
    test_prob = mnrval(logistic_regression,norm_test_data');
    for k=1:size(test_prob,1)
        if(test_prob(k,1)>test_prob(k,2))
            test_prob(k,3)=1
        else
            test_prob(k,3)=2
        end
    end
    test_results_regression=test_prob(:,3);
    tabulate(test_results_regression);
      Value    Count   Percent
          1      569     37.04%
          2      967     62.96%


    %%Creating a Confusion Matrix
    test_confusion_mat=confusionmat(ytest,test_results_regression);
    % 519	76
    % 50	891

    %%Calculating Accuracy for test data
    test_accuracy = trace(test_confusion_mat) ./sum(test_confusion_mat(:))

    test_accuracy =

        0.9180



    %% 2.5 NaiveBayes - creating model from training data
    O1= fitNaiveBayes(norm_train_data,ytrain_transpose);
    C1= O1.predict(norm_train_data);
    cMAT1 = confusionmat(ytrain_transpose,C1);   
    cMAT1 =
    1155         63
    476        1371
    
 train_accuracy_NB = trace(cMAT1) ./sum(cMAT1(:));
    
train_accuracy_NB =

    0.8241


    %Running the learnt NaÔve-Bayes model on normalized test data

    C1_test= O1.predict(norm_test_data);
    cMAT1_test = confusionmat(ytest_transpose,C1_test)

    cMAT1_test =

       566    29
       256   685

    test_accuracy_NB = trace(cMAT1_test) ./sum(cMAT1_test(:))

    %test_accuracy_NB =

     %   0.8145

    %% Question 2.6 SVM

    %% training svm model
    smoopt=svmsmoset('Maxiter',50000);
    svm=svmtrain(norm_train_data,ytrain_transpose,'autoscale','false','Method','SMO','SMO_Opts',smoopt);
    svm_train_prediction= svmclassify(svm,norm_train_data);
    svm_confusion_matrix = confusionmat(ytrain_transpose,svm_train_prediction)


    svm_confusion_matrix =

            1218           0
            1661         186

    train_accuracy_svm = trace(svm_confusion_matrix) ./sum(svm_confusion_matrix(:));
    %train_accuracy_svm =

     %  0.4581

    svm_test_prediction= svmclassify(svm,norm_test_data);

    svm_confusion_matrix = confusionmat(ytest',svm_test_prediction)

    svm_confusion_matrix =

       592     3
       853    88

    test_accuracy_svm = trace(svm_confusion_matrix) ./sum(svm_confusion_matrix(:))

    %test_accuracy_svm =

    % 0.4427







