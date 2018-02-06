# encoding: utf-8
import numpy as np
def batch_iterator(X, y=None, batch_size=64):

    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def accuracy_score(y_true, y_pred):

    pred = np.array( (y_pred>0.5) )
    accuracy = np.mean(y_true == pred, axis=0)
    return accuracy

def accuracy(y_true,y_pred):
    pred = (y_pred > 0.5)

    tp=0
    tn=0
    fp=0
    fn=0
    if len(y_pred.shape)>1:
        if y_pred.shape[1]>1:



            for i in range(len(pred)):

                if y_true[i][0]==pred[i][0] and y_true[i][0] == 1:
                    tp+=1

                if y_true[i][0]==pred[i][0] and y_true[i][0] == 0:
                    tn+=1

                if y_true[i][0]!=pred[i][0] and y_true[i][0] == 0:
                    fp+=1

                if y_true[i][0]!=pred[i][0] and y_true[i][0] == 1:
                    fn+=1
        else:
            for i in range(len(pred)):

                if y_true[i] == pred[i] and y_true[i] == 1:
                    tp += 1

                if y_true[i] == pred[i][0] and y_true[i] == 0:
                    tn += 1

                if y_true[i] != pred[i][0] and y_true[i] == 0:
                    fp += 1

                if y_true[i] != pred[i] and y_true[i] == 1:
                    fn += 1
    else:

        for i in range(len(pred)):

            if y_true[i] == pred[i] and y_true[i] == 1:
                tp += 1

            if y_true[i] == pred[i][0] and y_true[i] == 0:
                tn += 1

            if y_true[i] != pred[i][0] and y_true[i] == 0:
                fp += 1

            if y_true[i] != pred[i] and y_true[i] == 1:
                fn += 1



    res = (tp+tn)/(tp+tn+fp+fn)
    return res


def acc_test(Y_true,y_pred):

    check = (y_pred>0.5)

    g=0
    b=0


    if len(Y_true.shape)>1:
        if Y_true.shape[1]>1:
            for i in range(len(Y_true)):
                if Y_true[i][0] == check[i][0]:
                    g += 1
                else:
                    b += 1
        else:
            for i in range(len(Y_true)):
                if Y_true[i] == check[i]:
                    g += 1
                else:
                    b += 1

    else:

        for i in range(len(Y_true)):
            if Y_true[i]==check[i]:
                g+=1
            else:
                b+=1

    print('true: {0}'.format(g))
    print('false: {0}'.format(b))
