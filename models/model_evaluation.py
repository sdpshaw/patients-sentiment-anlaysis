from sklearn.metrics import confusion_matrix, classification_report


# Classification report
def report(y_pred, y_test, emotions):
    print("----Confusion Matrix----\n")
    print(confusion_matrix(y_test, y_pred, labels=emotions))
    print('\n----CLASSIFICATION METRICS----\n')
    print(classification_report(y_test, y_pred,
                                target_names=emotions))
