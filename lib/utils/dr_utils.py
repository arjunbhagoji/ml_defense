from sklearn.decomposition import PCA

def pca_dr(X_train,X_test,rd):
    train_len=len(X_train)
    test_len=len(X_test)
    #Reshaping for PCA function
    PCA_in_train=X_train.reshape(train_len,784)
    PCA_in_test=X_test.reshape(test_len,784)
    #Fitting the PCA model on training data
    pca=PCA(n_components=rd,random_state=10)
    pca_train=pca.fit(PCA_in_train)
    # Reconstructing training and test data
    X_train_dr=pca.transform(PCA_in_train)
    X_test_dr=pca.transform(PCA_in_test)

    X_train_dr=X_train_dr.reshape((train_len,1,rd))
    X_test_dr=X_test_dr.reshape((test_len,1,rd))
    return X_train_dr,X_test_dr,pca
