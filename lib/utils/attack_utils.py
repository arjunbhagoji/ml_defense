import numpy as np

def class_means(X,y):
    classes=np.unique(y)
    no_of_classes=len(classes)
    means=[]
    for item in classes:
        indices=np.where(y==item)[0]
        class_items=X[indices,:]
        mean=np.mean(class_items,axis=0)
        means.append(mean)
    return means

def length_scales(X,y):
    means=class_means(X, y)
    no_of_classes=len(means)
    scales=[]
    for i in range(no_of_classes):
        curr_mean=means[i]
        curr_scales=[]
        for j in range(no_of_classes):
            if i==j:
                continue
            elif i!=j:
                mean_diff=curr_mean-means[j]
                curr_scales.append(np.linalg.norm(mean_diff))
        scales.append(np.amin(curr_scales))
    return scales

def naive_untargeted_attack(X,y):
    scales=length_scales(X, y)
    print scales
    data_len=len(X)
    classes=np.unique(y)
    distances=[]
    for i in range(100):
        curr_data=X[i,:]
        curr_distances=[]
        for j in range(100):
            if i==j:
                continue
            elif i!=j:
                # if y[i]==y[j]:
                #     continue
                if y[i]!=y[j]:
                    data_diff=curr_data-X[j,:]
                    data_dist=np.linalg.norm(data_diff)
                    print data_dist
                    curr_distances.append(data_dist/scales[y[i]])
        distances.append(min(curr_distances))
    return distances
