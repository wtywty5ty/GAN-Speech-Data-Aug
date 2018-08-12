import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('train_xent.pkl', 'rb') as f:
    train_xent = pickle.load(f)

with open('test_xent.pkl', 'rb') as f:
    test_xent = pickle.load(f)

with open('cv_xent.pkl', 'rb') as f:
    cv_xent = pickle.load(f)

with open('fake_xent.pkl', 'rb') as f:
    fake_xent = pickle.load(f)
        

labels = ['Training set', 'CV set', 'Test set', 'Generated data']
    
data = []
data.append(train_xent)
data.append(cv_xent)
data.append(test_xent)
data.append(fake_xent)

plt.boxplot(data, labels=labels, showmeans=True, showfliers=False)
plt.ylabel('Cross entropy loss')
plt.savefig('average_xent.png')
plt.close