import pandas as pd

test = {
    'train_loss': 1,
    'train_acc': 2,
    'train_time': 3,
}

# res = pd.DataFrame([test])
# res.to_csv('my_csv.csv', mode='a')


#Add in headers to CSV
df = pd.DataFrame(columns=['train_loss','train_acc','train_time','val_loss','val_acc','val_time'])
df.to_csv('my_csv.csv')


res = pd.DataFrame([test])
res.to_csv('my_csv.csv', mode='a')