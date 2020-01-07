

ep = 1
train_loader = [1,2,3,4]
loss_meter = 5.444444
acc_meter = 12345
elapsed = 293.22
i = 2
print(f'Epoch {ep:03d} [{i}/{len(train_loader)}]: '
              f'Loss: {loss_meter / i:.4f} '
              f'Acc: {acc_meter / i:.4f} ({elapsed:.2f}s)', end='\r')