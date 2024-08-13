best_acc = 1
best_acc_loc = 2
with open('log.txt','a') as f:
        f.write('nr:{}-base:{}-epoch:{}\n'.format(1,best_acc,best_acc_loc))