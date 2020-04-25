
from trainTest.train_test_1 import train_v1,test_v1
from trainTest.train_test_2 import train_v2,test_v2
from trainTest.train_test_3 import train_v3,test_v3
from trainTest.train_test_4 import train_v4,test_v4
from trainTest.train_test_5 import train_v5,test_v5
from trainTest.train_test_6 import train_v6,test_v6
from exceptions.exceptions import InValidTestTrainMethod

def get_train_test_methods(train_test_version):

    #Normal model
    if train_test_version == 1:
        print("Train/Test Version 1 for normal models")
        return train_v1, test_v1

    #---- Multitask learning --- 
    #Not used. Use model 6 and change multiplications on loss
    elif train_test_version == 2:
        print("Train/Test Version 2 for BOXCARS (Auxillary Multitask learning - 2 features) ")
        return train_v2, test_v2
    # Not used. Use model 6 and change multiplications on loss

    elif train_test_version == 3: 
        print("Train/Test Version 3 for STANFORD (Multitask learning)")
        return train_v3, test_v3
    elif train_test_version == 4:
        print("Train/Test Version 4 for BOXCARS (Multitask learning - 3 features)")
        return train_v4, test_v4
    elif train_test_version == 5:
        print("Train/Test Version 5 for BOXCARS (Multitask learning - 4 features)")
        return train_v5, test_v5
    elif train_test_version == 6:
        print("Train/Test Version 6 for BOXCARS (Auxillary learning - 4 features")
        return train_v6, test_v6
    else:

        raise InValidTestTrainMethod(str(train_test_version) + "is not a valid trainTest method")