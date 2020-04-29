
from train_test.train_test_1 import train_v1,test_v1
from train_test.train_test_3 import train_v3,test_v3
from train_test.train_test_5 import train_v5,test_v5,test_v5_concat_multtlabels
from train_test.train_test_6 import train_v6,test_v6
from exceptions.exceptions import InValidTestTrainMethod

def get_train_test_methods(train_test_version):

    #Normal model
    if train_test_version == 1:
        print("Train/Test Version 1 for normal models")
        return train_v1, test_v1
    elif train_test_version == 3: 
        print("Train/Test Version 3 for STANFORD (Multitask learning)")
        return train_v3, test_v3
    elif train_test_version == 5:
        print("Train/Test Version 5 for BOXCARS (Multitask learning - 4 features)")
        return train_v5, test_v5
    elif train_test_version == 6:
        print("Train/Test Version 6 for BOXCARS (Auxillary learning - 4 features")
        return train_v6, test_v6
    elif train_test_version == 7:
        print("Train/Test Version 7 but really 4 for BOXCARS (Multitask learning - 4 Features) Will concateinate the multitask to get overall accuracy. Only use on testing")
        return test_v5_concat_multtlabels,test_v5_concat_multtlabels

    else:

        raise InValidTestTrainMethod(str(train_test_version) + "is not a valid trainTest method")