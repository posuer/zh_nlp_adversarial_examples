from models import NLImodel

model = NLImodel(is_train=True)

'''
Load traind model and predict
model = NLImodel(is_train=False)
model.predict(test_X, test_Y) #test_X, test_Y should be: len(test_X) >= 2 and len(test_Y) >= 2

'''
