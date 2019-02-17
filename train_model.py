from NLImodel.models import NLImodelClass
'''
model = NLImodel(is_train=True)

'''
#Load traind model and predict
model = NLImodelClass(is_train=False)
print(model.train_X[0:2], model.train_Y[0:2], model.train_Z[0:2])
result = model.predict(model.train_X[0:2], model.train_Y[0:2]) 
print(result)

