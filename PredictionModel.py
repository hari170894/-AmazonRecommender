from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.evaluate import GridSearch
from surprise.dataset import Reader
import os
from surprise.accuracy import rmse
from surprise.accuracy import mae
from surprise.dump import dump

train_file_path = os.path.expanduser('train.csv')
test_file_path = os.path.expanduser('test.csv')

reader = Reader(line_format='user item rating timestamp', sep=',')

traindata = Dataset.load_from_file(train_file_path, reader=reader)
testdata = Dataset.load_from_file(test_file_path,reader=reader)
traindata.split(n_folds=5)

param_grid = {'n_epochs': [1, 5, 10, 20], 'lr_all': [0.010,0.1,0.5,1],
              'reg_all': [0.1, 1 , 10]}
algo = SVD()
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])
grid_search.evaluate(traindata)

print(grid_search.best_score['RMSE'])
best_model=grid_search.best_params['RMSE']
print best_model

finalModel =SVD(n_epochs=best_model['n_epochs'],lr_all=best_model['lr_all'],reg_all=best_model['reg_all'])
data = Dataset.load_from_folds([('train.csv','test.csv')], reader=reader)
for trainset, testset in data.folds():
	finalModel.train(trainset)
	predictions=finalModel.test(testset)
	print 'TEST RMSE',rmse(predictions)
	print 'TEST MAE',mae(predictions)
	dump('./dump_SVD', predictions, trainset, finalModel)

