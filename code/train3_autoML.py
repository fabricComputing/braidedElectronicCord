from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
def autoML(address1,address2):
    train = pd.read_csv(address1)
    test = pd.read_csv(address2)

    predictor = TabularPredictor(label="type",problem_type='multiclass').fit(train)
    leaderboard = predictor.leaderboard(test)

if __name__ == '__main__':
    address1 = "" # train dataset
    address2 = "" # test dataset
    autoML(address1,address2)