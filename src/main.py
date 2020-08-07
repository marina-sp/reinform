import logging as log
import numpy as np
from Data import Data_loader
from Trainer import Trainer
from Agent import Agent
from Option import read_options

def main(option):
    log.basicConfig(level=log.INFO)
    trainer = Trainer()
    trainer.setup(option)

    if option.load_model:
        trainer.load_model()
        option.load_model = ''
    if option.train_batch != 0:
        trainer.train()
        trainer.save_model()
    if option.do_test:
        #trainer.load_model()
        trainer.test(data='valid')
        trainer.test(data='test')
    else:
        #trainer.load_model()
        trainer.test(data='train', short=100)
        trainer.test(data='valid', short=100)
        trainer.test(data='test', short=100)

if __name__ == "__main__":
    #torch.set_printoptions(threshold=100000)
    option = read_options()
    main(option)



