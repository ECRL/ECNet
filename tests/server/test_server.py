import unittest
from os.path import exists, isdir, join
from os import remove
from shutil import rmtree

from ecnet import Server
from ecnet.utils.server_utils import default_config
from ecnet.utils.logging import logger
from ecnet.utils.data_utils import DataFrame, PackagedData


DB_LOC = 'cn_model_v1.0.csv'


class TestServer(unittest.TestCase):

    def test_init(self):

        print('\nUNIT TEST: Server init')
        sv = Server()
        self.assertTrue(exists('config.yml'))
        self.assertEqual(sv._vars, default_config())
        remove('config.yml')

    def test_load_data(self):

        print('\nUNIT TEST: Server.load_data')
        sv = Server()
        sv.load_data(DB_LOC)
        self.assertEqual(len(sv._df), 482)
        self.assertEqual(type(sv._sets), PackagedData)
        remove('config.yml')

    def test_create_project(self):

        print('\nUNIT TEST: Server.create_project')
        sv = Server()
        sv.create_project('test_project', 3, 5)
        for pool in range(3):
            for candidate in range(5):
                self.assertTrue(isdir(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate)
                )))
        remove('config.yml')
        rmtree('test_project')

    def test_train_project(self):

        print('\nUNIT TEST: Server.train')
        sv = Server()
        sv.load_data(DB_LOC, random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        for pool in range(2):
            self.assertTrue(exists(join(
                'test_project',
                'pool_{}'.format(pool),
                'model.h5'
            )))
            for candidate in range(2):
                self.assertTrue(exists(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate),
                    'model.h5'
                )))
        remove('config.yml')
        rmtree('test_project')

    def test_use_project(self):

        print('\nUNIT TEST: Server.use')
        sv = Server()
        sv.load_data(DB_LOC, random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        results = sv.use()
        self.assertEqual(len(results), len(sv._df))
        remove('config.yml')
        rmtree('test_project')

    def test_save_project(self):

        print('\nUNIT TEST: Server.save_project')
        sv = Server()
        sv.load_data(DB_LOC, random=True, split=[0.7, 0.2, 0.1])
        sv.create_project('test_project', 2, 2)
        sv._vars['epochs'] = 100
        sv.train()
        sv.save_project()
        self.assertTrue(exists('test_project.prj'))
        self.assertTrue(not isdir('test_project'))
        remove('test_project.prj')
        remove('config.yml')

    def test_multiprocessing_train(self):

        print('\nUNIT TEST: multiprocessing training')
        sv = Server(num_processes=8)
        sv.load_data(DB_LOC)
        sv.create_project('test_project', 2, 4)
        sv._vars['epochs'] = 100
        sv.train()
        for pool in range(2):
            self.assertTrue(exists(join(
                'test_project',
                'pool_{}'.format(pool),
                'model.h5'
            )))
            for candidate in range(4):
                self.assertTrue(exists(join(
                    'test_project',
                    'pool_{}'.format(pool),
                    'candidate_{}'.format(candidate),
                    'model.h5'
                )))
        remove('config.yml')
        rmtree('test_project')

    def test_transform(self):

        print('\nUNIT TEST: PCA transformation')
        sv = Server()
        sv.load_data(DB_LOC, normalize=True, random=True,
                     split=[0.7, 0.2, 0.1], transform=True)
        for inp in sv._df._input_names:
            self.assertTrue('PC' in inp)
        n_inp = len(sv._df._input_names)
        self.assertEqual(n_inp, len(sv._sets.learn_x[0]))
        self.assertEqual(n_inp, len(sv._sets.valid_x[0]))
        self.assertEqual(n_inp, len(sv._sets.test_x[0]))


if __name__ == '__main__':

    DB_LOC = join('../', DB_LOC)
    unittest.main()
