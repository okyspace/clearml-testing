import argparse

from clearml import Task
task = Task.init(project_name="test", task_name="hello")

print("hello")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--symbol', help='symbol used for regression', default='AAPL')


parameters = {}
parameters = task.connect_configuration(
                     configuration=parameters, 
                     name='regressor selection',
                     description='set which regressor to run')
