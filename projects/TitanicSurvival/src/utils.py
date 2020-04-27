from pprint import PrettyPrinter 

__pp = PrettyPrinter().pprint

def print_params(estimator):
    __pp(list(estimator.get_params()))



