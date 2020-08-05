import torch


__all__ = ['LinearRegionCount']

class LinearRegionCount(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.ActPattern = set()
        self.n_LR = 0


    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        for i in range(n_batch):
            code_string = ''
            for j in range(n_neuron):
                if activations[i][j] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
            self.ActPattern.add(code_string)


    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        self.ActPattern.add(code_string)

    def getLinearReginCount(self):
        return len(self.ActPattern)


if __name__ == '__main__':
   counter = LinearRegionCount()
   a_2D=torch.randn(100,7)
   counter.update2D(a_2D)
   print(counter.getLinearReginCount())

#   for i in range(100):
#        a=torch.randn(3)
#        b=torch.randn(4)
#        list=dict()
#        list.setdefault('a', a)
#        list.setdefault('b', b)
#        counter.update1D(list)
#        print(counter.getLinearReginCount())
