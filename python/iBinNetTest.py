from iBinNet import Net

if __name__ == '__main__':

    net = Net('dataset/labels.txt', 'dataset', 0.2)

    net.train(2)
    net.validate()