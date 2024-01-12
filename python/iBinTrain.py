from iBinNet import Net

if __name__ == '__main__':

    net = Net('dataset/labels.txt', 'dataset', 0.2, device='cuda:0')

    net.train(n_epochs=2, save_path=None)
    net.validate()