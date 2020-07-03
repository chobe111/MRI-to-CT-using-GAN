from dataset import BrainM2C
from model import MriGAN


def main():
    model = MriGAN(batch_size=32, is_train=True)
    dataset = BrainM2C()

    print("start train model")
    model.train(dataset, iter_num=10000)

    print("train model end")


if __name__ == '__main__':
    main()
