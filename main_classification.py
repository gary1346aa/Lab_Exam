import numpy as np
import sys
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def OneHot(y):
    print(y)
    return np.eye(10, dtype = np.float32)[y]

def Accuracy(y,y_):
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
    return np.equal(y_digit, y_digit_).sum() / len(y_digit)

if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Data Processing
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    y_train = OneHot(y_train)
    y_test = OneHot(y_test)

    # Create NN Model
    if len(sys.argv) != 2:
        raise ValueError("Didn't Enter argument for NN selection")
    if sys.argv[1] == 'deep':
        nn = NeuralNetwork.Deep_NN(784, 204, 202, 10, "softmax")
    elif sys.argv[1] == 'wide':
        nn = NeuralNetwork.Wide_NN(784, 256, 10, "softmax")
    else:
        raise ValueError("Enter 'deep' for deep neural network, 'wide' for wide neural network")
    
    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        # Sample Data Batch)
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]
        y_batch = y_train[batch_id]

        # Forward & Backward & Update)
        nn.feed({"x": x_batch, "y": y_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-2)

        # Loss
        loss = nn.computeLoss()
        loss_rec.append(loss)

        # todo (Evaluation
        batch_id = np.random.choice(x_test.shape[0], batch_size)
        x_test_batch = x_test[batch_id]
        y_test_batch = y_test[batch_id]
        nn.feed({"x": x_test_batch})
        y_pred_batch = nn.forward()

        acc = Accuracy(y_pred_batch, y_test_batch)
    
        if i%100 == 0:
            print(f'\r[Iteration {i:5d}] Loss={loss:.4f} | Acc={acc:.3f}')

    nn.feed({"x":x_test})
    y_prob = nn.forward()
    total_acc = Accuracy(y_prob, y_test)
    print("Total Accuracy:", total_acc)

    plt.plot(loss_rec)
    plt.show()
