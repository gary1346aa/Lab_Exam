import numpy as np
import sys
import matplotlib.pyplot as plt
import MNISTtools
import NeuralNetwork

def OneHot(y):
    return np.eye(10, dtype = np.float32)[y]

def Noise(x):
    mean = 0
    stddev = 0.3
    noise = np.random.normal(mean, stddev, x.shape)
    return x + noise

if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    y_train, _ = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    y_test, _ = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Data Processing
    y_train = y_train.astype(np.float32) / 255.
    y_test = y_test.astype(np.float32) / 255.

    x_train = []
    x_test = []
    for y in y_train:
        x_train.append(Noise(y))
    for y in y_test:
        x_test.append(Noise(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)


    # Create NN Model)
    # Set dropout rate to 0 if don't want to use Dropout
    nn = NeuralNetwork.dAE_NN(784, 128, 784, "sigmoid", 0.5)

    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(10001):
        # Sample data batch
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

        # Evaluation
        batch_id = np.random.choice(x_test.shape[0], batch_size)
        x_test_batch = x_test[batch_id]
        y_test_batch = y_test[batch_id]
        nn.feed({"x": x_test_batch})
        y_pred_batch = nn.forward(train=False)

        # Print status
        if i%100 == 0:
            print(f'\r[Iteration {i:5d}] Loss={loss:.4f}')

    # Test and Visualize
    nn.feed({"x":x_test})
    y_prob = nn.forward(train=False)
    fig, axs = plt.subplots(4, 4)

    for i in range(8):
        axs[(i//4)*2, i%4].imshow(np.reshape(x_test[i], (28, 28)), cmap='gray')
        axs[(i//4)*2, i%4].axis('off')
    for i in range(8):
        axs[(i//4)*2 + 1, i%4].imshow(np.reshape(y_prob[i], (28, 28)), cmap='gray')
        axs[(i//4)*2 + 1, i%4].axis('off')

    plt.show()

    Filter = nn.W1.T

    fig, axs = plt.subplots(4, 4)

    for i in range(16):
        axs[i//4, i%4].imshow(np.reshape(Filter[i], (28, 28)), cmap = 'gray')
        axs[i//4, i%4].axis('off')

    plt.show()