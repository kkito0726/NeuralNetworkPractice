from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def read_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    t_train = t_train.astype('float32')
    x_test = x_test.astype('float32') / 255
    t_test = t_test.astype('float32')

    classes = 10
    t_train = to_categorical(t_train, classes)
    t_test = to_categorical(t_test, classes)

    return x_train, t_train, x_test, t_test


def model_cnn(activation="relu", output_neuron=10):
    model = Sequential([
        # 第1層 (入力層)
        Conv2D(32, (3, 3), input_shape=(28,28,1), activation=activation),

        # 第2層 (中間層)
        Conv2D(32, (3, 3), activation=activation),
        MaxPool2D(pool_size=(2, 2)),

        # 第3層 (中間層)
        Conv2D(32, (3, 3), activation=activation),
        MaxPool2D(pool_size=(2, 2)),

        # 出力層
        Flatten(),
        Dense(1024, activation=activation),
        Dense(output_neuron, activation="softmax")
    ])

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

    return model


def plot_result(model, x_test, history, epochs=10):
    plt.figure(figsize=(7,7))
    plt.subplots_adjust(hspace=0.5)
    for n in range(0,64):
        pre_num = n
        res = model.predict(x_test[pre_num:pre_num+1])
        plt.subplot(8, 8, n+1)
        plt.imshow(x_test[n], cmap="Greys")
        plt.title(res.argmax())
        plt.axis('off')
    plt.show()

    x = np.arange(1, epochs+1)
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(x, history.history["accuracy"], label="Accuracy")
    ax.plot(x, history.history["val_accuracy"], label="Val_Accuracy")
    ax.legend()

    ax = fig.add_subplot(1,2,2)
    ax.plot(x, history.history["loss"], label="Loss")
    ax.plot(x, history.history["val_loss"], label="Val_loss")
    ax.legend()
    plt.show()


def main():
    x_train, t_train, x_test, t_test = read_data()

    model = model_cnn()
    model.summary()

    history = model.fit(x_train, t_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)
    
    plot_result(model=model, x_test=x_test, history=history)



if __name__ == "__main__":
    main()