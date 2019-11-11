import pickle
import matplotlib.pyplot as plt
history = pickle.load(open("./snakenetHistory", "rb" ))

print(history)

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Loss over time')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history['precision'])
plt.plot(history['val_precision'])
plt.title('Precision over time')
plt.ylabel('precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()