import matplotlib.pyplot as plt

losses = [0.9, 0.7, 0.5]  # replace with your actual losses

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
