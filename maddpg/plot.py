import pickle as pkl
import matplotlib.pyplot as plt

f = open('experiments/learning_curves/5-9-simple-v1_rewards.pkl','rb')
data = pkl.load(f) # list

plt.plot(data)
plt.ylabel("Simple crypto scenario")
plt.show()
print(data)
