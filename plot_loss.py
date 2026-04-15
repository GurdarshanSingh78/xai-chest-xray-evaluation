import matplotlib.pyplot as plt

epochs = [1, 2, 3]
loss = [0.2656, 0.1373, 0.0958]

plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title('ResNet-50 Training Convergence (Scrambled Labels)')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.xticks([1, 2, 3])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
print("✅ Saved loss_curve.png")