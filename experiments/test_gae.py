from utils.gae import compute_gae


rewards = [5, 2, 8]

values = [10, 12, 7]

dones = [0, 0, 1]

advantages = compute_gae(
    rewards,
    values,
    dones
)

print("Advantages:")
print(advantages)