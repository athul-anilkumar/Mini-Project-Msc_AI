from utils.returns import compute_returns


rewards = [5, 2, 8]

returns = compute_returns(rewards)

print("Rewards:")
print(rewards)

print("\nReturns:")
print(returns)