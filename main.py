
def conversion_rate(users, purchases):
    if users <= 0:
        raise ValueError("Users must be greater than zero.")

    if purchases < 0 or purchases > users:
        raise ValueError("Purchases must be between zero and users.")

    return (purchases / users) * 100


users_a = 1000
purchases_a = 50

users_b = 1000
purchases_b = 60

rate_a = conversion_rate(users_a, purchases_a)
rate_b = conversion_rate(users_b, purchases_b)

print(f"Version A conversion rate: {rate_a:.2f}%")
print(f"Version B conversion rate: {rate_b:.2f}%")
print(f"Absolute improvement: {rate_b - rate_a:.2f} percentage points")
print(f"Relative improvement: {(rate_b - rate_a) / rate_a * 100:.2f}%")
