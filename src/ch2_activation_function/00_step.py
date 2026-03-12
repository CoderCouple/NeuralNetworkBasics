def step(x , threshold):
    if x < threshold:
        return 0
    elif x> threshold:
        return 1;
    else:
        return None


print(step(5, 1))
print(step(4, 1))
print(step(-2, 1))
print(step(1, 1))
print(step(0, 1))
