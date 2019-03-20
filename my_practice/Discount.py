arr = [1,1,1]
discount_rate = 0.99

def discount(array, rate):
    sum = 0
    result = []
    for i in range(0, len(array)):
        sum = sum * rate + array[len(array) - i - 1]
        result.insert(0, sum)

    return result

print(discount(arr, discount_rate))