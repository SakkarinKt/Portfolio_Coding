import math

def is_prime(num):
    if type(num) != int:
        raise TypeError('Invalid input type')
    if num < 0:
        raise ValueError('Input can"t be negative number, please try again')
    for i in range(2, int(math.sqrt(num)) + 1):
        if num%i == 0:
            return False
    return True
