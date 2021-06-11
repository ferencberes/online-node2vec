import random, functools
import numpy as np
from sympy import nextprime

# primes
def is_prime(n):
    if n % 2 == 0:
        return False
    else:
        for num in range(3,int(np.sqrt(n))+1,2):
            if n % num == 0:
                return False
        return True

# hash functions
def modulo_hash(p,x):
    return x%p

def map_hash(a, b, n, x):
    return (a*x + b) % nextprime(n)

def multiply_hash(alpha, b, x):
    if alpha >= 1:
        raise RuntimeError("'alpha' must be smaller than 1!")
    frac, _ = np.modf(alpha * x)
    return int(np.floor(frac*b))

# generators
class ModHashGenerator():
    def __init__(self, max_value=200000):
        self.max_value = max_value
    
    def __str__(self):
        return "modhash%i" % self.max_value
    
    def generate(self, num):
        numbers = list(range(1,self.max_value+1))
        random.shuffle(numbers)
        hash_functions, primes = [], [] 
        for p in numbers:
            if is_prime(p):
                primes.append(p)
                hash_functions.append(functools.partial(modulo_hash,p))
                if len(hash_functions) == num:
                    break
        #print(primes)
        return hash_functions
    
class MapHashGenerator():
    def __init__(self, max_value=200000):
        self.max_value = max_value
    
    def __str__(self):
        return "maphash%i" % self.max_value
    
    def generate(self, num):
        numbers = list(range(1,self.max_value+1))
        random.shuffle(numbers)
        hash_functions, primes = [], [] 
        for p in numbers:
            if is_prime(p):
                c = p
                break
        hash_functions = []
        for _ in range(num):
            a = np.random.randint(1, self.max_value)
            b = np.random.randint(1, self.max_value)
            hash_functions.append(functools.partial(map_hash, a, b, self.max_value))
        return hash_functions
    
class MulHashGenerator():
    def __init__(self, max_value=200000):
        self.max_value = max_value
    
    def __str__(self):
        return "mulhash%i" % self.max_value
    
    def generate(self, num):
        numbers = list(range(1,self.max_value+1))
        random.shuffle(numbers)
        hash_functions, primes = [], [] 
        for p in numbers:
            if is_prime(p):
                c = p
                break
        hash_functions = []
        for _ in range(num):
            alpha = np.random.random()
            b = np.random.randint(1, self.max_value)
            hash_functions.append(functools.partial(multiply_hash, alpha, b))
        return hash_functions