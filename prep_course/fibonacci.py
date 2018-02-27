# Written by Emile Badran on January/2018

def get_range():
    fib_range = input('Please indicate the maximum number for your Fibonacci sequence: ')
    while True:
        try:
            fib_range = int(fib_range)
            break
        except KeyboardInterrupt:
            print("Naughty, naughty. We're stopping here and calling the FBI.")
            return False
        except:
            print("Please insert an integer only.")
            fib_range = input('Please indicate the maximum number for your Fibonacci sequence: ')
    if False:
        break
    else:
        return int(fib_range)

def generate_fib(num):
    result = []
    for n in range(num):
        if n == 0:
            result.append(0)
        elif n == 1:
            result.append(1)
        else:
            result.append((n - 2) + (n - 1))
    return result

def display_values(values):
    for i in values:
        print(i)

def main():
    range_limit = get_range()
    fibonacci = generate_fib(range_limit)
    display_values(fibonacci)

main()
