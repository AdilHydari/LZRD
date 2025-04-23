import random
import argparse
import string
from collections import defaultdict

def generate_skewed_text(size, skew_factor=0.7):
    """Generate random text with a skewed distribution of characters."""
    # Create a skewed distribution for printable ASCII
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    weights = []
    
    # Apply skew factor to make some characters more common than others
    for i in range(len(chars)):
        weight = skew_factor ** (i / 10)  # Exponential decay
        weights.append(weight)
        
    # Normalize weights
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # Generate text based on weights
    result = ''.join(random.choices(chars, weights=weights, k=size))
    return result

def generate_markov_text(size, order=1, seed_text=None):
    """Generate text using a simple Markov chain model."""
    if not seed_text:
        seed_text = """
        This is a sample text that will be used to train a simple Markov model.
        The range coder should be able to compress text generated from this model
        effectively because it will have statistical regularities. The more text
        we use for training, the more realistic the output will be.
        """
    
    # Build Markov model from seed text
    model = defaultdict(list)
    for i in range(len(seed_text) - order):
        key = seed_text[i:i+order]
        next_char = seed_text[i+order]
        model[key].append(next_char)
    
    # Generate text using the model
    current = seed_text[:order]
    result = current
    
    for _ in range(size - order):
        if current in model and model[current]:
            next_char = random.choice(model[current])
            result += next_char
            current = current[1:] + next_char
        else:
            # If we reach a dead end, add a random character and continue
            next_char = random.choice(seed_text)
            result += next_char
            current = current[1:] + next_char
    
    return result

def generate_repeated_patterns(size, pattern_length=10, num_patterns=5):
    """Generate text with repeated patterns."""
    # Create a few random patterns
    patterns = []
    for _ in range(num_patterns):
        pattern = ''.join(random.choices(string.ascii_letters + string.digits, k=pattern_length))
        patterns.append(pattern)
    
    # Repeat patterns with some randomness
    result = ""
    while len(result) < size:
        pattern = random.choice(patterns)
        result += pattern
    
    return result[:size]

def main():
    parser = argparse.ArgumentParser(description='Generate test data for range coding')
    parser.add_argument('--output', '-o', default='test_data.bin', help='Output file')
    parser.add_argument('--size', '-s', type=int, default=10000, help='Size in bytes')
    parser.add_argument('--type', '-t', choices=['skewed', 'markov', 'patterns'], 
                      default='markov', help='Type of data to generate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--order', type=int, default=3, help='Order for Markov model')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    if args.type == 'skewed':
        data = generate_skewed_text(args.size)
    elif args.type == 'markov':
        data = generate_markov_text(args.size, args.order)
    elif args.type == 'patterns':
        data = generate_repeated_patterns(args.size)
    
    with open(args.output, 'wb') as f:
        f.write(data.encode('utf-8'))
    
    print(f"Generated {args.size} bytes of {args.type} data to {args.output}")
    
    # Print character frequency statistics
    char_freq = defaultdict(int)
    for char in data:
        char_freq[char] += 1
    
    print("\nTop 10 most frequent characters:")
    for char, freq in sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        if char in string.whitespace:
            char_repr = repr(char)[1:-1]  # Remove quotes
        else:
            char_repr = char
        print(f"'{char_repr}': {freq} ({freq/args.size:.2%})")

if __name__ == "__main__":
    main() 