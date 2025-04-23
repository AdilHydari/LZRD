import argparse
import random
from collections import defaultdict
import os

def build_markov_model(text, order=3):
    """Build a Markov model from input text."""
    model = defaultdict(list)
    for i in range(len(text) - order):
        key = text[i:i+order]
        next_char = text[i+order]
        model[key].append(next_char)
    return model

def generate_text(model, seed, size, order=3):
    """Generate text using a Markov model."""
    current = seed
    result = current
    
    for _ in range(size - order):
        if current in model and model[current]:
            next_char = random.choice(model[current])
            result += next_char
            current = current[1:] + next_char
        else:
            # If we reach a dead end, reset with the seed
            next_char = random.choice(seed)
            result += next_char
            current = current[1:] + next_char
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Train Markov model on text and generate samples')
    parser.add_argument('--input', '-i', required=True, help='Input text file for training')
    parser.add_argument('--output', '-o', default='markov_text.bin', help='Output file')
    parser.add_argument('--size', '-s', type=int, default=10000, help='Size in bytes to generate')
    parser.add_argument('--order', type=int, default=3, help='Order of the Markov model')
    parser.add_argument('--seed', help='Initial seed text. If not provided, will use first few chars from input')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Read the input file
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    print(f"Training Markov model (order {args.order}) on {len(text)} characters")
    model = build_markov_model(text, args.order)
    
    # Use provided seed or get from input text
    if args.seed:
        seed_text = args.seed
    else:
        seed_text = text[:args.order]
    
    print(f"Generating {args.size} bytes with seed '{seed_text}'")
    generated_text = generate_text(model, seed_text, args.size, args.order)
    
    with open(args.output, 'wb') as f:
        f.write(generated_text.encode('utf-8'))
    
    print(f"Generated text saved to {args.output}")
    
    # Print a sample
    print("\nSample of generated text:")
    sample_size = min(200, args.size)
    print(generated_text[:sample_size] + "...")

if __name__ == "__main__":
    main() 