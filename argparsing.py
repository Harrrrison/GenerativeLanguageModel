import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')

    parser.add_argument('-bs', type=str, required=True, help='please provide a batch_size')

def main():
    args = parse_args()

    print(f'batch size: {args.bs}')

if __name__ == '__main__':
    main()