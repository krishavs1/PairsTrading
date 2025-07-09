import re

def parse_pairs(file_path):
    pairs = set()
    with open(file_path, 'r') as f:
        for line in f:
            # Extract ticker symbols (2–5 uppercase letters)
            tickers = re.findall(r'\b[A-Z]{2,5}\b', line)
            if len(tickers) >= 2:
                pairs.add((tickers[0], tickers[1]))
    return pairs

def compare_pairs(file1, file2):
    set1 = parse_pairs(file1)
    set2 = parse_pairs(file2)

    common = set1 & set2
    only_in_file1 = set1 - set2
    only_in_file2 = set2 - set1

    print("Common pairs:")
    for p in sorted(common):
        print(f"  {p[0]}/{p[1]}")

    print("\nOnly in first file:")
    for p in sorted(only_in_file1):
        print(f"  {p[0]}/{p[1]}")

    print("\nOnly in second file:")
    for p in sorted(only_in_file2):
        print(f"  {p[0]}/{p[1]}")

    return common, only_in_file1, only_in_file2

if __name__ == "__main__":
    compare_pairs(
        "engle_integrated_pairs.txt",
        "johansen_integrated_pairs.txt"
    )
