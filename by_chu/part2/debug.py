import argparse

parser = argparse.ArgumentParser(description="与えられた数値を二乗します。")

parser.add_argument("number", type=int, help="二乗したい値を入力してください", action="store")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="計算過程を詳細に表示します")

args = parser.parse_args()

result = args.number ** 2

if args.verbose:
    print(f"{args.number}の二乗を計算します")
    print(f"結果は{result}です")
else:
    print(result)
