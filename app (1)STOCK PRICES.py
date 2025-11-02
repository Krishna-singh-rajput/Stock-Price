import sys
import argparse
from utils import fetch_history, save_csv, plot_close

def main():
    parser = argparse.ArgumentParser(description="Fetch and plot stock price history")
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD or leave empty for today")
    parser.add_argument("--out", default=None, help="CSV output path (optional)")
    args = parser.parse_args()

    df = fetch_history(args.ticker, args.start, args.end)
    if df is None or df.empty:
        print("No data returned for", args.ticker)
        sys.exit(1)

    if args.out:
        save_csv(df, args.out)
        print(f"Saved CSV to {args.out}")

    plot_close(df, args.ticker)

if __name__ == "__main__":
    main()