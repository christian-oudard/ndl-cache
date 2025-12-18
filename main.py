from cache import SEPTable


def main():
    sep = SEPTable()

    # Query AAPL data for a date range
    df = sep.query(
        columns=['openunadj', 'open', 'openadj', 'closeunadj', 'close', 'closeadj', 'volumeunadj', 'volume', 'volumeadj'],
        ticker='AAPL',
        date_gte='2020-08-25',
        date_lte='2020-09-05'
    )

    print("=== SEP Query Result ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
