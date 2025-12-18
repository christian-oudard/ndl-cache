import pandas as pd


class SEPTable:
    table_name = 'SHARADAR/SEP'
    # Which columns form the key for each row?
    index_columns = ['date', 'ticker']
    # What columns are requested from the server?
    query_columns = ['open', 'low', 'high', 'close', 'volume', 'closeadj', 'closeunadj']
    # What is stored in the database?
    immutable_columns = ['split_factor', 'split_dividend_factor', 'openunadj', 'lowunadj', 'highunadj', 'closeunadj', 'volumeunadj']

    def immutable_data(queried):
        immutable = pd.DataFrame()
        immutable['split_factor'] = queried['close'] / queried['closeunadj']
        immutable['split_dividend_factor'] = queried['closeadj'] / queried['closeunadj']
        for column in ['open', 'low', 'high', 'close', 'volume']:
            immutable[f'{column}unadj'] = immutable[f'{column}'] / immutable['split_factor']

        assert immutable['closeunadj'] == queried['closeunadj']

        return immutable

    def derived_data(immutable):
        derived = pd.DataFrame()
        for column in ['open', 'low', 'high', 'close', 'volume']:
            derived[f'{column}'] = immutable[f'{column}unadj'] * immutable['split_factor']
            derived[f'{column}adj'] = immutable[f'{column}unadj'] * immutable['split_dividend_factor']
        return derived



def main():
    sep = SEPTable()
    sep.query('2024-03-04', 'AAPL', 'openunadj')



if __name__ == "__main__":
    main()
