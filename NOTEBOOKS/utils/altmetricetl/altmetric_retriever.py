import pandas as pd
from time import sleep
import os
import re
from unidecode import unidecode


class AltmetricRetriever:

    import hashlib
    import csv
    from pyaltmetric import Altmetric, Citation
    cacheDir = r'./data/.altmetric'
    print(f'instantiation of AltmetricRetriever with cache at {cacheDir}')
    altmetric = Altmetric(r"f6db5700cc57afcc4c014d6c42374125")

    def __init__(self):
        return

    def process_query(self, query=None, refresh=None):

        queryCode = self.hashlib.md5(query.encode('utf-8')).hexdigest()
        queryFile = self.cacheDir + f'\\{queryCode}'

        # print(f'process_query -> query {query} -> hash {queryCode} -> file {queryFile} -> refresh {refresh}')
        if refresh:
            record = self.retrieve_web_record(query=query)
            self.cache_web_record(record=record, file=queryFile)

        else:
            if os.path.isfile(queryFile):
                record = self.retrieve_disk_record(file=queryFile)
                # print(f'query apparently exists {query} -> {queryFile}: {record}')
                if not record:
                    print(f'queryFile apparently exists but {query} -> {queryFile} FAILS')
                    return None
            else:
                print(f'file {queryFile} not found so retrieve from web using query -> {query}')
                record = self.retrieve_web_record(query=query)
                # print(f'write to cache {record = }')
                self.cache_web_record(record=record, file=queryFile)
                if not record:
                    print(f'query {query} FAILS')
                    return None
        return record

    def retrieve_disk_record(self, file=None):
        with open(file, 'r') as inFile:
            for line in self.csv.reader(inFile, dialect='excel-tab'):
                return line

    def retrieve_web_record(self, query=None):
        item = self.altmetric.doi(query)
        if not item:
            print(f'retrieved {item = } return [{query}, None]')
            return [query, None]
        # print(f'retrieved web record {query = } {item = }')
        citation = self.Citation(item)

        result = citation.get_fields('doi', 'title', 'journal', 'score', 'readers_count',
                                     'cited_by_posts_count', 'cited_by_tweeters_count',
                                     'cited_by_fbwalls_count', 'cited_by_gplus_count',
                                     'cited_by_rdts_count', 'cited_by_feeds_count')
        result[1] = unidecode(str(result[1]))
        # print(f'retrieved record from web {result[0] = }')
        return result

    def cache_web_record(self, record=None, file=None):
        # print(f'write cache -> {record = }')
        with open(file, 'w') as outFile:
            self.csv.writer(outFile, dialect='excel-tab').writerow(record, )
            if len(record) == 2:
                print(f'wrote empty record {record = }')
        return

    def getAltmetrics(self, doi_list=None, refresh=None):

        cols = ['doi', 'title', 'journal', 'score', 'readers_count',
                                             'posts_count', 'tweet_count',
                                             'fb_count', 'gplus_count',
                                             'reddit_count', 'feeds_count']

        altmetrics_df = pd.DataFrame(columns=cols)
        altmetric_list = []
        for j, doi in enumerate(doi_list):
            # sleep(0.01)
            # search for article using doi
            result = self.process_query(query=doi, refresh=False)
            # write row to list
            if result:
                altmetric_list.append(result)
                if j % 100 == 0:
                    print(f'j = {j} -> result {result}')

        altmetrics_df = altmetrics_df.from_records(altmetric_list, columns=cols)
        print(f'{altmetrics_df.shape = }\n{altmetrics_df.head()}')
        altmetrics_df.info()

        return altmetrics_df

