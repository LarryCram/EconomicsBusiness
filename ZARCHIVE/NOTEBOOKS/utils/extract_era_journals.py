from abc import ABCMeta, abstractmethod
import pandas as pd
import diskcache as dc
from pathlib import Path

class IExtractEraJournals(metaclass=ABCMeta):
    """
    abstract class for interface management
    """

    @classmethod
    @abstractmethod
    def get_era_journal_table(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_era_journal_table(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def filter_era_journal_table(cls):
        raise NotImplementedError


class ExtractEraJournals(IExtractEraJournals):
    """
    concrete class to read ERA journals, FOR codes and OA sources and assemble them
    """

    def __init__(self, repository=None, refresh=None):
        self.repository = repository
        self.filtered_journals = None
        self.summary_journal_table = None
        self.sources = None
        self.journals = None
        self.for_codes = None
        self.for_dict = None
        self.refresh = refresh

    def get_era_journal_table(self):
        """
        read the ERA/FOR table if it exists, else make it
        """
        pkl_path_journals = f'data/{self.repository}_journals.pkl'
        if Path(pkl_path_journals).exists() and not self.refresh:
            self.journals = pd.read_pickle(pkl_path_journals)

        pkl_path_for = f'data/{self.repository}_for_codes.pkl'
        if Path(pkl_path_for).exists() and not self.refresh:
            self.for_codes = pd.read_pickle(pkl_path_for)
            self.for_dict_maker()
            return self

        self.build_era_journal_table()
        self.for_dict_maker()
        self.journals.to_pickle(pkl_path_journals)
        self.for_codes.to_pickle(pkl_path_for)
        self.journals.to_excel(pkl_path_journals.replace('.pkl', '.xlsx'))
        self.for_codes.to_excel(pkl_path_for.replace('.pkl', '.xlsx'))
        return self

    def build_era_journal_table(self):
        """
        build the ERA table and save it to disk
        """
        self.read_era_journal_list()
        self.attach_for_to_journal()
        # self.attach_issn_to_journal()
        self.get_oa_source_ids()
        return self

    def read_era_journal_list(self):
        """
        read the ERA journal list and FOR cade table from the ARC Excel file
        """
        in_file = './data/ERA 2023 Submission Journal List.xlsx'
        sheets = pd.read_excel(in_file, sheet_name=['ERA2023 Submission Journal List', 'FoR Codes'])
        self.journals, self.for_codes = sheets.values()
        self.journals.columns = [c.replace(' ', '_').lower() for c in self.journals.columns]
        self.journals = self.journals
        print(self.journals.head())
        print(self.for_codes.head())
        return self

    def for_dict_maker(self):
        """
        construct a FOR table. Map 'MD' -> 99 and 2D xx -> xx00. FOR Codes are INTEGERS
        """
        self.for_codes.columns = [c.lower().replace(' ', '_') for c in self.for_codes.columns]
        self.for_codes['for_code'] = [99 if c == 'MD' else c for c in self.for_codes.for_code]
        self.for_codes['for_code'] = [int(c) * 100 if int(c) < 100 else int(c) for c in self.for_codes.for_code]
        self.for_dict = dict(zip(self.for_codes.for_code, self.for_codes.for_description))
        return self

    def attach_for_to_journal(self):
        """
        attach list of its FOR codes to every ERA journal
        """
        for_codes = []
        for f1, f2, f3 in zip(self.journals.for_1, self.journals.for_2, self.journals.for_3):
            lst = []
            for f in (f1, f2, f3):
                if f == f:
                    if f == 'MD':
                        f = 99
                    f = int(f)
                    if f < 100:
                        f *= 100
                    # lst.append(str(f).zfill(4))
                    lst.append(f)
            for_codes.append(lst)
        self.journals = self.journals.drop(columns=[c.lower() for c in self.journals.columns if 'for' in c.lower()])
        self.journals['for_codes'] = for_codes
        return self

    def get_oa_source_ids(self):
        """
        match ERA journal to OA source using ISSN and then TITLE
        """
        self.extract_sources()
        issn_dict = dict(zip(self.sources.issn, self.sources.id))
        title_dict = dict(zip(self.sources.display_name.str.lower(), self.sources.id))
        for row in self.journals.itertuples(index=True):
            self.journals.loc[row.Index, 'source_id'] = None
            self.journals.loc[row.Index, 'issn'] = None
            
            hold = self.journals.loc[row.Index, [c for c in self.journals.columns if 'issn' in c]].tolist()
            self.journals.loc[row.Index, 'issns'] = '|'.join([item for item in hold if isinstance(item, str)])
            if not hold:
                print(f'ERA JOURNAL HAS NO ISSNS {row = }')
                raise SystemExit
            for i in hold:
                if i in issn_dict:
                    self.journals.loc[row.Index, 'source_id'] = issn_dict.get(i)
                    self.journals.loc[row.Index, 'issn'] = i
                    break
                if row.title.lower() in title_dict:
                    self.journals.loc[row.Index, 'source_id'] = title_dict.get(row.title.lower())
                    self.journals.loc[row.Index, 'issn'] = None
                    break
                else:
                    self.journals.loc[row.Index, 'source_id'] = None
                    self.journals.loc[row.Index, 'issn'] = None
        self.journals = self.journals.drop(columns=[c for c in self.journals.columns if 'issn_' in c])
        return self

    def extract_sources(self):
        """
        load the OA sources list from cache_pandas_core
        """
        for path in [r'C:/Users/LC.RSPE-056310/Dropbox/RESEARCH/openalex-snapshot/.cache_pandas_core',\
                      r'C:/Users/thecr/Dropbox/RESEARCH/openalex-snapshot/.cache_pandas_core']:
            print(f'{path = } {Path(path).exists() = }')
            if Path(path).exists():                      
                cache_core = dc.Cache(path, size_limit=int(100e9))
                continue
        self.sources = cache_core['sources'][['id', 'issn', 'display_name']].explode('issn')
        print(self.sources.head())
        return self

    def report(self):
        """
        report on the ERA journal list mapped to FOR and OA sources
        """
        self.summarise_journal_table_by_for_code()
        print(f'{self.journals.era_journal_id.nunique() = }')
        print(f'{self.journals.era_journal_id.count() = }')
        print(f'{self.journals.title.nunique() = }')
        print(f'{self.journals.title.count() = }')
        return self

    def summarise_journal_table_by_for_code(self):
        """
        make a pd Dataframe for the ERA table
        """
        journals = self.journals.explode('for_codes')
        for_codes = sorted(list(set(journals.for_codes)))
        hold = []
        for code in for_codes:
            temp = journals[journals.for_codes == code]
            t1 = temp.era_journal_id.nunique()
            t2 = temp.issn.nunique()
            t3 = temp.source_id.nunique()
            hold.append([code, self.for_dict.get(code), len(temp), t1, t2, t3])
        cols = ['code', 'description', 'ERA_count', 'journal_count', 'with_ISSN', 'with_source_id']
        self.summary_journal_table = pd.DataFrame(hold, columns=cols)
        print(f'{self.summary_journal_table.shape = }\n{self.summary_journal_table.head()}')
        return self

    def filter_era_journal_table(self, code_list=None):
        """
        from full matched set select the target set by FOR code
        """
        if isinstance(code_list, int):
            code_list = [code_list, ]
        mask = [not set(code_list).isdisjoint(codes) for codes in self.journals.for_codes]
        self.filtered_journals = self.journals[mask]
        self.filtered_journals = self.filtered_journals  #.drop(columns=['issns'], errors='ignore')
        msc_journals = self.extract_msc_journals()
        wos_journals = self.extract_wos_journals()
        self.filtered_journals = pd.concat([self.filtered_journals, msc_journals, wos_journals], axis=0).reset_index(drop=True)
        return self
    
    def extract_msc_journals(self):
        # msc_journals = pd.read_excel(r'C:/Users/LC.RSPE-056310/Projects/MathsBibliometrics/data/msc_from_oa.xlsx')
        msc_journals = pd.read_excel(r'./data/msc_in_oa_not_era.xlsx')
        msc_journals['era_journal_id'] = range(30000, 30000+len(msc_journals))
        cols = ['era_journal_id', 'title', 'for_codes',  'source_id', 'issn']
        msc_journals = msc_journals[cols]
        msc_journals['for_codes'] = msc_journals['for_codes'].astype(object)
        for row in msc_journals.itertuples():
            msc_journals.at[row.Index, 'for_codes'] = [4999]
        print(msc_journals.info())
        print(msc_journals.head())
        return msc_journals
    
    def extract_wos_journals(self):
        # msc_journals = pd.read_excel(r'C:/Users/LC.RSPE-056310/Projects/MathsBibliometrics/data/msc_from_oa.xlsx')
        wos_journals = pd.read_excel(r'./data/wos_in_oa_not_era.xlsx')
        wos_journals['era_journal_id'] = range(60000, 60000+len(wos_journals))
        cols = ['era_journal_id', 'title', 'for_codes',  'source_id', 'issn']
        wos_journals = wos_journals[cols]
        wos_journals['for_codes'] = wos_journals['for_codes'].astype(object)
        for row in wos_journals.itertuples():
            wos_journals.at[row.Index, 'for_codes'] = [4998]
        print(wos_journals.info())
        print(wos_journals.head())
        return wos_journals


