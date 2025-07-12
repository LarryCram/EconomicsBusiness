import pandas as pd
import re

class MakeAbstract:
    """
    A class to generate an abstract from an inverted index.

    Methods:
    - abstract(abstract_inverted_index) -> object: Generates an abstract from the provided inverted index.
    """

    def __init__(self):
        """
        Initializes the MakeAbstract class.
        """

    def abstract(self, abstract_inverted_index: dict = None) -> str:
        """
        Generates an abstract from the provided inverted index.

        Args:
            abstract_inverted_index: The inverted index containing the abstract information.

        Returns:
            object: The generated abstract.

        Raises:
            ValueError: If an exception occurs during the construction of the abstract.
        """
        if isinstance(abstract_inverted_index, (type(pd.NA), type(None), float,)):
            return ''
        try:
            return self._extracted_from_abstract(abstract_inverted_index)
        except:
            raise ValueError(f'exception in construction of abstract from inverted index ({abstract_inverted_index = })')

    def _extracted_from_abstract(self, abstract_inverted_index):
        """
        Extracts the abstract from the inverted index.

        Args:
            abstract_inverted_index: The inverted index containing the abstract information.

        Returns:
            str: The extracted abstract.
        """
        abstract_dict = {pos: word for word, posList in abstract_inverted_index.items() for pos in posList}
        abstract = ' '.join([abstract_dict[pos] for pos in sorted(abstract_dict)])
        abstract = re.sub('^abstract *?', '', abstract, count=1, flags=re.I)
        s = abstract.split(" ")
        abstract = " ".join([i for i in s if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", i)])
        return abstract.strip()
