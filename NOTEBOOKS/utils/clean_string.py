import unidecode
import string
import re
def clean_string(s) -> str:
    """Cleans and formats a given string by removing punctuation and normalizing whitespace.

    This function processes the input string by converting it to lowercase, removing punctuation, 
    and ensuring that multiple spaces are reduced to a single space. It also capitalizes the first 
    letter of the resulting string. If the input is None or empty, it returns the input unchanged.

    Args:
        s (str): The input string to be cleaned.

    Returns:
        str: The cleaned and formatted string, or the original input if it is None or empty.
    """
def clean_string(s) -> str:
    if s is None or len(s) < 1:
        return s
    return re.sub(' +', ' ',  ''.join([c if c not in PUNC else ' ' for c in unidecode(s).lower()])).strip().capitalize()