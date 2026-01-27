toy = {"work_id": "W1", "journal_id": "J1", "publication_year": 1, "referenced_works": [], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}, {"author_id": "A2", "institution_id": ["I2"]}]}
{"work_id": "W2", "journal_id": "J2", "publication_year": 1, "referenced_works": ["W1"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}]}
{"work_id": "W3", "journal_id": "J1", "publication_year": 2, "referenced_works": ["W1", "W2"], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}]}
{"work_id": "W4", "journal_id": "J3", "publication_year": 2, "referenced_works": ["W1", "W3"], "authorships": [{"author_id": "A2", "institution_id": ["I2"]}, {"author_id": "A4", "institution_id": ["I1"]}]}
{"work_id": "W5", "journal_id": "J2", "publication_year": 2, "referenced_works": ["W2"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}]}
{"work_id": "W6", "journal_id": "J1", "publication_year": 3, "referenced_works": ["W3", "W4", "W5"], "authorships": [{"author_id": "A1", "institution_id": ["I1"]}, {"author_id": "A4", "institution_id": ["I1"]}]}
{"work_id": "W7", "journal_id": "J3", "publication_year": 3, "referenced_works": ["W4", "W6"], "authorships": [{"author_id": "A2", "institution_id": ["I2"]}]}
{"work_id": "W8", "journal_id": "J2", "publication_year": 3, "referenced_works": ["W1", "W5"], "authorships": [{"author_id": "A3", "institution_id": ["I3"]}, {"author_id": "A4", "institution_id": ["I1"]}]}


print(toy)