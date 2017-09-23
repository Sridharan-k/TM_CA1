# KE5205-TextMining-CA

1. Verify data integrity of MsiaAccidentCases.xslx (finish by 24 Sept)
Rows 2-41 (Person1)
Rows 42-81 (Person2)
Rows 82-121 (Person3)
Rows 122-161 (Person4)
Rows 162-201 (Person5)
Rows 202-240 (Person6)

2. Build a model for osha.xlsx: preprocess x[train], make sure y[train] categorization is correct
3. Categorize the cause from osha.xslx using the built model
4. Check for model accuracy
5. Additionally verify the osha.xlsx cause, normalize the final training data to contain at least 100 6. per category
6. 8 categories = 800 dataset (if 100/category), train a model and apply to osha.xslx

(#2 to #5 is iterative)

Finally, answer:
Q1: highest count(cause)
Q2: highest(extract occupation from Q1 entries)
Q3: highest(extract body part from Q1 entries)
Q4: highest(extract activities from Q1 entries)
