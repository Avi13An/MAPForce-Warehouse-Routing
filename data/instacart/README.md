# Instacart data

`product_frequencies.csv` holds how often each of the 49,677 products was
ordered, counted from `order_products__prior.csv` of the *Instacart Market
Basket Analysis* data (32,434,489 order lines). Columns: `product_id`,
`count`, most-ordered first. Every reported run uses it; with this file
present the loader in `src/demand/orders.py` never needs the raw download.

To rebuild the file from the raw data:

1. Download the data from Kaggle (sign in first). The file needed is
   `order_products__prior.csv`, about 578 MB.
2. Put it in this folder as `order_products__prior.csv`, as
   `order_products__prior.csv.zip`, or leave Kaggle's `archive.zip` as it is.
   The loader looks inside any zip here. Raw files are git-ignored.
3. Delete `product_frequencies.csv` and run any experiment once. The loader
   counts the orders and writes the file again.

Without `product_frequencies.csv` and without the raw data, the code falls
back to a synthetic Zipf distribution and marks every result row
`demand_source = synthetic`.
