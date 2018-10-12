-- count total sales by product and join three tables to get the product name and state
WITH sales_count
AS (
	SELECT
		transactions.customer_id cust_id,
		customers.state cust_state,
		transactions.product_id prod_id,
		products.product_name prod_name,
		COUNT(transactions.product_id) prod_count
	FROM
		transactions
	JOIN
		customers
	ON
		transactions.customer_id = customers.customer_id
   JOIN
		products
   ON
        products.product_id = transactions.product_id
	WHERE
		transactions.payment_success = 1
	GROUP BY cust_state, prod_name
	ORDER BY cust_state ASC
)

-- select the max sales count
SELECT
	prod_name,
	cust_state,
	MAX(prod_count) sale_count
FROM
   sales_count
GROUP BY
	cust_state;


---------


-- join two tables to get user name and transaction amounts and sum the total purchase
WITH
	total_purchases
AS (
	SELECT
		customers.name cust_name,
		customers.state cust_state,
		SUM(transactions.transact_amt) amount
	FROM
		customers
	JOIN
		transactions
	ON
		customers.customer_id = transactions.customer_id
	GROUP BY
		cust_state,
		cust_name
	ORDER BY cust_state, amount
)

-- rank users by total purchase and state
SELECT *
FROM
	total_purchases
AS
	table_copy_a
WHERE table_copy_a.cust_name
IN (
      SELECT table_copy_b.cust_name
      FROM total_purchases AS table_copy_b
      WHERE table_copy_a.cust_state = table_copy_b.cust_state
      ORDER BY table_copy_b.amount DESC
      LIMIT 5
      )
;



-------------


-- convert transact_at column to datetime
UPDATE transactions SET transact_at=DATE(transact_at);

-- filter all transactions from Gmail users in the past 30 days and count the number of purchases of each product
WITH
	purchase_counts
AS (
	SELECT
		transactions.customer_id cust_id,
		transactions.product_id prod_id,
		transactions.transact_at trans_date,
		COUNT(transactions.transact_amt) trans_count,
		customers.email email
	FROM
		transactions
	JOIN
		customers
	ON
		transactions.customer_id = customers.customer_id
	WHERE
		email LIKE '%gmail%' AND
		trans_date >= DATE('now','-5 day')
	GROUP BY
		cust_id,
		prod_id
	ORDER BY
		cust_id, trans_count DESC
);

-- select the five most popular items per gmail user
SELECT *
FROM
	purchase_counts
AS
	table_copy_a
WHERE table_copy_a.prod_id -- iterable variable here
IN (
     SELECT table_copy_b.prod_id -- iterable variable here
     FROM purchase_counts AS table_copy_b
     WHERE table_copy_a.cust_id = table_copy_b.cust_id -- filtering variable here
	  ORDER BY table_copy_b.trans_count DESC  ranking variable here
	  LIMIT 5
);
